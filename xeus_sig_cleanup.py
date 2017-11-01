from __future__ import division
import sys
sys.path.append('/home/sciortino/ML')
import profile_unc_estimation
import numpy as np
import collections
import profiletools
import MDSplus
import gptools
import os
import scipy
import Tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import profiletools
import profiletools.gui
import re
from VUV_gui_classes import VUVData,interp_max
import cPickle as pkl

# example addition

class Injection(object):
    """Class to store information on a given injection.
    """
    def __init__(self, t_inj, t_start, t_stop):
        self.t_inj = t_inj
        self.t_start = t_start
        self.t_stop = t_stop

shot=1101014019
LBO_inj=[Injection(1.25, 1.23, 1.4),]

# Get data. If this wasn't saved before, then use the following parameters to load it:
# included lines: 6
# lambda_min=2.02, lambda_max=2.04
# baseline start = 1.11, baseline end = 1.22 
# ---> then press "apply" and forcely close the GUI
try:
	with open('vuv_signals_1101014019.pkl', 'rb') as f:
            vuv_data = pkl.load(f)
except IOError:
	vuv_data = VUVData(shot, LBO_inj, debug_plots=True)
	with open('vuv_signals_1101014019.pkl', 'wb') as f:
		pkl.dump(vuv_data, f, protocol=pkl.HIGHEST_PROTOCOL)

# Extract signal in simple form
y=vuv_data.signal.y
y_clean=np.asarray([y[i] if y[i]>0 else np.array([0.0,]) for i in range(len(y))])[:,0]
y_unc=vuv_data.signal.std_y[:,0]
t=vuv_data.signal.t
# xeus=r.signals[1]
# t=xeus.t
# sig=np.atleast_2d(xeus.y)
# sig_clean=np.asarray([sig[i] if sig[i]>0 else np.array([0.0,]) for i in range(len(sig))])
# sig_unc=np.atleast_2d(xeus.std_y)

# bad=[9,10,43]
# bad=[43]
# mask=np.asarray([True if i not in bad else False for i in range(len(t))])
plt.figure()
plt.errorbar(t,y,y_unc,marker='s',mfc='red',mec='green')

def profile_fitting(x, y, err_y=None, kernel='SE', s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False, method='GPR', noiseLevel=2):
    """Interpolate profiles and uncertainties over a dense grid. Also return the maximum 
    value of the smoothed data.
    
    This function can use Gaussian process regression or splines. When the former is adopted, both the 
    mean and the standard deviation of the updated profile are returned. Use the spline interpolation
    only as a cross-check.
    
    We allow the use of several GPR kernels so as to assess their performance in confidently 
    predicting profiles and assessing their uncertainty consistency. 

    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    kernel : str, optional
        Type of kernel to be used. At this stage, we create the kernel internally, but in the future
        it would be better to do it externally and just give a gptools kernel object as an argument
        More kernels should be added over time. 
    s_guess : float, optional
        Initial guess for the signal variance. Default is 0.2.
    s_max : float, optional
        Maximum value for the signal variance. Default is 10.0
    l_guess : float, optional
        Initial guess for the covariance length scale. Default is 0.03.
    fixed_l : bool, optional
        Set to True to hold the covariance length scale fixed during the MAP
        estimate. This helps mitigate the effect of bad points. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    method : {'GPR', 'spline'}, optional
        Method to use when interpolating. Default is 'GPR' (Gaussian process
        regression). Can also use a cubic spline.
    noiseLevel : float, optional
        Initial guess for a noise multiplier. Default: 2
    """
    # grid = scipy.linspace(max(0, x.min()), min(0.08, x.max()), 1000)
    grid = scipy.linspace(x.min(), x.max(), 1000)
    if method == 'GPR':
        # hp is the hyperprior. A product of kernels is a kernel, so the joint hyperprior is 
        # just the product of hyperpriors for each of the hyperparameters of the individual 
        # priors. gptools offers convenient functionalities to create joint hyperpriors.

        # Define the kernel type amongst the implemented options. 
        if kernel=='SE':
            hp = (
            gptools.UniformJointPrior([(0, s_max),]) *
            gptools.GammaJointPriorAlt([l_guess,], [0.1,])
            )
            k = gptools.SquaredExponentialKernel(
                # param_bounds=[(0, s_max), (0, 2.0)],
                hyperprior=hp,
                initial_params=[s_guess, l_guess],
                fixed_params=[False, fixed_l]
            )
            
        elif kernel=='gibbs':
            # Set this with 
            hprior=(
                # Set a uniform prior for sigmaf
                gptools.UniformJointPrior([(0,10),])*
                # Set Gamma distribution('alternative form') for the other 4 priors of the Gibbs 1D Tanh kernel
                gptools.GammaJointPriorAlt([1.0,0.5,0.0,1.0],[0.3,0.25,0.1,0.05])
                )

            k = gptools.GibbsKernel1dTanh(
                #= ====== =======================================================================
                #0 sigmaf Amplitude of the covariance function
                #1 l1     Small-X saturation value of the length scale.
                #2 l2     Large-X saturation value of the length scale.
                #3 lw     Length scale of the transition between the two length scales.
                #4 x0     Location of the center of the transition between the two length scales.
                #= ====== =======================================================================
                initial_params=self.initial_params,
                fixed_params=[False]*5,
                hyperprior=hprior,
                )
        elif kernel=='matern':
            ValueError('Implementation not completed yet')
            k = gptools.Matern52Kernel( # this has 3 hyperparameters in 1D
                # # param_bounds=[(0, s_max), (0, 2.0)],
                # hyperprior=hp,
                # initial_params=[s_guess, l_guess],
                # fixed_params=[False, fixed_l]
            )
        elif isinstance(kernel,gptools.Kernel):
            k=kernel
        else:
            ValueError('Only the SE kernel is currently defined! Break here.')

        # Create additional noise to optimize over
        nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=np.mean(err_y)*noiseLevel,
                        fixed_noise=False, noise_bound=(np.min(err_y), np.max(err_y)*noiseLevel))#, enforce_bounds=True)
        print "noise_bound= [", np.min(err_y), ",",np.max(err_y)*noiseLevel,"]"

        gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y, noise_k=nk)
        gp.optimize_hyperparameters(verbose=True, random_starts=100)
        m_gp, s_gp = gp.predict(grid)
        i = m_gp.argmax()

    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=2*len(x)
        )(grid)
        if scipy.isnan(m_gp).any():
            print(x)
            print(y)
            print(err_y)
        i = m_gp.argmax()
    else:
        raise ValueError("Undefined method: %s" % (method,))
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        if method == 'GPR':
            a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        a.axvline(grid[i])
    
    if method == 'GPR':
        return (m_gp, s_gp, m_gp[i], s_gp[i])
    else:
        return (m_gp,m_gp[i])


# rel_unc=sig_unc/sig
#m_gp,m_gp_max = interp_max(t, y_clean, err_y=y_unc, s_guess=20, s_max=10.0, l_guess=0.5, fixed_l=False, debug_plots=True, method='GP')

m_gp, m_gp_max = profile_fitting(t, y_clean, err_y=y_unc, s_guess=0.2, s_max=10.0, l_guess=0.005, 
    fixed_l=False, debug_plots=True, method='spline',kernel='SE',noiseLevel=1)

m_gp, s_gp, m_gp_max, s_gp_max = profile_fitting(t, y_clean, err_y=y_unc, s_guess=0.2, s_max=10.0, l_guess=0.005, 
    fixed_l=False, debug_plots=True, method='GPR',kernel='SE',noiseLevel=1)

m_gp, s_gp, m_gp_max, s_gp_max = profile_fitting(t, y_clean, err_y=y_unc, s_guess=0.2, s_max=10.0, l_guess=0.005, 
    fixed_l=False, debug_plots=True, method='GPR',kernel='SE',noiseLevel=1)

m_gp, s_gp, m_gp_max, s_gp_max = profile_fitting(t, y_clean, err_y=y_unc, s_guess=0.2, s_max=10.0, l_guess=0.005, 
    fixed_l=False, debug_plots=True, method='GPR',kernel='SE',noiseLevel=1)

# m_gp, s_gp, m_gp_max, s_gp_max = profile_fitting(t, y_clean, err_y=y_unc, s_guess=0.2, s_max=10.0, l_guess=0.005, 
#     fixed_l=False, debug_plots=True, method='GPR',kernel='SE',noiseLevel=2)