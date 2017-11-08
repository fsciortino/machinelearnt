#  Given (x,y,sigma_y), check statistical consistency of uncertainties and provide an 
# improved estimate based on their spatio-temporal scattering using different machine
# learning techniques. mostly Gaussian Process Regression.
#
# F.Sciortino, 10/10/17


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
import warnings

def profile_fitting(x, y, err_y=None, optimize=True, kernel='SE', num_dim=1, s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False, method='GPR', noiseLevel=2):
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
    optimize : bool, optional
        Specify whether optimization over hyperparameters should occur or not. Default is True.
    kernel : str, optional
        Type of kernel to be used. At this stage, we create the kernel internally, but in the future
        it would be better to do it externally and just give a gptools kernel object as an argument
        More kernels should be added over time. 
    num_dim : int, optional
        Number of dimensions of the input/output data. Default is 1
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
    #grid = scipy.linspace(x.min(), x.max(), 1000)
    grid = x
    # Create empty object for results:
    res= type('', (), {})()

    if method == 'GPR':
        # hp is the hyperprior. A product of kernels is a kernel, so the joint hyperprior is 
        # just the product of hyperpriors for each of the hyperparameters of the individual 
        # priors. gptools offers convenient functionalities to create joint hyperpriors.

        # Define the kernel type amongst the implemented options. 
        if kernel=='SE':
            hprior = (
            # gptools.UniformJointPrior([(0, s_max),]) *
            gptools.GammaJointPriorAlt([2.0,], [5.0,])*
            gptools.GammaJointPriorAlt([l_guess,], [0.1,])
            )
            k = gptools.SquaredExponentialKernel(
                #= ====== =======================================================================
                #0 sigmaf Amplitude of the covariance function
                #1 l1     Small-X saturation value of the length scale.
                #2 l2     Large-X saturation value of the length scale.
                #= ====== =======================================================================
                # param_bounds=[(0, s_max), (0, 2.0)],
                hyperprior=hprior,
                initial_params=[s_guess, l_guess],
                fixed_params=[False, fixed_l]
            )
            
        elif kernel=='gibbs':
            # Set this with 
            hprior=(
                # Set a uniform prior for sigmaf
                gptools.UniformJointPrior([(0,10),])*
                # Set Gamma distribution('alternative form') for the other 4 priors of the Gibbs 1D Tanh kernel
                gptools.GammaJointPriorAlt([0.3,0.5,0.0,0.0],[0.3,0.25,0.3,0.3])
                )

            k = gptools.GibbsKernel1dTanh(
                #= ====== =======================================================================
                #0 sigmaf Amplitude of the covariance function
                #1 l1     Small-X saturation value of the length scale.
                #2 l2     Large-X saturation value of the length scale.
                #3 lw     Length scale of the transition between the two length scales.
                #4 x0     Location of the center of the transition between the two length scales.
                #= ====== =======================================================================
                initial_params=[2.0, 0.5, 0.05, 0.1, 0.5], # for random_starts!= 0, the initial state of the hyperparameters is not actually used.,
                fixed_params=[False]*5,
                hyperprior=hprior,
                )
        elif kernel=='matern52':
            hprior=( 
                gptools.GammaJointPriorAlt([2.0,0.05],[5.0,0.25])
                ) 
            k = gptools.Matern52Kernel( # this has 2 hyperparameters in 1D
                #= ===== ===========================================
                #0 sigma Prefactor to the kernel  
                #2 l1    Length scale for first dimension
                #3 l2    Length scale for second dimension
                #4 ...   More length scales for more dimensions
                #= ===== =======================================
                hyperprior=hprior,
                initial_params=[0.5, 0.5],
                fixed_params=[False]*2
            ) 
        elif kernel == 'RQ': # rational quadratic
            hprior=(   
                gptools.GammaJointPriorAlt([0.3,0.5,0.0],[1.3,1.25,1.3])
                )   
            gptools.RationalQuadraticKernel(
                #= ===== ===========================================
                #0 sigma Prefactor to the kernel  
                #1 alpha Order of kernel
                #2 l1    Length scale for first dimension
                #3 l2    Length scale for second dimension
                #4 ...   More length scales for more dimensions
                #= ===== =======================================
                hyperprior=hprior,
                initial_params=[1.0, 0.5, 1.0],
                fixed_params=[False]*3
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

        for i in range(len(y)):
            if y[i]==0:
                gp.add_data(x[i], 0, n=0, err_y=0.0)
                gp.add_data(x[i], 0, n=0, err_y=0.0)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            if optimize: 
                gp.optimize_hyperparameters(verbose=True, random_starts=200)
            else:
                print 'Optimization is turned off. Using initial guesses for hyperparameters!'

        m_gp, s_gp = gp.predict(grid,noise=True)
        res.free_params = gp.free_params[:]
        res.free_param_names = gp.free_param_names[:]
        res.free_param_bounds = gp.free_param_bounds[:]
    
        #i = m_gp.argmax()

        # Check percentage of points within 3 sd:
        points_in_1sd=0.0; points_in_2sd=0.0; points_in_3sd=0.0
        for i in range(len(y)):
            # Find value of grid that is the closest to x[i]:
            gidx = np.argmin(abs(grid - x[i]))
            if abs(m_gp[gidx]-y[i]) < s_gp[gidx]:
                points_in_1sd += 1.0
            if abs(m_gp[gidx]- y[i]) > s_gp[gidx] and abs(m_gp[gidx]- y[i]) < 2*s_gp[gidx]:
                points_in_2sd += 1.0
            if abs(m_gp[gidx]- y[i]) > 2*s_gp[gidx] and abs(m_gp[gidx]- y[i]) < 3*s_gp[gidx]:
                points_in_3sd += 1.0
    
        frac_within_1sd = float(points_in_1sd)/ len(y)
        frac_within_2sd = float(points_in_2sd)/ len(y)
        frac_within_3sd = float(points_in_3sd)/ len(y)
        print 'Fraction of points within 1 sd: {}'.format(frac_within_1sd)
        print 'Fraction of points within 2 sd: {}'.format(frac_within_2sd)
        print 'Fraction of points within 3 sd: {}'.format(frac_within_3sd)

    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=2*len(x)
        )(grid)
        if scipy.isnan(m_gp).any():
            print(x)
            print(y)
            print(err_y)
        #i = m_gp.argmax()
    else:
        raise ValueError("Undefined method: %s" % (method,))
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        if method == 'GPR':
            gptools.univariate_envelope_plot(grid, m_gp, s_gp, ax=a,label='Inferred')
            #a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        plt.plot(grid[m_gp.argmax()],m_gp.max(),'r*')
        #a.axvline(grid[i])

    if method == 'GPR':
        res.m_gp=m_gp
        res.s_gp=s_gp 
        res.frac_within_1sd=frac_within_1sd
        res.frac_within_2sd=frac_within_2sd
        res.frac_within_3sd=frac_within_3sd

        # res = results(m_gp=m_gp, s_gp=s_gp, frac_within_1sd=frac_within_1sd, frac_within_2sd=frac_within_2sd, frac_within_3sd=frac_within_3sd)
        # return (m_gp, s_gp, frac_within_1sd, frac_within_2sd, frac_within_3sd)
    else:
        res.m_gp=m_gp
        #return m_gp
    return res