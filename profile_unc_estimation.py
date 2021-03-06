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

class hyperparams:
    def __init__(self,**kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
            #print "Set hparams.%s = %d" %(key, value) 


def profile_fitting(x, y, err_y=None, optimize=True, method='GPR', kernel='SE', num_dim=1, debug_plots=True, noiseLevel=2., **kwargs): #sigma_max=10.0, l_min = 0.005, 
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
    method : {'GPR', 'spline'}, optional
        Method to use when interpolating. Default is 'GPR' (Gaussian process
        regression). Can also use a cubic spline.
    kernel : str, optional
        Type of kernel to be used. At this stage, we create the kernel internally, but in the future
        it would be better to do it externally and just give a gptools kernel object as an argument
        More kernels should be added over time. 
    num_dim : int, optional
        Number of dimensions of the input/output data. Default is 1
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    noiseLevel : float, optional
        Initial guess for a noise multiplier. Default: 2
    kwargs : dictionary
        arguments to be passed on to set the hyper-prior bounds for the kernel of choice. 
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
            assert len(kwargs) == 4
            hparams = hyperparams(**kwargs); #hparams.set_kwargs(**kwargs)
            # Defaults:
            if not hasattr(hparams,'sigma_mean'): hparams.sigma_mean = 2.0
            if not hasattr(hparams,'l_mean'): hparams.l_mean = 0.005
            if not hasattr(hparams,'sigma_sd'): hparams.sigma_sd = 10.0
            if not hasattr(hparams,'l_sd'): hparams.l_sd = 0.1

            hprior = (
            gptools.GammaJointPriorAlt([hparams.sigma_mean, hparams.l_mean], [hparams.sigma_sd,hparams.l_sd])
            )
            k = gptools.SquaredExponentialKernel(
                #= ====== =======================================================================
                #0 sigma  Amplitude of the covariance function
                #1 l1     Small-X saturation value of the length scale.
                #2 l2     Large-X saturation value of the length scale.
                #= ====== =======================================================================
                # param_bounds=[(0, sigma_max), (0, 2.0)],
                hyperprior=hprior,
                initial_params=[10000.0, 400000.0], # random, doesn't matter because we do random starts anyway
                fixed_params=[False]*2
            )
            
        elif kernel=='gibbs':
            #if num_dim == 1: assert len(kwargs) == 10
            hparams = hyperparams(**kwargs); 
            # Defaults:
            if not hasattr(hparams,'sigma_min'): hparams.sigma_min = 0.0
            if not hasattr(hparams,'sigma_max'): hparams.sigma_max = 10.0

            if not hasattr(hparams,'l1_mean'): hparams.l1_mean = 0.3
            if not hasattr(hparams,'l1_sd'): hparams.l1_sd = 0.3

            if not hasattr(hparams,'l2_mean'): hparams.l2_mean = 0.5
            if not hasattr(hparams,'l2_sd'): hparams.l2_sd = 0.25
            
            if not hasattr(hparams,'lw_mean'): hparams.lw_mean = 0.0
            if not hasattr(hparams,'lw_sd'): hparams.lw_sd = 0.3
            
            if not hasattr(hparams,'x0_mean'): hparams.x0_mean = 0.0
            if not hasattr(hparams,'x0_sd'): hparams.x0_sd = 0.3

            hprior=(
                gptools.UniformJointPrior([(hparams.sigma_min,hparams.sigma_max),])*
                gptools.GammaJointPriorAlt([hparams.l1_mean,hparams.l2_mean,hparams.lw_mean,hparams.x0_mean],
                                           [hparams.l1_sd,hparams.l2_sd,hparams.lw_sd,hparams.x0_sd])
                )

            k = gptools.GibbsKernel1dTanh(
                #= ====== =======================================================================
                #0 sigma  Amplitude of the covariance function
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
            if num_dim == 1: assert len(kwargs) == 4
            hparams = hyperparams(**kwargs); 
            # Defaults:
            if not hasattr(hparams,'sigma_mean'): hparams.sigma_mean = 2.0
            if not hasattr(hparams,'l_mean'): hparams.l_mean = 0.005

            if not hasattr(hparams,'sigma_sd'): hparams.sigma_sd = 10.0
            if not hasattr(hparams,'l_sd'): hparams.l_sd = 0.1

            hprior=( 
                gptools.GammaJointPriorAlt([hparams.sigma_mean,hparams.l_mean],
                                           [hparams.sigma_sd,hparams.l_sd])
                ) 
            k = gptools.Matern52Kernel( # this has 2 hyperparameters in 1D
                #= ===== ===========================================
                #0 sigma Prefactor to the kernel  
                #2 l1    Length scale for first dimension
                #3 ...   More length scales for more dimensions
                #= ===== =======================================
                hyperprior=hprior,
                initial_params=[0.5, 0.5],
                fixed_params=[False]*2
            ) 
        elif kernel == 'RQ': # rational quadratic
            if num_dim == 1: assert len(kwargs) == 6
            hparams = hyperparams(**kwargs); 

            # Defaults:
            if not hasattr(hparams,'sigma_mean'): hparams.sigma_mean = 2.0
            if not hasattr(hparams,'alpha_mean'): hparams.alpha_mean = 0.005
            if not hasattr(hparams,'l1_mean'): hparams.l1_mean = 0.005

            if not hasattr(hparams,'sigma_sd'): hparams.sigma_sd = 10.0
            if not hasattr(hparams,'alpha_sd'): hparams.alpha_sd = 0.1
            if not hasattr(hparams,'l1_sd'): hparams.l1_sd = 0.1

            hprior=( 
                gptools.GammaJointPriorAlt([hparams.sigma_mean, hparams.alpha_mean, hparams.l1_mean],
                                           [hparams.sigma_sd, hparams.alpha_sd, hparams.l1_sd])
                ) 

            k = gptools.RationalQuadraticKernel(
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

        # Create additional noise to optimize over (the first argument is n_dims)
        nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=np.mean(err_y)*noiseLevel,
                        fixed_noise=True)#, noise_bound=(np.mean(err_y)*noiseLevel*(4.0/5.0),np.mean(err_y)*noiseLevel*(6.0/5.0)))    #(np.min(err_y), np.max(err_y)*noiseLevel))#, enforce_bounds=True)
        #print "noise_bound= [", np.min(err_y), ",",np.max(err_y)*noiseLevel,"]"

        gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y, noise_k=nk)

        for i in range(len(y)):
            if y[i]==0:
                gp.add_data(x[i], 0, n=0, err_y=0.0)
                gp.add_data(x[i], 0, n=0, err_y=0.0)
                gp.add_data(x[i], 0, n=0, err_y=0.0)
                gp.add_data(x[i], 0, n=1, err_y=0.0)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            if optimize: 
                res_min, ll_trials = gp.optimize_hyperparameters(verbose=False, random_starts=100)
            else: 
                print 'Optimization is turned off. Using initial guesses for hyperparameters!'

        m_gp, s_gp = gp.predict(grid,noise=True)
        res.free_params = gp.free_params[:]
        res.free_param_names = gp.free_param_names[:]
        res.free_param_bounds = gp.free_param_bounds[:]

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
        
        ###
        print("Estimating AIC, BIC...")
        sum2_diff = 0 
        for i in range(len(y)):
            # Find value of grid that is the closest to x[i]:
            gidx = np.argmin(abs(grid - x[i]))
            sum2_diff = (m_gp[gidx]-y[i])**2
                
        chi_squared = float(sum2_diff) / len(y)
        num_params = len(hparams.__dict__) / 2
        num_data = len(y)

        AIC = chi_squared + 2.0 * num_params 
        BIC = chi_squared + num_params * scipy.log(num_data)

    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=None #2*len(x)
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
        #plt.plot(grid[m_gp.argmax()],m_gp.max(),'r*')
        plt.xlabel('time (s)', fontsize=14)
        plt.ylabel('Signal Amplitude (A.U.)', fontsize=14)
        plt.tick_params(axis='both',which='major', labelsize=14)

    if method == 'GPR':
        res.m_gp=m_gp
        res.s_gp=s_gp 
        res.frac_within_1sd=frac_within_1sd
        res.frac_within_2sd=frac_within_2sd
        res.frac_within_3sd=frac_within_3sd
        if optimize: 
            res.ll = res_min.fun 
            res.ll_trials = ll_trials
        res.BIC = BIC
        res.AIC = AIC
    else:
        res.m_gp=m_gp

    return res


