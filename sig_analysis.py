from __future__ import division
import sys
sys.path.append('/home/sciortino/ML')
import profile_unc_estimation as  prof_fit
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
import bayesimp_helper
import scipy.special
import nlopt
from get_expt_data import get_data

from profiletools import errorbar3d

# Select shot for which 
shot_train=1101014019
shot_val = 1101014029

t_min_train = 1.23
t_max_train = 1.4
t_min_val = 0.99
t_max_val = 1.08

vuv_data_train = get_data(query='xeus', shot = shot_train)
vuv_data_val = get_data(query='xeus', shot = shot_val)
ne_train = get_data(query='ne', shot = shot_train, t_min = t_min_train, t_max = t_max_train)
ne_val = get_data(query='ne', shot = shot_val, t_min = t_min_train, t_max = t_max_train)
Te_train = get_data(query='Te', shot = shot_train, t_min = t_min_val, t_max = t_max_val)
Te_val = get_data(query='Te', shot = shot_val,t_min = t_min_val, t_max = t_max_val)



# ================================== ne  ==============================
#f = plt.figure()
#ax = f.add_subplot(111, projection = '3d')
#errorbar3d(ax, ne_train.X[:,0],ne_train.X[:,1], ne_train.y, xerr = ne_train.err_X[:,0], 
#    yerr = ne_train.err_X[:,1], zerr = ne_train.err_y)
# plot in 3d
ne_train.plot_data()

# clean data
bad_idx = [index for index, value in enumerate(ne_train.err_y) if value>0.4]
true_bad_idx = np.asarray([False,]*len(ne_train.y))
true_bad_idx[bad_idx] = True
ne_train_clean_roa = ne_train.X[~true_bad_idx,1]
ne_train_clean_y = ne_train.y[~true_bad_idx]
ne_train_clean_err_y = ne_train.err_y[~true_bad_idx]

# sort
sorted_idx = [i[0] for i in sorted(enumerate(ne_train_clean_roa), key = lambda x: x[1])]
sorted_ne_train_roa = ne_train_clean_roa[sorted_idx]
sorted_ne_train_y= ne_train_clean_y[sorted_idx]
sorted_ne_train_y_err= ne_train_clean_err_y[sorted_idx]

# plot
f = plt.figure()
ax = f.add_subplot(111)
plt.errorbar(sorted_ne_train_roa, sorted_ne_train_y, sorted_ne_train_y_err,marker='s',mfc='red',mec='green')

gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
                'l1_sd':0.3,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
res = prof_fit.profile_fitting(sorted_ne_train_roa, sorted_ne_train_y, err_y=sorted_ne_train_y_err, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
gibbs_logposterior = res.ll
gibbs_BIC = res.BIC

# ================================== Te  ==============================
# plot in 3d
ne_train.plot_data()

# clean data
bad_idx = [index for index, value in enumerate(Te_train.err_y) if value>0.4]
true_bad_idx = np.asarray([False,]*len(Te_train.y))
true_bad_idx[bad_idx] = True
Te_train_clean_roa = Te_train.X[~true_bad_idx,1]
Te_train_clean_y = Te_train.y[~true_bad_idx]
Te_train_clean_err_y = Te_train.err_y[~true_bad_idx]

# sort
sorted_idx = [i[0] for i in sorted(enumerate(Te_train_clean_roa), key = lambda x: x[1])]
sorted_Te_train_roa = Te_train_clean_roa[sorted_idx]
sorted_Te_train_y= Te_train_clean_y[sorted_idx]
sorted_Te_train_y_err= Te_train_clean_err_y[sorted_idx]

# plot
f = plt.figure()
ax = f.add_subplot(111)
plt.errorbar(sorted_ne_train_roa, sorted_ne_train_y, sorted_ne_train_y_err,marker='s',mfc='red',mec='green')

gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
                'l1_sd':0.3,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
res = prof_fit.profile_fitting(sorted_ne_train_roa, sorted_ne_train_y, err_y=sorted_ne_train_y_err, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
gibbs_logposterior = res.ll
gibbs_BIC = res.BIC

# ================================== XEUS  ==============================

# Extract signal in simple form
signal = vuv_data_train.signal
y_train=signal.y
y_clean_train=np.asarray([y_train[i] if y_train[i]>0 else np.array([0.0,]) for i in range(len(y_train))])[:,0]
y_unc_train=signal.std_y[:,0]
t_train=signal.t

# Benchmark
if True: 
    signal_u = bayesimp_helper.get_systematic_uncertainty(signal, plot=True)


# ==================================================================
# 
#                             TRAINING
#
# ===================================================================
report_figs = True
if report_figs:
    plt.figure()
    plt.errorbar(t_train,y_clean_train,y_unc_train,marker='s',mfc='red',mec='green')

    res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=True, sigma_max=10.0, 
        debug_plots=True, method='spline',kernel='SE',noiseLevel=1)

# SE kernel test: need low length-scales
SE_params={'sigma_mean': 1.0, 'l_mean': 1e-4, 'sigma_sd': 1.5, 'l_sd':0.001}
SE_params={'sigma_mean': 2, 'l_mean': 1e-4, 'sigma_sd': 10, 'l_sd': 0.01}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',
                                kernel='SE',noiseLevel=2, **SE_params)
SE_logposterior0 = res.ll
SE_BIC0 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-3, 'sigma_sd': 10.0, 'l_sd':0.1}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=2, **SE_params)
SE_logposterior1 = res.ll
SE_BIC1 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-2, 'sigma_sd': 10.0, 'l_sd':0.1}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=2, **SE_params)
SE_logposterior2 = res.ll
SE_BIC2 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-4, 'sigma_sd': 10.0, 'l_sd':0.1}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=3, **SE_params)
SE_logposterior3 = res.ll
SE_BIC3 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-3, 'sigma_sd': 10.0, 'l_sd':0.1}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=3, **SE_params)
SE_logposterior4 = res.ll
SE_BIC4 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-2, 'sigma_sd': 10.0, 'l_sd':0.1}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=3, **SE_params)
SE_logposterior5 = res.ll
SE_BIC5 = res.BIC

# Try gibbs kernel:
gibbs_params={'sigma_min':0.0,'sigma_max':2.0,'l1_mean':0.005,'l2_mean':0.1,'lw_mean':0.01,'x0_mean':0.0,
                'l1_sd':0.05,'l2_sd':0.05,'lw_sd':0.02,'x0_sd':0.002}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
gibbs_logposterior = res.ll
gibbs_BIC = res.BIC

# Try matern52 kernel:
matern52_params={'sigma_mean': 2.0, 'l_mean': 0.05, 'sigma_sd': 10.0, 'l_sd':10.0}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='matern52',noiseLevel=2, **matern52_params)
matern52_logposterior = res.ll
gmatern52_BIC = res.BIC

# Try RQ kernel:
RQ_params={'sigma_mean': 2.0, 'l_mean': 0.005, 'sigma_sd': 10.0, 'l_sd':0.1}
res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='RQ',noiseLevel=2, **RQ_params)
RQ_logposterior = res.ll
RQ_BIC = res.BIC

#==================================================================
#
#                            VALIDATION
#
#===================================================================
range_1sd = scipy.special.erf(1/np.sqrt(2))
range_2sd = scipy.special.erf(2/np.sqrt(2)) - range_1sd
range_3sd = scipy.special.erf(3/np.sqrt(2)) - range_2sd

nL = np.linspace(1.0, 3.0, 3)
hyperparams = np.zeros((len(nL),3))
for j in range(len(nL)):
    noiseLevel = nL[j]
    res_train = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=True, 
        method='GPR',kernel='SE', noiseLevel=noiseLevel)

    hyperparams[j,:] = res_train.free_params[:]
    sigma_f_opt = res_train.free_params[:][0]
    l_1_opt = res_train.free_params[:][1]
    sigma_n_opt = res_train.free_params[:][2]
    noiseLevel_opt = sigma_n_opt / np.mean(y_unc_train)

    # Find validation set error
    res_val = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=False, 
        method='GPR', kernel='SE', noiseLevel=noiseLevel_opt)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd

    loss = 0.5 * ((range_1sd - frac_within_1sd)**2 + (range_2sd - frac_within_2sd)**2 + (range_3sd - frac_within_3sd)**2)

    print loss


# ===========================================================
#
#                           OPTIMIZATION
#
# ============================================================
#SE_params = type('', (), {})()
range_1sd = scipy.special.erf(1/np.sqrt(2))
range_2sd = scipy.special.erf(2/np.sqrt(2)) - range_1sd
range_3sd = scipy.special.erf(3/np.sqrt(2)) - range_2sd

SE_params={'sigma_mean': 1.0, 'l_mean': 1e-4, 'sigma_sd': 0.5, 'l_sd': 0.001}

def SE_func(t_train, y_clean_train, y_unc_train, nL, SE_params):
    f_output = type('', (), {})()
    f_output.t_train = t_train; f_output.y_clean_train = y_clean_train; f_output.y_unc_train = y_unc_train; 
    f_output = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',
                            kernel='SE', noiseLevel= nL, debug_plots=False, **SE_params)
    
    f_output.SE_logposterior = f_output.ll
    f_output.SE_BIC = f_output.BIC

    f_output.sigma_f_opt = f_output.free_params[:][0]
    f_output.l1_opt = f_output.free_params[:][1]
    sigma_n_opt = f_output.free_params[:][2]
    f_output.noiseLevel_opt = sigma_n_opt / np.mean(y_unc_train)

    return f_output

def MSE_loss(x,grad): 
    #assert len(grad) == 0, "grad is not empty, but it should"

    nL = x    
    f_output = SE_func(t_train, y_clean_train, y_unc_train, nL, SE_params)   
    # Find validation set error
    res_val = prof_fit.profile_fitting(t_train,y_clean_train, err_y=y_unc_train, optimize=False,
         method='GPR',kernel='SE',noiseLevel=f_output.noiseLevel_opt,debug_plots=False, **SE_params)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd

    loss = 0.5 * ((range_1sd - frac_within_1sd)**2 + (range_2sd - frac_within_2sd)**2 + (range_3sd - frac_within_3sd)**2)
    print '***************** Computed loss for given parameters **********************'
    return loss


opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
opt.set_min_objective(MSE_loss)
opt.set_lower_bounds([1.0,] * opt.get_dimension())
opt.set_upper_bounds([5.0,] * opt.get_dimension())
# opt.set_ftol_abs(1.0)
opt.set_ftol_rel(1e-6)
# opt.set_maxeval(40000)#(100000)
opt.set_maxtime(3600)

uopt = opt.optimize(np.asarray([2]))
