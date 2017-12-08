from __future__ import division
import sys
sys.path.append('/home/sciortino/ML')
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

import cPickle as pkl
import warnings
import bayesimp_helper
import scipy.special
import nlopt
from profiletools import errorbar3d

from VUV_gui_classes import VUVData,interp_max
from get_expt_data import get_data, datasets_org
from profile_unc_estimation import profile_fitting, MSE_Gaussian_loss #as  prof_fit


# =================================
# Select shots to analyze:
shot_train = 1101014019
shot_val1 = 1101014029
shot_val2 = 1101014030
shot_val3 = 1101014011

nTshot_val1 = 1101014030
nTshot_val2 = 1101015012
nTshot_val3 = 1101015013
nTshot_val4 = 1101014007
nTshot_val5 = 1100903006
nTshot_val6 = 1100811017
nTshot_val7 = 1100811018
nTshot_val8 = 1100812004
nTshot_val9 = 1100722019
nTshot_val10 = 1100722020
# ==================================

# =================== GET  DATA  ====================
# XEUS data
xeux_train = get_data(query='xeus', shot = shot_train, t_min= 1.23, t_max = 1.4, inj_idx = 1)
xeux_val1 = get_data(query='xeus', shot = shot_val1, t_min= 0.99, t_max = 1.08, inj_idx = 1)
xeux_val2 = get_data(query='xeus', shot = shot_val2, t_min= 0.77, t_max = 0.94, inj_idx = 2)
xeux_val3 = get_data(query='xeus', shot = shot_val2, t_min= 0.99, t_max = 1.14, inj_idx = 3)
xeux_val4 = get_data(query='xeus', shot = shot_val2, t_min= 1.19, t_max = 1.34, inj_idx = 4)
xeux_val5 = get_data(query='xeus', shot = shot_val3, t_min= 0.76, t_max = 0.91, inj_idx = 1)
xeux_val6 = get_data(query='xeus', shot = shot_val3, t_min= 0.95, t_max = 1.12, inj_idx = 2)
#xeux_val7 = get_data(query='xeus', shot = shot_val3, t_min= 1.15, t_max = 1.3, inj_idx = 3)

# Organize training and validation data sets:
xeux_val_sets = [xeux_val1,xeux_val2,xeux_val3,xeux_val4,xeux_val5,xeux_val6]
t_train_xeus,y_train_xeus,y_unc_train_xeus,t_val_xeus, y_val_xeus, y_unc_val_xeus = datasets_org(training = xeux_train, validation = xeux_val_sets)

# ne data
ne_train = get_data(query='ne', shot = shot_train, t_min = 1.23, t_max = 1.40)
ne_val1 = get_data(query='ne', shot = nTshot_val2, t_min = 1.00, t_max = 1.20)
ne_val2 = get_data(query='ne', shot = nTshot_val4, t_min = 0.80, t_max = 1.00)
ne_val3 = get_data(query='ne', shot = nTshot_val5, t_min = 1.00, t_max = 1.15)
ne_val4 = get_data(query='ne', shot = nTshot_val6, t_min = 0.80, t_max = 1.00)
ne_val5 = get_data(query='ne', shot = nTshot_val7, t_min = 0.90, t_max = 1.10)
ne_val6 = get_data(query='ne', shot = nTshot_val8, t_min = 0.90, t_max = 1.10)
ne_val7 = get_data(query='ne', shot = nTshot_val10, t_min = 0.75, t_max = 0.90)
#ne_val8 = get_data(query='ne', shot = nTshot_val1, t_min = 1.00, t_max = 1.20)  ######
#ne_val9 = get_data(query='ne', shot = nTshot_val9, t_min = 0.90, t_max = 1.10)
#ne_val10 = get_data(query='ne', shot = nTshot_val3, t_min = 1.25, t_max = 1.40)

# Organize training and validation data sets:
ne_val_sets = [ne_val1, ne_val2, ne_val3, ne_val4, ne_val5, ne_val6, ne_val7]#, ne_val8, ne_val9, ne_val9, ne_val10] 
t_train_ne,y_train_ne,y_unc_train_ne,t_val_ne, y_val_ne, y_unc_val_ne = datasets_org(training = xeux_train, validation = xeux_val_sets)

# Te data
Te_train = get_data(query='Te', shot = shot_train, t_min = 1.23, t_max = 1.40)
Te_val1 = get_data(query='ne', shot = nTshot_val2, t_min = 1.00, t_max = 1.20)
Te_val2 = get_data(query='ne', shot = nTshot_val4, t_min = 0.80, t_max = 1.00)
Te_val3 = get_data(query='ne', shot = nTshot_val5, t_min = 1.00, t_max = 1.15)
Te_val4 = get_data(query='ne', shot = nTshot_val6, t_min = 0.80, t_max = 1.00)
Te_val5 = get_data(query='ne', shot = nTshot_val7, t_min = 0.90, t_max = 1.10)
Te_val6 = get_data(query='ne', shot = nTshot_val8, t_min = 0.90, t_max = 1.10)
Te_val7 = get_data(query='ne', shot = nTshot_val10, t_min = 0.75, t_max = 0.90)

# Te_val8 = get_data(query='ne', shot = nTshot_val1, t_min = 1.00, t_max = 1.20)
# Te_val9 = get_data(query='ne', shot = nTshot_val3, t_min = 1.25, t_max = 1.40)
# Te_val10 = get_data(query='ne', shot = nTshot_val9, t_min = 0.90, t_max = 1.10)

# Organize training and validation data sets:
Te_val_sets = [Te_val1, Te_val2, Te_val3, Te_val4, Te_val5, Te_val6, Te_val7]#, Te_val8, Te_val9, Te_val9, Te_val10] 
t_train_Te,y_train_Te,y_unc_train_Te,t_val_Te, y_val_Te, y_unc_val_Te = datasets_org(training = xeux_train, validation = xeux_val_sets)

# ================================== XEUS  ==============================
# Benchmark: plot using augmented XEUS uncertainties and Monte Carlo interpolation
signal_u = bayesimp_helper.get_systematic_uncertainty(xeux_train.signal, plot=True)
signal_u_val1 = bayesimp_helper.get_systematic_uncertainty(xeux_val1.signal, plot=True)
signal_u_val2 = bayesimp_helper.get_systematic_uncertainty(xeux_val2.signal, plot=True)
signal_u_val3 = bayesimp_helper.get_systematic_uncertainty(xeux_val3.signal, plot=True)
signal_u_val4 = bayesimp_helper.get_systematic_uncertainty(xeux_val4.signal, plot=True)
signal_u_val5 = bayesimp_helper.get_systematic_uncertainty(xeux_val5.signal, plot=True)
signal_u_val6 = bayesimp_helper.get_systematic_uncertainty(xeux_val6.signal, plot=True)

# ===========================================================
#
#                           OPTIMIZATION
#
# ============================================================
range_1sd = scipy.special.erf(1/np.sqrt(2))
range_2sd = scipy.special.erf(2/np.sqrt(2)) - range_1sd
range_3sd = scipy.special.erf(3/np.sqrt(2)) - range_2sd

SE_params={'sigma_mean': 2, 'l_mean': 1e-4, 'sigma_sd': 10, 'l_sd': 0.01}

opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
opt.set_min_objective(MSE_loss)
opt.set_lower_bounds([1.0,] * opt.get_dimension())
opt.set_upper_bounds([5.0,] * opt.get_dimension())
opt.set_xtol_rel(0.1)
opt.set_maxtime(1000)

# Launch optimization
uopt = opt.optimize(np.asarray([2.0]))

# find statistics for optimized result:
res_val = profile_fitting(t_val,y_clean_val, err_y=y_unc_val, optimize=True,
     method='GPR',kernel='SE',noiseLevel=uopt[0],debug_plots=True, **SE_params)
frac_within_1sd = res_val.frac_within_1sd
frac_within_2sd = res_val.frac_within_2sd
frac_within_3sd = res_val.frac_within_3sd
print 'Fraction of points within 1 sd: {}'.format(frac_within_1sd)
print 'Fraction of points within 2 sd: {}'.format(frac_within_2sd)
print 'Fraction of points within 3 sd: {}'.format(frac_within_3sd)

######################################
# with Gibbs kernel
gibbs_params={'sigma_min':0.0,'sigma_max':2.0,'l1_mean':0.005,'l2_mean':0.1,'lw_mean':0.01,'x0_mean':0.0,
                'l1_sd':0.05,'l2_sd':0.05,'lw_sd':0.02,'x0_sd':0.002}
gibbs_params2={'sigma_min':0.0,'sigma_max':1.0,'l1_mean':0.1,'l2_mean':0.005,'lw_mean':0.02,'x0_mean':0.02,
                'l1_sd':0.005,'l2_sd':0.005,'lw_sd':0.3,'x0_sd':0.03}
def MSE_loss(x,grad): 
    nL = x[0]    
    print nL
    res_val = profile_fitting(t_val[1],y_clean_val[1], err_y=y_unc_val[1], optimize=True,
         method='GPR',kernel='gibbs',noiseLevel=nL,debug_plots=True, **gibbs_params)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd
    loss = 0.5 * ((range_1sd - frac_within_1sd)**2 + (range_2sd - frac_within_2sd)**2 + (range_3sd - frac_within_3sd)**2)
    print '***************** Validation loss = ', loss, ' ******************'
    return loss

opt = nlopt.opt(nlopt.LN_SBPLX, 1)  
opt.set_min_objective(MSE_loss)
opt.set_lower_bounds([1.0,] * opt.get_dimension())
opt.set_upper_bounds([8.0,] * opt.get_dimension())
opt.set_xtol_rel(0.1)
opt.set_maxtime(1000)

# Launch optimization
uopt = opt.optimize(np.asarray([2.0]))
print ' HEY! The optimal noise is ', uopt[0]
# find statistics for optimized result:
res_val = profile_fitting(t_val, y_clean_val, err_y=y_unc_val, optimize=True,
         method='GPR',kernel='gibbs',noiseLevel=uopt[0],debug_plots=True, **gibbs_params)
frac_within_1sd = res_val.frac_within_1sd
frac_within_2sd = res_val.frac_within_2sd
frac_within_3sd = res_val.frac_within_3sd
print 'Fraction of points within 1 sd: {}'.format(frac_within_1sd)
print 'Fraction of points within 2 sd: {}'.format(frac_within_2sd)
print 'Fraction of points within 3 sd: {}'.format(frac_within_3sd)

gibbs_params2={'sigma_min':0.0,'sigma_max':1.0,'l1_mean':0.1,'l2_mean':0.005,'lw_mean':0.02,'x0_mean':0.02,
                'l1_sd':0.005,'l2_sd':0.005,'lw_sd':0.3,'x0_sd':0.03}
res_val = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=True,
         method='GPR',kernel='gibbs',noiseLevel=4,debug_plots=True, **gibbs_params2)



# ==================================================================
# 
#                             TRAINING
#
# ===================================================================
report_figs = True
if report_figs:
    plt.figure()
    plt.errorbar(t_train,y_clean_train,y_unc_train,marker='s',mfc='red',mec='green')

    res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=True, sigma_max=10.0, 
        debug_plots=True, method='spline',kernel='SE',noiseLevel=1)

# SE kernel test: need low length-scales
SE_params={'sigma_mean': 1.0, 'l_mean': 1e-4, 'sigma_sd': 1.5, 'l_sd':0.001}
SE_params={'sigma_mean': 1.0, 'l_mean': 1e-4, 'sigma_sd': 0.5, 'l_sd': 0.001}
SE_params={'sigma_mean': 2, 'l_mean': 1e-4, 'sigma_sd': 10, 'l_sd': 0.01}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',
                                kernel='SE',noiseLevel=4, **SE_params)
SE_logposterior0 = res.ll
SE_BIC0 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-3, 'sigma_sd': 10.0, 'l_sd':0.1}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=2, **SE_params)
SE_logposterior1 = res.ll
SE_BIC1 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-2, 'sigma_sd': 10.0, 'l_sd':0.1}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=2, **SE_params)
SE_logposterior2 = res.ll
SE_BIC2 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-4, 'sigma_sd': 10.0, 'l_sd':0.1}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=3, **SE_params)
SE_logposterior3 = res.ll
SE_BIC3 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-3, 'sigma_sd': 10.0, 'l_sd':0.1}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=3, **SE_params)
SE_logposterior4 = res.ll
SE_BIC4 = res.BIC

SE_params={'sigma_mean': 2.0, 'l_mean': 1e-2, 'sigma_sd': 10.0, 'l_sd':0.1}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=3, **SE_params)
SE_logposterior5 = res.ll
SE_BIC5 = res.BIC

# Try gibbs kernel:
gibbs_params={'sigma_min':0.0,'sigma_max':2.0,'l1_mean':0.005,'l2_mean':0.1,'lw_mean':0.01,'x0_mean':0.0,
                'l1_sd':0.05,'l2_sd':0.05,'lw_sd':0.02,'x0_sd':0.002}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
gibbs_logposterior = res.ll
gibbs_BIC = res.BIC

# Try matern52 kernel:
matern52_params={'sigma_mean': 2.0, 'l_mean': 0.05, 'sigma_sd': 10.0, 'l_sd':10.0}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='matern52',noiseLevel=2, **matern52_params)
matern52_logposterior = res.ll
gmatern52_BIC = res.BIC

# Try RQ kernel:
RQ_params={'sigma_mean': 2.0, 'l_mean': 0.005, 'sigma_sd': 10.0, 'l_sd':0.1}
res = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='RQ',noiseLevel=2, **RQ_params)
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
    res_train = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=True, 
        method='GPR',kernel='SE', noiseLevel=noiseLevel)

    hyperparams[j,:] = res_train.free_params[:]
    sigma_f_opt = res_train.free_params[:][0]
    l_1_opt = res_train.free_params[:][1]
    sigma_n_opt = res_train.free_params[:][2]
    noiseLevel_opt = sigma_n_opt / np.mean(y_unc_train)

    # Find validation set error
    res_val = profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=False, 
        method='GPR', kernel='SE', noiseLevel=noiseLevel_opt)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd

    loss = 0.5 * ((range_1sd - frac_within_1sd)**2 + (range_2sd - frac_within_2sd)**2 + (range_3sd - frac_within_3sd)**2)

    print loss

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
res = profile_fitting(sorted_ne_train_roa, sorted_ne_train_y, err_y=sorted_ne_train_y_err, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
gibbs_logposterior = res.ll
gibbs_BIC = res.BIC

# ================================== Te  ==============================
# plot in 3d
Te_train.plot_data()

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
plt.errorbar(sorted_Te_train_roa, sorted_Te_train_y, sorted_Te_train_y_err,marker='s',mfc='red',mec='green')

gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
                'l1_sd':0.3,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
res = profile_fitting(sorted_Te_train_roa, sorted_Te_train_y, err_y=sorted_Te_train_y_err, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
gibbs_logposterior = res.ll
gibbs_BIC = res.BIC

