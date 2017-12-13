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
from profile_unc_estimation import profile_fitting #as  prof_fit

#################### BEGIN #################################
# Get data:
xeus_test, xeus_val_sets = get_data('XEUS')
ne_test, ne_val_sets = get_data('ne')
Te_test, Te_val_sets = get_data('Te')

# organize data in testing and (multiple) validation sets:
x_test_xeus,y_test_xeus,y_unc_test_xeus,x_val_xeus, y_val_xeus, y_unc_val_xeus = datasets_org(testing = xeus_test, validation = xeus_val_sets)
x_test_ne,y_test_ne,y_unc_test_ne,x_val_ne, y_val_ne, y_unc_val_ne = datasets_org(testing = ne_test, validation = ne_val_sets,query ='ne')
x_test_Te,y_test_Te,y_unc_test_Te,x_val_Te, y_val_Te, y_unc_val_Te = datasets_org(testing = Te_test, validation = Te_val_sets,query ='Te')

# XEUS benchmark: plot using augmented XEUS uncertainties and Monte Carlo interpolation
signal_u_test = bayesimp_helper.plot_benchmark_prof(x_test_xeus,y_test_xeus,y_unc_test_xeus, plot=True)

signal_u_val1 = bayesimp_helper.plot_benchmark_prof(x_val_xeus[0],y_val_xeus[0],y_unc_val_xeus[0], plot=True)
signal_u_val2 = bayesimp_helper.plot_benchmark_prof(x_val_xeus[1],y_val_xeus[1],y_unc_val_xeus[1], plot=True)
signal_u_val3 = bayesimp_helper.plot_benchmark_prof(x_val_xeus[2],y_val_xeus[2],y_unc_val_xeus[2], plot=True)
signal_u_val4 = bayesimp_helper.plot_benchmark_prof(x_val_xeus[3],y_val_xeus[3],y_unc_val_xeus[3], plot=True)
signal_u_val5 = bayesimp_helper.plot_benchmark_prof(x_val_xeus[4],y_val_xeus[4],y_unc_val_xeus[4], plot=True)
#signal_u_val6 = bayesimp_helper.get_systematic_uncertainty(xeus_val_sets[5].signal, plot=True)

# ===========================================================
#
#                           OPTIMIZATION
#
# ============================================================
# def Gaussian_frac_loss(x, grad, xx, y, y_unc, params): 
#     #assert len(grad) == 0, "grad is not empty, but it should"
#     nL = x[0]; print nL
#     res_val = profile_fitting(xx, y, err_y=y_unc, optimize=True,
#          method='GPR',kernel='SE',noiseLevel=nL,debug_plots=True, **params)

#     frac_within_1sd = res_val.frac_within_1sd
#     frac_within_2sd = res_val.frac_within_2sd
#     frac_within_3sd = res_val.frac_within_3sd

#     loss = 0.5 * ((1 - range_1sd/(frac_within_1sd+eps))**2 + (1 - frac_within_2sd/(range_2sd+eps))+ (1 - frac_within_31sd/(range_3sd+eps)))# + lam * reg
#     print '***************** Validation loss = ', loss, ' ******************'
#     return loss

range_1sd = scipy.special.erf(1/np.sqrt(2))
range_2sd = scipy.special.erf(2/np.sqrt(2)) - scipy.special.erf(1/np.sqrt(2))
range_3sd = scipy.special.erf(3/np.sqrt(2)) - scipy.special.erf(2/np.sqrt(2))

def MSE_Gaussian_loss(x, grad, xx, y, y_unc, params, kernel): 
    #assert len(grad) == 0, "grad is not empty, but it should"
    nL = x[0]; print nL
    res_val = profile_fitting(xx, y, err_y=y_unc, optimize=True,
         method='GPR',kernel=kernel,noiseLevel=nL,debug_plots=False, **params)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd

    beta = 2.0 # try 2
    loss = 0.5 * (range_1sd**(-beta)*(range_1sd - frac_within_1sd)**2 + range_2sd**(-beta)*(range_2sd - frac_within_2sd)**2 + range_3sd**(-beta)*(range_3sd - frac_within_3sd)**2)# + lam * reg
    #print '***************** Validation loss = ', loss, ' ******************'
    return loss

################# XEUS SE KERNEL ########################
#SE_params={'sigma_mean': 2, 'l_mean': 1e-4, 'sigma_sd': 10, 'l_sd': 1e-2}
SE_params={'sigma_mean': 1, 'l_mean': 1e-1, 'sigma_sd': 5, 'l_sd': 1e-2}
# res_test = profile_fitting(x_test_xeus,y_test_xeus, err_y=y_unc_test_xeus, optimize=True,
#      method='GPR',kernel='SE',noiseLevel=2.5,debug_plots=True, **SE_params)

# Obtain optimized estimate of \psi by averaging over optimized results on validation set:
psi_val_XEUS_SE = np.zeros(len(x_val_xeus))
for i_val in range(0,len(x_val_xeus)):
    opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
    opt.set_lower_bounds([1.0,] * opt.get_dimension())
    opt.set_upper_bounds([3.0,] * opt.get_dimension())
    opt.set_xtol_abs(0.1)
    objective = lambda x,grad: MSE_Gaussian_loss(x,grad,x_val_xeus[i_val],y_val_xeus[i_val],
        y_unc_val_xeus[i_val], SE_params, kernel='SE')
    opt.set_min_objective(objective)
    
    # Launch optimization
    psi_val_XEUS_SE[i_val] = opt.optimize(np.asarray([2.0]))[0]
    print ' -----> Completed validation set %d'%(i_val)

# find statistics for optimized result:
res_test = profile_fitting(x_test_xeus,y_test_xeus, err_y=y_unc_test_xeus, optimize=True,
     method='GPR',kernel='SE',noiseLevel=np.mean(psi_val_XEUS_SE),debug_plots=True, **SE_params)
print 'SE kernel free params: ', res_test.free_params
print 'SE kernel ll: ', res_test.ll
print 'SE kernel BIC: ', res_test.BIC
print 'Fraction of points within 1 sd: {}'.format(res_test.frac_within_1sd)
print 'Fraction of points within 2 sd: {}'.format(res_test.frac_within_2sd)
print 'Fraction of points within 3 sd: {}'.format(res_test.frac_within_3sd)

######################## GIBBS KERNEL ######################
gibbs_params={'sigma_min':0.0,'sigma_max':2.0,'l1_mean':0.05,'l2_mean':0.3,'lw_mean':0.02,'x0_mean':0.0,
                'l1_sd':0.05,'l2_sd':0.05,'lw_sd':0.02,'x0_sd':0.002}
# gibbs_params={'sigma_min':0.0,'sigma_max':1.0,'l1_mean':0.1,'l2_mean':0.005,'lw_mean':0.02,'x0_mean':0.02,
#                 'l1_sd':0.005,'l2_sd':0.005,'lw_sd':0.3,'x0_sd':0.03}
# res_xeus_final = profile_fitting(x_val_xeus[3],y_val_xeus[3], err_y=y_unc_val_xeus[3], optimize=True,
#      method='GPR',kernel='gibbs',noiseLevel=2,debug_plots=True, **gibbs_params)

# Obtain optimized estimate of \psi by averaging over optimized results on validation set:
psi_val_XEUS_gibbs = np.zeros(len(x_val_xeus))
for i_val in range(len(x_val_xeus)):
    opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
    opt.set_lower_bounds([1.0,] * opt.get_dimension())
    opt.set_upper_bounds([5.0,] * opt.get_dimension())
    opt.set_xtol_abs(0.1)
    objective = lambda x,grad: MSE_Gaussian_loss(x,grad,x_val_xeus[i_val],y_val_xeus[i_val],y_unc_val_xeus[i_val], gibbs_params, kernel='gibbs')
    opt.set_min_objective(objective)
    
    # Launch optimization
    psi_val_XEUS_gibbs[i_val] = opt.optimize(np.asarray([2.0]))[0]
    print ' -----> Completed validation set %d'%(i_val)

# find statistics for optimized result:
res_test_xeus_final = profile_fitting(x_test_xeus,y_test_xeus, err_y=y_unc_test_xeus, optimize=True,
     method='GPR',kernel='gibbs',noiseLevel=np.mean(psi_val_XEUS_gibbs),debug_plots=True, **gibbs_params)
print 'Gibbs kernel free params: ', res_test.free_params
print 'Gibbs kernel ll: ', res_test.ll
print 'Gibbs kernel BIC: ', res_test.BIC
print 'Fraction of points within 1 sd: {}'.format(res_test_xeus_final.frac_within_1sd)
print 'Fraction of points within 2 sd: {}'.format(res_test_xeus_final.frac_within_2sd)
print 'Fraction of points within 3 sd: {}'.format(res_test_xeus_final.frac_within_3sd)



# ==================================================================
# 
#                             kernel tests
#
# ===================================================================
SE_params={'sigma_mean': 1, 'l_mean': 1e-1, 'sigma_sd': 5, 'l_sd': 1e-2}
res = profile_fitting(x_test_xeus,y_test_xeus, err_y=y_unc_test_xeus, optimize=True,
     method='GPR',kernel='SE',noiseLevel=2,debug_plots=True, **gibbs_params)
gibbs_ll = res.ll
gibbs_BIC = res.BIC
print 'gibbs kernel free params: ', res.free_params
print 'gibbs kernel ll: ', gibbs_ll
print 'gibbs kernel BIC: ', gibbs_BIC

#######################
gibbs_params={'sigma_min':0.0,'sigma_max':1.0,'l1_mean':0.1,'l2_mean':0.005,'lw_mean':0.02,'x0_mean':0.02,
                'l1_sd':0.005,'l2_sd':0.005,'lw_sd':0.3,'x0_sd':0.03}
res = profile_fitting(x_test_xeus,y_test_xeus, err_y=y_unc_test_xeus, optimize=True,
     method='GPR',kernel='gibbs',noiseLevel=2,debug_plots=True, **gibbs_params)
gibbs_ll = res.ll
gibbs_BIC = res.BIC
print 'gibbs kernel free params: ', res.free_params
print 'gibbs kernel ll: ', gibbs_ll
print 'gibbs kernel BIC: ', gibbs_BIC

##########################
# # Try matern52 kernel:
# matern52_params={'sigma_mean': 2.0, 'l_mean': 0.05, 'sigma_sd': 10.0, 'l_sd':10.0}
# res = profile_fitting(x_test_xeus, y_test_xeus, err_y=y_unc_test_xeus, method='GPR',kernel='matern52',noiseLevel=2, **matern52_params)
# matern52_logposterior = res.ll
# gmatern52_BIC = res.BIC

# Try RQ kernel:
RQ_params={'sigma_mean': 2.0, 'alpha_mean': 5, 'l1_mean': 5, 
            'sigma_sd': 10.0, 'alpha_sd': 2,   'l1_sd':0.1}
res = profile_fitting(x_test_xeus, y_test_xeus, err_y=y_unc_test_xeus, optimize=True,
    method='GPR',kernel='RQ',noiseLevel=2, debug_plots=True,**RQ_params)
RQ_ll = res.ll
RQ_BIC = res.BIC
print 'RQ kernel free params: ', res.free_params
print 'RQ kernel ll: ', RQ_ll
print 'RQ kernel BIC: ', RQ_BIC


# ==================================================================
# 
#                             ne and Te
#
# ===================================================================

# ================================== ne  ==============================
#f = plt.figure()
#ax = f.add_subplot(111, projection = '3d')
#errorbar3d(ax, ne_test.X[:,0],ne_test.X[:,1], ne_test.y, xerr = ne_test.err_X[:,0], 
#    yerr = ne_test.err_X[:,1], zerr = ne_test.err_y)
# plot in 3d
ne_test.plot_data()

# plot
f = plt.figure()
ax = f.add_subplot(111)
plt.errorbar(x_test_ne, y_test_ne, y_unc_test_ne,marker='s',mfc='red',mec='green')

gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':1.0,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
                'l1_sd':0.3,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
# res = profile_fitting(x_test_ne, y_test_ne, err_y=y_unc_test_ne, optimize=True,
#      method='GPR',kernel='gibbs',noiseLevel=2,debug_plots=True, **gibbs_params)

psi_val_ne = np.zeros(len(x_val_ne))
for i_val in range(len(x_val_ne)):
    opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
    opt.set_lower_bounds([1.0,] * opt.get_dimension())
    opt.set_upper_bounds([5.0,] * opt.get_dimension())
    opt.set_xtol_abs(0.1)
    objective = lambda x,grad: MSE_Gaussian_loss(x,grad,x_val_ne[i_val],y_val_ne[i_val],y_unc_val_ne[i_val], gibbs_params, kernel='gibbs')
    opt.set_min_objective(objective)
    
    # Launch optimization
    psi_val_ne[i_val] = opt.optimize(np.asarray([2.0]))[0]
    print ' -----> Completed validation set %d'%(i_val)

#psi_val_ne = [1.25,1.25,1.1171875,1.25,1.25,1.1328125]
# find statistics for optimized result:
res_test_ne_final = profile_fitting(x_test_ne,y_test_ne, err_y=y_unc_test_ne, optimize=True,
     method='GPR',kernel='gibbs',noiseLevel=np.mean(psi_val_ne),debug_plots=True, **gibbs_params)
res_test_ne_final = profile_fitting(x_test_ne,y_test_ne, err_y=y_unc_test_ne, optimize=True,
     method='GPR',kernel='gibbs',noiseLevel=1.0,debug_plots=True, **gibbs_params)
plt.xlabel(r'$\rho$', fontsize=16)
plt.ylabel(r'$n_e [m^{-3}]$', fontsize=16)
print 'gibbs kernel free params for ne: ', res_test_ne_final.free_params
print 'gibbs kernel ll for ne: ', res_test_ne_final.ll
print 'gibbs kernel BIC for ne: ', res_test_ne_final.BIC
print 'Fraction of points within 1 sd for ne: {}'.format(res_test_ne_final.frac_within_1sd)
print 'Fraction of points within 2 sd for ne: {}'.format(res_test_ne_final.frac_within_2sd)
print 'Fraction of points within 3 sd for ne: {}'.format(res_test_ne_final.frac_within_3sd)

# ================================== Te  ==============================
# plot in 3d
Te_test.plot_data()
Te_val_sets[3].plot_data()

# plot
f = plt.figure()
ax = f.add_subplot(111)
plt.errorbar(x_test_Te, y_test_Te, y_unc_test_Te,marker='s',mfc='red',mec='green')

gibbs_params={'sigma_min':0.0,'sigma_max':10.0,'l1_mean':0.5,'l2_mean':0.5,'lw_mean':0.01,'x0_mean':1.0,
                'l1_sd':0.02,'l2_sd':0.25,'lw_sd':0.1,'x0_sd':0.05}
res = profile_fitting(x_test_Te, y_test_Te, err_y=y_unc_test_Te, optimize=True,
     method='GPR',kernel='gibbs',noiseLevel=1.25,debug_plots=True, **gibbs_params)
# gibbs_ll = res.ll
# gibbs_BIC = res.BIC
# print 'gibbs kernel free params: ', res.free_params
# print 'gibbs kernel ll: ', gibbs_ll
# print 'gibbs kernel BIC: ', gibbs_BIC

psi_val_Te = np.zeros(len(x_val_Te))
for i_val in range(len(x_val_Te)-3):
    opt = nlopt.opt(nlopt.LN_SBPLX, 1)  # LN_SBPLX
    opt.set_lower_bounds([1.0,] * opt.get_dimension())
    opt.set_upper_bounds([5.0,] * opt.get_dimension())
    opt.set_xtol_abs(0.1)
    objective = lambda x,grad: MSE_Gaussian_loss(x,grad,x_val_Te[i_val],y_val_Te[i_val],y_unc_val_Te[i_val], gibbs_params, kernel='gibbs')
    opt.set_min_objective(objective)
    
    # Launch optimization
    psi_val_Te[i_val] = opt.optimize(np.asarray([2.0]))[0]
    print ' -----> Completed validation set %d'%(i_val)

# find statistics for optimized result:
res_test_Te_final = profile_fitting(x_test_Te,y_test_Te, err_y=y_unc_test_Te, optimize=True,
     method='GPR',kernel='gibbs',noiseLevel=np.mean(psi_val_Te),debug_plots=True, **gibbs_params)
plt.xlabel(r'$\rho$', fontsize=16)
plt.ylabel(r'$T_e$ [keV]', fontsize=16)
print 'gibbs kernel free params for Te: ', res_test_Te_final.free_params
print 'gibbs kernel ll for Te: ', res_test_Te_final.ll
print 'gibbs kernel BIC for Te: ', res_test_Te_final.BIC
print 'Fraction of points within 1 sd for Te: {}'.format(res_test_Te_final.frac_within_1sd)
print 'Fraction of points within 2 sd for Te: {}'.format(res_test_Te_final.frac_within_2sd)
print 'Fraction of points within 3 sd for Te: {}'.format(res_test_Te_final.frac_within_3sd)

left,bottom, width, height = [0.55,0.55,0.3,0.3]
ax2 = plt.gcf().add_axes([left,bottom,width,height])
gptools.univariate_envelope_plot(x_test_Te[x_test_Te>0.9], 
    res_test_Te_final.m_gp[x_test_Te>0.9], res_test_Te_final.s_gp[x_test_Te>0.9], ax=ax2,color='green')
ax2.set_xlabel(r'$\rho$', fontsize=14)
ax2.set_ylabel(r'$T_e$ [keV]', fontsize=14)