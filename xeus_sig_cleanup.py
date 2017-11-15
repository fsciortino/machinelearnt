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

class Injection(object):
    """Class to store information on a given injection.
    """
    def __init__(self, t_inj, t_start, t_stop):
        self.t_inj = t_inj
        self.t_start = t_start
        self.t_stop = t_stop

# Select shot for which 
shot_train=1101014019
shot_val = 1101014029

LBO_inj_train=[Injection(1.25, 1.23, 1.4),] #1101019019
LBO_inj_val=[Injection(1.0, 0.99, 1.08),]  #1101014029

# ========================================
# Get data. If this wasn't saved before, then use the following parameters to load it:
# included lines: 6
# lambda_min=2.02, lambda_max=2.04
# baseline start = 1.11, baseline end = 1.22 
# ---> then press "apply" and forcely close the GUI

# Training data:
try:
    with open('vuv_signals_%d.pkl'%shot_train, 'rb') as f:
            vuv_data_train = pkl.load(f)
except IOError:
    vuv_data_train = VUVData(shot_train, LBO_inj_train, debug_plots=True)
    with open('vuv_signals_%d.pkl'%shot_train, 'wb') as f:
        pkl.dump(vuv_data_train, f, protocol=pkl.HIGHEST_PROTOCOL)

# Validation data:
try:
    with open('vuv_signals_%d.pkl'%shot_val, 'rb') as f:
            vuv_data_val = pkl.load(f)
except IOError:
    vuv_data_val = VUVData(shot_val, LBO_inj_val, debug_plots=True)
    with open('vuv_signals_%d.pkl'%shot_val, 'wb') as f:
        pkl.dump(vuv_data_val, f, protocol=pkl.HIGHEST_PROTOCOL)
# ==========================================

# Extract signal in simple form
y_train=vuv_data_train.signal.y
y_clean_train=np.asarray([y_train[i] if y_train[i]>0 else np.array([0.0,]) for i in range(len(y_train))])[:,0]
y_unc_train=vuv_data_train.signal.std_y[:,0]
t_train=vuv_data_train.signal.t


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

    # SE kernel test:
    SE_params={'sigma_mean': 2.0, 'l_mean': 0.1, 'sigma_sd': 10.0, 'l_sd':0.1}
    res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='SE',noiseLevel=2, **SE_params)
    logposterior_SE = 


    # Try gibbs kernel:
    gibbs_params={'sigma_min':0.0,'sigma_max':2.0,'l1_mean':0.005,'l2_mean':0.1,'lw_mean':0.01,'x0_mean':0.0,
                    'l1_sd':0.05,'l2_sd':0.05,'lw_sd':0.02,'x0_sd':0.002}
    res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='gibbs',noiseLevel=2, **gibbs_params)
    
    # Try matern52 kernel:
    matern52_params={'sigma_mean': 2.0, 'l_mean': 0.05, 'sigma_sd': 10.0, 'l_sd':10.0}
    res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='matern52',noiseLevel=2, **matern52_params)
    
    # Try RQ kernel:
    RQ_params={'sigma_mean': 2.0, 'l_mean': 0.005, 'sigma_sd': 10.0, 'l_sd':0.1}
    res = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',kernel='RQ',noiseLevel=2, **RQ_params)


#==================================================================
#
#                            VALIDATION
#
#===================================================================
range_1sd = 0.6827
range_2sd = 0.9545 - range_1sd
range_3sd = 0.9973 - range_2sd

nL = np.linspace(1.0, 3.0, 3)
hyperparams = np.zeros((len(nL),3))
for j in range(len(nL)):
    noiseLevel = nL[j]
    res_train = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=True, sigma_max=10.0, 
        method='GPR',kernel='SE', noiseLevel=noiseLevel)

    hyperparams[j,:] = res_train.free_params[:]
    sigma_f_opt = res_train.free_params[:][0]
    l_1_opt = res_train.free_params[:][1]
    sigma_n_opt = res_train.free_params[:][2]
    noiseLevel_opt = sigma_n_opt / np.mean(y_unc_train)

    # Find validation set error
    res_val = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, optimize=False, sigma_max=10.0, 
        method='GPR', kernel='SE', noiseLevel=noiseLevel_opt)

    frac_within_1sd = res_val.frac_within_1sd
    frac_within_2sd = res_val.frac_within_2sd
    frac_within_3sd = res_val.frac_within_3sd

    loss = 0.5 * ((range_1sd - frac_within_1sd)**2 + (range_2sd - frac_within_2sd)**2 + (range_3sd - frac_within_3sd)**2)

    print loss
