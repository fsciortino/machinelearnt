from __future__ import division
import sys
sys.path.append('/home/sciortino/ML')
import numpy as np
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
import re
from VUV_gui_classes import VUVData,interp_max
import cPickle as pkl
import warnings
import bayesimp_helper
import scipy.special
import profiletools


def get_data(query='xeus', shot = 1101014019, t_min = None, t_max = None, inj_idx = None):
    ''' Load data from several diagnostics from C-Mod

    '''
    # Training data:
    if t_min == None and t_max ==None:
        if shot == 1101014019: 
            t_min = 1.23; t_max= 1.4
        elif shot == 1101014029: 
            t_min = 0.99; t_max = 1.08
    else: 
        ValueError('t_min and t_max not correctly given')

    if inj_idx == None:
        inj_idx = 1

    if query == 'xeus':
        # ========================================
        # Get data. If this wasn't saved before, then use the following parameters to load it:
        # included lines: 6
        # lambda_min=2.02, lambda_max=2.04
        # baseline start = 1.11, baseline end = 1.22 
        # ---> then press "apply" and forcely close the GUI
        # ========================================
        try:
            with open('vuv_signals_%d_%d.pkl'%(shot,inj_idx), 'rb') as f:
                    vuv_data = pkl.load(f)
        except IOError:
            class Injection(object):
                """Class to store information on a given injection. """
                def __init__(self, t_inj, t_start, t_stop):
                    self.t_inj = t_inj
                    self.t_start = t_start
                    self.t_stop = t_stop

            LBO_inj=[Injection(t_min+0.01, t_min, t_max),]  

            vuv_data = VUVData(shot, LBO_inj, debug_plots=True)
            with open('vuv_signals_%d_%d.pkl'%(shot,inj_idx), 'wb') as f:
                pkl.dump(vuv_data, f, protocol=pkl.HIGHEST_PROTOCOL)

        return vuv_data

    elif query == 'ne':
        p_ne=profiletools.ne(shot, include=['CTS','ETS'],abscissa='r/a',t_min=t_min,t_max=t_max) #,'GPC','GPC2'],abscissa='r/a',t_min=1.0,t_max=1.08)
        return p_ne

    elif query == 'Te':
        p_Te=profiletools.Te(shot, include=['CTS','ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)  #,'GPC','GPC2'],abscissa='r/a',t_min=1.0,t_max=1.08)
        return p_Te


##############
def datasets_org(training = None, validation = None, pos_constraint = True):
    ''' Function to organize training and validation datasets. 
    This assumes that 'training' only contains one dataset, whilst validation 
    contains an array of datasets. This is not yet well generalized! 

    MWE:
    val_sets = [vuv_data_val1,vuv_data_val2,vuv_data_val3,vuv_data_val4,vuv_data_val5,vuv_data_val6]
    t_train,y_train,y_unc_train,t_val, y_val, y_unc_val = datasets_org(training = vuv_data_train, validation = val_sets)

    
    '''
    if training == None and validation == None:
        raise ValueError('Missing input data sets!')

    if training != None:
        signal = training.signal
        _y_train=signal.y

        if pos_constraint:
            y_train = np.asarray([_y_train[i] if _y_train[i]>0 else np.array([0.0,]) for i in range(len(_y_train))])[:,0]
        else:
            y_train = _y_train

        y_unc_train=signal.std_y[:,0]
        t_train=signal.t

    if validation != None:
        t_val = {}
        y_val = {}
        y_unc_val = {}
        
        for i_set in range(len(validation)):
            vuv_data_val = validation[i_set]
            signal_val = vuv_data_val.signal
            _y_val = signal_val.y

            if pos_constraint:
                y_val[i_set] = np.asarray([_y_val[i] if _y_val[i]>0 else np.array([0.0,]) for i in range(len(_y_val))])[:,0]
            else:
                y_val[i_set] = _y_val

            y_unc_val[i_set]=signal_val.std_y[:,0]
            t_val[i_set]=signal_val.t

    if training != None:
        if validation != None:
            return (t_train,y_train,y_unc_train,t_val, y_val, y_unc_val)
        else:
            return (t_train,y_train)
    else:
        return (t_val, y_val, y_unc_val)



def SE_func(x,grad):
    nL = x[0]    
    print nL

    f_output = type('', (), {})()
    f_output.t_train = t_train; 
    f_output.y_clean_train = y_clean_train; 
    f_output.y_unc_train = y_unc_train; 
    f_output = prof_fit.profile_fitting(t_train, y_clean_train, err_y=y_unc_train, method='GPR',
                            kernel='SE', noiseLevel= nL, debug_plots=False, **SE_params)
    
    f_output.SE_logposterior = f_output.ll
    f_output.SE_BIC = f_output.BIC

    f_output.sigma_f_opt = f_output.free_params[:][0]
    f_output.l1_opt = f_output.free_params[:][1]
    sigma_n_opt = f_output.free_params[:][2]
    f_output.noiseLevel_opt = sigma_n_opt / np.mean(y_unc_train)

    return f_output