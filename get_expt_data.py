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


def get_data(sys='XEUS'): 
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
    
    if sys=='XEUS':
        xeus_train = load_data(query='xeus', shot = shot_train, t_min= 1.23, t_max = 1.4, inj_idx = 1)
        xeus_val1 = load_data(query='xeus', shot = shot_val1, t_min= 0.99, t_max = 1.08, inj_idx = 1)
        xeus_val2 = load_data(query='xeus', shot = shot_val2, t_min= 0.77, t_max = 0.94, inj_idx = 2)
        xeus_val3 = load_data(query='xeus', shot = shot_val2, t_min= 0.99, t_max = 1.14, inj_idx = 3)
        xeus_val4 = load_data(query='xeus', shot = shot_val2, t_min= 1.19, t_max = 1.34, inj_idx = 4)
        xeus_val5 = load_data(query='xeus', shot = shot_val3, t_min= 0.76, t_max = 0.91, inj_idx = 1)
        xeus_val6 = load_data(query='xeus', shot = shot_val3, t_min= 0.95, t_max = 1.12, inj_idx = 2)
        #xeus_val7 = load_data(query='xeus', shot = shot_val3, t_min= 1.15, t_max = 1.3, inj_idx = 3)

        # Organize training and validation data sets:
        xeus_val_sets = [xeus_val1,xeus_val2,xeus_val3,xeus_val4,xeus_val5,xeus_val6]
        return (xeus_train, xeus_val_sets)

    elif sys=='ne':
        ne_train = load_data(query='ne', shot = shot_train, t_min = 1.23, t_max = 1.40)
        ne_val1 = load_data(query='ne', shot = nTshot_val2, t_min = 1.00, t_max = 1.20)
        ne_val2 = load_data(query='ne', shot = nTshot_val4, t_min = 0.80, t_max = 1.00)
        ne_val3 = load_data(query='ne', shot = nTshot_val5, t_min = 1.00, t_max = 1.15)
        ne_val4 = load_data(query='ne', shot = nTshot_val6, t_min = 0.80, t_max = 1.00)
        ne_val5 = load_data(query='ne', shot = nTshot_val7, t_min = 0.90, t_max = 1.10)
        ne_val6 = load_data(query='ne', shot = nTshot_val8, t_min = 0.90, t_max = 1.10)
        ne_val7 = load_data(query='ne', shot = nTshot_val10, t_min = 0.75, t_max = 0.90)
        #ne_val8 = load_data(query='ne', shot = nTshot_val1, t_min = 1.00, t_max = 1.20)  ######
        #ne_val9 = load_data(query='ne', shot = nTshot_val9, t_min = 0.90, t_max = 1.10)
        #ne_val10 = load_data(query='ne', shot = nTshot_val3, t_min = 1.25, t_max = 1.40)

        # Organize training and validation data sets:
        ne_val_sets = [ne_val1, ne_val2, ne_val3, ne_val4, ne_val5, ne_val6, ne_val7]#, ne_val8, ne_val9, ne_val9, ne_val10] 
        return (ne_train, ne_val_sets)

    elif sys=='Te':
        Te_train = load_data(query='Te', shot = shot_train, t_min = 1.23, t_max = 1.40)
        Te_val1 = load_data(query='ne', shot = nTshot_val2, t_min = 1.00, t_max = 1.20)
        Te_val2 = load_data(query='ne', shot = nTshot_val4, t_min = 0.80, t_max = 1.00)
        Te_val3 = load_data(query='ne', shot = nTshot_val5, t_min = 1.00, t_max = 1.15)
        Te_val4 = load_data(query='ne', shot = nTshot_val6, t_min = 0.80, t_max = 1.00)
        Te_val5 = load_data(query='ne', shot = nTshot_val7, t_min = 0.90, t_max = 1.10)
        Te_val6 = load_data(query='ne', shot = nTshot_val8, t_min = 0.90, t_max = 1.10)
        Te_val7 = load_data(query='ne', shot = nTshot_val10, t_min = 0.75, t_max = 0.90)
        # Te_val8 = load_data(query='ne', shot = nTshot_val1, t_min = 1.00, t_max = 1.20)
        # Te_val9 = load_data(query='ne', shot = nTshot_val3, t_min = 1.25, t_max = 1.40)
        # Te_val10 = load_data(query='ne', shot = nTshot_val9, t_min = 0.90, t_max = 1.10)

        # Organize training and validation data sets:
        Te_val_sets = [Te_val1, Te_val2, Te_val3, Te_val4, Te_val5, Te_val6, Te_val7]#, Te_val8, Te_val9, Te_val9, Te_val10] 
        return (Te_train, Te_val_sets)


###############################################################
def load_data(query='xeus', shot = 1101014019, t_min = None, t_max = None, inj_idx = None):
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
def datasets_org(training = None, validation = None, query ='XEUS', clean = True):
    ''' Function to organize training and validation datasets. 
    This assumes that 'training' only contains one dataset, whilst validation 
    contains an array of datasets. This is not yet well generalized! 

    MWE:
    val_sets = [vuv_data_val1,vuv_data_val2,vuv_data_val3,vuv_data_val4,vuv_data_val5,vuv_data_val6]
    t_train,y_train,y_unc_train,t_val, y_val, y_unc_val = datasets_org(training = vuv_data_train, validation = val_sets)

    
    '''
    if training == None and validation == None:
        raise ValueError('Missing input data sets!')

    #################   XEUS   #######################
    if query == 'XEUS':
        if training != None:
            signal = training.signal
            _y_train=signal.y

            if clean:
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

                if clean:
                    y_val[i_set] = np.asarray([_y_val[i] if _y_val[i]>0 else np.array([0.0,]) for i in range(len(_y_val))])[:,0]
                else:
                    y_val[i_set] = _y_val

                y_unc_val[i_set]=signal_val.std_y[:,0]
                t_val[i_set]=signal_val.t

        if training != None:
            if validation != None:
                return (t_train,y_train,y_unc_train,t_val, y_val, y_unc_val)
            else:
                return (t_train,y_train,y_unc_train)
        else:
            return (t_val, y_val, y_unc_val)

    ################   ne and Te   ######################
    elif query == 'ne' or query == 'Te':
        # import pdb
        # pdb.set_trace()
        if training != None:
            data_train = training
            x_train = data_train.X[:,1]
            y_train = data_train.y
            y_unc_train = data_train.err_y

            if clean:
                # clean data
                bad_idx = [index for index, value in enumerate(y_unc_train) if value>0.4]
                true_bad_idx = np.asarray([False,]*len(y_train))
                true_bad_idx[bad_idx] = True
                clean_x = x_train[~true_bad_idx]
                clean_y = y_train[~true_bad_idx]
                clean_err_y = y_unc_train[~true_bad_idx]

                # sort
                sorted_idx = [i[0] for i in sorted(enumerate(clean_x), key = lambda x: x[1])]
                x_train = clean_x[sorted_idx]
                y_train = clean_y[sorted_idx]
                y_unc_train = clean_err_y[sorted_idx]

        if validation != None:
            x_val = {}
            y_val = {}
            y_unc_val = {}
            
            for i_set in range(len(validation)):
                data_val = validation[i_set]
                _x_val = data_val.X[:,1]
                _y_val = data_val.y
                _y_unc_val = data_val.err_y

                if clean:
                    # clean data
                    bad_idx = [index for index, value in enumerate(_y_unc_val) if value>0.4]
                    true_bad_idx = np.asarray([False,]*len(_y_val))
                    true_bad_idx[bad_idx] = True
                    clean_x = _x_val[~true_bad_idx]
                    clean_y = _y_val[~true_bad_idx]
                    clean_err_y = _y_unc_val[~true_bad_idx]

                    # sort
                    sorted_idx = [i[0] for i in sorted(enumerate(clean_x), key = lambda x: x[1])]
                    _x_val = clean_x[sorted_idx]
                    _y_val = clean_y[sorted_idx]
                    _y_unc_val = clean_err_y[sorted_idx]

                x_val[i_set] = _x_val
                y_val[i_set] = _y_val
                y_unc_val[i_set] = _y_unc_val

        if training != None:
            if validation != None:
                return (x_train,y_train,y_unc_train,x_val, y_val, y_unc_val)
            else:
                return (x_train,y_train,y_unc_train)
        else:
            return (x_val, y_val, y_unc_val)

##################################
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