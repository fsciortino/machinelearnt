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
import operator


def get_data(sys='XEUS'): 
    # Select shots to analyze:
    shot_test = 1101014019
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
        #xeus_test = load_data(query='xeus', shot = shot_test, t_min= 1.23, t_max = 1.4, inj_idx = 1)
        xeus_test = load_data(query='xeus', shot = shot_val1, t_min= 0.99, t_max = 1.08, inj_idx = 1)
        #xeus_val1 = load_data(query='xeus', shot = shot_val1, t_min= 0.99, t_max = 1.08, inj_idx = 1)
        xeus_val1 = load_data(query='xeus', shot = shot_val2, t_min= 0.77, t_max = 0.94, inj_idx = 2)
        xeus_val2 = load_data(query='xeus', shot = shot_val2, t_min= 0.99, t_max = 1.14, inj_idx = 3)
        xeus_val3 = load_data(query='xeus', shot = shot_val2, t_min= 1.19, t_max = 1.34, inj_idx = 4)
        xeus_val4 = load_data(query='xeus', shot = shot_val3, t_min= 0.76, t_max = 0.91, inj_idx = 1)
        xeus_val5 = load_data(query='xeus', shot = shot_val3, t_min= 0.95, t_max = 1.12, inj_idx = 2)
        #xeus_val7 = load_data(query='xeus', shot = shot_val3, t_min= 1.15, t_max = 1.3, inj_idx = 3)

        # Organize testing and validation data sets:
        xeus_val_sets = [xeus_val1,xeus_val2,xeus_val3,xeus_val4,xeus_val5]#,xeus_val6]
        return (xeus_test, xeus_val_sets)

    elif sys=='ne':
        ne_test = load_data(query='ne', shot = shot_test, t_min = 1.23, t_max = 1.40)
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

        # Organize testing and validation data sets:
        ne_val_sets = [ne_val1, ne_val2, ne_val3, ne_val4, ne_val5, ne_val6, ne_val7]#, ne_val8, ne_val9, ne_val9, ne_val10] 
        return (ne_test, ne_val_sets)

    elif sys=='Te':
        Te_test = load_data(query='Te', shot = shot_test, t_min = 1.23, t_max = 1.40)
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

        # Organize testing and validation data sets:
        Te_val_sets = [Te_val1, Te_val2, Te_val3, Te_val4, Te_val5, Te_val6, Te_val7]#, Te_val8, Te_val9, Te_val9, Te_val10] 
        return (Te_test, Te_val_sets)


###############################################################
def load_data(query='xeus', shot = 1101014019, t_min = None, t_max = None, inj_idx = None):
    ''' Load data from several diagnostics from C-Mod

    '''
    # testing data:
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
def datasets_org(testing = None, validation = None, query ='XEUS', clean = True):
    ''' Function to organize testing and validation datasets. 
    This assumes that 'testing' only contains one dataset, whilst validation 
    contains an array of datasets. This is not yet well generalized! 

    MWE:
    val_sets = [vuv_data_val1,vuv_data_val2,vuv_data_val3,vuv_data_val4,vuv_data_val5,vuv_data_val6]
    t_test,y_test,y_unc_test,t_val, y_val, y_unc_val = datasets_org(testing = vuv_data_test, validation = val_sets)

    
    '''
    if testing == None and validation == None:
        raise ValueError('Missing input data sets!')

    #################   XEUS   #######################
    if query == 'XEUS':
        if testing != None:
            signal = testing.signal
            _y_norm_test=signal.y_norm - np.mean(signal.y_norm[signal.t<-0.005])

            maxx_index, maxx = max(enumerate(_y_norm_test), key = operator.itemgetter(1))
            maxx_unc = signal.std_y_norm[maxx_index,0]

            # When shifting up and normalizing again, one must re-propagate uncertainties:
            _y_norm_unc_test = (scipy.sqrt((np.atleast_2d(signal.std_y_norm[:,0] / maxx).T)**2.0 + ((_y_norm_test / maxx) * (maxx_unc / maxx))**2.0 ))

            # normalize again:
            _y_norm_test = _y_norm_test / np.max(_y_norm_test)

            # Set to 0 all negative XEUS signals:
            if clean:
                y_norm_test = np.asarray([_y_norm_test[i] if _y_norm_test[i]>0 else np.array([0.0,]) for i in range(len(_y_norm_test))])[:,0]
            else:
                y_norm_test = _y_norm_test

            y_norm_unc_test = _y_norm_unc_test[:,0]
            t_test=signal.t

        if validation != None:
            t_val = {}
            y_norm_val = {}
            y_norm_unc_val = {}
            
            for i_set in range(len(validation)):
                vuv_data_val = validation[i_set]
                signal_val = vuv_data_val.signal
                
                # if i_set == 3:
                #     import pdb
                #     pdb.set_trace()
                _y_norm_val = signal_val.y_norm - np.mean(signal_val.y_norm[signal_val.t<-0.005])
                
                maxx_index, maxx = max(enumerate(_y_norm_val), key = operator.itemgetter(1))
                maxx_unc = signal_val.std_y_norm[maxx_index,0]

                # When shifting up and normalizing again, one must re-propagate uncertainties:
                _y_norm_unc_val = (scipy.sqrt((np.atleast_2d(signal_val.std_y_norm[:,0] / maxx).T)**2.0 + ((_y_norm_val / maxx) * (maxx_unc / maxx))**2.0 ))

                # normalize again:
                _y_norm_val = _y_norm_val / np.max(_y_norm_val)

                # Set to 0 all negative XEUS signals:
                if clean:
                    y_norm_val[i_set] = np.asarray([_y_norm_val[i] if _y_norm_val[i]>0 else np.array([0.0,]) for i in range(len(_y_norm_val))])[:,0]
                else:
                    y_norm_val[i_set] = _y_norm_val

                #y_norm_unc_val[i_set]=signal_val.std_y_norm[:,0]
                y_norm_unc_val[i_set] = _y_norm_unc_val[:,0]
                t_val[i_set]=signal_val.t

        if testing != None:
            if validation != None:
                return (t_test,y_norm_test,y_norm_unc_test,t_val, y_norm_val, y_norm_unc_val)
            else:
                return (t_test,y_norm_test,y_norm_unc_test)
        else:
            return (t_val, y_norm_val, y_norm_unc_val)

    ################   ne and Te   ######################
    elif query == 'ne' or query == 'Te':
        # import pdb
        # pdb.set_trace()
        if testing != None:
            data_test = testing
            x_test = data_test.X[:,1]
            y_test = data_test.y
            y_unc_test = data_test.err_y

            if clean:
                # clean data
                bad_idx = [index for index, value in enumerate(y_unc_test) if value>0.4]
                true_bad_idx = np.asarray([False,]*len(y_test))
                true_bad_idx[bad_idx] = True
                clean_x = x_test[~true_bad_idx]
                clean_y = y_test[~true_bad_idx]
                clean_err_y = y_unc_test[~true_bad_idx]

                # sort
                sorted_idx = [i[0] for i in sorted(enumerate(clean_x), key = lambda x: x[1])]
                x_test = clean_x[sorted_idx]
                y_test = clean_y[sorted_idx]
                y_unc_test = clean_err_y[sorted_idx]

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

        if testing != None:
            if validation != None:
                return (x_test,y_test,y_unc_test,x_val, y_val, y_unc_val)
            else:
                return (x_test,y_test,y_unc_test)
        else:
            return (x_val, y_val, y_unc_val)

##################################
def SE_func(x,grad):
    nL = x[0]    
    print nL

    f_output = type('', (), {})()
    f_output.t_test = t_test; 
    f_output.y_clean_test = y_clean_test; 
    f_output.y_unc_test = y_unc_test; 
    f_output = prof_fit.profile_fitting(t_test, y_clean_test, err_y=y_unc_test, method='GPR',
                            kernel='SE', noiseLevel= nL, debug_plots=False, **SE_params)
    
    f_output.SE_logposterior = f_output.ll
    f_output.SE_BIC = f_output.BIC

    f_output.sigma_f_opt = f_output.free_params[:][0]
    f_output.l1_opt = f_output.free_params[:][1]
    sigma_n_opt = f_output.free_params[:][2]
    f_output.noiseLevel_opt = sigma_n_opt / np.mean(y_unc_test)

    return f_output