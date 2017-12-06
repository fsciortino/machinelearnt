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



# ========================================
# Get data. If this wasn't saved before, then use the following parameters to load it:
# included lines: 6
# lambda_min=2.02, lambda_max=2.04
# baseline start = 1.11, baseline end = 1.22 
# ---> then press "apply" and forcely close the GUI
# ========================================

def get_data(query='xeus', shot = 1101014019, t_min = None, t_max = None):
    #data= type('', (), {})()
    # Training data:
    if t_min == None and t_max ==None:
        if shot == 1101014019: 
            t_min = 1.23; t_max= 1.4
        elif shot == 1101014029: 
            t_min = 0.99; t_max = 1.08
    else: 
        ValueError('t_min and t_max not correctly given')

    if query == 'xeus':
        try:
            with open('vuv_signals_%d.pkl'%shot, 'rb') as f:
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
            with open('vuv_signals_%d.pkl'%shot, 'wb') as f:
                pkl.dump(vuv_data, f, protocol=pkl.HIGHEST_PROTOCOL)

        return vuv_data

    elif query == 'ne':
        p_ne=profiletools.ne(shot, include=['CTS','ETS'],abscissa='r/a',t_min=t_min,t_max=t_max) #,'GPC','GPC2'],abscissa='r/a',t_min=1.0,t_max=1.08)
        return p_ne

    elif query == 'Te':
        p_Te=profiletools.Te(shot, include=['CTS','ETS'],abscissa='r/a',t_min=t_min,t_max=t_max)  #,'GPC','GPC2'],abscissa='r/a',t_min=1.0,t_max=1.08)
        return p_Te


