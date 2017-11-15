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

#Input independent variable x and y=f(x) (same dimensions), along with the number of splines to be
#fit, N. 

def spline_MC(xin,yin,xout,yin_err,n_trial):
        
    # Change 'yin' and 'yin_err' inputs to 2D arrays and augment for number of trials

	yin = np.atleast_2d(yin).T
	yin_err = np.atleast_2d(yin_err)
	y =  np.tile(yin,(n_trial,1))
	yin_err = np.tile(yin_err,(n_trial,1))

    # Randomly assign y values to the input data within the error bounds
	y = y + yin_err*np.random.standard_normal(y.shape)
    
    # Interpolate onto locations specified in 'xout'
	f = scipy.interpolate.interp1d(xin,y,kind='cubic')
	y_fit = f(xout)

    # Calculate mean and variance at each xout

	m = np.mean(y_fit,axis=0)
	var = np.var(y_fit,axis=0)

	plt.plot(xin,yin,'.')
	plt.plot(xout,y_fit,'.')

	return m, var, y_fit
