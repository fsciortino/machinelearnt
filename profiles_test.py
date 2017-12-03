import profiletools
import gptools
import eqtools
import os
import numpy as np
import scipy
import MDSplus
import matplotlib.pyplot as plt

plt.ion()
shot=1101014006

# First, test whether profiletools is working
# profiletools is a package that was written by Mark Chilenski specifically
# to work with tokamak profiles. It has a lot of functions that do funky stuff
# but unless we go out to explore how they work, it might be better to 
# "pick and choose" them rather than just use the package as a whole, given 
# that we'll need to implement more analysis routines for our ML project

p=profiletools.ne(shot, include=['CTS','ETS'],abscissa='r/a',t_min=1.0,t_max=1.08) #,'GPC','GPC2'],abscissa='r/a',t_min=1.0,t_max=1.08)
p.time_average(weighted=True)
p.plot_data()

p.create_gp()
p.find_gp_MAP_estimate()
p.plot_gp(ax='gca')

roa=scipy.linspace(0,1.2,100)
ax,mean,stdev = p.smooth(roa,plot=True)

# Compute gradients
mean_gradient, stddev_gradient = p.smooth(roa, n=1)
roa2 = scipy.concatenate((roa, roa))
n = scipy.concatenate((scipy.zeros_like(roa), scipy.ones_like(roa)))

# Get mean and stddev:
mean, stddev = p.smooth(roa2, n=n)

# or get the entire covariance:
mean, cov = p.smooth(roa2, n=n, return_cov=True)

# Compute volume average:
p.compute_volume_average()

# Compute normalized inverse gradient length scales
p.compute_a_over_L(roa,plot=True)

########################################
# 
#	   Try accessing data directly
#
########################################
efit_tree = eqtools.CModEFITTree(shot)
electrons = MDSplus.Tree('electrons', shot)
Z_shift=0.0

############################
## Core Thomson Scattering
#############################
N_ne_CTS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:ne_rz')
t_ne_CTS = N_ne_CTS.dim_of().data()
ne_CTS = N_ne_CTS.data() / 1e20
dev_ne_CTS = electrons.getNode(r'yag_new.results.profiles:ne_err').data() / 1e20

Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data() + Z_shift
R_CTS = (electrons.getNode(r'yag.results.param:r').data() * scipy.ones_like(Z_CTS))
channels_CTS = range(0, len(Z_CTS))

t_grid_CTS, Z_grid_CTS = scipy.meshgrid(t_ne_CTS, Z_CTS)
t_grid_CTS, R_grid_CTS = scipy.meshgrid(t_ne_CTS, R_CTS)
t_grid_CTS, channel_grid_CTS = scipy.meshgrid(t_ne_CTS, channels_CTS)

ne_CTS_flat = ne_CTS.flatten()
ne_err_CTS_flat = dev_ne_CTS.flatten()
Z_CTS = scipy.atleast_2d(Z_grid_CTS.flatten())
R_CTS = scipy.atleast_2d(R_grid_CTS.flatten())
channels_CTS = channel_grid_CTS.flatten()
t_CTS = scipy.atleast_2d(t_grid_CTS.flatten())

X_CTS = scipy.hstack((t_CTS.T, R_CTS.T, Z_CTS.T))

#########################
# Edge Thomson Scattering
#########################
N_ne_ETS = electrons.getNode(r'yag_edgets.results:ne')
t_ne_ETS = N_ne_ETS.dim_of().data()
ne_ETS = N_ne_ETS.data() / 1e20
dev_ne_ETS = electrons.getNode(r'yag_edgets.results:ne:error').data() / 1e20

Z_ETS = electrons.getNode(r'yag_edgets.data:fiber_z').data() + Z_shift
R_ETS = (electrons.getNode(r'yag.results.param:R').data() *
         scipy.ones_like(Z_ETS))
channels_ETS = range(0, len(Z_ETS))
    
t_grid_ETS, Z_grid_ETS = scipy.meshgrid(t_ne_ETS, Z_ETS)
t_grid_ETS, R_grid_ETS = scipy.meshgrid(t_ne_ETS, R_ETS)
t_grid_ETS, channel_grid_ETS = scipy.meshgrid(t_ne_ETS, channels_ETS)

ne_ETS_flat = ne_ETS.flatten()
ne_err_ETS_flat= dev_ne_ETS.flatten()
Z_ETS = scipy.atleast_2d(Z_grid_ETS.flatten())
R_ETS = scipy.atleast_2d(R_grid_ETS.flatten())
channels_ETS = channel_grid_ETS.flatten()
t_ETS = scipy.atleast_2d(t_grid_ETS.flatten())

X_ETS = scipy.hstack((t_ETS.T, R_ETS.T, Z_ETS.T))
    