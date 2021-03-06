import gptools
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# SE_params={'sigma_mean': 2, 'l_mean': 1e-4, 'sigma_sd': 10, 'l_sd': 0.01}
# gibbs_params2={'sigma_min':0.0,'sigma_max':1.0,'l1_mean':0.1,'l2_mean':0.005,'lw_mean':0.02,'x0_mean':0.02,
#                'l1_sd':0.005,'l2_sd':0.005,'lw_sd':0.3,'x0_sd':0.03}

m=0.005 #1.0
s=0.005#1e-3
dist=gptools.GammaJointPriorAlt(m,s)

print 'Standard Gamma Joint Prior parameters: ', dist.a,dist.b

dist = gptools.GammaJointPrior(400,4e3)
N=1000000
n=N/100
samples=scipy.stats.gamma.rvs(dist.a,loc=0,scale=1.0/dist.b,size=N)
p,x=np.histogram(samples,bins=n)

# convert bin edges to centers
x=x[:-1]+(x[1]-x[0])/2

fig,ax=plt.subplots(4,2,figsize=(10,12))

for b,k in zip(ax.flatten(),range(len(ax.flatten()))):
	f=UnivariateSpline(x,p,s=N*k)
	b.plot(x,f(x))
	b.set_title('s=N*k = %d '%(N*k))

plt.ion()
plt.show() 


#gptools.GammaJointPriorAlt([1.0,0.5,0.0,1.0],[0.3,0.25,0.3,0.05])                )

#k = gptools.GibbsKernel1dTanh(
    #= ====== =======================================================================
    #0 sigmaf Amplitude of the covariance function
    #1 l1     Small-X saturation value of the length scale.
    #2 l2     Large-X saturation value of the length scale.
    #3 lw     Length scale of the transition between the two length scales.
    #4 x0     Location of the center of the transition between the two length scales.
    #= ====== ==============================