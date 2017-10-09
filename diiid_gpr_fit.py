 #-*-Python-*-
# Created by sciortinof at 20 Jun 2017  15:44

"""
The functions and classes in this script allow Gaussian Process Regression (GPR) fits on DIII-D,
based on M.Chilenski's gptools package and O.Meneghini's 2014 implementation for DIII-D.
http://gptools.readthedocs.io/en/latest/index.html

Example fit of data:
fit=OMFITlib_GPRfit.GPRfit(x,y,e,n=0,use_MCMC=False,optimize_noise=optimize_noise,
                        random_starts=20, zero_value_outside=False)

"""

from numpy import *
from uncertainties import *
from uncertainties import unumpy
from uncertainties.unumpy import nominal_values, std_devs, uarray
import copy
import pickle

def GPRfit(x,y,e,optimize_noise=True,random_starts=20, zero_value_outside=True, ntanh=1):
    """
    Inputs:
    --------
    x: array or array of arrays
        Independent variable data points.

    y: array or array of arrays
        Dependent variable data points.

    e: array or array of arrays
        Uncertainty in the dependent variable data points.

    optimize_noise: bool, optional
        Set to True to multiply the (heteroschedastic) diagnostic uncertainties by a factor whose
        value is included as one of the model's hyperparameters. All optimization is done in parallel
        with all the processors that are available. Default is True.

    random_starts: int, optional
        Number of random starts for the optimization of the hyperparameters of the GP. Each random
        starts begins sampling the posterior distribution in a different way. The optimization that
        gives the largest posterior probability is chosen. It is recommended to increase this value
        if the fit results difficult. If the regression fails, it might be necessary to vary the
        constraints given in the _fit method of the class GPfit2 below, which has been kept rather
        general for common usage. Default is 20.

    zero_value_outside: bool, optional
        Set to True if the profile to be evaluated is expected to go to zero beyond the LCFS, e.g. for
        electron temperature and density; note that this option does NOT force the value to be 0 at the LCFS,
        but only attempts to constrain the fit to stabilize to 0 well beyond rho=1. Profiles like those of
        omega_tor_12C6 and T_12C6 are experimentally observed not to go to 0 at the LCFS, so this option
        should be set to False for these. Default is True.

    ntanh: integer, optional
        Set to 2 if an internal transport barrier is expected. Default is 1 (no ITB expected).
        This parameter has NOT been tested recently.

    Returns:
    ----------
    (object) fit: call at points at which the profile is to be evaluated, e.g. if locations are stored in
        an array ``xo'', call fo = fit(xo). For an example, see 7_fit.py in OMFITprofiles.

"""
    if optimize_noise:
        print('*** Noise optimization activated ***')
        noiseLevel=4  #this is essentially an upper bound on the factor by which noise could be underestimated
    else:
        noiseLevel=1
    if random_starts<10:
        printe('It is recommended that at least 10 random starts are used. '
            'Setting random_starts to 10 for effective profile evaluation. ')
        random_starts=10
    if ntanh!=1 and ntanh!=2:
        printe('Value of ntanh should be set to either 1 or 2; set to 2 if an Internal Transport Barrier (ITB) is'
              ' expected in the core plasma. Setting ntanh=1 now.s')
    fitdata={'x':x,'y':y,'e':e,'noiseLevel':noiseLevel,'random_starts':random_starts,
             'zero_value_outside': zero_value_outside, 'ntanh': ntanh}
    out_data=OMFITx.remote_python(root,
            script=root['LIB']['OMFITlib_GPRfit'],
            target_function='fitGP',
            namespace=fitdata)
    obj_data=fitGP2([],[],[],[],[],[],[])
    obj_data.__dict__.update(out_data)

    return obj_data

# Helper funtion
def fitGP(x,y,e,noiseLevel,random_starts, zero_value_outside, ntanh):
    """
    This function is useful to run python processes outside of OMFIT using OMFITx.remote_python.
    This is currently a needed hack because Tkinter seems to clash with multiprocessing.

    """
    tmp=fitGP2(x,y,e,noiseLevel,random_starts, zero_value_outside, ntanh)
    return tmp.__dict__

# Helper function
def predict_MCMC(Xstar, n, gp):
    """
    Helper function to call gptool's predict method with MCMC

    """
    out=gp.predict_MCMC(Xstar,n=n,full_output=True, noise=True, return_std=True, full_MCMC=True)
    return out

# Main GPR fitting class
class fitGP2(object):
    """
    Class used for fitting of experimental profiles using Gaussian Process Regression.

    This class is initialized in the function GPRfit.
    Refer to the description at the top of this script for a detailed explanation of the
    inputs and outputs of the methods of this class.  Additionally, it has an input parameter
    "verbose" that controls verbosity of the non-linear kernel optimization (currently
    set to False for low-verbosity)

    """

    def __init__(self, xx, yy, ey, noiseLevel=2.0, random_starts=20, zero_value_outside=True, ntanh=1, verbose=False):

        self.xx=atleast_2d(xx)
        self.yy=atleast_2d(yy)
        self.ey=atleast_2d(ey)
        #self.n=n
        #self.use_MCMC=use_MCMC
        self.ntanh=ntanh
        self.noiseLevel=noiseLevel
        self.random_starts=random_starts
        self.initial_params=[2.0, 0.5, 0.05, 0.1, 0.5] # for random_starts!= 0, the initial state of the hyperparameters is not actually used.
        self.verbose=verbose
        self.zero_value_outside=zero_value_outside

        self.gp=[]
        if not self.xx.size:
            return

        for k in range(self.xx.shape[0]):
            if verbose:
                printi('fitting profile '+str(k+1)+' of '+str(self.xx.shape[0]))
            i=~isnan(self.yy[k,:])&~isnan(self.xx[k,:])&~isnan(self.ey[k,:])
            self.gp.append(self._fit(self.xx[k,i],self.yy[k,i],self.ey[k,i]))

    def _fit(self, xx, yy, ey):
        import gptools
        norm=mean(abs(yy))
        yy=yy/norm
        ey=ey/norm
        print('********** GPR version 2.0 ***************')

        for kk in range(self.ntanh):

            hprior=(
                # Set a uniform prior for sigmaf
                gptools.UniformJointPrior([(0,10),])*
                # Set Gamma distribution('alternative form') for the other 4 priors of the Gibbs 1D Tanh kernel
                gptools.GammaJointPriorAlt([1.0,0.5,0.0,1.0],[0.3,0.25,0.1,0.05])
                )

            k = gptools.GibbsKernel1dTanh(
                #= ====== =======================================================================
                #0 sigmaf Amplitude of the covariance function
                #1 l1     Small-X saturation value of the length scale.
                #2 l2     Large-X saturation value of the length scale.
                #3 lw     Length scale of the transition between the two length scales.
                #4 x0     Location of the center of the transition between the two length scales.
                #= ====== =======================================================================
                initial_params=self.initial_params,
                fixed_params=[False]*5,
                hyperprior=hprior,
                )

            if kk==0:
                nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=mean(ey)*self.noiseLevel,
                        fixed_noise=False, noise_bound=(min(ey), max(ey)*self.noiseLevel))#, enforce_bounds=True)
                print "noise_bound= [", min(ey), ",",max(ey)*self.noiseLevel,"]"
                ke=k
            else:  #the following is from Orso's initial implementation. Not tested on ITBs!
                nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=gp.noise_k.params[0], fixed_noise=False)
                k1 = gptools.GibbsKernel1dTanh(
                    initial_params=copy.deepcopy(gp.k.params[-5:]),
                    fixed_params=[False]*5)
                ke+=k1

            # Create and populate GP:
            gp = gptools.GaussianProcess(ke, noise_k=nk)
            gp.add_data(xx, yy, err_y=ey)
            gp.add_data(0, 0, n=1, err_y=0.0) #zero derivative on axis

            #================= Add constraints ====================
            # Impose constraints on values in the SOL
            if self.zero_value_outside:
                    gp.add_data(max([1.1,max(xx)])+0.1, 0, n=0, err_y=mean(ey)) #zero beyond edge
                    gp.add_data(max([1.1,max(xx)])+0.2, 0, n=0, err_y=mean(ey)) #zero beyond edge

            # Impose constraints on derivatives in the SOL
            grad=gradient(yy,xx) # rough estimate of gradients
            gp.add_data(max([1.1,max(xx)]),0,n=1, err_y=max(grad)*max(ey/yy)) # added uncertainty in derivative
            print "Added {:.0f}% of max(gradient) in max(grad) on GPR derivative constraints outside of the LCFS'".format(max(ey/yy)*100)
            gp.add_data(max([1.1,max(xx)])+0.1, 0, n=1) #zero derivative far beyond at edge

            for kk1 in range(1,3):
                if self.zero_value_outside:
                    gp.add_data(max([1.1,max(xx)])+0.1*kk1, 0, n=0, err_y=mean(ey)) #zero at edge
                gp.add_data(max([1.1,max(xx)])+0.1*kk1, 0, n=1) #zero derivative beyond the edge

            # In shots where data is missing at the edge, attempt forcing outer stabilization
            if max(xx)<0.8:
                print "Missing data close to the edge. Fit at rho>0.8 might be rather wild."
                if self.zero_value_outside:
                    if max(ey/yy)<0.1:
                        gp.add_data(1.0, 0, n=0, err_y=max(ey)*2)
                    else:
                        gp.add_data(1.0, 0, n=0, err_y=max(ey))
                # pad SOL with zero-derivative constraints
                for i in arange(5):
                    gp.add_data(1.05+0.02*i,0,n=1) #exact derivative=0

            #============ Optimization of hyperparameters ===========
            print 'Number of random starts: ', self.random_starts
            if kk==0:
                # Optimize hyperparameters:
                gp.optimize_hyperparameters(
                    method='SLSQP',
                    verbose=self.verbose,
                    num_proc=None,    #if 0, optimization with 1 processor in series; if None, use all available processors
                    random_starts=self.random_starts,
                    opt_kwargs={ 'bounds': (ke+nk).free_param_bounds,})

            else:
                # Optimize hyperparameters:
                gp.optimize_hyperparameters(
                    method='SLSQP',
                    verbose=self.verbose,
                    num_proc=None,
                    random_starts=self.random_starts,
                    opt_kwargs={ 'bounds': ke.free_param_bounds,},)

        gp.norm=norm
        self.inferred_params=copy.deepcopy(gp.k.params)
        self.final_noise=copy.deepcopy(gp.noise_k.params)
        print '------> self.inferred_params: ', self.inferred_params
        print '-------> self.final_noise: ', self.final_noise
        print '-------> mean(ey) =', mean(ey)
        print '-------> self.final_noise/ mean(ey) =', self.final_noise/mean(ey)
        return gp

    def __call__(self, Xstar, n=0, use_MCMC=False, profile=None):
        """
        Evaluate the fit at specific locations.

        Inputs:
        ----------
        Xstar: array
            Independent variable values at which we wish to evaluate the fit.

        n: int, optional
            Order of derivative to evaluate. Default is 0 (data profile)

        use_MCMC: bool, optional
            Set whether MCMC sampling and a fully-Bayesian estimate for a fitting
            should be used. This is recommended for accurate computations of gradients
            and uncertainties.

        Profile: int, optional
            Profile to evaluate if more than one has been computed and include in the gp
            object. To call the nth profile, set profile=n. If None, it will return an
            array of arrays.

        Outputs:
        ----------
        Value and error of the fit evaluated at the Xstar points

        """

        if profile is None:
            profile=range(len(self.gp))
            print "len(profile) = ", len(profile)
        M=[]; D=[]
        run_on_engaging=False
        for k in atleast_1d(profile):
            if n>1:
                print 'Trying to evaluate higher derivatives than 1. Warning: *NOT* TESTED!'
            else:
                print 'Proceeding with the evaluation of {:}-derivative'.format(n)

            predict_data={'Xstar':Xstar,'n': n, 'gp': self.gp[k]}
            if use_MCMC:
                print '*************** Using MCMC for predictions ********************'
                if run_on_engaging: # set up to run on engaging for the moment
                    out=OMFITx.remote_python(module_root=None,
                                         script=root['LIB']['OMFITlib_GPRfit'],
                                         target_function='predict_MCMC',
                                         namespace=predict_data,
                                         workdir='tmp',
                                         server=OMFIT['MainSettings']['SERVER']['engaging']['server'],
                                         tunnel=OMFIT['MainSettings']['SERVER']['engaging']['tunnel'])
                else:
                    out=OMFITx.remote_python(root,
                                         script=root['LIB']['OMFITlib_GPRfit'],
                                         target_function='predict_MCMC',
                                         namespace=predict_data)
            else:
                out=self.gp[k].predict(Xstar,n=n,full_output=True, noise=True)

            m=out['mean'] #covd=out['cov'] has size len(Xstar) x len(Xstar)
            std= out['std'] # equivalent to squeeze(sqrt(diagonal(covd)))

            # Multiply the outputs by the norm, since data were divided by this before fitting
            m=m*self.gp[k].norm
            d=std*self.gp[k].norm
            M.append(m)
            D.append(d)
        M=squeeze(M); D=squeeze(D)
        return unumpy.uarray(M, D)



    def plot(self, profile=None, ngauss=1):
        """
        Function to conveniently plot the input data and the result of the fit.

        Inputs:
        -----------

        Profile: int, optional
            Profile to evaluate if more than one has been computed and include in the gp
            object. To call the nth profile, set profile=n. If None, it will return an
            array of arrays.

        ngauss: int, optional
            Number of shaded standard deviations

        Outputs:
        -----------
        None

        """
        if profile is None:
            profile=range(len(self.gp))

        Xstar=linspace(0,numpy.nanmin([1.2,numpy.nanmax(self.xx+0.1)]),1000)
        for k in profile:
            ua = self(dd,0,k)
            m, d = nominal_values(ua), std_devs(ua)
            pyplot.errorbar(self.xx[k,:],self.yy[k,:],self.ey[k,:],color='b',linestyle='')
            pyplot.plot(Xstar,m,linewidth=2,color='g')
            for kk in range(1,ngauss+1):
                pyplot.fill_between(Xstar, m-d*kk, m+d*kk, alpha=0.25, facecolor='g')
        pyplot.axvline(0,color='k')
        pyplot.axhline(0,color='k')
