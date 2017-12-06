# bayesimp_helper script
# Contains functions that support operation of bayesimp
#
# sciortino, Sep.2017
###########################################
import scipy
import matplotlib.pyplot as plt
import numpy as np
from spline_MC import spline_MC

def get_systematic_uncertainty(signal, t0=0, plot=False):
    ''' Update signal uncertainties based on the scattering that occurs before the 
    expected signal time.  

    Add the expected systematic uncertainty in quadrature with the current 
    diagnostic uncertainties    

    '''
    t = signal.t
    y= signal.y
    std_y = signal.std_y

    std_y_sys = np.zeros(std_y.shape)
    for chord in range(y.shape[1]):
        std_y_sys[:,chord] = abs(min(y[:,chord][t<t0]))+std_y[:,chord][np.argmin(y[:,chord][t<t0])]
    updated_std_y = np.sqrt(std_y**2 + std_y_sys**2)

    signal_u = signal
    signal_u.std_y = updated_std_y
    
    if plot:
        plt.figure()
        plt.errorbar(t, y, std_y, marker='s',mfc='blue',mec='green', ecolor='blue')
        plt.xlabel('t [s]', fontsize = 14)
        plt.ylabel('Signal Amplitude [A.U.]', fontsize = 14)
        plt.figure()
        plt.errorbar(signal_u.t, signal_u.y, signal_u.std_y, marker='s',mfc='red',mec='green',ecolor='red')
        plt.xlabel('t [s]', fontsize = 14)
        plt.ylabel('Signal Amplitude [A.U.]', fontsize = 14)

        # Monte Carlo interpolation of augmented uncertainties
        xout = np.linspace(min(signal_u.t),max(signal_u.t), 1000)
        spline_MC(signal_u.t,signal_u.y,xout,signal_u.std_y[:,0],1000)

    return signal_u

def set_min_uncertainty(signal, threshold=0.05):
    """ Set a minimum relative uncertainty for the Hirex-Sr signals.

    Parameters
    -----------
    signal: instance of the Signal class of bayesimp
        Diagnostic signal of which we wish to modify the uncertainties
    threshold: float, optional
        The minimum relative uncertainty to be accepted. All relative uncertainties
        that are smaller than this are set to the value of threshold. This is applied 
        to the unnormalized signals and then the uncertainty is propagated to the 
        normalized quantities. 

    """
    # Increase Hirex-Sr uncertainties to be a rel error of 5% minimum (JUST FOR TESTING)
    corrected_unc=signal.std_y/signal.y<=0.05
    signal.std_y[corrected_unc]=0.05*signal.y[corrected_unc]

    # correction for normalized uncertainties
    if signal.s/signal.m<=0.05:
        signal.s=0.05*signal.m

    signal.std_y_norm=scipy.sqrt((signal.std_y / signal.m)**2.0 + ((signal.y / signal.m)*(signal.s / signal.m))**2.0)

#####################
def get_tau_imp(signal, plots=True):
    """
    Calculate the impurity confinement time, defined as the time constant 
    obtained from an exponential fit of the spectroscopic signal after peaking. 

    If signal is an array of signals, finds the impurity confinement time for each, 
    then  provides the mean and standard deviation within the set. 

    NB: signal is assumed to be an instance of the Signal class defined in bayesimp.
    """
    tau_imp=np.zeros((signal.y.shape[1],1))
    time=signal.t
    if plots==True:
        f1 = plt.figure()
        ax1 = f1.add_subplot(2,1,1)
        ax2 = f1.add_subplot(2,1,2)

    for i in range(signal.y.shape[1]):    
        sig=signal.y[:,i]
        
        # Subtract the minimum/background
        minsig=np.nanmin(sig)
        sig=sig-minsig

        time_clean=np.asarray([value for value in time if not np.isnan(sig[np.where(time==value)[0]])])
        sig=np.asarray([value for value in sig if not np.isnan(value)])

        maxidx=np.where(sig==max(sig))[0][0]

        log_sig=np.log(sig)
        log_sig_to_fit=log_sig[maxidx+3:]
        time_to_fit=np.asarray(time_clean[maxidx+3:])
        fit_coeffs=np.polyfit(time_to_fit,log_sig_to_fit,1)

        tau_imp[i]=-1/fit_coeffs[0]
        #print i,tau_imp[i]

        if plots==True:
            ax1.plot(time_clean,sig)
            ax1.plot(time_clean[maxidx],sig[maxidx],'ro')
            ax1.plot(time_clean[maxidx+3],sig[maxidx+3],'go')

            ax2.plot(time_to_fit,log_sig_to_fit)
            ax2.plot(time_to_fit, fit_coeffs[1]+fit_coeffs[0]*np.asarray(time_to_fit),'r')

    tau_imp_mean=np.nanmean(tau_imp)
    tau_imp_std=np.nanstd(tau_imp)

    return (tau_imp,tau_imp_mean, tau_imp_std)