#  Given (x,y,sigma_y), check statistical consistency of uncertainties and provide an 
# improved estimate based on their spatio-temporal scattering using different machine
# learning techniques. mostly Gaussian Process Regression.
#
# F.Sciortino, 10/10/17


def profile_fitting(x, y, err_y=None, kernel='SE', s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False, method='GPR'):
    """Interpolate profiles and uncertainties over a dense grid. Also return the maximum 
    value of the smoothed data.
    
    This function can use Gaussian process regression or splines. When the former is adopted, both the 
    mean and the standard deviation of the updated profile are returned. Use the spline interpolation
    only as a cross-check.
    
    We allow the use of several GPR kernels so as to assess their performance in confidently 
    predicting profiles and assessing their uncertainty consistency. 

    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    kernel : str, optional
        Type of kernel to be used. At this stage, we create the kernel internally, but in the future
        it would be better to do it externally and just give a gptools kernel object as an argument
        More kernels should be added over time. 
    s_guess : float, optional
        Initial guess for the signal variance. Default is 0.2.
    s_max : float, optional
        Maximum value for the signal variance. Default is 10.0
    l_guess : float, optional
        Initial guess for the covariance length scale. Default is 0.03.
    fixed_l : bool, optional
        Set to True to hold the covariance length scale fixed during the MAP
        estimate. This helps mitigate the effect of bad points. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    method : {'GPR', 'spline'}, optional
        Method to use when interpolating. Default is 'GP' (Gaussian process
        regression). Can also use a cubic spline.
    """
    grid = scipy.linspace(max(0, x.min()), min(0.08, x.max()), 1000)
    if method == 'GPR':
        # hp is the hyperprior. A product of kernels is a kernel, so the joint hyperprior is 
        # just the product of hyperpriors for each of the hyperparameters of the individual 
        # priors. gptools offers convenient functionalities to create joint hyperpriors.
        hp = (
            gptools.UniformJointPrior([(0, s_max),]) *
            gptools.GammaJointPriorAlt([l_guess,], [0.1,])
        )
        # Define the kernel type amongst the implemented options. 
        if kernel=='SE':
            k = gptools.SquaredExponentialKernel(
                # param_bounds=[(0, s_max), (0, 2.0)],
                hyperprior=hp,
                initial_params=[s_guess, l_guess],
                fixed_params=[False, fixed_l]
            )
        elif kernel=='gibbs':
            ValueError('Implementation not completed yet')
            k = gptools.GibbsKernel1dTanh( # this has 5 hyperparameters
                # # param_bounds=[(0, s_max), (0, 2.0)],
                # hyperprior=hp,
                # initial_params=[s_guess, l_guess],
                # fixed_params=[False, fixed_l]
            )
        elif kernel=='matern':
            ValueError('Implementation not completed yet')
            k = gptools.Matern52Kernel( # this has 3 hyperparameters in 1D
                # # param_bounds=[(0, s_max), (0, 2.0)],
                # hyperprior=hp,
                # initial_params=[s_guess, l_guess],
                # fixed_params=[False, fixed_l]
            )
        elif isinstance(kernel,gptools.Kernel):
            k=kernel
        else:
            ValueError('Only the SE kernel is currently defined! Break here.')

        gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y)
        gp.optimize_hyperparameters(verbose=True, random_starts=100)
        m_gp, s_gp = gp.predict(grid)
        i = m_gp.argmax()
    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=2*len(x)
        )(grid)
        if scipy.isnan(m_gp).any():
            print(x)
            print(y)
            print(err_y)
        i = m_gp.argmax()
    else:
        raise ValueError("Undefined method %s" % (method,))
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        if method == 'GP':
            a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        a.axvline(grid[i])
    
    if method == 'GPR':
        return (m_gp, s_gp, m_gp[i], s_gp[i])
    else:
        return (m_gp,m_gp[i])