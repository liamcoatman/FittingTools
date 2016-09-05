"""
Total least squares fit to data with intrinsic variance. 

This borrows heavily from the accompanying code to chapter 8 of 
Statistics, Data Mining, and Machine Learning in Astronomy by 
Zeljko Ivezic, Andrew Connolly, Jacob VanderPlas, and Alex Gray.

Some code is also borrowed from the blog of Jake VanderPlas:
http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/ 

Following Hogg et al. 2010, arXiv 1008.4686, the fit is done in terms of 
(theta, b), where theta is the angle the line makes with the horizontal axis and 
b is the perpendicular distance from the line to the origin. Uniform priors are placed
on both of these parameters.  

The corner_plot function, written by Angus Williams, can be found here
https://github.com/anguswilliams91/CornerPlot 

"""

import pandas as pd 
import numpy as np 
import emcee 
from corner_plot import corner_plot
import matplotlib.pyplot as plt 
from PlottingTools.plot_setup import figsize, set_plot_properties
import palettable 


def scatter_plot(trace):


    # Configure style of plot 
    set_plot_properties() 

    # Change color map 
    cs = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    # Set up figure 
    fig, ax = plt.subplots(figsize=figsize(0.75, vscale=0.9))

    # Get data
    xi, yi, dxi, dyi, rho_xy = get_data() 
    xi = xi + 3.0

    # Plot data 
    ax.errorbar(xi, yi, xerr=dxi, yerr=dyi, linestyle='', color='grey', alpha=0.4, zorder=2)
    ax.scatter(xi, yi, color=cs[1], s=8, zorder=3)

    logx = np.linspace(3.2, 4, 50)
  
    #----------------Plot fit--------------------------------   
    m, b = trace[:2]
    yfit = b[:, None] + m[:, None] * (logx - 3.0) 
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(logx, mu, 'k', linestyle='-', zorder=5)
    ax.fill_between(logx, mu - sig, mu + sig, color=palettable.colorbrewer.qualitative.Pastel1_6.mpl_colors[1], zorder=1)

    #--------------------------------------------------------

    ax.set_xlim(3.3, 4)
    ax.set_ylim(ax.get_xlim())

    ax.set_xlabel(r'log FWHM H$\alpha$ [km~$\rm{s}^{-1}$]')
    ax.set_ylabel(r'log FWHM H$\beta$ [km~$\rm{s}^{-1}$]')

    fig.tight_layout()

    # plt.show()

    return None 

def get_m_b(beta):

    """Translate between typical slope-intercept representation, 
    and the normal vector representation"""

    b = np.dot(beta, beta) / beta[1]
    m = -beta[0] / beta[1]
    return m, b


def TLS_log_likelihood(theta, X, dX, log=True):

    """
    This is taken from chapter 8 of Statistics, Data Mining, and Machine Learning 

    The only modification I have made is to add the intrinsic variance V as an extra
    parameter in the model. 

    Compute the total least squares log-likelihood

    This uses Hogg et al eq. 29-32

    Parameters
    ----------
    v : ndarray
        The normal vector to the linear best fit.  shape=(D,).
        Note that the magnitude |v| is a stand-in for the intercept.
    X : ndarray
        The input data.  shape = [N, D]
    dX : ndarray
        The covariance of the errors for each point.
        For diagonal errors, the shape = (N, D) and the entries are
        dX[i] = [sigma_x1, sigma_x2 ... sigma_xD]
        For full covariance, the shape = (N, D, D) and the entries are
        dX[i] = Cov(X[i], X[i]), the full error covariance.
    V: intrinsic variance 
    log: return log likelihood 
 
    Returns
    -------
    logL : float
        The log-likelihood of the model v given the data.
    or p: probability of getting measuremnt given true value 

    Notes
    -----
    This implementation follows Hogg 2010, arXiv 1008.4686
    """
    
    v = theta[:2]

    if len(theta) == 3:
        V = theta[2]**2
    else:
        V = 0.0 

    # check inputs
    X, dX, v = map(np.asarray, (X, dX, v))
    N, D = X.shape
    assert v.shape == (D,)
    assert dX.shape in ((N, D), (N, D, D))

    v_norm = np.linalg.norm(v)
    v_hat = v / v_norm

    # eq. 30
    Delta = np.dot(X, v_hat) - v_norm

    # eq. 31
    if dX.ndim == 2:
        # diagonal covariance
        Sig2 = np.sum((v_hat * dX) ** 2, 1)
    else:
        # full covariance
        Sig2 = np.dot(np.dot(v_hat, dX), v_hat)

    if log:
        return (-0.5 * np.sum(np.log(2 * np.pi * (Sig2 + V))) - np.sum(0.5 * Delta** 2 / (Sig2 + V)))
    
    else: 
        return (2.*np.pi*(Sig2 + V))**-0.5 * np.exp(-Delta**2.0 / (2.0*(Sig2 + V)))


def log_prior_sigma_i(sigma_i):

    """
    Prior on the intrinsic dispersion sigma_i 
    """

    if sigma_i < 0.0 or sigma_i > 100.0:
        return -np.inf 
    else:
        return -np.log10(sigma_i) # Jeffreys prior 


def log_posterior(theta, X, dX):

    if log_prior_sigma_i(theta[2]) == -np.inf:
        return -np.inf
    else:
        return log_prior_sigma_i(theta[2]) + TLS_log_likelihood(theta, X, dX, log=True)

def get_data():

    # Read file 
    df = pd.read_csv('balmer_widths.csv', index_col=0)
    
    # Convert to log quantities 
    xi = df.FWHM_Broad_Ha.apply(np.log10) - 3.0
    dxi = df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha / np.log(10)  
    yi = df.FWHM_Broad_Hb.apply(np.log10)
    dyi = df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb / np.log(10) 

    # Get numpy arrays 
    xi = xi.values
    yi = yi.values
    dxi = dxi.values
    dyi = dyi.values 

    # assume zero covariance 
    rho_xy = np.zeros(len(xi))

    return xi, yi, dxi, dyi, rho_xy 

def fit_model():

    # get data
    xi, yi, dxi, dyi, rho_xy = get_data() 

    # change format of input 
    X = np.vstack((xi, yi)).T
    dX = np.zeros((len(xi), 2, 2))
    dX[:, 0, 0] = dxi ** 2
    dX[:, 1, 1] = dyi ** 2
    dX[:, 0, 1] = dX[:, 1, 0] = rho_xy * dxi * dyi

    # Set up computation. The number of trace results will be nwalkers * nsteps
    
    ndim = 3  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 10000  # "burn-in" period to let chains stabilize
    nsteps = 20000  # number of MCMC steps to take
    ncores = 8
    
    np.random.seed(0)

    # Start walkers in tight gaussian ball around best fit parameters 
    p0 = [-1.54, 1.59, 0.04]
    std = 0.01*np.ones_like(p0)
    starting_guesses = emcee.utils.sample_ball(p0, std, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[X, dX], threads=ncores)
    sampler.run_mcmc(starting_guesses, nsteps)

    # Reshape trace, remove burnin. 
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    
    # Convert to slope, intercept representation 
    for i in range(trace.shape[1]):
        trace[:2, i] = get_m_b(trace[:2, i])

    np.save('trace', trace) 

    return None

if __name__ == '__main__':

    # fit model 
    # fit_model()

    # plot data + model 
    trace = np.load('trace.npy')
    scatter_plot(trace)

    # triangle plot of chains 
    trace[1, :] = (10**trace[1, :]) / 1e3 

    corner_plot(trace.T, 
                axis_labels=[r'$\alpha$', r'$\beta$', r'$\sigma_I$'], 
                wspace=0.0,
                hspace=0.0,
                nticks=4,
                nbins=80,
                figsize=figsize(0.7, 1),
                fontsize=11, 
                tickfontsize=11)  

    plt.show() 