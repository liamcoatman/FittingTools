from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import emcee
from scipy.stats import multivariate_normal
from GusUtils import gus_utils as gu
import palettable
import matplotlib as mpl
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm




# compute the ellipse pricipal axes and rotation from covariance
def get_principal(sigma_x, sigma_y, rho_xy):

    sigma_xy2 = rho_xy * sigma_x * sigma_y

    alpha = 0.5 * np.arctan2(2 * sigma_xy2,
                             (sigma_x ** 2 - sigma_y ** 2))
    tmp1 = 0.5 * (sigma_x ** 2 + sigma_y ** 2)
    tmp2 = np.sqrt(0.25 * (sigma_x ** 2 - sigma_y ** 2) ** 2 + sigma_xy2 ** 2)

    return np.sqrt(tmp1 + tmp2), np.sqrt(tmp1 - tmp2), alpha


# plot ellipses
def plot_ellipses(x, y, sigma_x, sigma_y, rho_xy, factor=2, ax=None):

    from matplotlib.patches import Ellipse

    if ax is None:
        ax = plt.gca()

    sigma1, sigma2, alpha = get_principal(sigma_x, sigma_y, rho_xy)

    for i in range(len(x)):
        ax.add_patch(Ellipse((x[i], y[i]),
                             factor * sigma1[i], factor * sigma2[i],
                             alpha[i] * 180. / np.pi,
                             fc='none', ec='k'))

def convert_to_stdev(logL):

    """
    Given a grid of log-likelihood values, convert them to cumulative
    standard deviation.  This is useful for drawing contours from a
    grid of likelihoods.
    """
    sigma = np.exp(logL)

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)

def compute_sigma_level(trace1, trace2, nbins=20):

    """From a set of traces, bin by number of standard deviations"""

    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    
    """Plot traces and contours"""
    
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])

    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)

    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    
    
def plot_MCMC_model(ax, xdata, ydata, sigma_y, trace, linestyle='--'):

    """Plot the linear model and 2sigma contours"""
    
    # ax.plot(xdata, ydata, 'ok')
    # ax.errorbar(xdata, ydata, yerr=sigma_y, linestyle='', color='black')

    m, b = trace[:2]
    xfit = np.linspace(xdata.min(), xdata.max(), 10)
    yfit = b[:, None] + m[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, 'k', linestyle=linestyle, zorder=5)
    ax.fill_between(xfit, mu - sig, mu + sig, color=palettable.colorbrewer.qualitative.Pastel1_3.mpl_colors[1], zorder=1)

def plot_MCMC_results(xdata, ydata, sigma_y, trace, colors='k'):

    """Plot both the trace and the model together"""

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, sigma_y, trace)

def get_m_b(beta):

    """Translate between typical slope-intercept representation, 
    and the normal vector representation"""

    b = np.dot(beta, beta) / beta[1]
    m = -beta[0] / beta[1]
    return m, b


def get_beta(m, b):
    denom = (1 + m * m)
    return np.array([-b * m / denom, b / denom])


def TLS_log_likelihood(theta, X, dX, log=True):

    """Compute the total least squares log-likelihood

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
    or p: promability of getting measuremnt given true value 

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

def outlier_likelihood(Vb, X, dX):

    """
    I don't think this works - updated by Gus. 
    """
    
    root2pi = np.sqrt(2 * np.pi) 

    # Surely can eliminate this loop
    tmp = []
    for i in np.arange(len(X)):
        m = [0., 0.]
        cov = dX[i] + np.identity(2) * Vb
        tmp.append(multivariate_normal(m, cov).pdf([X[i, 0], X[i, 1]]))

    return np.array(tmp)  


def full_log_likelihood(theta, X, dX):

    """Full likelihood of the model (including outliers)"""

    v = theta[:2] # Normal vector to linear best fit
    Pout = theta[3] # Outlier fraction
    Sout = 100 # Sigma of outlier distribution 

    return np.sum(np.log((1 - Pout) * TLS_log_likelihood(theta, X, dX, log=False) + Pout * outlier_likelihood(Sout, Pout, X, dX))) 

def full_log_likelihood(params,data):
    """Full likelihood of the model (including outliers)"""
    m,b,sigma_intrinsic,sigma_outlier,outlier_fraction = params
    return np.sum(np.log((1.-outlier_fraction)*likelihood_line([m,b,sigma_intrinsic],data)+\
            outlier_fraction*outlier_distribution(sigma_outlier,data)))


def full_log_posterior(theta, X, dX):

    """
    Full posterior, including outlier distribution
    """
    
    if log_prior_V(theta[2]) + log_prior_Pb(theta[3]) == -np.inf:
        return -np.inf
    else:
        return log_prior_Sin(theta[2]) + log_prior_Pb(theta[3]) + log_prior_Sout(theta[4]) + full_TLS_log_likelihood(theta, X, dX)


def log_prior_Sin(Sin):

    """
    Prior on the intrinsic sigma Sin
    """

    if Sin < 0.0 or Sin > 100.0:
        return -np.inf 
    else:
        return -np.log10(Sin) # Jeffreys prior 

def log_prior_Sout(Sout):

    """
    Prior on the outlier sigma Sout
    """

    if Sout < 0.0:
        return -np.inf 
    else:
        return -np.log10(Sout) # Jeffreys prior 

def log_prior_Pout(Pout):

    """
    Uniform prior on Pb, the fraction of bad points
    """

    if Pout < 0.0 or Pout > 1.0:
        return -np.inf 
    else:
        return 0.0

def log_posterior(theta, X, dX):

    if log_prior_Sin(theta[2]) == -np.inf:
        return -np.inf
    else:
        return log_prior_Sin(theta[2]) + TLS_log_likelihood(theta, X, dX, log=True)

def get_data(data = 'linewidths'):

    if data == 'hogg':

        from astroML.datasets import fetch_hogg2010test
        data = fetch_hogg2010test()
        data = data[5:]  # no outliers
        xi = data['x']
        yi = data['y']
        dxi = data['sigma_x']
        dyi = data['sigma_y']
        rho_xy = data['rho_xy'] 

  
    elif data == 'linewidths_corrected_ha':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort('Blueshift_CIV_Ha')

        xi = df.Blueshift_CIV_Ha.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Ha_Corr 
        blueshift_err = np.sqrt(df.Median_Broad_Ha_Err**2 + df.Median_CIV_BEST_Err**2) 
        dxi = blueshift_err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha_Corr)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    elif data == 'linewidths_corrected_hb':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Hb == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort('Blueshift_CIV_Hb')

        xi = df.Blueshift_CIV_Hb.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / df.FWHM_Broad_Hb
        blueshift_err = np.sqrt(df.Median_Broad_Hb_Err**2 + df.Median_CIV_BEST_Err**2) 
        dxi = blueshift_err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))


    elif data == 'linewidths_hb_equiv':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_CIV_BEST == 0]
        df = df[df.BAL_FLAG != 1]
        
        df = df.sort('Blueshift_CIV_Ha')

        fwhm = df['FWHM_Broad_Ha'] * 1.e-3 
        fwhm_err = df['FWHM_Broad_Ha_Err'] * 1.e-3 
        
        fwhm_hb = 1.22e3 * np.power(fwhm, 0.97)
        fwhm_hb_err = 1.22e3 * np.power(fwhm, 0.97-1.0) * 0.97 * fwhm_err

        xi = df.Blueshift_CIV_Ha.values / 1.0e3 
        yi = df.FWHM_CIV_BEST.values / fwhm_hb
        blueshift_err = np.sqrt(df.Median_Broad_Ha_Err**2 + df.Median_CIV_BEST_Err**2) 
        dxi = blueshift_err.values / 1.0e3 
        dyi = yi * np.sqrt((df.FWHM_CIV_BEST_Err / df.FWHM_CIV_BEST)**2 + (fwhm_hb_err / fwhm_hb)**2) 
        dyi = dyi.values

        # minimum 10% error 
        # dxi[(dxi / xi) < 0.1] = 0.1 * xi[(dxi / xi) < 0.1] 
        # dyi[(dyi / yi) < 0.1] = 0.1 * yi[(dyi / yi) < 0.1] 
    
        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    elif data == 'balmer_widths':

        df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    
        df = df[df.WARN_Ha == 0]
        df = df[df.WARN_Hb == 0]
    
        xi = df.FWHM_Broad_Ha.apply(np.log10) - 3.0 # /1e3 
        dxi = df.FWHM_Broad_Ha_Err / df.FWHM_Broad_Ha / np.log(10)  
        yi = df.FWHM_Broad_Hb.apply(np.log10)
        dyi = df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb / np.log(10) 

        xi = xi.values
        yi = yi.values
        dxi = dxi.values
        dyi = dyi.values 

        # assume zero covariance 
        rho_xy = np.zeros(len(xi))

    return xi, yi, dxi, dyi, rho_xy 

def get_mock_data():
    
    """Generate a mock data set plus a few outliers"""
    
    slope = 2.0
    intercept = 0
    sigma_intrinsic = 0.5 
    npoints = 20 

    #generate random slope and intercept
    # slope = np.float(np.random.uniform(0.1,4.,1))
    # intercept = np.float(np.random.uniform(-10.,10.,1))
    # sigma_intrinsic = np.float(np.random.uniform(0.1,2.,1))

    theta = np.arctan(slope)
    
    #generate coordinates along the line with random intrinsic spread
    gamma = np.random.uniform(1.,10.,npoints)
    delta = np.random.normal(loc=0., scale=sigma_intrinsic, size=npoints)
    
    #now transform to x and y
    sint, cost = np.sin(theta), np.cos(theta)
    xp = cost*gamma - sint*delta 
    yp = sint*gamma + cost*delta + intercept
    
    #now generate x and y errors
    dx = np.abs(np.random.normal(loc=0., scale=0.3, size=npoints))
    dy = np.abs(np.random.normal(loc=0., scale=0.3, size=npoints))
    rho_xy = np.random.uniform(-1., 1., size=npoints) #correlation parameters
       
    #now scatter xp and yp by these errors
    x, y = np.zeros_like(xp), np.zeros_like(xp)
    for i in np.arange(npoints):
        cov = [[dx[i]**2., rho_xy[i]*dx[i]*dy[i]], [rho_xy[i]*dx[i]*dy[i], dy[i]**2.]]
        mean = [xp[i], yp[i]]
        xi, yi = np.random.multivariate_normal(mean, cov, 1).T
        x[i], y[i] = np.float(xi), np.float(yi)
    
    return np.vstack((x, y, dx, dy, rho_xy)).T, [slope, intercept, sigma_intrinsic]


def fit_model_linewidths():

    # data, ptrue = get_mock_data() 
    # xi, yi, dxi, dyi, rho_xy = data.T  

    # get data
    xi, yi, dxi, dyi, rho_xy = get_data(data='linewidths_hb_equiv') 

    # change format of input 
    X = np.vstack((xi, yi)).T
    dX = np.zeros((len(xi), 2, 2))
    dX[:, 0, 0] = dxi ** 2
    dX[:, 1, 1] = dyi ** 2
    dX[:, 0, 1] = dX[:, 1, 0] = rho_xy * dxi * dyi

    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    
    ndim = 3  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 100000  # "burn-in" period to let chains stabilize
    nsteps = 200000  # number of MCMC steps to take
    ncores = 24
    
    # set theta near the maximum likelihood, with 
    np.random.seed(0)
    
    #starting_guesses = np.random.random((nwalkers, ndim))
    # start in tight gassian ball around best fit paramters - best way to explore space 
    p0 = [-0.3, 0.6, 0.1]
    # p0 = [-10.4, 4.6] # hogg data, zero intrinsic variance
    std = 0.01*np.ones_like(p0)
    starting_guesses = emcee.utils.sample_ball(p0, std, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[X, dX], threads=ncores)
    sampler.run_mcmc(starting_guesses, nsteps)

    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    
    # convert to m, b 
    for i in range(trace.shape[1]):
        trace[:2, i] = get_m_b(trace[:2, i])

    np.save('trace', trace)

    return None

def fit_model_balmerwidths():

    # data, ptrue = get_mock_data() 
    # xi, yi, dxi, dyi, rho_xy = data.T  

    # get data
    xi, yi, dxi, dyi, rho_xy = get_data(data='balmer_widths') 

    # change format of input 
    X = np.vstack((xi, yi)).T
    dX = np.zeros((len(xi), 2, 2))
    dX[:, 0, 0] = dxi ** 2
    dX[:, 1, 1] = dyi ** 2
    dX[:, 0, 1] = dX[:, 1, 0] = rho_xy * dxi * dyi

    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    
    ndim = 3  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 100000  # "burn-in" period to let chains stabilize
    nsteps = 200000  # number of MCMC steps to take
    ncores = 24
    
    # set theta near the maximum likelihood, with 
    np.random.seed(0)

    # start in tight gassian ball around best fit paramters - best way to explore space 
    # remember: this is v not the slope and intercept
    p0 = [-1.54, 1.59, 0.04]
    std = 0.01*np.ones_like(p0)
    starting_guesses = emcee.utils.sample_ball(p0, std, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[X, dX], threads=ncores)
    sampler.run_mcmc(starting_guesses, nsteps)

    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    
    # convert to m, b 
    for i in range(trace.shape[1]):
        trace[:2, i] = get_m_b(trace[:2, i])

    np.save('trace', trace)

    return None

if __name__ == "__main__":

    fit_model_linewidths()

    # xi, yi, dxi, dyi, rho_xy = get_data(data='balmer_widths') 
    # fig, ax = plt.subplots()
    # ax.scatter(xi, yi)
    # from scipy import stats
    # slope, intercept, r_value, p_value, std_err = stats.linregress(xi, yi)
    # xs = np.arange(0, 1, 0.01)
    # ax.plot(xs, slope*xs + intercept)
    # print slope, intercept
    # plt.show() 
    



















#--------------------------------------------------------------------------------------------------


def least_squares_emcee():

    """
    Perform least squares fit using EMCEE 
    """

    from astroML.datasets import fetch_hogg2010test
    import emcee
    

    df = pd.read_csv('/home/lc585/Dropbox/IoA/nirspec/tables/masterlist_liam.csv', index_col=0)
    
    df = df[df.WARN_Ha == 0]
    df = df[df.WARN_Hb == 0]
    
    xi = df.FWHM_Broad_Ha.apply(np.log10)
    yi = df.FWHM_Broad_Hb.apply(np.log10)
    dyi = df.FWHM_Broad_Hb_Err / df.FWHM_Broad_Hb / np.log(10) 

    xi = xi.values
    yi = yi.values
    dyi = dyi.values 

    # Define our posterior
    # Note that emcee requires log-posterior
    
    def log_prior(theta):
        alpha, beta = theta
        # uniform prior on arctan(slope)
        return -1.5 * np.log(1 + beta** 2) 
    
    def log_likelihood(theta, x, y, dy):
        alpha, beta = theta
        y_model = alpha + beta * x
        return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + (y - y_model)** 2 / dy** 2)
    
    def log_posterior(theta, x, y, dy):
        return log_prior(theta) + log_likelihood(theta, x, y, dy)
    

    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    
    ndim = 2  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 2000  # number of MCMC steps to take
    
    p0 = [0.05, 1.0] # hogg data, zero intrinsic variance
    std = 0.01*np.ones_like(p0)
    starting_guesses = emcee.utils.sample_ball(p0, std, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xi, yi, dyi])
    o = sampler.run_mcmc(starting_guesses, nsteps)

    # sampler.chain is of shape (nwalkers, nsteps, ndim)
    # we'll throw-out the burn-in points and reshape:
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

    for i in range(trace.shape[1]):
        trace[:2, i] = get_m_b(trace[:2, i])

    plot_MCMC_results(xi, yi, dyi, trace)

    plt.show()


def TLS_log_likelihood(theta, X, dX):
   
    """Compute the total least squares log-likelihood
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
    Returns
    -------
    logL : float
        The log-likelihood of the model v given the data.
    Notes
    -----
    This implementation follows Hogg 2010, arXiv 1008.4686
    """
    
    #get v from m and b 
    v = get_beta(theta[0], theta[1])
    
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
    return (-0.5 * np.sum(np.log(2 * np.pi * Sig2))
            - np.sum(0.5 * Delta ** 2 / Sig2))

def tls_optimisation():

    """
    TLS fit by optimising likelihood
    """

    from astroML.datasets import fetch_hogg2010test
    from scipy import optimize

    data = fetch_hogg2010test()
    data = data[5:]  # no outliers
    x = data['x']
    y = data['y']
    sigma_x = data['sigma_x']
    sigma_y = data['sigma_y']
    rho_xy = data['rho_xy']

    X = np.vstack((x, y)).T
    dX = np.zeros((len(x), 2, 2))
    dX[:, 0, 0] = sigma_x ** 2
    dX[:, 1, 1] = sigma_y ** 2
    dX[:, 0, 1] = dX[:, 1, 0] = rho_xy * sigma_x * sigma_y

    # Maximise function:
    min_func = lambda theta: -TLS_log_likelihood(theta, X, dX)
    theta_fit = optimize.fmin(min_func,
                             x0=[-1, 1])
    
    # ------------------------------------------------------------
    # Plot the data and fits
    fig, ax = plt.subplots()
        
    #------------------------------------------------------------
    # first let's visualize the data
    ax.scatter(x, y, c='k', s=9)
    plot_ellipses(x, y, sigma_x, sigma_y, rho_xy, ax=ax)
    
    #------------------------------------------------------------
    # plot the best-fit line
    m_fit, b_fit = theta_fit
    x_fit = np.linspace(0, 300, 10)
    ax.plot(x_fit, m_fit * x_fit + b_fit, '-k')
    
    ax.set_xlim(40, 250)
    ax.set_ylim(100, 600)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    
    print 'Best m: {0}, Best b: {1}'.format(m_fit, b_fit)
    print get_beta(theta_fit[0], theta_fit[1])

    plt.show()

# tls_optimisation()

def tls_brute():

    from astroML.datasets import fetch_hogg2010test
    from astroML.plotting.mcmc import convert_to_stdev

    data = fetch_hogg2010test()
    data = data[5:]  # no outliers
    x = data['x']
    y = data['y']
    sigma_x = data['sigma_x']
    sigma_y = data['sigma_y']
    rho_xy = data['rho_xy']

    X = np.vstack((x, y)).T
    dX = np.zeros((len(x), 2, 2))
    dX[:, 0, 0] = sigma_x ** 2
    dX[:, 1, 1] = sigma_y ** 2
    dX[:, 0, 1] = dX[:, 1, 0] = rho_xy * sigma_x * sigma_y

    fig, ax = plt.subplots(figsize=(5,4))
    m = np.linspace(1.7, 2.8, 100)
    b = np.linspace(-60, 110, 100)
    logL = np.zeros((len(m), len(b)))
    
    for i in range(len(m)):
        for j in range(len(b)):
            logL[i, j] = TLS_log_likelihood([m[i], b[j]], X, dX)
    
    ax.contour(m, b, convert_to_stdev(logL.T),
               levels=(0.683, 0.955, 0.997),
               colors='k')


    plt.show()


def tls_emcee():

    from astroML.datasets import fetch_hogg2010test
    import emcee
    import corner

    data = fetch_hogg2010test()
    data = data[5:]  # no outliers
    x = data['x']
    y = data['y']
    sigma_x = data['sigma_x']
    sigma_y = data['sigma_y']
    rho_xy = data['rho_xy']

    X = np.vstack((x, y)).T
    dX = np.zeros((len(x), 2, 2))
    dX[:, 0, 0] = sigma_x ** 2
    dX[:, 1, 1] = sigma_y ** 2
    dX[:, 0, 1] = dX[:, 1, 0] = rho_xy * sigma_x * sigma_y

    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    
    ndim = 2  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 2000  # number of MCMC steps to take
    
    # set theta near the maximum likelihood, with 
    np.random.seed(0)
    
    #starting_guesses = np.random.random((nwalkers, ndim))
    # start in tight gassian ball around best fit paramters - best way to explore space 
    p0 = [2.3, 30.0]
    std = [0.01, 0.1]
    starting_guesses = emcee.utils.sample_ball(p0, std, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, TLS_log_likelihood, args=[X, dX])
    o = sampler.run_mcmc(starting_guesses, nsteps)

    emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    corner.corner(emcee_trace)

    plt.show()

def outlier_rejection_emcee():

    """
    Same as Figure 8.9 in astroML book, but using EMCEE 
    """

    from astroML.datasets import fetch_hogg2010test
    from astroML.plotting.mcmc import convert_to_stdev
    import emcee
    import corner
 
    #----------------------------------------------------------------------
    # This function adjusts matplotlib settings for a uniform feel in the textbook.
    # Note that with usetex=True, fonts are rendered with LaTeX.  This may
    # result in an error if LaTeX is not installed on your system.  In that case,
    # you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=True)
    
    np.random.seed(0)
    
    #------------------------------------------------------------
    # Get data: this includes outliers
    data = fetch_hogg2010test()
    xi = data['x']
    yi = data['y']
    dyi = data['sigma_y']
    
    
    #----------------------------------------------------------------------
    # First model: no outlier correction
    # define priors on beta = (slope, intercept)

    # define priors on theta = (slope, intercept)
    def log_prior_slope(theta):
        # uniform prior on arctan(slope)
        # d[arctan(x)]/dx = 1 / (1 + x^2)
        m, b = theta[0], theta[1]
        prob_b = 0. 
        prob_m = -1.5 * np.log(1 + m**2)  
        return prob_m + prob_b 
    
    def log_likelihood_M0(theta, x, y, dy):
        m, b = theta[0], theta[1]
        y_model = b + m * x
        return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + (y - y_model)** 2 / dy** 2)
    
    def log_posterior_M0(theta, x, y, dy):
        return log_prior_slope(theta) + log_likelihood_M0(theta, x, y, dy)


    #----------------------------------------------------------------------
    # Second model: nuisance variables correcting for outliers
    # This is the mixture model given in equation 17 in Hogg et al
    
    # uniform prior on Pb, the fraction of bad points
    def log_prior_Pb(theta):
        if theta[2] < 0.0 or theta[2] > 1.0:
            return -np.inf 
        else:
            return 0.
        
    # uniform prior on Yb, the centroid of the outlier distribution
    def log_prior_Yb(theta):
        if theta[3] < -10000.0 or theta[3] > 10000.0:
            return -np.inf  
        else:
            return 0.
    
    # uniform prior on log(sigmab), the spread of the outlier distribution
    def log_prior_sigma_b(theta):
        if theta[4] < 0.0 or np.log(theta[4]) < 0. or np.log(theta[4]) > 10.0:
            return -np.inf
        else:
            return -np.log(theta[4])
    
    # set up the expression for likelihood
    def mixture_log_likelihood(theta, x, y, dyi):
        
        """
        Equation 8.67 in ML textbook
        
        Pb = probability of any point is an outlier
        Yb = mean of background
        sigma_b = square root of variance of background 
        """
        
        m, b, Pb, Yb, sigma_b = theta 
        
        model = b + m * x 
        
        Vi = dyi ** 2
        Vb = sigma_b ** 2
    
        root2pi = np.sqrt(2 * np.pi)
    
        L_in = (1. / root2pi / dyi * np.exp(-0.5 * (y - model) ** 2 / Vi))
    
        L_out = (1. / root2pi / np.sqrt(Vi + Vb) * np.exp(-0.5 * (y - Yb) ** 2 / (Vi + Vb)))
    
        return np.sum(np.log((1 - Pb) * L_in + Pb * L_out))
    
    def log_posterior(theta, x, y, dyi):
        
        if log_prior_slope(theta) + log_prior_Pb(theta) + log_prior_Yb(theta) + log_prior_sigma_b(theta) == -np.inf:
            return -np.inf
        else:
            return log_prior_slope(theta) + log_prior_Pb(theta) + log_prior_Yb(theta) + log_prior_sigma_b(theta) + mixture_log_likelihood(theta, x, y, dyi)

    #------------------------------------------------------------
    # Third model - cannot have a discrete variable in emcee. 

    #------------------------------------------------------------
    # plot the data
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25,
                        bottom=0.1, top=0.95, hspace=0.2)
    
    # first axes: plot the data
    ax1 = fig.add_subplot(221)
    ax1.errorbar(xi, yi, dyi, fmt='.k', ecolor='gray', lw=1)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    
    #------------------------------------------------------------
    # Go through models; compute and plot likelihoods
    models = [log_posterior_M0]
    linestyles = [':', '--', '-']
    labels = ['no outlier correction\n(dotted fit)',
              'mixture model\n(dashed fit)',
              'outlier rejection\n(solid fit)']
    
    
    x = np.linspace(0, 350, 10)
    
    bins = [(np.linspace(140, 300, 51), np.linspace(0.6, 1.6, 51)),
            (np.linspace(-40, 120, 51), np.linspace(1.8, 2.8, 51)),
            (np.linspace(-40, 120, 51), np.linspace(1.8, 2.8, 51))]
    

    #---------------------------------------------------------------
    # First model 

    ndim = 2  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 2000  # number of MCMC steps to take
    
    # set theta near the maximum likelihood, with 
    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_M0, args=[xi, yi, dyi])
    sampler.run_mcmc(starting_guesses, nsteps)
    
    # sampler.chain is of shape (nwalkers, nsteps, ndim)
    # we'll throw-out the burn-in points and reshape:
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

    plot_MCMC_model(ax1, xi, yi, dyi, trace, linestyle=linestyles[0])
       
    # plot the likelihood contours
    ax = plt.subplot(222 + 0)

    plot_MCMC_trace(ax, xi, yi, trace, False, colors='k')

    ax.set_xlabel('slope')
    ax.set_ylabel('intercept')

    ax.grid(color='gray')
    ax.yaxis.set_major_locator(plt.MultipleLocator(40))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    
    ax.text(0.98, 
            0.98, 
            labels[0], 
            ha='right', 
            va='top',
            bbox=dict(fc='w', ec='none', alpha=0.5),
            transform=ax.transAxes)

    ax.set_ylim(bins[0][0][0], bins[0][0][-1])
    ax.set_xlim(bins[0][1][0], bins[0][1][-1])
    
    #---------------------------------------------------------------
    # Second model 

    ndim = 5  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 2000  # number of MCMC steps to take
    
    # set theta near the maximum likelihood, with 
    np.random.seed(0)
    
    # starting guess
    theta0 = [2.0, 26.0, 0.1, 400, 5.0]
    std = [0.1, 1.0, 0.001, 10.0, 0.1]
    starting_guesses = emcee.utils.sample_ball(theta0, std, nwalkers)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xi, yi, dyi])
    sampler.run_mcmc(starting_guesses, nsteps)


    # sampler.chain is of shape (nwalkers, nsteps, ndim)
    # we'll throw-out the burn-in points and reshape:
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
 
    plot_MCMC_model(ax1, xi, yi, dyi, trace[:2, :], linestyle=linestyles[1])
       
    # plot the likelihood contours
    ax = plt.subplot(222 + 1)

    plot_MCMC_trace(ax, xi, yi, trace[:2, :], False, colors='k')


    ax.set_ylabel('intercept')
    ax.set_xlabel('slope')
    ax.grid(color='gray')

    ax.yaxis.set_major_locator(plt.MultipleLocator(40))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    
    ax.text(0.98, 
            0.98, 
            labels[1], 
            ha='right', 
            va='top',
            bbox=dict(fc='w', ec='none', alpha=0.5),
            transform=ax.transAxes)

    ax.set_ylim(bins[1][0][0], bins[1][0][-1])
    ax.set_xlim(bins[1][1][0], bins[1][1][-1])
    

    #---------------------------------------------------------------

    ax1.set_xlim(0, 350)
    ax1.set_ylim(100, 700)     

    plt.show() 





