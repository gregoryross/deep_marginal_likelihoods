import numpy as np
from scipy import stats
from scipy.special import loggamma, gammaln

class NormalInverseGamma(object):
    """
    The normal inverse-gamma distribution. Used as a conjugate prior in Bayesian linear regression. This class
    expresses this distribution from using it's conditional and marginal distributions.

    The conditional distributions are:
    sigma2 ~ InverseGamma(a, b)
    beta | sigma2 ~ Normal(mu, cov * sigma2)

    The marginal over the variance (sigma2) is a multivariate t-distribution:
    beta ~ MultiT(2a, mu, cov * b / a )
    """

    def __init__(self, a, b, mu, cov):
        """
        Parameters
        ----------
        a: float
            The first parameter of the inverse-gamma distribution.
        b: float
            The scale parameter of the inverse-gamma distribution.
        mu: numpy.ndarray
            The center (of the conditional normal distribution) for the regression variables. A column vector is expected.
            Note that 1D regression requires 2 entries, one for the intercept and the other for the gradient.
        cov: numpy.ndarray
            the covariance matrix for the regression variables.
        """
        self.a = a
        self.b = b
        self.mu = mu
        self.cov = cov

        # The precision matrix
        self._prec = np.linalg.inv(cov)

        # The the marginal distribution of the sigma2 (the variance) parameter: an inverse-gamma distribution.
        self.invgamma = stats.invgamma(self.a, loc=0, scale=self.b)

        # The marginal distribution of the regression parameters: a multivariate t disribution.
        # self.marg_beta = multivariate_t(mean=self.mu, shape=self.cov*b/a, df=self.a*2)

    def rvs(self, n):
        """
        Draw random samples from the normal inverse-gamma distribution.

        Parameter
        ---------
        n: int
            The number for samples you wish to draw.

        Returns
        -------
        samples: numpy.ndarray
            Random samples arranged row wise. The first column are the sigma2 (variance) samples, the other columns
            are the regression parameter samples.
        """
        samples = np.zeros(shape=(n, len(self.mu) + 1))
        for i in range(n):
            s2_samp = self.invgamma.rvs(1)
            samples[i, 0] = s2_samp
            samples[i, 1:samples.shape[1]] = stats.multivariate_normal(self.mu.flatten(), self.cov * s2_samp).rvs(1)

        return samples


def gen_toy_data(n, norm_inv_gamma):
    """
    Generate toy regression data by drawing regression parameters from a normal-inverse-gamma model. The explanatory
    variables are drawn from a standard normal distributions.
    """
    params = norm_inv_gamma.rvs(1)
    sigma2 = params[0][0]
    mu = params[:, 1:].T

    nvars = mu.shape[0] - 1

    # The first column is the intercept, the second is the random variable.
    variables = np.random.normal(loc=0, scale=1, size=n * nvars).reshape((n, nvars))
    intercept = np.repeat(1, n).reshape((n, 1))
    explanatory = np.hstack((intercept, variables))

    #explanatory = np.vstack((np.repeat(1, n), np.random.normal(loc=0, scale=1, size=n))).T

    noise = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=n)
    noise = np.reshape(noise, (n, 1))

    # The response is the linear relationship plus noise
    response = np.dot(explanatory, mu) + noise

    return explanatory, response, params


def fit_norm_inverse_gamma(prior, explanatory, response):
    """
    Calibrate a Bayesian normal inverse gamma model to explanatory and response data.

    Parameters
    ----------
    prior: norm_inv_gamma.NormalInverseGamma
        The normal inverse gamma prior.
    explanatory: numpy.ndarray
        The explanatory variables, arranged column-wise. To fit an intercept, one column (usually the first) should be
        an array of ones.
    response: numpy.ndarray
        The response variable. A 1D column vector is expected.
    """
    # The number of data points.
    n = explanatory.shape[0]

    # The prior precision matrix
    prior_prec = np.linalg.inv(prior.cov)

    # The posterior precision matrix and posterior covariance matrix
    post_prec = prior_prec + np.dot(explanatory.T, explanatory)
    post_cov = np.linalg.inv(post_prec)

    # The posterior gradient and intercept parameters
    right = np.dot(prior_prec, prior.mu) + np.dot(explanatory.T, response)
    post_mu = np.dot(post_cov, right)

    # The posterior inverse gamma parameters. The "a" parameter is straightforward:
    post_a = prior.a + 0.5 * n

    # The "b" (scale) parameter can be more involved.
    post_b = prior.b + 0.5 * np.dot(np.dot(prior.mu.T, prior_prec), prior.mu)
    post_b -= 0.5 * np.dot(np.dot(post_mu.T, post_prec), post_mu)
    post_b += 0.5 * np.dot(response.T, response)

    return post_mu, post_cov, post_a, post_b


class PostNormalInverseGamma(NormalInverseGamma):
    """
    The posterior of a Bayesian normal inverse gamma model that will be calibrated to explanatory and response data.
    """

    def __init__(self, prior, explanatory, response):
        """
        Parameters
        ----------
        prior: norm_inv_gamma.NormalInverseGamma
            The normal inverse gamma prior.
        explanatory: numpy.ndarray
            The explanatory variables, arranged column-wise. To fit an intercept, one column (usually the first) should be
            an array of ones.
        response: numpy.ndarray
            The response variable. A 1D column vector is expected.
        """
        self.response = response
        self.explanatory = explanatory

        post_mu, post_cov, post_prec, post_a, post_b = self.fit(prior, explanatory, response)

        self.mu = post_mu
        self.cov = post_cov
        self._prec = post_prec
        self.a = post_a
        self.b = post_b

        # The mode of the IG:
        self.sigma2_map = self.b / (self.a + 1)

        # The the marginal distribution of the sigma2 (the variance) parameter: an inverse-gamma distribution.
        self.invgamma = stats.invgamma(self.a, loc=0, scale=self.b)

        # The marginal likelihood
        self.log_marg_like = self.calc_log_marg_like(prior)

    def pred_rvs(self, n):
        """
        Sample from the postior and predict the values of the response from the explanatory variables.

        Parameter
        ---------
        n: int
            The number of prediction samples you wish to draw.

        Returns
        -------
        predictions: numpy.ndarray
            The predictions of the response variable for each draw from the posterior distribution. The samples are
            arranged in a column-wise.
        """
        #TODO: optimize!
        predictions = np.zeros((self.response.shape[0], n))

        for i in range(n):
            params = self.rvs(1)
            samp_mu = params[:, 1:].T
            pred = np.dot(self.explanatory, samp_mu)
            predictions[:, i] = pred[:, 0]

        return predictions

    @staticmethod
    def fit(prior, explanatory, response):
        """
        Calibrate a Bayesian normal inverse gamma model to explanatory and response data.

        Parameters
        ----------
        prior: norm_inv_gamma.NormalInverseGamma
            The normal inverse gamma prior.
        explanatory: numpy.ndarray
            The explanatory variables, arranged column-wise. To fit an intercept, one column (usually the first) should be
            an array of ones.
        response: numpy.ndarray
            The response variable. A 1D column vector is expected.
        """
        # The number of data points.
        n = explanatory.shape[0]

        # The prior precision matrix
        #prior_prec = np.linalg.inv(prior.cov)

        # The posterior precision matrix and posterior covariance matrix
        post_prec = prior._prec + np.dot(explanatory.T, explanatory)
        post_cov = np.linalg.inv(post_prec)

        # The posterior gradient and intercept parameters
        right = np.dot(prior._prec, prior.mu) + np.dot(explanatory.T, response)
        post_mu = np.dot(post_cov, right)

        # The posterior inverse gamma parameters. The "a" parameter is straightforward:
        post_a = prior.a + 0.5 * n

        # The "b" (scale) parameter can be more involved.
        post_b = prior.b + 0.5 * np.dot(np.dot(prior.mu.T, prior._prec), prior.mu)
        post_b -= 0.5 * np.dot(np.dot(post_mu.T, post_prec), post_mu)
        post_b += 0.5 * np.dot(response.T, response)

        return post_mu, post_cov, post_prec, post_a, post_b[0][0]


    def calc_log_marg_like(self, prior):
        """
        Calculate the natural logarithm of the marginal likelihood for the linear model.

        Parameters
        ----------
        prior: NormalInverseGamma
            The prior normal-inverse-gamma model

        Returns
        -------
        log_marg_like: float
            The logarithm of the marginal likelihood.
        """
        return _calc_log_marg_like(prior, self)

    def old_calc_log_marg_like(self, prior):
        """
        Calculate the logarithm of the marginal likelihood for the linear model.

        Parameters
        ----------
        prior: NormalInverseGamma
            The prior normal-inverse-gamma model

        Returns
        -------
        log_marg_like: float
            The logarithm of the marginal likelihood.
        """
        # Assuming the sign of the determinate is +ve, otherwise the marginal likelihood is imaginary.
        prior_sign, ln_prior_det = np.linalg.slogdet(prior.cov)
        post_sign, ln_post_det = np.linalg.slogdet(self.cov)
        if prior_sign < 0:
            raise Exception('Error: prior preciscion matrix has a negative determinant.')
        if post_sign < 0:
            raise Exception('Error: post preciscion matrix has a negative determinant.')

        n = self.explanatory.shape[0]
        log_marg_like = 0
        log_marg_like += -0.5 * n * np.log(2 * np.pi) + 0.5 * ln_post_det  - 0.5 * ln_prior_det
        log_marg_like += prior.a * np.log(prior.b) - self.a * np.log(self.b)
        log_marg_like += gammaln(self.a) - gammaln(prior.a)
        return log_marg_like


def _calc_log_marg_like_multivart(prior, post):
    """
    Calculate the marginal likelihood of the normal inverse gamma posterior using the multivariate t-distribution.

    Multivariate t method has been adapted from https://gregorygundersen.com/blog/2020/01/20/multivariate-t/

    Note
    ----
    This can be very slow for large data sets.

    Parameters
    ----------
    prior: NormalInverseGamma
    post: PostNormalInverseGamma
    """
    def multi_t_logpdf(x, mean, shape, df):
        # Adapted from https://gregorygundersen.com/blog/2020/01/20/multivariate-t/
        dim = mean.size

        vals, vecs = np.linalg.eigh(shape)
        logdet = np.log(vals).sum()
        valsinv = np.array([1. / v for v in vals])
        U = vecs * np.sqrt(valsinv)
        dev = x - mean
        # maha       = np.square(np.dot(dev, U)).sum(axis=-1) What was originally stated in the blog
        maha = np.square(np.dot(dev.T, U)).sum(axis=-1)

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim / 2. * np.log(df * np.pi)
        D = 0.5 * logdet
        E = -t * np.log(1 + (1. / df) * maha)

        return A - B - C - D + E

    center = np.dot(post.explanatory, prior.mu)
    shape = prior.b / prior.a * np.eye(post.explanatory.shape[0]) + np.dot(np.dot(post.explanatory, prior.cov), post.explanatory.T)

    return multi_t_logpdf(post.response, mean=center, shape=shape, df=2 * prior.a)

def _calc_log_marg_like_multivart_scipy(prior, post):
    """
    Calculate the natural logarithm of the marginal likelihood for the linear model using scipy's multivariate t.

    It's written by the same author as _calc_log_marg_like_multivart and is discussed here:

    https://gregorygundersen.com/blog/2020/01/20/multivariate-t/

    Parameters
    ----------
    prior: NormalInverseGamma
        The prior normal-inverse-gamma model

    Returns
    -------
    log_marg_like: float
        The logarithm of the marginal likelihood.
    """
    center = np.dot(post.explanatory, prior.mu)
    shape = prior.b / prior.a * np.eye(post.explanatory.shape[0]) + np.dot(np.dot(post.explanatory, prior.cov), post.explanatory.T)
    df = 2 * prior.a

    return stats.multivariate_t.logpdf(post.response.flatten(), loc=center.flatten(), shape=shape, df=df)

def _calc_log_marg_like(prior, post):
    """
    Calculate the log marginal likelihood of the normal inverse gamma posterior.

    Note
    ----
    This may be inaccurate for small data sets, but it's fast.

    Parameters
    ----------
    prior: NormalInverseGamma
    post: PostNormalInverseGamma
    """
    # Assuming the sign of the determinate is +ve, otherwise the marginal likelihood is imaginary.
    prior_sign, ln_prior_det = np.linalg.slogdet(prior.cov)
    post_sign, ln_post_det = np.linalg.slogdet(post.cov)
    if prior_sign < 0:
        raise Exception('Error: prior preciscion matrix has a negative determinant.')
    if post_sign < 0:
        raise Exception('Error: post preciscion matrix has a negative determinant.')

    n = post.explanatory.shape[0]
    log_marg_like = 0.0
    log_marg_like += -0.5 * n * np.log(2 * np.pi) + 0.5 * ln_post_det - 0.5 * ln_prior_det
    log_marg_like += prior.a * np.log(prior.b) - post.a * np.log(post.b)
    log_marg_like += gammaln(post.a) - gammaln(prior.a)

    return log_marg_like


def log_likelihood(explanatory, response, sigma2, mu):
    """
    The log likelihood for Bayesian linear regression.

    Parameters
    ----------
    explanatory: numpy.ndarray
        The explanatory variables, arranged column-wise. To fit an intercept, one column (usually the first) should be
        an array of ones.
    response: numpy.ndarray
        The response variable. A 1D column vector is expected.
    sigma2: float
        The variance of the noise on the regression variables.
    mu: numpy.ndarray
        The estimated regression parameters. A column vector is expected.
    """
    n = explanatory.shape[0]
    pred = np.dot(explanatory, mu)

    ll = -(n / 2) * np.log(2 * np.pi * sigma2)
    ll -= 1 / (2 * sigma2) * np.dot((response - pred).T, (response - pred))

    return ll[0][0]


def entropy_of_normal(sigma2):
    """
    Get the entropy of a normal distribution.

    Paramters
    ---------
    sigma2: float
        The variance of the normal distribution.

    Returns
    -------
    entropy: float
        The entropy of a normal distribution.
    """
    return np.log(np.sqrt(sigma2 * 2 * np.pi * np.e))


def sample_loglike(model, explanatory, response, nsamps=1000):
    """
    Samples over the parameters of a normal-inverse-gamma model and return the log likelihood for each sample.

    Parameters
    ----------
    model: NormalInverseGamma
        The normal-inverse-gamma model that will be sampled over.
    explanatory: numpy.ndarray
        The explanatory variables, arranged column-wise. To fit an intercept, one column (usually the first) should be
        an array of ones.
    response: numpy.ndarray
        The response variable. A 1D column vector is expected.
    """
    post_ll = np.zeros(nsamps)

    for i in range(nsamps):
        params = model.rvs(1)
        samp_sigma2 = params[:, 0][0]
        samp_mu = params[:, 1:].T
        post_ll[i] = log_likelihood(explanatory, response, samp_sigma2, samp_mu)

    return post_ll


def estimate_mean_loglike(model, explanatory, response, nsamps=1000):
    """
    Samples over a normal-inverse-gamma model and estimates the mean of the log likelihood.

    Parameters
    ----------
    model: NormalInverseGamma
        The normal-inverse-gamma model that will be sampled over.
    explanatory: numpy.ndarray
        The explanatory variables, arranged column-wise. To fit an intercept, one column (usually the first) should be
        an array of ones.
    response: numpy.ndarray
        The response variable. A 1D column vector is expected.
    """
    post_ll = np.zeros(nsamps)

    for i in range(nsamps):
        params = model.rvs(1)
        samp_sigma2 = params[:, 0]
        samp_mu = params[:, 1:].T
        post_ll[i] = log_likelihood(explanatory, response, samp_sigma2[0], samp_mu)

    return post_ll.mean()