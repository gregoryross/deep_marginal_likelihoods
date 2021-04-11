import numpy as np
from scipy import stats

import norm_inv_gamma as nig

"""
python -m pytest test_norm_inv_gamma.py
"""
def test_norm_inv_gamma():
    """
    Ensure that samples from the normal inverse gamma have the correct stats.

    Testing the mean, standard deviation, and quantiles of the marginal samples of the regression parameters. These
    should be t-distributions for uncorrelated covariance matrices.
    """
    np.random.seed(seed=1)

    # Initialize parameters
    a = 2
    b = 1

    # 1D regression problem, intercept and gradient. No correlation between parameters.
    param_vars = np.array((4, 2))
    cov_mat = np.array(((param_vars[0], 0), (0, param_vars[1])))
    mu = np.array((0, 0)).reshape((2, 1))

    # Now sample a large number of data points
    model = nig.NormalInverseGamma(a=a, b=b, mu=mu, cov=cov_mat)
    samps = model.rvs(5000)

    # Test whether the first 2 moments of the regression variables are close to a 1D t distribution.
    for i in range(2):
        tdist = stats.t(loc=mu[i], df=2 * a, scale=np.sqrt(param_vars[i] * b / a))
        assert np.isclose(tdist.mean(), samps[:, i + 1].mean(), rtol=0.1, atol=0.1)
        assert np.isclose(tdist.std(), samps[:, i + 1].std(), rtol=0.1, atol=0.1)

    # Next assert that the quantiles of the empirical distribution are closer to a t-dist than a normal.
    for i in range(2):
        qq_t = stats.probplot(samps[:, i + 1], sparams=(2 * a, mu[i], np.sqrt(param_vars[i] * b / a)), dist='t')
        qq_norm = stats.probplot(samps[:, i + 1], sparams=(mu[i], np.sqrt(param_vars[i] * b / a)), dist='norm')

        err_t = np.mean((qq_t[0][0] - qq_t[0][1]) ** 2)
        err_norm = np.mean(np.abs(qq_norm[0][0] - qq_norm[0][1]))
        assert err_norm > err_t

    np.random.seed(None)

def test_fit_normal_inverse_gamma():
    """
    Ensure the conjugate bayesian linear model has been properly updated to reflect the data.

    This function will validate the posterior value for the posterior "b" (i.e. scale) parameter of the inverse
    gamma distribution by comparing the computed value to a the value obtained using a different expression. The
    posterior value of "b" used in FittedNormalInverseGamma.fit() is a function of the posterior "mu" and posterior
    covariance matrix - making this a decent marker of validity.

    """
    # Generate toy data
    cov = np.eye(2)
    mu = np.zeros(shape=(2, 1))

    # Instantiate the prior
    prior = nig.NormalInverseGamma(a=3, b=1, mu=mu, cov=cov)

    n = 10
    explanatory, response, params = nig.gen_toy_data(n, prior)
    post_mu, post_cov, post_prec, post_a, post_b = nig.PostNormalInverseGamma.fit(prior, explanatory, response)

    # Now compare post_b to the value obtained using a different calculation method. This method requires inverting a
    # n by by n matrix, which is expensive for large data sets/
    # Coded below as b = prior.b + 0.5 * A * B * C
    A = (response - np.dot(explanatory, prior.mu)).T
    B = np.linalg.inv(np.eye(n) + np.dot(np.dot(explanatory, prior.cov), explanatory.T))
    C = response - np.dot(explanatory, prior.mu)
    diff_post_b = prior.b + 0.5 * np.dot(A, np.dot(B, C))

    assert np.isclose(post_b, diff_post_b)


def test_post_norm_inv_gamma():
    """
    Test the basic functioning of the posterior class of the normal inverse gamma model.
    """
    # Generate toy data
    cov = np.eye(2)
    mu = np.zeros(shape=(2, 1))

    # Instantiate the prior
    prior = nig.NormalInverseGamma(a=3, b=1, mu=mu, cov=cov)

    # Generate a random linear model by drawing from the prior
    n = 100
    explanatory, response, params = nig.gen_toy_data(n, prior)

    post = nig.PostNormalInverseGamma(prior, explanatory, response)


def test_log_marg_like():
    """
    Making sure the marginal likelihood of the normal inverse is calculated correctly. In the asymptotic sampling
    limit, the -ve of the log of the marginal likelihood should equal the number of samples times the conditional
    entropy of the response variable - IF the "true model" is within the model space.

    Note
    ----
    This is a stochastic test but has a pass rate of AT LEAST 1 in 5000.
    """
    prior = nig.NormalInverseGamma(a=3, b=1, mu=np.array([[0], [0]]), cov=np.eye(2))
    # Generate toy data from the prior
    n = 100000  # Number of data points
    explanatory, response, params = nig.gen_toy_data(n, prior)
    sigma2 = params[0][0]  # The true variance of the data

    # Fit the posterior model to the toy data
    post = nig.PostNormalInverseGamma(prior, explanatory, response)

    # The conditional entropy of the resonse given the explanatory variables is that of a normal
    condition_ent = nig.entropy_of_normal(sigma2)

    # The difference between the conditional entropy and the negative of the log marginal likelihood (divided by n)
    diff = condition_ent + post.log_marg_like / n

    # Should at least be within 5 %
    assert diff / condition_ent < 0.05


def test_gen_toy_data():
    # 1D regression problem, intercept and gradient. No correlation between regression parameters.
    cov = np.eye(2)
    mu = np.zeros(shape=(2, 1))

    # Instantiate the model"
    model = nig.NormalInverseGamma(a=3, b=1, mu=mu, cov=cov)

    n = 10
    explanatory, response, params = nig.gen_toy_data(n, model)
    assert explanatory.shape[0] == n
    assert response.shape[0] == n


def test_log_likelihood():
    """
    Ensuring my log likelihood function matches the result from a different calculation method.
    """
    prior = nig.NormalInverseGamma(a=3, b=1, mu=np.array([[0], [0]]), cov=np.eye(2))

    # Draw a toy regression parameters from the prior and 1D explanatory variable from standard normal
    n = 100
    explanatory, response, params = nig.gen_toy_data(n, prior)
    params = prior.rvs(1)
    mu = params[:, 1:].T
    sigma2 = params[0][0]

    # Scipy's log pdf
    pred = np.dot(explanatory, mu)
    ll_scipy = stats.norm.logpdf(response, loc=pred, scale=np.sqrt(sigma2)).sum()

    # My own function
    ll = nig.log_likelihood(explanatory, response, sigma2, mu)

    assert np.isclose(ll, ll_scipy)


def run_all():
    """
    Run all the tests.
    """
    test_norm_inv_gamma()
    test_fit_normal_inverse_gamma()
    test_post_norm_inv_gamma()
    test_log_marg_like()
    test_gen_toy_data()
    test_log_likelihood()

if __name__ == "__main__":
    run_all()
