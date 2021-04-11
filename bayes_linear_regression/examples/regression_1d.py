import numpy as np
import matplotlib.pylab as plt

import norm_inv_gamma as nig


def main():
    """
    Run through an example regression problem.
    """
    # 1. Generate a 1D toy regression problem by sampling from the prior
    cov = np.eye(2)
    mu = np.zeros(shape=(2,1))

    ## Instantiate the model
    prior = nig.NormalInverseGamma(a=3, b=1, mu=mu, cov=cov)

    ## Draw toy parameters and data from the prior
    n = 100
    explanatory, response, params = nig.gen_toy_data(n, prior)

    ## Plot the toy data
    fig, ax = plt.subplots(1, figsize=(6, 5))

    ax.scatter(explanatory[:,1], response)
    ax.set_xlabel('Explanatory variable', fontsize=14)
    ax.set_ylabel('Response variable', fontsize=14)
    ax.set_title('1D toy regression data', fontsize=15)

    plt.tight_layout()
    plt.savefig('example_1d_regression_data.png', dpi=150)

    # 2. Fit the model
    post = nig.PostNormalInverseGamma(prior, explanatory, response)

    ## Draw samples from the posterior
    samples = post.rvs(1000)

    ## Plot the posterior samples against the true values
    fig, axis = plt.subplots(1,3, figsize=(18, 6))
    titles = ['Parameter 0: the standard deviation', 'Parameter 1: the intercept', 'Parameter 2: the gradient']
    for i in range(3):
        ax = axis[i]
        ax.hist(samples[:,i], bins=50, label='Posterior samples')
        ax.axvline(params[0][i], color='k', lw=3, ls='--', label='True value')
        if i==0:
            ax.legend(fontsize=15)
        ax.set_title(titles[i], fontsize=16)
        ax.set_ylabel('Counts', fontsize=16)

    plt.tight_layout()
    plt.savefig('example_1d_regression_posterior.png', dpi=150)


if __name__ == "__main__":
    main()

