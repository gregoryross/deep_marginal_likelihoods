import numpy as np
import matplotlib.pylab as plt

import sys
sys.path.append('../')
import norm_inv_gamma as nig

def main():
    """
        Demonstrating that the marginal likelihood converges to the entropy as the number of samples goes to infinity.
    """
    # 1. Draw a toy regression parameters from the prior and 1D explanatory variable from standard normal
    prior = nig.NormalInverseGamma(a=3, b=1, mu=np.array([[0], [0]]), cov=np.eye(2))

    n = 100
    explanatory, response, params = nig.gen_toy_data(n, prior)
    params = prior.rvs(1)
    mu = params[:, 1:].T
    sigma2 = params[0][0]

    # 2. Now perform multiple rounds of Bayesian updating and calculate the marginal likelihood each time.
    nrounds = 500
    l = []
    ntotal = n
    for i in range(nrounds):
        # Calibrate Bayesian model
        post = nig.PostNormalInverseGamma(prior, explanatory, response)
        l.append(post.log_marg_like / ntotal)

        # Now add new data to the model
        new_explanatory = np.vstack((np.repeat(1, n), np.random.normal(loc=0, scale=1, size=n))).T
        noise = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=n)
        noise = np.reshape(noise, (n, 1))
        new_response = np.dot(new_explanatory, mu) + noise

        explanatory = np.vstack((new_explanatory, explanatory))
        response = np.vstack((new_response, response))
        ntotal += n

    # 3. Plot the convergence of the data towards the entropy
    condition_ent = nig.entropy_of_normal(sigma2)

    fig, ax = plt.subplots(1, figsize=(7, 6))

    ndata = np.arange(n - 1, nrounds * n, n)
    ax.plot(ndata, l, color='C0')
    ax.scatter(ndata, l, alpha=0.9, color='C0', label='Log marginal likelihood / N')
    ax.axhline(-condition_ent, lw=3, color='k', label='Negative conditional entropy')

    ax.legend(fontsize=13)
    ax.set_xlabel('Number of data points (N)', fontsize=14)
    ax.set_ylabel('Information (nats)', fontsize=14)
    ax.set_title('Convergence of log marginal likelihood with increasing data', fontsize=14)
    plt.tight_layout()
    plt.savefig('example_convergence_margelike.png', dpi=150)


if __name__ == "__main__":
    main()