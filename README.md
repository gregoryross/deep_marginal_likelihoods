# Bounding Bayesian marginal likelihoods with deep learning
Model selection in Bayesian analysis is critically important but remains immensely challenging
despite the increasing ease of sampling methods. To aid formal model comparison, the repo explores 
a novel bound to the marginal likelihood (also known as the marginal evidence) that can be computed using deep
learning techniques. Central to this approach is calculating the lower bound of the Kullback-Leibler divergence between
prior and posterior. More details of this idea can be in the rough write-up [here](Bounding_marginal_likelihoods_with_deep_learning.pdf).

To help test this new bound, this repo currently implements 2 modules. As described below, the first is for Bayesian linear regression
using normal-inverse-gamma priors and the second is for estimating lower bounds of the Kullback-Leibler divergence between
between 2 distributions using samples from those distributions. 

## Manifest
* `kl_lower_bounds/` Jax-implementated neural nets that can estimate the lower bound the of KL divergences using samples from distrubutions.  
* `bayes_linear_regression/` a set of classes and functions for performing Bayesian linear regression.
* `bayes_linear_regression/examples/` a set of examples that demonstrate some of the functionality of the code.
* `notebooks/` Jupyter notebooks that are used to play with the code. 

This repository also contains a rough write-up of the theory behind the methods implemented here. 
* `Bounding_marginal_likelihoods_with_deep_learning.pdf` a rough write-up of a way to estimate marginal likelihoods.
* `write-up/` the Tex file and notes for the write-up PDF.
* `references` PDFs of prior relevant work.

## Usage
### Linear regression using normal-inverse-gamma priors
Bayesian regression problems using normal-inverse-gamma priors have analytical marginal likelihoods. This makes these models
useful for validating methods that estimate marginal likelihoods, as we are trying to do here.
```
from bayes_linear_regression import norm_inv_gamma as nig
import numpy as np
```
First, we need to specify the parameters of the prior. In this demonstration, we want our model to have 2 regression parameters: one
intercept and one gradient. We need to set the prior covariance matrix between these 2 parameters as well as the mean of
the prior:
```python
cov = np.eye(2)   
mu = np.zeros(shape=(2,1))
```
Next, will initiate the prior object. The inverse-gamma distribution has 2 parameters `a` and `b` which we will enter
directly below: 
```python
prior = nig.NormalInverseGamma(a=3, b=1, mu=mu, cov=cov)
```
To make module validation easier, we can generate a toy regression problem by drawing regression parameters as well as
data points from the prior:
```python
explanatory, response, regression_params = nig.gen_toy_data(100, prior)
```
In the above, we are only drawing 100 data points.

One neat aspect of normal-inverse-gamma priors is that they are conjugate with the posterior. Calculating this posterior
disribution is simple with this module:
```python
post = nig.PostNormalInverseGamma(prior, explanatory, response)
```
Estimates of the regression parameters can be obtained directly from `post`. Importantly for this repo, the (log of the)
marginal likelihood of the posterior distribution can be accessed like so:
```python
print(post.log_marg_like)
```

### The lower bounds to the KL divergence 
As described [here](Bounding_marginal_likelihoods_with_deep_learning.pdf), calculating a lower bound of the KL divergence
between the prior and posterior allows us to get an _upper_ bound to the marginal likelihood. The module `kl_lower_bounds`
can estimate this lower bound using neural nets implemented with `Jax`.
```python
from kl_lower_bounds import kl_lower_bounds as klb
```
Using the `klb.get_kl_lower_bound`, we can estimate the KL divergence between the prior and the postirior as long as we
have samples from both. So let's quickly generate 5000 samples from each:
```python
prior_samps = prior.rvs(5000)
post_samps = post.rvs(5000)
```
For the sake of this example, let's make the neural net very simple, with 1 hidden layer that contains 100 hidden units.
The output layer must only have one unit for the KL divergence estimate. Currently, the activation
functions are hard coded as softplus functions. 
```python
layer_shapes = (prior_samps.shape[1], 100, 1)
```
We'll intialize the parameters of neural net using zero-centered Gaussians and run the optimization:
```python
nn_params = klb.init_nn_params(layer_shapes, scale=0.1, seed=1)
kl_lb, final_nn_params = klb.get_kl_lower_bound(nn_params,
                                                prior_samps, 
                                                post_samps, 
                                                batch_size=100,
                                                nsteps=10000,
                                                record_freq=5)
``` 
`kl_lb` is an array whose entries are lower bounds to the KL divergence between the prior and posterior. The final entry
our final estimate for the KL divergence.

### An upper bound to the marginal likelihood
To calculate an upper bound to marginal likelihood we need 2 ingredients: the posterior mean of the log likelihood and an
upper bound of the KL divergence between the prior and posterior. The posterior mean of the log likelihood can be estimated
like this:
```python
post_mean_log_like = nig.nig.estimate_mean_loglike(post, explanatory, response)
```
and our upper bound for the marginal likelihood is the following:
```python
print(post_mean_log_like - kl_lb[-1])
```
We can compare this upper bound to the true value `post.log_marg_like`.

