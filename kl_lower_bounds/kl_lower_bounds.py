import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.experimental import optimizers


def kl_normal(mu1, sigma1, mu2, sigma2):
    """
    The Kullback Leibler divergence between two 1D normal distributions. The direction of the KL divergence is
    D(1 || 2).

    Parameters
    ----------
    mu1, sigma1: floats
        The mean and standard deviation of the 1st distribution.
    mu2, sigma2: floats
        The mean and standard deviation of the 2nd distribution.

    Returns
    -------
    numpy.float64
        The KL divergence.
    """
    return onp.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 1 / 2


def nn_predict(params, inputs):
    """
    For a given set of weights (params) and input data, return the output of a neural net. The non-linearity is
    hyperbolic tangent and the final layer is linear.

    Parameters
    ----------
    params: list of a list of jaxlib.xla_extension.DeviceArray
        The NN weights.
    inputs: jaxlib.xla_extension.DeviceArray
        The explanatory variables of the NN. Shape = (n, k), where n is the number of data points and k is the
        dimension of the data.

    Returns
    -------
    output: jaxlib.xla_extension.DeviceArray
        The 1 dimensional output of the NN.
    """
    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)

    W, b = params[-1]
    return np.dot(inputs, W) + b


def init_nn_params(layer_shapes, scale, seed):
    """
    Build a list of (weights, biases) tuples, one for each layer. Adapted from JAX. Each parameter is drawn from a
    Guassian with a mean of zero and a standard deviation given by the scale parameter.

    Parameters
    ----------
    layer_shapes: tuple/list of ints
        The number of units in each layer. The first entry and last entry must much the dimensionality of the inputs and
        outputs. E.g. later_shapes = (1,50,50,1) implies a NN with 1 input, 2 hidden layers each of 50 units, and one
        output.
    scale: float
        The standard deviation of the Gaussian from which _all_ weights and biases are drawn from.
    seed: int
        The random seed used to generate the parameters.

    Returns
    -------
    params: list
        The parameters of the NN. Must be compatiple with nn_predict.
    """
    key = random.PRNGKey(seed)
    split_keys = random.split(key, 2 * len(layer_shapes))
    return [[random.normal(split_keys[i], (size[0], size[1])) * scale,  # weight matrix
             random.normal(split_keys[-(i + 1)], (size[1],)) * scale]  # bias vector
            for i, size in enumerate(zip(layer_shapes[:-1], layer_shapes[1:]))]


@jit
def update_params(params, grads, learning_rate):
    """
    Using a set of gradients, update the parameters (weights) of a neural net by gradient descent.
    """
    for i in range(len(params)):
        for j in range(len(params[i])):
            params[i][j] += learning_rate * grads[i][j]
    return params


@jit
def kl_lb(params, x1, x2):
    """
    The f-gan lower bound to the KL divergence.

    Parameters
    ----------
    params: list of a list of jaxlib.xla_extension.DeviceArray
        The neural net parameters
    x1: jaxlib.xla_extension.DeviceArray
        Expressing the KL divergences as D(p || q), x1 are samples from the p.
    x2: jaxlib.xla_extension.DeviceArray
        Expressing the KL divergences as D(p || q), x2 are samples from the q.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        A lower bound to the KL divergence.
    """
    return np.mean(nn_predict(params, x1)) - np.mean(np.exp(nn_predict(params, x2) - 1))


@jit
def neg_kl_lb(params, x1, x2):
    """
    The negative of the f-gan lower bound to the KL divergence.
    """
    return -np.mean(nn_predict(params, x1)) + np.mean(np.exp(nn_predict(params, x2) - 1))


def get_kl_lower_bound_sgd(params, prior_samps, post_samps, learning_rate=1e-4, batch_size=30, nsteps=1000, record_freq=1):
    """
    Get a lower bound of the KL divergence of the prior and posterior using the f-gan lower bound and vanilla
    stochastic gradient decent.

    Parameters
    ----------
    params: list of a list of jaxlib.xla_extension.DeviceArray
        The starting NN weights.
    prior_samps, post_samps: 2 jaxlib.xla_extension.DeviceArray
        Samples from the prior and posterior distributions.
    learning_rate: float
        At each time step, the parameters are updated by this amount multiplied by the gradient.
    batch_size: int
        The number of mini-batch samples used at each iteration to estimate the gradient.
    nsteps: int
        The total number of time steps.
    record_freq: int
        The interval at which the KL lower bound will be saved. If a value less than 1 is used (e.g. 0 or -1), only the
        value at nsteps is recorded.

    Returns
    -------
    lower_bound: list
        Lower bounds to the KL divergences for each time step. If record_freq < 0, then only the final lower bound is
        returned.
    """
    if record_freq < 1:
        freq = nsteps
    else:
        freq = record_freq

    lb_grad = jit(grad(kl_lb))

    lower_bound = []
    for n in range(1, nsteps + 1):
        # Mini-batch sampling of the gradient using bootstrap sampling
        inds1 = onp.random.choice(post_samps.shape[0], size=batch_size, replace=False)
        inds2 = onp.random.choice(prior_samps.shape[0], size=batch_size, replace=False)
        grads = lb_grad(params, post_samps[inds1, :], prior_samps[inds2, :])
        # Update the parameters
        params = update_params(params, grads, learning_rate)

        # Save the lower bound to the kl divergence
        if n % freq == 0:
            lower_bound.append(kl_lb(params, post_samps, prior_samps))

    return lower_bound, params


def get_kl_lower_bound(params, prior_samps, post_samps, step_size=1e-3, batch_size=30, nsteps=1000, record_freq=1):
    """
    Get a lower bound of the KL divergence of the prior and posterior using the f-gan lower bound and stochastic
    gradient decent with Adam.

    Parameters
    ----------
    params: list of a list of jaxlib.xla_extension.DeviceArray
        The starting NN weights.
    prior_samps, post_samps: 2 jaxlib.xla_extension.DeviceArray
        Samples from the prior and posterior distributions.
    step_size: float
        The step size, or 'alpha', parameter in the Adam optimizer. The learning rate at a given time step is
        proportional to this parameter.
    batch_size: int
        The number of mini-batch samples used at each iteration to estimate the gradient.
    nsteps: int
        The total number of time steps.
    record_freq: int
        The interval at which the KL lower bound will be saved. If a value less than 1 is used (e.g. 0 or -1), only the
        value at nsteps is recorded.

    Returns
    -------
    lower_bound: list
        Lower bounds to the KL divergences for each time step. If record_freq < 0, then only the final lower bound is
        returned.
    """
    if record_freq < 1:
        freq = nsteps
    else:
        freq = record_freq

    # Using the adam method for gradient descent.
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    opt_state = opt_init(params)

    # The gradient of the negative of the lower bound to the KL divergence
    lb_grad = jit(grad(neg_kl_lb))

    @jit
    def step(i, opt_state, x1, x2):
        p = get_params(opt_state)
        g = lb_grad(p, x1, x2)
        return opt_update(i, g, opt_state)

    lower_bound = []
    for n in range(1, nsteps + 1):
        # Mini-batch sampling of the gradient using bootstrap sampling
        inds1 = onp.random.choice(post_samps.shape[0], size=batch_size, replace=False)
        inds2 = onp.random.choice(prior_samps.shape[0], size=batch_size, replace=False)
        opt_state = step(n, opt_state, post_samps[inds1, :], prior_samps[inds2, :])

        # Save the lower bound to the kl divergence
        if n % freq == 0:
            p = get_params(opt_state)
            lower_bound.append(kl_lb(p, post_samps, prior_samps))

    return lower_bound, p
