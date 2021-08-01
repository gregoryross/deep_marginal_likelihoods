import numpy as onp
from scipy.stats import norm
import jax.numpy as np
import kl_lower_bounds as klb

def test_kl_estimation():
    """
    Test the function of the NN lower bound estimation with a 1D Gaussian. Comparing to a known value by fixing the
    random seed.
    """
    # The random seed for the minibatch and prior and posterior sampling
    onp.random.seed(1)

    nsamps = 5000  # How many samples to draw from posterior and prior
    p1 = norm(loc=4, scale=3)
    p2 = norm(loc=0, scale=1)
    #t = klb.kl_normal(0, 1, 4, 3)

    prior_samps = np.array(p1.rvs(nsamps)).reshape((nsamps, 1))
    post_samps = np.array(p2.rvs(nsamps)).reshape((nsamps, 1))

    # NN parameters
    nsteps = 200
    batch_size = 30
    save_freq = -1

    layer_shapes = (1, 100, 100, 100, 1)
    params = klb.init_nn_params(layer_shapes, 0.1, 3)

    # Run the optimization:
    lb, final_params = klb.get_kl_lower_bound(params, prior_samps, post_samps, batch_size=batch_size, nsteps=nsteps,
                                          record_freq=save_freq)

    assert float(lb[0]) == 1.4583255052566528


def run_all():
    """
    Run the KL lower bound tests.
    """
    test_kl_estimation()


if __name__ == "__main__":
    run_all()
