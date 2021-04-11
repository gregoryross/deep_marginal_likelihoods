import tempfile
import os
import norm_inv_gamma as nig

def test_regression_1d():
    """
    A very simple regression example that plots the posterior distribution samples.
    """
    from examples import regression_1d
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        regression_1d.main()
        os.chdir(cwd)

def test_convergence_marglike():
    """
    A simple demonstration of the convergence of the marginal likelihood.
    """
    from examples import convergence_of_marginal_likelihood as conv
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        conv.main()
        os.chdir(cwd)

if __name__ == "__main__":
    test_regression_1d()
    test_convergence_marglike()

