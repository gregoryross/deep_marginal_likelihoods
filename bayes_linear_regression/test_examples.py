import tempfile
import os

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

def test_poly_model_comp():
    """
    Polynomial model comparison with the marginal likelihood and mean log likelihood.
    """
    from examples import model_comparison_with_loglike as comp
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        comp.main(nsamps=100)
        os.chdir(cwd)

if __name__ == "__main__":
    test_regression_1d()
    test_convergence_marglike()
    test_poly_model_comp()
