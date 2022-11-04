import pytest

from lf2i.simulator.gaussian import GaussianMean
from lf2i.test_statistics.waldo import Waldo


@pytest.fixture
def gaussian_simulator(request):
    return GaussianMean(
        likelihood_cov=1,
        prior='uniform',
        parameter_space_bounds={'low': -1, 'high': 1},
        param_grid_size=1000,
        param_dim=request.param[0],
        data_dim=request.param[1],
        data_sample_size=request.param[2]
    )


@pytest.fixture
def waldo_prediction(request):
    return Waldo(
        estimator='gb',
        param_dim=request.param[0],
        method='prediction',
        num_posterior_samples=100
    )


@pytest.fixture
def waldo_posterior(request):
    return Waldo(
        estimator='snpe',
        param_dim=request.param[0],
        method='posterior',
        num_posterior_samples=100,
        cond_variance_estimator='gb'
    )
