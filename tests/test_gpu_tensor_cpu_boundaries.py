import numpy as np
import pytest
import torch
from sklearn.base import BaseEstimator

from lf2i.calibration.p_values import augment_calibration_set
from lf2i.calibration.torch_utils import FeedForwardNN, LearnerRegression, QuantileLoss
from lf2i.test_statistics.waldo import Waldo
from lf2i.utils.calibration_diagnostics_inputs import (
    preprocess_diagnostics,
    preprocess_fit_p_values,
    preprocess_indicators_lf2i,
    preprocess_indicators_prediction,
    preprocess_predict_p_values,
    preprocess_predict_quantile_regression,
    preprocess_train_quantile_regression,
)
from lf2i.utils.odds_inputs import (
    preprocess_for_odds_cs,
    preprocess_for_odds_cv,
    preprocess_odds_estimation,
)
from lf2i.utils.waldo_inputs import preprocess_waldo_computation, preprocess_waldo_estimation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_learner_regression_predict_returns_cpu_tensor_on_cuda():
    learner = LearnerRegression(
        model=FeedForwardNN(input_d=1, output_d=1, hidden_layer_shapes=[4]),
        optimizer=torch.optim.Adam,
        loss=QuantileLoss(quantiles=[0.5]),
        device="cuda",
        verbose=False,
    )
    pred = learner.predict(torch.randn(3, 1, device="cuda"))
    assert isinstance(pred, torch.Tensor)
    assert pred.device.type == "cpu"


class MockPosteriorTrainer:
    def append_simulations(self, theta, x, **kwargs):
        return self

    def train(self, **kwargs):
        return self

    def build_posterior(self, **kwargs):
        return self

    def sample(self, sample_shape, x, show_progress_bars=True, **kwargs):
        return torch.randn(sample_shape[0], 1, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_waldo_evaluate_posterior_sampling_cuda_outputs_numpy():
    waldo = Waldo(
        estimator=MockPosteriorTrainer(),
        poi_dim=1,
        estimation_method="posterior",
        num_posterior_samples=8,
        verbose=False,
        n_jobs=1,
    )
    waldo._estimator_trained["posterior"] = True
    out = waldo.evaluate(
        parameters=torch.randn(4, 1, device="cuda"),
        samples=torch.randn(4, 1, device="cuda"),
        mode="critical_values",
    )
    assert isinstance(out, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_train_quantile_regression_cuda_tensor_to_numpy():
    test_statistics = torch.randn(4, device="cuda")
    parameters = torch.randn(4, 1, device="cuda")
    ts_out, p_out = preprocess_train_quantile_regression(test_statistics, parameters, 1, object())
    assert isinstance(ts_out, np.ndarray)
    assert isinstance(p_out, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_predict_quantile_regression_cuda_tensor_to_numpy():
    out = preprocess_predict_quantile_regression(torch.randn(5, 1, device="cuda"), object(), 1)
    assert isinstance(out, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_fit_p_values_cuda_tensor_to_numpy():
    out = preprocess_fit_p_values(torch.randn(5, device="cuda"), object())
    assert isinstance(out, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_predict_p_values_cuda_tensor_to_numpy():
    out = preprocess_predict_p_values(
        mode="critical_values",
        test_stats=torch.randn(5, device="cuda"),
        poi=torch.randn(5, 1, device="cuda"),
        rejection_probs_model=object(),
    )
    assert isinstance(out, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_diagnostics_cuda_tensor_to_numpy():
    indicators, parameters, new_parameters = preprocess_diagnostics(
        indicators=torch.randint(0, 2, (5,), device="cuda"),
        parameters=torch.randn(5, 1, device="cuda"),
        new_parameters=torch.randn(3, 1, device="cuda"),
        param_dim=1,
    )
    assert isinstance(indicators, np.ndarray)
    assert isinstance(parameters, np.ndarray)
    assert isinstance(new_parameters, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_indicators_prediction_cuda_tensor_to_numpy():
    parameters, samples = preprocess_indicators_prediction(
        parameters=torch.randn(4, 1, device="cuda"),
        samples=torch.randn(4, 1, device="cuda"),
        param_dim=1,
    )
    assert isinstance(parameters, np.ndarray)
    assert isinstance(samples, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_indicators_lf2i_cuda_tensor_to_numpy():
    out = preprocess_indicators_lf2i(
        test_statistics=torch.randn(4, device="cuda"),
        critical_values=torch.randn(4, device="cuda"),
        p_values=torch.randn(4, device="cuda"),
        parameters=torch.randn(4, 1, device="cuda"),
        param_dim=1,
    )
    assert all(isinstance(x, np.ndarray) for x in out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_odds_estimation_cuda_tensor_to_numpy():
    labels, params_samples = preprocess_odds_estimation(
        parameters=torch.randn(4, 1, device="cuda"),
        samples=torch.randn(4, 1, device="cuda"),
        param_dim=1,
        estimator=BaseEstimator(),
    )
    assert isinstance(labels, np.ndarray)
    assert isinstance(params_samples, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_for_odds_cv_cuda_tensor_to_numpy():
    out = preprocess_for_odds_cv(
        parameters=torch.randn(2, 1, device="cuda"),
        samples=torch.randn(2, 2, 1, device="cuda"),
        param_dim=1,
        batch_size=2,
        data_dim=1,
        estimator=BaseEstimator(),
    )
    assert all(isinstance(x, np.ndarray) for x in out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_for_odds_cs_cuda_tensor_to_numpy():
    out = preprocess_for_odds_cs(
        parameter_grid=torch.randn(3, 1, device="cuda"),
        samples=torch.randn(2, 2, 1, device="cuda"),
        param_dim=1,
        batch_size=2,
        data_dim=1,
        estimator=BaseEstimator(),
    )
    assert all(isinstance(x, np.ndarray) for x in out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_waldo_estimation_cuda_tensor_to_numpy():
    parameters, samples = preprocess_waldo_estimation(
        parameters=torch.randn(4, 1, device="cuda"),
        samples=torch.randn(4, 1, device="cuda"),
        estimation_method="prediction",
        estimator=BaseEstimator(),
        param_dim=1,
    )
    assert isinstance(parameters, np.ndarray)
    assert isinstance(samples, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_preprocess_waldo_computation_cuda_tensor_to_numpy():
    parameters, cond_mean, cond_var = preprocess_waldo_computation(
        parameters=torch.randn(4, 1, device="cuda"),
        conditional_mean=np.random.randn(4, 1),
        conditional_var=np.random.rand(4, 1) + 1e-3,
        param_dim=1,
    )
    assert isinstance(parameters, np.ndarray)
    assert isinstance(cond_mean, np.ndarray)
    assert isinstance(cond_var, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_augment_calibration_set_cuda_tensor_to_numpy():
    inputs, rejection_indicators = augment_calibration_set(
        test_statistics=torch.randn(6, device="cuda"),
        poi=torch.randn(6, 1, device="cuda"),
        num_augment=2,
        acceptance_region="left",
        conditional_resampling=False,
    )
    assert isinstance(inputs, np.ndarray)
    assert isinstance(rejection_indicators, np.ndarray)
