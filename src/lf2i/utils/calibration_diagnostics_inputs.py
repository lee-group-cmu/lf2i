from typing import Union, Tuple, Any, Sequence, Iterator
import warnings

import itertools
import numpy as np
import torch
from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBModel


def preprocess_train_quantile_regression(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    param_dim: int
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    if isinstance(test_statistics, torch.Tensor):
        test_statistics = test_statistics.numpy()
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.numpy()
    return test_statistics.reshape(-1, ), parameters.reshape(-1, param_dim)


def preprocess_predict_quantile_regression(
    parameters: Union[np.ndarray, torch.Tensor],
    estimator: Any,
    param_dim: int
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(estimator, torch.nn.Module):
        # PyTorch models
        if isinstance(parameters, np.ndarray):
            parameters = torch.from_numpy(parameters)
    if isinstance(estimator, (BaseEstimator, XGBModel)):
        # Scikit-Learn or XGBoost models
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy()
    return parameters.reshape(-1, param_dim)


def preprocess_neyman_inversion(
    test_statistic: np.ndarray,
    critical_values: np.ndarray,
    parameter_grid: Union[np.ndarray, torch.Tensor],
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(parameter_grid, torch.Tensor):
        parameter_grid = parameter_grid.numpy()
    parameter_grid = parameter_grid.reshape(-1, param_dim)
    return test_statistic.reshape(-1, parameter_grid.shape[0]), critical_values.reshape(1, parameter_grid.shape[0]), parameter_grid


def preprocess_diagnostics(
    indicators: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    new_parameters: Union[np.ndarray, torch.Tensor, None],
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(indicators, torch.Tensor):
        indicators = indicators.numpy()
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.numpy()
    if isinstance(new_parameters, torch.Tensor):
        new_parameters = new_parameters.numpy()
    if new_parameters is not None:
        new_parameters = new_parameters.reshape(-1, param_dim)
    return indicators.reshape(-1, ), parameters.reshape(-1, param_dim), new_parameters


def preprocess_indicators_lf2i(
    test_statistics: np.ndarray,
    critical_values: np.ndarray,
    parameters: np.ndarray,
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return test_statistics.reshape(-1, ), critical_values.reshape(-1, ), parameters.reshape(-1, param_dim)


def preprocess_indicators_posterior(
    parameters: torch.Tensor,
    samples: torch.Tensor,
    parameter_grid: torch.Tensor,
    param_dim: int,
    batch_size: int,
    posterior: Union[Any, Sequence[Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Iterator[Any]]:
    if isinstance(posterior, Sequence):
        posterior = iter(posterior)
    else:
        posterior = itertools.cycle([posterior])
    return parameters.reshape(-1, param_dim), samples.reshape(parameters.shape[0], batch_size, -1), parameter_grid.reshape(-1, param_dim), posterior


def preprocess_indicators_prediction(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.numpy()
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if (len(samples.shape) == 3) and (samples.shape[1] > 1):
        warnings.warn(f"You provided a simulated set with single-sample size = {samples.shape[1]}. This dimension will be flattened to compute indicators. Is this the desired behaviour?")
    return parameters.reshape(-1, param_dim), samples.reshape(-1, samples.shape[-1])
