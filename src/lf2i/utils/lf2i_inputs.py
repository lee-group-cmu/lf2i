from typing import Union, Tuple

import numpy as np
import torch


def preprocess_quantile_regression(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    prediction_grid: Union[np.ndarray, torch.Tensor],
    param_dim: int
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    if isinstance(test_statistics, torch.Tensor):
        test_statistics = test_statistics.numpy()
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.numpy()
    if isinstance(prediction_grid, torch.Tensor):
        prediction_grid = prediction_grid.numpy()
    return test_statistics.reshape(-1, ), parameters.reshape(-1, param_dim), prediction_grid.reshape(-1, param_dim)


def preprocess_neyman_inversion(
    test_statistic: np.ndarray,
    critical_values: np.ndarray,
    parameter_grid: np.ndarray,
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    parameters: np.ndarray,
    samples: np.ndarray,
    parameter_grid: np.ndarray,
    sample_size: int,
    samples_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameters = parameters.reshape(-1, sample_size)
    return parameters, samples.reshape(parameters.shape[0], samples_dim, sample_size), parameter_grid.reshape(-1, sample_size)
