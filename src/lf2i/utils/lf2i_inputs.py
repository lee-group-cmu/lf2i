from typing import Union, Tuple
import warnings

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
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    parameter_grid: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    data_sample_size: int,
) -> Tuple[np.ndarray, Union[np.ndarray, torch.Tensor], np.ndarray]:
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.float().numpy()
    if isinstance(parameter_grid, torch.Tensor):
        parameter_grid = parameter_grid.float().numpy()
    parameters = parameters.reshape(-1, param_dim)
    return parameters, samples.reshape(parameters.shape[0], data_sample_size, param_dim), parameter_grid.reshape(-1, param_dim)


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
