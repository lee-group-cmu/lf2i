from typing import Union, Tuple, Any, Sequence, Iterator, Optional
import warnings

import itertools
import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor, TabularDataset

from lf2i.utils.miscellanea import check_for_nans, to_np_if_torch


def preprocess_train_quantile_regression(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    estimator: Any
) -> Union[Tuple[Union[np.ndarray, torch.Tensor]], TabularDataset]:
    check_for_nans(test_statistics)
    check_for_nans(parameters)

    if isinstance(estimator, TabularPredictor):
        if isinstance(test_statistics, torch.Tensor):
            test_statistics = test_statistics.numpy().reshape(-1, )
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy().reshape(-1, param_dim)
        test_statistics = pd.Series(test_statistics, name='target')
        parameters = pd.DataFrame(parameters, columns=[f'feature{i+1}' for i in range(parameters.shape[1])])
        dataset = TabularDataset(data=pd.concat([test_statistics, parameters], axis=1))
        return dataset
    else:
        if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
            # PyTorch models
            if isinstance(test_statistics, np.ndarray):
                test_statistics = torch.from_numpy(test_statistics).reshape(-1, )
            if isinstance(parameters, np.ndarray):
                parameters = torch.from_numpy(parameters).reshape(-1, param_dim)
        else:
            # numpy-based models
            if isinstance(test_statistics, torch.Tensor):
                test_statistics = test_statistics.numpy().reshape(-1, )
            if isinstance(parameters, torch.Tensor):
                parameters = parameters.numpy().reshape(-1, param_dim)

        return test_statistics, parameters


def preprocess_predict_quantile_regression(
    parameters: Union[np.ndarray, torch.Tensor],
    estimator: Any,
    param_dim: int
) -> Union[np.ndarray, torch.Tensor, TabularDataset]:
    check_for_nans(parameters)
    
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        # PyTorch models
        if isinstance(parameters, np.ndarray):
            parameters = torch.from_numpy(parameters).reshape(-1, param_dim)
    elif isinstance(estimator, TabularPredictor):
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy().reshape(-1, param_dim)
        parameters = TabularDataset(data=pd.DataFrame(parameters, columns=[f'feature{i+1}' for i in range(parameters.shape[1])]))
    else:
        # numpy-based models
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy().reshape(-1, param_dim)
    return parameters


def preprocess_fit_p_values(
    inp: Union[np.ndarray, torch.Tensor],
    rejection_probs_model: Any
) -> Union[np.ndarray, torch.Tensor]:
    check_for_nans(inp)
    if isinstance(rejection_probs_model, torch.nn.Module) or (hasattr(rejection_probs_model, 'model') and isinstance(rejection_probs_model.model, torch.nn.Module)):
        # PyTorch models
        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp)
        if inp.ndim == 1:
            inp = inp.unsqueeze(1)
    else:  # assume anything else works with numpy arrays
        # Scikit-Learn, XGBoost, CatBoost, etc...
        if isinstance(inp, torch.Tensor):
            inp = inp.numpy()
        if inp.ndim == 1:
            inp = np.expand_dims(inp, axis=1)
    return inp


def preprocess_predict_p_values(
    mode: str,
    test_stats: Union[np.ndarray, torch.Tensor],
    poi: Union[np.ndarray, torch.Tensor],
    rejection_probs_model: Any
) -> Union[np.ndarray, torch.Tensor]:
    check_for_nans(test_stats)
    check_for_nans(poi)
    if isinstance(rejection_probs_model, torch.nn.Module) or (hasattr(rejection_probs_model, 'model') and isinstance(rejection_probs_model.model, torch.nn.Module)):
        # PyTorch models
        if isinstance(test_stats, np.ndarray):
            test_stats = torch.from_numpy(test_stats)
        if isinstance(poi, np.ndarray):
            poi = torch.from_numpy(poi)
        if poi.ndim == 1:
            poi = poi.unsqueeze(1)
        if mode == 'confidence_sets':
            poi = torch.tile(poi, dims=(test_stats.reshape(-1, poi.shape[0]).shape[0], 1))
        stacked_inp = torch.hstack((test_stats.reshape(-1, 1), poi))
    else:  # assume anything else works with numpy arrays
        # Scikit-Learn, XGBoost, CatBoost, etc...
        if isinstance(test_stats, torch.Tensor):
            test_stats = test_stats.numpy()
        if isinstance(poi, torch.Tensor):
            poi = poi.numpy()
        if poi.ndim == 1:
            poi = np.expand_dims(poi, axis=1)
        if mode == 'confidence_sets':
            poi = np.tile(poi, reps=(test_stats.reshape(-1, poi.shape[0]).shape[0], 1))
        stacked_inp = np.hstack((test_stats.reshape(-1, 1), poi))
    return stacked_inp


def preprocess_neyman_inversion(
    test_statistics: Optional[Union[np.ndarray, torch.Tensor]],
    critical_values: Optional[Union[np.ndarray, torch.Tensor]],
    p_values: Optional[Union[np.ndarray, torch.Tensor]],
    parameter_grid: Union[np.ndarray, torch.Tensor],
    param_dim: int
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    check_for_nans(parameter_grid)
    parameter_grid = parameter_grid.reshape(-1, param_dim)
    parameter_grid = to_np_if_torch(parameter_grid)

    # either p_values alone or both test statistics and critical values should be given
    if test_statistics is not None:
        check_for_nans(test_statistics)
        test_statistics = to_np_if_torch(test_statistics).reshape(-1, parameter_grid.shape[0])
        num_obs = test_statistics.shape[0]
    if critical_values is not None:
        check_for_nans(critical_values)
        critical_values = to_np_if_torch(critical_values).reshape(1, parameter_grid.shape[0])
    if p_values is not None:
        check_for_nans(p_values)
        p_values = to_np_if_torch(p_values).reshape(-1, parameter_grid.shape[0])
        num_obs = p_values.shape[0]
    
    return (
        num_obs, 
        test_statistics if test_statistics is not None else None,
        critical_values if critical_values is not None else None,
        p_values if p_values is not None else None,
        parameter_grid
    )


def preprocess_diagnostics(
    indicators: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    new_parameters: Union[np.ndarray, torch.Tensor, None],
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    check_for_nans(indicators)
    check_for_nans(parameters)
    if new_parameters:
        check_for_nans(new_parameters)
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
    critical_values: Optional[np.ndarray],
    p_values: Optional[np.ndarray],
    parameters: np.ndarray,
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    check_for_nans(test_statistics)
    if critical_values is not None:
        check_for_nans(critical_values)
    if p_values is not None:
        check_for_nans(p_values)
    check_for_nans(parameters)
    return (
        test_statistics.reshape(-1, ),
        critical_values.reshape(-1, ) if critical_values is not None else None,
        p_values.reshape(-1, ) if p_values is not None else None,
        parameters.reshape(-1, param_dim)
    )


def preprocess_indicators_posterior(
    parameters: torch.Tensor,
    samples: torch.Tensor,
    parameter_grid: torch.Tensor,
    param_dim: int,
    batch_size: int,
    posterior: Union[Any, Sequence[Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Iterator[Any]]:
    check_for_nans(parameters)
    check_for_nans(samples)
    check_for_nans(parameter_grid)
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
    check_for_nans(parameters)
    check_for_nans(samples)
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.numpy()
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if (len(samples.shape) == 3) and (samples.shape[1] > 1):
        warnings.warn(f"You provided a simulated set with single-sample size = {samples.shape[1]}. This dimension will be flattened to compute indicators. Is this the desired behaviour?")
    return parameters.reshape(-1, param_dim), samples.reshape(-1, samples.shape[-1])
