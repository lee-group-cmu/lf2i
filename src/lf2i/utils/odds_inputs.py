from typing import Union, Tuple, Any, List
import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBModel

from lf2i.utils.miscellanea import to_np_if_torch, check_for_nans


def preprocess_odds_estimation(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    estimator: Any
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    check_for_nans(parameters)
    check_for_nans(samples)
    # TODO: this is not general, i.e. assumes our torch “construction” with a Learner that has a model attribute
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        # PyTorch models
        if isinstance(parameters, np.ndarray):
            parameters = torch.from_numpy(parameters)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
    if isinstance(estimator, (BaseEstimator, XGBModel)):
        # Scikit-Learn or XGBoost models
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy()
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
    
    if (len(samples.shape) == 3) and (samples.shape[1] > 1):
        warnings.warn(
            f"""You provided a simulated set with single-sample size = {samples.shape[1]}.\n
            This dimension will be flattened for estimation. Is this the desired behaviour?"""
        )
    if isinstance(parameters, np.ndarray):
        params_samples = np.hstack((
            parameters.reshape(-1, param_dim),
            samples.reshape(-1, samples.shape[-1])
        ))
    else:
        params_samples = torch.hstack((
            parameters.reshape(-1, param_dim),
            samples.reshape(-1, samples.shape[-1])
        ))

    # Relabel via permutation
    labels = preprocess_odds_relabel(parameters, samples)

    return labels.reshape(-1, ), params_samples


def preprocess_odds_relabel(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Create labels from parameters and samples.

    Behavior:
    - If `samples` has shape (n_configs, batch_size, data_dim) it returns
      labels = repeat(arange(n_configs), repeats=batch_size).
    - If `samples` has shape (n_configs, data_dim) it returns labels = arange(n_configs).
    - Works when `parameters` / `samples` are torch Tensors or numpy arrays.
      Both inputs must be of the same type.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        1D integer label array / tensor of length equal to number of generated samples.
    """
    # require inputs to be of the same type
    params_is_torch = isinstance(parameters, torch.Tensor)
    samples_is_torch = isinstance(samples, torch.Tensor)
    params_is_np = isinstance(parameters, np.ndarray)
    samples_is_np = isinstance(samples, np.ndarray)

    if not ((params_is_torch and samples_is_torch) or (params_is_np and samples_is_np)):
        raise TypeError("parameters and samples must both be numpy arrays or both torch tensors")

    if params_is_torch:
        # infer number of parameter configurations from parameters if possible
        try:
            n_configs = parameters.reshape(-1, parameters.shape[-1]).shape[0]
        except Exception:
            n_configs = parameters.shape[0]

        if samples.ndim == 3:
            batch_size = samples.shape[1]
            labels = torch.repeat_interleave(torch.arange(n_configs, dtype=torch.long), repeats=batch_size)
        elif samples.ndim == 2:
            if samples.shape[0] == n_configs:
                labels = torch.arange(n_configs, dtype=torch.long)
            else:
                labels = torch.arange(samples.shape[0], dtype=torch.long)
        else:
            labels = torch.arange(samples.shape[0], dtype=torch.long)

        expected = n_configs * (samples.shape[1] if samples.ndim == 3 else 1)
        if labels.numel() != expected:
            warnings.warn("Generated labels length does not match expected number of samples from parameters/samples.")
        return labels

    else:
        # numpy branch (original behavior)
        try:
            n_configs = parameters.reshape(-1, parameters.shape[-1]).shape[0]
        except Exception:
            n_configs = parameters.shape[0]

        if samples.ndim == 3:
            batch_size = samples.shape[1]
            labels = np.repeat(np.arange(n_configs), repeats=batch_size)
        elif samples.ndim == 2:
            if samples.shape[0] == n_configs:
                labels = np.arange(n_configs)
            else:
                labels = np.arange(samples.shape[0])
        else:
            labels = np.arange(samples.shape[0])

        if labels.size != (n_configs * (samples.shape[1] if samples.ndim == 3 else 1)):
            warnings.warn("Generated labels length does not match expected number of samples from parameters/samples.")

        return labels


def preprocess_for_odds_cv(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    batch_size: int,
    data_dim: int,
    estimator: Any
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    """Flatten samples along `batch_size` dimension and stack them with corresponding repeated parameters column-wise.
    This is done to simultaneously estimate odds at all samples, given the corresponding parameters.
    
    Inputs are converted to correct format depending on estimator type.

    Parameters
    ----------
    parameters : Union[np.ndarray, torch.Tensor]
        Array of parameters, one for each batch of size `batch_size`.
    samples : Union[np.ndarray, torch.Tensor]
        Array of samples. Assumed to have shape `(n_samples, batch_size, data_dim)`.
    param_dim : int
        Dimensionality of the parameter space.
    batch_size : int
        Number of samples in a batch from a specific parameter configuration.
    data_dim: int
        Dimensionality of each single sample.

    Returns
    -------
    Tuple[Union[np.ndarray, torch.Tensor]]
        Parameters, samples, and stacked parameters and samples. The stacked vector is flattened along dim 1, with output shape `(n_samples*batch_size, param_dim+data_dim)`.
    """
    check_for_nans(parameters)
    check_for_nans(samples)

    # Ensure samples have shape (n_samples, batch_size, data_dim).
    # Accept inputs that are (batch_size, data_dim) and treat them as single sample (n_samples=1).
    if isinstance(samples, torch.Tensor):
        if samples.ndim == 2:
            samples = samples.reshape(1, batch_size, data_dim)
    else:
        if samples.ndim == 2:
            samples = samples.reshape(1, batch_size, data_dim)

    # TODO: this is not general, i.e. assumes our torch “construction” with a Learner that has a model attribute
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        if isinstance(parameters, np.ndarray):
            parameters = torch.from_numpy(parameters)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        params_samples = torch.hstack((
            torch.repeat_interleave(parameters.reshape(-1, param_dim), repeats=batch_size, dim=0),
            samples.reshape(-1, data_dim)
        ))
    else:
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy()
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        params_samples = np.hstack((
            np.repeat(parameters.reshape(-1, param_dim), repeats=batch_size, axis=0),
            samples.reshape(-1, data_dim)
        ))

    return parameters, samples, params_samples


def preprocess_for_odds_cs(
    parameter_grid: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    poi_dim: int,
    batch_size: int,
    data_dim: int,
    estimator: Any
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    """Repeat and tile both parameter_grid and samples to achieve the following data structure:
        param_grid_0, samples_0_0
        param_grid_0, samples_0_1
        param_grid_1, samples_0_0
        param_grid_1, samples_0_1
        ...
        param_grid_0, samples_1_0
        param_grid_0, samples_1_1
        param_grid_1, samples_1_0
        param_grid_1, samples_1_1
        ...
    
    This is done to simultaneously estimated odds across all parameters *for each* sample.
 
    Parameters
    ----------
    parameter_grid : Union[np.ndarray, torch.Tensor]
        Array of parameters over which odds have to be evaluated *for each* sample.
    samples : Union[np.ndarray, torch.Tensor]
        Array of samples. Should have shape `(n_samples, batch_size, data_dim)`.
    poi_dim : int
        Dimensionality of the space of parameters of interest.
    batch_size : int
        Number of samples in a batch from a specific parameter configuration.
    data_dim: int
        Dimensionality of each single sample.

    Returns
    -------
    np.ndarray
        Parameter grid, samples, and stacked parameter grid and samples. The stacked vector has output shape `(param_grid_size*n_samples*batch_size, param_dim+data_dim)`.
    """    
    check_for_nans(parameter_grid)
    check_for_nans(samples)
    parameter_grid = parameter_grid.reshape(-1, poi_dim)
    samples = samples.reshape(-1, batch_size, data_dim)   
    # TODO: this is not general, i.e. assumes our torch “construction” with a Learner that has a model attribute
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        if isinstance(parameter_grid, np.ndarray):
            parameter_grid = torch.from_numpy(parameter_grid)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        params_samples = torch.hstack((
            torch.tile(
                torch.repeat_interleave(parameter_grid, repeats=batch_size, dim=0), 
                dims=(samples.shape[0], 1)
            ),
            torch.tile(samples, dims=(1, parameter_grid.shape[0], 1)).reshape(-1, data_dim)
        ))
    else:
        if isinstance(parameter_grid, torch.Tensor):
            parameter_grid = parameter_grid.numpy()
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        params_samples = np.hstack((
            np.tile(
                np.repeat(parameter_grid, repeats=batch_size, axis=0), 
                reps=(samples.shape[0], 1)
            ),
            np.tile(samples, reps=(1, parameter_grid.shape[0], 1)).reshape(-1, data_dim)
        ))
        
    return parameter_grid, samples, params_samples


def preprocess_odds_integration(
    estimator: Any,
    fixed_poi: Union[np.ndarray, torch.Tensor],
    integ_params: List[float],
    sample: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    batch_size: int
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        estimator_inputs = torch.hstack((
            torch.repeat_interleave(torch.cat((fixed_poi, torch.Tensor(integ_params))).reshape(1, param_dim), repeats=batch_size).reshape(-1, param_dim), 
            sample
        ))
    else:
        estimator_inputs = np.hstack((
            np.repeat(np.concatenate((to_np_if_torch(fixed_poi), np.array(integ_params))).reshape(1, param_dim), repeats=batch_size).reshape(-1, param_dim), 
            sample
        ))
    return estimator_inputs


def preprocess_odds_maximization(
    estimator: Any,
    fixed_poi: Union[np.ndarray, torch.Tensor],
    opt_params: Tuple[np.ndarray],
    sample: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    batch_size: int
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        estimator_inputs = torch.hstack((
            torch.repeat_interleave(torch.cat((fixed_poi, torch.from_numpy(np.concatenate(opt_params)))).reshape(1, param_dim), repeats=batch_size).reshape(-1, param_dim), 
            sample
        ))
    else:
        estimator_inputs = np.hstack((
            np.repeat(np.concatenate((to_np_if_torch(fixed_poi), np.concatenate(opt_params))).reshape(1, param_dim), repeats=batch_size).reshape(-1, param_dim), 
            sample
        ))
    return estimator_inputs
