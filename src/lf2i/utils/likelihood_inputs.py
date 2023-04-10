from typing import Union, Tuple, Any
import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBModel


def preprocess_odds_estimation(
    labels: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    estimator: Any
) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    if isinstance(estimator, torch.nn.Module):
        # PyTorch models
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(parameters, np.ndarray):
            parameters = torch.from_numpy(parameters)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
    if isinstance(estimator, (BaseEstimator, XGBModel)):
        # Scikit-Learn or XGBoost models
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
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
    return labels.reshape(-1, ), params_samples


def preprocess_odds_cv(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    data_sample_size: int,
    estimator: Any
) -> np.ndarray:
    """Flatten samples along `data_sample_size` dimension and stack them with corresponding repeated parameters column-wise.
    If `parameters` or `samples` are of type `torch.Tensor`, they are converted to `np.ndarray`.

    This is done to simultaneously estimate odds at all samples, given the corresponding parameters.

    Parameters
    ----------
    parameters : Union[np.ndarray, torch.Tensor]
        Array of parameters, one for each sample of size `data_sample_size`.
    samples : Union[np.ndarray, torch.Tensor]
        Array of samples. Assumed to have shape `(n_samples, data_sample_size, data_dim)`.
    param_dim : int
        Dimensionality of the parameter.
    data_sample_size : int
        Dimensionality of a sample from a specific parameter.

    Returns
    -------
    np.ndarray
        Array containing both parameters and samples, with shape `(n_samples*data_sample_size, param_dim+data_dim)`.
    """
    if isinstance(parameters, np.ndarray):
        parameters = torch.from_numpy(parameters)
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
    params_samples = np.hstack((
        np.repeat(parameters.reshape(-1, param_dim), repeats=data_sample_size, axis=0),
        samples.reshape(-1, samples.shape[-1])
    ))
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        return torch.from_numpy(params_samples).float()
    else:
        return params_samples


def preprocess_odds_cs(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    data_sample_size: int,
    estimator: Any
) -> np.ndarray:
    """Tile samples along `data_sample_size` dimension for `parameters.shape[0]` times and flatten them, 
    then stack column-wise with parameters repeated `data_sample_size` times and tiled `n_samples` times.

    This is done to simultaneously estimated odds across all parameters *for each* sample.
 
    Parameters
    ----------
    parameters : Union[np.ndarray, torch.Tensor]
        Array of parameters over which odds have to be evaluated *for each* sample.
    samples : Union[np.ndarray, torch.Tensor]
        Array of samples. Assumed to have shape `(n_samples, data_sample_size, data_dim)`.
    param_dim : int
        Dimensionality of the parameter.
    data_sample_size : int
        Dimensionality of a sample from a specific parameter.

    Returns
    -------
    np.ndarray
        Array containing both parameters and samples ready for evaluation, with shape `(n_samples*data_sample_size*n_parameters, param_dim+data_dim)`.
    """
    if len(samples.shape) < 3:
        warnings.warn(
            f"""Samples shape is {samples.shape}. Dimension 0 is treated as the data sample size 
            (i.e., all Xs from a specific parameter).\nTo suppress this warning, pass samples with shape (n_samples, data_sample_size, data_dim)"""
        )
        samples = samples.reshape(1, -1, -1)
    if isinstance(parameters, np.ndarray):
        parameters = torch.from_numpy(parameters)
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
    params_samples = np.hstack((
        np.tile(
            np.repeat(parameters.reshape(-1, param_dim), repeats=data_sample_size, axis=0), 
            reps=(samples.shape[0])
        ),
        np.tile(samples, reps=(1, parameters.reshape(-1, param_dim).shape[0], 1)).reshape(-1, samples.shape[-1])
    ))
    if isinstance(estimator, torch.nn.Module):
        return torch.from_numpy(params_samples).float()
    else:
        return params_samples
