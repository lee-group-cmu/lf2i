from random import sample
from typing import Tuple, Union, List
import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBModel


def preprocess_waldo_estimation(
    parameters: Union[np.ndarray, torch.Tensor], 
    samples: Union[np.ndarray, torch.Tensor], 
    estimation_method: str,
    estimator: object,
    param_dim: int
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    if estimation_method == 'prediction':
        # make sure types are correct for:
        if isinstance(estimator, torch.nn.Module):
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
    else:
        # for posterior estimation we currently support `SBI` from mackelab, which uses PyTorch
        if isinstance(parameters, np.ndarray):
            parameters = torch.from_numpy(parameters)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
    
    if (len(samples.shape) == 3) and (samples.shape[1] > 1):
        warnings.warn(f"You provided a simulated set with single-sample size = {samples.shape[1]}. This dimension will be flattened for estimation or evaluation. Is this the desired behaviour?")
    if (param_dim == 1) and (estimation_method == 'prediction'):
        parameters = parameters.reshape(-1, )
    else:
        parameters = parameters.reshape(-1, param_dim)
    return parameters, samples.reshape(-1, samples.shape[-1])


def preprocess_waldo_evaluation(
    parameters: Union[np.ndarray, torch.Tensor], 
    samples: Union[np.ndarray, torch.Tensor], 
    estimation_method: str,
    estimator: object,
    param_dim: int
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    return preprocess_waldo_estimation(parameters, samples, estimation_method, estimator, param_dim)


def preprocess_waldo_computation(
    parameters: Union[np.ndarray, torch.Tensor],
    conditional_mean: Union[np.ndarray, List],
    conditional_var: Union[np.ndarray, List],
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.numpy()
    if param_dim == 1:
        parameters = parameters.reshape(-1, param_dim)
        if isinstance(conditional_mean, List):
            # method == posterior
            conditional_mean = np.vstack(conditional_mean)
            conditional_var = np.vstack(conditional_var)
        else:
            # method == prediction
            conditional_mean = conditional_mean.reshape(-1, param_dim)
            conditional_var = conditional_var.reshape(-1, param_dim)
    else:
        # so that when sliced it is still 2-dimensional
        parameters = parameters.reshape(-1, 1, param_dim)
        if isinstance(conditional_mean, np.ndarray):
            # only when method == prediction; this should rarely be used
            conditional_mean = conditional_mean.reshape(-1, 1, param_dim)
            conditional_var = conditional_var.reshape(-1, param_dim, param_dim)
    
    return parameters, conditional_mean, epsilon_variance_correction(conditional_var, param_dim)


def epsilon_variance_correction(
    conditional_var: Union[List, np.ndarray],
    param_dim: int,
    epsilon: float = 1e-3
) -> Union[List, np.ndarray]:
    # Make sure the estimated conditional variance is always >= 0+epsilon to avoid ZeroDivisionError or exploding test statistics.
    warning_msg = f"""At least one element of `conditional_var` is negative.\n
                        You should make sure your conditional variance estimator output is non-negative.\n"""
    if param_dim == 1:
        if np.sum(conditional_var < 0) > 0:
            warnings.warn(warning_msg + f"Applying eps={epsilon} correction.")
        return np.maximum(conditional_var, epsilon)
    else:
        # not implemented   
        return conditional_var
