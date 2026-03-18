from typing import Union, Tuple, Any, List, Optional
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
    estimator: Any,
    parameter_space_bounds: Optional[List[Tuple[float]]] = None
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

    # Normalize parameters if bounds are provided
    parameters = preprocess_normalize_parameters(parameters, parameter_space_bounds)

    # Relabel via permutation
    parameters, samples, labels = preprocess_odds_relabel(parameters, samples)

    if (len(samples.shape) == 3) and (samples.shape[1] > 1):
        if isinstance(parameters, np.ndarray):
            pass
        else:
            data_set_size, batch_size, data_dim = samples.shape
            parameters_expanded = parameters.unsqueeze(1).expand(data_set_size, batch_size, param_dim)
            params_samples = torch.cat([parameters_expanded, samples], dim=-1)
    else:
        if isinstance(parameters, np.ndarray):
            params_samples = np.hstack((
                parameters.reshape(-1, param_dim),
                samples.reshape(-1, samples.shape[-1]) if samples.ndim == 2 else samples.reshape(samples.shape[0], -1)
            ))
        else:
            params_samples = torch.hstack((
                parameters.reshape(-1, param_dim),
                samples.reshape(-1, samples.shape[-1]) if samples.ndim == 2 else samples.reshape(samples.shape[0], -1)
            ))

    return labels, params_samples


def preprocess_odds_relabel(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    use_distant_pairs: bool = False
) -> Tuple:
    """
    Create labels by splitting data into two halves, keeping one matched and permuting the other.
    
    Parameters
    ----------
    parameters : array of shape (n_samples, param_dim)
    samples : array of shape (n_samples, batch_size, data_dim)
    use_distant_pairs : bool
        If True, use farthest parameter pairs for negative class.
        If False, use random permutation within the second half.
    
    Returns
    -------
    Tuple of (all_parameters, all_samples, all_labels)
    """
    params_is_torch = isinstance(parameters, torch.Tensor)
    samples_is_torch = isinstance(samples, torch.Tensor)
    
    if not ((params_is_torch and samples_is_torch) or 
            (isinstance(parameters, np.ndarray) and isinstance(samples, np.ndarray))):
        raise TypeError("parameters and samples must both be numpy arrays or both torch tensors")
    
    n_samples = samples.shape[0]
    
    # Ensure even number of samples
    if n_samples % 2 != 0:
        n_samples = n_samples - 1
        if params_is_torch:
            parameters = parameters[:n_samples]
            samples = samples[:n_samples]
        else:
            parameters = parameters[:n_samples]
            samples = samples[:n_samples]
    
    half = n_samples // 2
    
    if params_is_torch:
        # Split into two halves
        params_first = parameters[:half].clone()
        samples_first = samples[:half].clone()
        
        params_second = parameters[half:].clone()
        samples_second = samples[half:].clone()
        
        # Class 1: First half with matched pairs
        params_pos = params_first
        samples_pos = samples_first
        labels_pos = torch.ones(half, dtype=torch.int64)
        
        # Class 0: Second half with permuted parameters
        if use_distant_pairs:
            # If half <= 1 there is no meaningful "distant" partner, fall back to random permutation
            if half <= 1:
                permutation = torch.randperm(half)
            else:
                # Calculate pairwise distances within second half
                params_expanded = params_second.unsqueeze(1)  # (half, 1, param_dim)
                params_tiled = params_second.unsqueeze(0)     # (1, half, param_dim)
                distances = torch.norm(params_expanded - params_tiled, dim=2)  # (half, half)

                # For each sample, find a distant parameter (not necessarily the farthest to avoid always using same pairs)
                distances.fill_diagonal_(float('-inf'))
                # Get top-k farthest, then randomly choose from them
                k = min(5, half - 1)  # Consider up to 5 farthest parameters
                # topk requires k >= 1; half > 1 ensures this
                _, top_k_indices = torch.topk(distances, k, dim=1)
                # Randomly select one from top-k for each sample (on the same device)
                device = top_k_indices.device
                random_k_idx = torch.randint(0, k, (half,), device=device)
                permutation = top_k_indices[torch.arange(half, device=device), random_k_idx]
        else:
            # Random permutation ensuring no i->i mapping
            permutation = torch.randperm(half)
            # Ensure derangement (no fixed points)
            for i in range(half):
                if permutation[i] == i:
                    # Swap with next position (with wraparound)
                    j = (i + 1) % half
                    permutation[i], permutation[j] = permutation[j], permutation[i]
        
        params_neg = params_second[permutation]
        samples_neg = samples_second  # Keep samples in original order
        labels_neg = torch.zeros(half, dtype=torch.int64)
        
        # Combine
        all_parameters = torch.cat([params_pos, params_neg], dim=0)
        all_samples = torch.cat([samples_pos, samples_neg], dim=0)
        all_labels = torch.cat([labels_pos, labels_neg], dim=0)
        
        # Shuffle the combined dataset
        shuffle_idx = torch.randperm(n_samples)
        all_parameters = all_parameters[shuffle_idx]
        all_samples = all_samples[shuffle_idx]
        all_labels = all_labels[shuffle_idx]
        
        return all_parameters, all_samples, all_labels
    
    else:  # numpy version
        # Split into two halves
        params_first = parameters[:half].copy()
        samples_first = samples[:half].copy()
        
        params_second = parameters[half:].copy()
        samples_second = samples[half:].copy()
        
        # Class 1: First half with matched pairs
        params_pos = params_first
        samples_pos = samples_first
        labels_pos = np.ones(half, dtype=np.int64)
        
        # Class 0: Second half with permuted parameters
        if use_distant_pairs:
            from scipy.spatial.distance import cdist
            distances = cdist(params_second, params_second)
            np.fill_diagonal(distances, -np.inf)
            
            # Get top-k farthest, then randomly choose
            k = min(5, half - 1)
            top_k_indices = np.argpartition(distances, -k, axis=1)[:, -k:]
            random_k_idx = np.random.randint(0, k, size=half)
            permutation = top_k_indices[np.arange(half), random_k_idx]
        else:
            permutation = np.random.permutation(half)
            # Ensure derangement
            for i in range(half):
                if permutation[i] == i:
                    j = (i + 1) % half
                    permutation[i], permutation[j] = permutation[j], permutation[i]
        
        params_neg = params_second[permutation]
        samples_neg = samples_second
        labels_neg = np.zeros(half, dtype=np.int64)
        
        # Combine
        all_parameters = np.vstack([params_pos, params_neg])
        all_samples = np.vstack([samples_pos, samples_neg]) if samples.ndim == 2 else np.concatenate([samples_pos, samples_neg], axis=0)
        all_labels = np.concatenate([labels_pos, labels_neg])
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        all_parameters = all_parameters[shuffle_idx]
        all_samples = all_samples[shuffle_idx]
        all_labels = all_labels[shuffle_idx]
        
        return all_parameters, all_samples, all_labels


def preprocess_for_odds_cv(
    parameters: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    batch_size: int,
    data_dim: int,
    estimator: Any,
    parameter_space_bounds: Optional[List[Tuple[float]]] = None
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

        if parameter_space_bounds is not None:
            parameters = preprocess_normalize_parameters(parameters, parameter_space_bounds)

        if samples.ndim == 3 and samples.shape[1] > 1:
            data_set_size, batch_size, data_dim = samples.shape
            parameters_expanded = parameters.unsqueeze(1).expand(data_set_size, batch_size, param_dim)
            params_samples = torch.cat([parameters_expanded, samples], dim=-1)
        else:
            params_samples = torch.hstack((
                torch.repeat_interleave(parameters.reshape(-1, param_dim), repeats=batch_size, dim=0),
                samples.reshape(-1, data_dim)
            ))
    else:
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.numpy()
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()

        if parameter_space_bounds is not None:
            parameters = preprocess_normalize_parameters(parameters, parameter_space_bounds)

        if samples.ndim == 3 and samples.shape[1] > 1:
            pass
        else:
            params_samples = np.hstack((
                np.repeat(parameters.reshape(-1, param_dim), repeats=batch_size, axis=0),
                samples.reshape(-1, data_dim)
            ))

    return parameters, samples, params_samples


def preprocess_for_odds_cs(
    parameter_grid: Union[np.ndarray, torch.Tensor],
    samples: Union[np.ndarray, torch.Tensor],
    param_dim: int,
    batch_size: int,
    data_dim: int,
    estimator: Any,
    parameter_space_bounds: Optional[List[Tuple[float]]] = None
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
        Note that parameter_grid is expected to be of shape (-1, poi_dim + nuisance_dim).
    samples : Union[np.ndarray, torch.Tensor]
        Array of samples. Should have shape `(n_samples, batch_size, data_dim)`.
    param_dim : int
        Dimensionality of the parameter space.
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
    parameter_grid = parameter_grid.reshape(-1, param_dim)
    samples = samples.reshape(-1, batch_size, data_dim)   
    # TODO: this is not general, i.e. assumes our torch “construction” with a Learner that has a model attribute
    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        if isinstance(parameter_grid, np.ndarray):
            parameter_grid = torch.from_numpy(parameter_grid)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)

        if parameter_space_bounds is not None:
            parameter_grid = preprocess_normalize_parameters(parameter_grid, parameter_space_bounds)

        if samples.ndim == 3 and samples.shape[1] > 1:
            data_set_size, batch_size, data_dim = samples.shape
            parameter_grid_expanded = parameter_grid.unsqueeze(1).expand(-1, batch_size, param_dim)  # shape (param_grid_size, batch_size, param_dim)
            parameter_grid_expanded_repeated = torch.repeat_interleave(parameter_grid_expanded, repeats=samples.shape[0], dim=0)
            samples_tiled = torch.tile(samples, dims=(parameter_grid.shape[0], 1, 1))  # shape (param_grid_size*n_samples, batch_size, data_dim)
            params_samples = torch.cat([parameter_grid_expanded_repeated, samples_tiled], dim=-1)
        else:
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

        if parameter_space_bounds is not None:
            parameter_grid = preprocess_normalize_parameters(parameter_grid, parameter_space_bounds)

        if samples.ndim == 3 and samples.shape[1] > 1:
            pass
        else:
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
    raise NotImplementedError


def preprocess_odds_maximization(
    estimator: Any,
    nominal_params: torch.Tensor,
    opt_param: Union[np.ndarray, torch.Tensor],
    opt_param_index: int,
    sample: Union[np.ndarray, torch.Tensor],
    parameter_space_bounds: Optional[List[Tuple[float]]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Preprocessing for one-at-a-time optimization of the odds ratio. Given a 
    sample, a nominal parameter vector, and a 
    particular component j of the full parameter vector, concatenate the 
    parameters by swapping the j-th component of the nominal parameter with the 
    optimization variable.

    Args:
        estimator
        nominal_params: Of shape (poi_dim + nuisance_dim,)
        opt_param: Scalar value for the optimization variable (the j-th component of the parameter vector)
        opt_param_index: Index of the optimization variable in the full parameter vector
        sample: One sample, of shape (batch_size, data_dim)
    """
    batch_size, data_dim = sample.shape
    param_dim = len(nominal_params)

    if not isinstance(opt_param, torch.Tensor):
        opt_param = torch.tensor(opt_param, dtype=nominal_params.dtype)
    else:
        opt_param = opt_param.to(nominal_params.dtype)

    if isinstance(estimator, torch.nn.Module) or (hasattr(estimator, 'model') and isinstance(estimator.model, torch.nn.Module)):
        # Reshape to start
        sample = sample.reshape(1, batch_size, data_dim)  # shape (1, batch_size, data_dim)
        nominal_params[opt_param_index] = opt_param  # swap in the optimization variable
        nominal_params = nominal_params.reshape(1, param_dim)  # shape (1, param_dim)
        nominal_params_clone = nominal_params.clone()  # Avoid in-place modification of original nominal_params

        if parameter_space_bounds is not None:
            nominal_params_clone = preprocess_normalize_parameters(nominal_params_clone, parameter_space_bounds)

        parameter_expanded = nominal_params_clone.unsqueeze(1).expand(1, batch_size, param_dim)  # shape (1, batch_size, param_dim)
        estimator_inputs = torch.cat([parameter_expanded, sample], dim=-1).float()  # shape (1, batch_size, param_dim + data_dim)
    else:
        pass

    return estimator_inputs


def preprocess_normalize_parameters(
    parameters: Union[np.ndarray, torch.Tensor],
    parameter_space_bounds: Optional[List[Tuple[float]]] = None
) -> Union[np.ndarray, torch.Tensor]:
    check_for_nans(parameters)

    if isinstance(parameters, torch.Tensor):
        parameters = parameters.clone()
    else:
        parameters = np.copy(parameters)

    if parameter_space_bounds is not None:
        for i, (lower, upper) in enumerate(parameter_space_bounds):
            if lower < upper:
                parameters[:, i] = (parameters[:, i] - lower) / (upper - lower)
            else:
                warnings.warn(f"Invalid bounds for parameter {i}: lower {lower} is not less than upper {upper}. Skipping normalization for this parameter.")
    return parameters
