from typing import Union, Tuple, Any, Optional
import warnings

import numpy as np
from scipy.stats import norm
import torch
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.kde import KDEWrapper


def hpd_region(
    posterior: Union[NeuralPosterior, KDEWrapper, Distribution],
    param_grid: torch.Tensor, 
    x: torch.Tensor, 
    credible_level: float, 
    num_level_sets: int = 100_000,
    norm_posterior_samples: Optional[int] = None, 
    tol: float = 0.01
) -> Tuple[float, torch.Tensor]:
    """
    Compute a high-posterior-density region at the desired credibility level.

    Parameters
    ----------
    posterior : Union[NeuralPosterior, KDEWrapper, Distribution]
        Estimated posterior distribution. Must implement `log_prob(...)` method.
    param_grid : torch.Tensor
        Grid of evaluation points to be considered for the construction of the HPD region.
    x : torch.Tensor
        Observed sample given which the posterior :math:`p(\theta|x)` is evaluated.
    credible_level : float
        Desired credible level for the HPD region. Must be in (0, 1).
    num_level_sets : int, optional
        Number of level sets to examine, by default 100_000. A high number of level sets ensures the actual credible level is as close as possible to the specified one.
    norm_posterior_samples : int, optional
        Number of samples to use to estimate the leakage correction factor, by default None. More samples lead to better estimates of the normalization constant when returning a normalized posterior.
        If `None`, uses the un-normalized posterior (but note that the density is already being explicitly normalized over the `param_grid`).
    tol : float, optional
        Actual credible levels within `tol` of the specified `credible_level` will be considered acceptable, by default 0.01.
        NOTE: this is used as a stopping criterion, but if the closest actual credible level is not within `tol` of `credible_level`, a warning is raised but the HPD region is still returned.

    Returns
    -------
    Tuple[float, torch.Tensor]
        Actual credible level and HPD region.

    Raises
    ------
    ValueError
        If posterior is not an instance of NeuralPosterior or KDEWrapper from the `sbi` package, or torch.Distribution.
    """
    # evaluate posterior over grid of values
    if isinstance(posterior, NeuralPosterior):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ... when using NSF
            posterior_probs = torch.exp(posterior.log_prob(
                theta=param_grid, x=x, 
                norm_posterior=True if norm_posterior_samples else False,
                leakage_correction_params={'num_rejection_samples': norm_posterior_samples}
            ).double()).double()
    elif isinstance(posterior, (KDEWrapper, Distribution)):
        posterior_probs = torch.exp(posterior.log_prob(param_grid).double()).double()
    else:
        raise ValueError
    posterior_probs /= torch.sum(posterior_probs)  # normalize

    # descend the level sets of the posterior and stop when the area above a level equals credible level (up to tolerance)
    level_sets = torch.linspace(torch.max(posterior_probs).item(), 0, num_level_sets)  # thresholds to include or not parameters
    credible_levels_trajectory = []
    idx = 0
    current_credible_level, current_level_set_idx = 0, idx
    while abs(current_credible_level - credible_level) > tol:
        if idx == (num_level_sets-1):  # no more to examine
            warnings.warn(f'All level sets analyzed. Actual credible level: {current_credible_level}. Actual tolerance: {current_credible_level - credible_level}')
            break
        new_credible_level = torch.sum(posterior_probs[posterior_probs >= level_sets[idx]])
        credible_levels_trajectory.append(new_credible_level)
        if abs(new_credible_level - credible_level) < abs(current_credible_level - credible_level):
            current_credible_level = new_credible_level
            current_level_set_idx = idx
        idx += 1
    # all params such that p(params|x) > level_set, where level_set is the last chosen one
    return current_credible_level, param_grid[posterior_probs >= level_sets[current_level_set_idx], :]


def gaussian_prediction_sets(
    conditional_mean_estimator: Any,
    conditional_variance_estimator: Any,
    samples: Union[torch.Tensor, np.ndarray],
    confidence_level: float,
    param_dim: int
) -> np.ndarray:
    r"""Compute prediction sets centered around the point estimate using a Gaussian approximation: :math:`\mathbb{E}[\theta|X] \pm z_{1-\alpha/2} \cdot \sqrt{\mathbb{V}[\theta|X]}`.

    Parameters
    ----------
    conditional_mean_estimator : Any
        Prediction algorithm to estimate the conditional mean under squared error loss. Must implement `predict(X=...)` method.
    conditional_variance_estimator : Any
        Prediction algorithm to estimate the conditional variance under squared error loss. Must implement `predict(X=...)` method.
        One way to get this is to use the `conditional_mean_estimator`, compute the squared residuals, and regress them against the data.
    samples : Union[torch.Tensor, np.ndarray]
        Array of samples given which to compute the prediction sets. The 0-th dimension indexes samples coming from different parameters.
        One prediction set for each “row” will be computed.
    confidence_level : float
        Desired confidence level of the resulting prediction sets. It determines the Gaussian percentile to use as multiplier for the error estimate. 
    param_dim : int
        Dimensionality of the parameter.

    Returns
    -------
    np.ndarray
        Array of dimensions (n_samples, 2), where the columns are for the lower and upper bounds of the prediction sets.

    Raises
    ------
    NotImplementedError
        Not yet implemented for non-scalar parameters.
    """
    if param_dim == 1:
        conditional_mean = conditional_mean_estimator.predict(X=samples).reshape(-1, 1)
        conditional_var = conditional_variance_estimator.predict(X=samples).reshape(-1, 1)
        z_percentile = norm(loc=0, scale=1).ppf(1-((1-confidence_level)/2))  # two-tailed
        prediction_sets_bounds = np.hstack((
            conditional_mean - z_percentile*np.sqrt(conditional_var),
            conditional_mean + z_percentile*np.sqrt(conditional_var)
        ))
        assert prediction_sets_bounds.shape == (conditional_mean.shape[0], 2)
    else:
        raise NotImplementedError

    return prediction_sets_bounds
