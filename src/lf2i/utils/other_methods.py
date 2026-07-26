from typing import Union, Tuple, Any
import warnings

import numpy as np
from scipy.stats import norm
import torch
from torch.distributions import Distribution

from lf2i.estimators.base_posteriors import (
    AbstractNeuralPosterior,
    AbstractKDE
)
from lf2i.diagnostics import monte_carlo_methods as _monte_carlo_methods


def hpd_region(
    posterior: Union[AbstractNeuralPosterior, AbstractKDE, Distribution],
    param_grid: torch.Tensor,
    x: torch.Tensor,
    credible_level: float,
    num_level_sets: int = 100_000,
    tol: float = 0.01,
    **posterior_kwargs
) -> Tuple[float, torch.Tensor]:
    r"""
    Compute the highest posterior density (HPD) region for an estimated posterior distribution. Currently compatible with the posterior estimators commonly seen in the `sbi` and `bayesflow` software libraries.

    Parameters
    ----------
    posterior : Union[AbstractNeuralPosterior, AbstractKDE, Distribution]
        The estimated posterior distribution from which to compute the HPD region. These types of objects are typically returned by the `sbi` and `bayesflow` software libraries:
        - `AbstractNeuralPosterior` from `sbi` methods involving underlying neural networks, e.g. `SNPE`, `FMPE`.
        - `AbstractKDE` from `sbi` methods involving kernel density estimation, e.g. `SBCABC`.
        - `Distribution` from `torch.distributions` or other libraries.
    param_grid : torch.Tensor
        Grid of parameter values over which to evaluate the posterior.
    x : torch.Tensor
        Observed data or summary statistics.
    credible_level : float
        The desired credible level for the HPD region (e.g., 0.95 for a 95% credible region).
    num_level_sets : int, optional
        Number of level sets to consider when descending the posterior vis a vis a binary search, by default 100_000.
    tol : float, optional
        Tolerance for the credible level, by default 0.01.
    **posterior_kwargs: Any
        Any keyword argument needed when calling the `log_prob` method of the `posterior`.

    Returns
    -------
    Tuple[float, torch.Tensor]
        The achieved credible level and the parameter values within the HPD region.

    Raises
    ------
    ValueError
        If the posterior type is not recognized.
    """
    assert 0 < credible_level < 1, "Credible level must be in (0, 1)."
    x = x if (len(x.shape) > 1) else x.unsqueeze(0)

    if isinstance(posterior, AbstractNeuralPosterior):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ... when using NSF
            posterior_probs = torch.exp(posterior.log_prob(
                theta=param_grid, x=x, **posterior_kwargs
            ).double()).double()
    elif isinstance(posterior, (AbstractKDE, Distribution)):
        posterior_probs = torch.exp(posterior.log_prob(param_grid).double()).double()
    else:
        raise ValueError
    posterior_probs /= torch.sum(posterior_probs)  # normalize

    # descend the level sets of the posterior and stop when the area above a level equals credible level (up to tolerance)
    level_sets = torch.linspace(0, torch.max(posterior_probs).item(), num_level_sets)  # thresholds to include or not parameters
    credible_levels_trajectory = []
    idx = 0
    current_credible_level, current_level_set_idx = 0, idx

    # Binary search to find the level set that gives the credible level
    left = 0
    right = num_level_sets - 1
    while left <= right:
        mid = (left + right) // 2
        new_credible_level = torch.sum(posterior_probs[posterior_probs >= level_sets[mid]])
        credible_levels_trajectory.append(new_credible_level)
        if abs(new_credible_level - credible_level) < abs(current_credible_level - credible_level):
            current_credible_level = new_credible_level
            current_level_set_idx = mid
        if new_credible_level < credible_level:
            right = mid - 1
        else:
            left = mid + 1

    # all params such that p(params|x) > level_set, where level_set is the last chosen one
    accepted = (posterior_probs >= level_sets[current_level_set_idx]).flatten()

    return float(current_credible_level), param_grid[accepted, :]


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


def _deprecated_monte_carlo_shim(name):
    def shim(*args, **kwargs):
        warnings.warn(
            f"lf2i.utils.other_methods.{name} is deprecated; "
            f"import from lf2i.diagnostics.monte_carlo_methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_monte_carlo_methods, name)(*args, **kwargs)
    shim.__name__ = name
    return shim


monte_carlo_confidence_region = _deprecated_monte_carlo_shim('monte_carlo_confidence_region')
monte_carlo_critical_values = _deprecated_monte_carlo_shim('monte_carlo_critical_values')
monte_carlo_coverage = _deprecated_monte_carlo_shim('monte_carlo_coverage')
monte_carlo_coverage_posterior = _deprecated_monte_carlo_shim('monte_carlo_coverage_posterior')
monte_carlo_pvalue_diagnostics = _deprecated_monte_carlo_shim('monte_carlo_pvalue_diagnostics')
