from typing import Dict, Union, Tuple, Any, Optional, Sequence
import warnings

import numpy as np
from scipy.stats import norm
import torch
from torch.distributions import Distribution

from lf2i.estimators.base_posteriors import (
    AbstractNeuralPosterior,
    AbstractKDE
)
from lf2i.simulator import Simulator
from lf2i.test_statistics import TestStatistic
from lf2i.utils.miscellanea import to_torch_if_np, to_np_if_torch


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
    posterior : Union[AbstractNeuralPosterior, AbstractKDE, Distribution, AmortizedPosterior]
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


def monte_carlo_confidence_region(
    test_statistic: TestStatistic,
    simulator: Simulator,
    test_param: torch.Tensor,
    param_grid: torch.Tensor, 
    x: torch.Tensor, 
    confidence_level: float, 
    monte_carlo_size: int = 2_000,
    critical_values: torch.Tensor=None,
):
    # evaluate posterior over grid of values
    if isinstance(test_statistic, TestStatistic):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ... when using NSF
            ts_values = test_statistic.evaluate(parameters=param_grid, samples=x, mode='confidence_sets').reshape(-1)
    else:
        raise ValueError

    if critical_values is None:
        critical_values = monte_carlo_critical_values(test_statistic, simulator, param_grid, confidence_level, monte_carlo_size)
    critical_values = np.array(critical_values)

    if test_statistic.acceptance_region == 'left':
        return param_grid[ts_values < critical_values, :]
    else:
        return param_grid[ts_values > critical_values, :]


def monte_carlo_critical_values(
    test_statistic: TestStatistic,
    simulator: Simulator,
    param_grid: torch.Tensor,
    confidence_level: Union[float, list],
    monte_carlo_size: int
):
    parameters_mc = param_grid.repeat_interleave(monte_carlo_size, dim=0)
    samples_mc = simulator(parameters_mc)
    ts_values_mc = test_statistic.evaluate(
        parameters=parameters_mc,
        samples=samples_mc,
        mode='critical_values'
    ).reshape(-1, monte_carlo_size)

    # return a 1D array for a single confidence level, or a dict mapping each level to its 1D array of critical values
    if np.ndim(confidence_level) == 0:
        confidence_level = confidence_level if test_statistic.acceptance_region == 'left' else 1-confidence_level
        mc_critical_values = np.quantile(ts_values_mc, confidence_level, axis=1)
        return mc_critical_values
    else:
        q = np.asarray(confidence_level)
        q = q if test_statistic.acceptance_region == 'left' else 1-q
        mc_q = np.quantile(ts_values_mc, q, axis=1)  # shape (len(q), n_params)
        return mc_q # {float(level): mc_q[i, :] for i, level in enumerate(confidence_level)}


def monte_carlo_coverage(
    test_statistic: TestStatistic,
    calibration_model,
    simulator: Simulator,
    evaluation_grid: np.ndarray,
    confidence_level: Union[float, Sequence[float]],
    calibration_method: str,
    monte_carlo_size: int = 500,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[float, np.ndarray]]]:
    """MC-exact coverage at each point of ``evaluation_grid``.

    Parameters
    ----------
    confidence_level : Union[float, Sequence[float]]
        One or more nominal confidence levels in (0, 1). Simulation and test-statistic
        evaluation are performed only once regardless of how many levels are requested.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(evaluation_grid, coverage_per_grid_point)`` if ``confidence_level`` is a
        scalar float, where ``coverage_per_grid_point`` has shape ``(n_grid,)``.
    Tuple[np.ndarray, Dict[float, np.ndarray]]
        ``(evaluation_grid, {cl: coverage_per_grid_point, ...})`` if
        ``confidence_level`` is a sequence, one entry per requested level.
    """
    # Deferred to avoid circular import: coverage_probability imports from other_methods
    from lf2i.diagnostics.coverage_probability import compute_indicators_lf2i
    from lf2i.utils.calibration_diagnostics_inputs import (
        preprocess_predict_p_values,
        preprocess_predict_quantile_regression,
    )

    assert calibration_method in ['critical-values', 'p-values'], \
        "calibration_method must be 'critical-values' or 'p-values'"

    scalar_input = isinstance(confidence_level, float)
    cls: list = [confidence_level] if scalar_input else list(confidence_level)

    evaluation_grid = np.asarray(evaluation_grid)
    n_grid = evaluation_grid.shape[0]
    param_dim = evaluation_grid.shape[1] if evaluation_grid.ndim > 1 else 1

    # Simulate once — this is the expensive step, shared across all levels.
    parameters_mc = np.repeat(evaluation_grid.reshape(n_grid, param_dim), monte_carlo_size, axis=0)
    parameters_mc_torch = to_torch_if_np(parameters_mc)
    samples_mc = simulator(parameters_mc_torch)
    ts_values = to_np_if_torch(
        test_statistic.evaluate(parameters=parameters_mc_torch, samples=samples_mc, mode='diagnostics')
    ).reshape(-1)

    calib_key = (
        'multiple_levels' if 'multiple_levels' in calibration_model
        else f'{cls[0]:.2f}'
    )

    # Pre-compute quantities that are shared across levels.
    if calibration_method == 'critical-values':
        all_critical_values = to_np_if_torch(calibration_model[calib_key].predict(
            preprocess_predict_quantile_regression(parameters_mc, calibration_model[calib_key], param_dim)
        ))
        p_values = None
    else:
        all_critical_values = None
        p_values = to_np_if_torch(calibration_model[calib_key].predict_proba(
            X=preprocess_predict_p_values('diagnostics', ts_values, parameters_mc, calibration_model[calib_key])
        )[:, 1])

    # Compute coverage for each confidence level cheaply (no re-simulation).
    results: Dict[float, np.ndarray] = {}
    for cl in cls:
        if calibration_method == 'critical-values':
            if calib_key == 'multiple_levels':
                idx_cl = np.argmin(np.abs(
                    cl - (1 - np.array(
                        calibration_model['multiple_levels'].estimator.get_params()['loss_function']
                        .split('=')[1].split(',')
                    ).astype(float))
                ))
                critical_values_cl = all_critical_values[:, idx_cl]
            else:
                # One model per level: look up the model for this specific level.
                key_cl = f'{cl:.2f}'
                if key_cl not in calibration_model:
                    raise KeyError(
                        f"No calibration model found for confidence_level={cl} "
                        f"(tried key '{key_cl}'). Available keys: {list(calibration_model.keys())}"
                    )
                critical_values_cl = to_np_if_torch(calibration_model[key_cl].predict(
                    preprocess_predict_quantile_regression(parameters_mc, calibration_model[key_cl], param_dim)
                ))
            alpha_cl = None
        else:
            critical_values_cl = None
            alpha_cl = 1 - cl

        indicators = compute_indicators_lf2i(
            calibration_method=calibration_method,
            test_statistics=ts_values,
            parameters=parameters_mc,
            critical_values=critical_values_cl,
            p_values=p_values,
            alpha=alpha_cl,
            acceptance_region=test_statistic.acceptance_region,
            param_dim=param_dim,
        )
        results[cl] = indicators.reshape(n_grid, monte_carlo_size).mean(axis=1)

    if scalar_input:
        return evaluation_grid, results[cls[0]]
    return evaluation_grid, results


def monte_carlo_pvalue_diagnostics(
    test_statistic: TestStatistic,
    calibration_model,
    simulator: Simulator,
    evaluation_grid: np.ndarray,
    monte_carlo_size: int = 500,
    pinball_levels: Sequence[float] = None,
):
    """
    For each theta in evaluation_grid, draw MC samples, evaluate the test statistic,
    and compare the calibration model's predicted CDF to the empirical CDF.

    Returns a dict with per-theta arrays:
      'mse'              – mean((p_hat_(i) - u_i)^2) over sorted samples
      'crps'             – integral (F_hat - F_emp)^2 dt via trapezoid rule at sample points
      'pinball_<alpha>'  – pinball loss L_alpha(u_i, p_hat_i) for each alpha in pinball_levels
    where u_i = (i - 0.5) / M is the midpoint empirical CDF estimate.
    """
    from lf2i.utils.calibration_diagnostics_inputs import (
        preprocess_predict_p_values,
    )

    if pinball_levels is None:
        pinball_levels = np.linspace(0.05, 0.95, 19)

    n_grid = evaluation_grid.shape[0]
    M = monte_carlo_size

    # Tile: repeat each theta M times -> (n_grid*M, param_dim)
    params_mc = evaluation_grid.repeat_interleave(M, dim=0)
    samples_mc = simulator(params_mc)

    # Evaluate test statistic for all (theta, x) pairs at once
    ts_all = to_np_if_torch(
        test_statistic.evaluate(
            parameters=params_mc,
            samples=samples_mc,
            mode='diagnostics',
        )
    ).reshape(-1)  # (n_grid*M,)

    # Resolve calibration model key (p-values: one shared model, possibly 'multiple_levels')
    calib_key = (
        'multiple_levels' if 'multiple_levels' in calibration_model
        else next(iter(calibration_model))
    )
    calib_model = calibration_model[calib_key]

    # Evaluate calibration model for all (theta, T) pairs at once
    params_mc_np = to_np_if_torch(params_mc)
    X_pred = preprocess_predict_p_values('diagnostics', ts_all, params_mc_np, calib_model)
    p_hat_all = calib_model.predict_proba(X=X_pred)[:, 1]  # (n_grid*M,)

    # Reshape to (n_grid, M)
    ts_grid = ts_all.reshape(n_grid, M)
    p_hat_grid = p_hat_all.reshape(n_grid, M)

    mse = np.zeros(n_grid)
    crps = np.zeros(n_grid)
    pinball_per_alpha = {alpha: np.zeros(n_grid) for alpha in pinball_levels}

    for j in range(n_grid):
        sort_idx = np.argsort(ts_grid[j])
        p_hat_sorted = p_hat_grid[j][sort_idx]
        T_sorted = ts_grid[j][sort_idx]

        # Empirical CDF: midpoint estimate (i - 0.5) / M for i = 1, ..., M
        u = (np.arange(1, M + 1) - 0.5) / M
        residuals = p_hat_sorted - u  # predicted minus empirical

        mse[j] = np.mean(residuals ** 2)

        # CRPS: trapezoid integral of (F_hat(t) - F_emp(t))^2 over the MC sample range
        dT = np.diff(T_sorted)  # (M-1,)
        seg_err_sq = 0.5 * (residuals[:-1] ** 2 + residuals[1:] ** 2)
        crps[j] = np.sum(seg_err_sq * dT)

        # Pinball loss per alpha: L_alpha(u_i, p_hat_i) = (1-alpha)*residual if residual>=0
        #                                                = -alpha*residual     if residual<0
        for alpha in pinball_levels:
            pinball_per_alpha[alpha][j] = np.mean(
                np.where(residuals >= 0, (1 - alpha) * residuals, -alpha * residuals)
            )

    estimation_errors = {'mse': mse, 'crps': crps}
    for alpha, pb in pinball_per_alpha.items():
        estimation_errors[f'pinball_{alpha:.2f}'] = pb
    return evaluation_grid, estimation_errors


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
