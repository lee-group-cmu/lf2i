from typing import List, Optional, Tuple, Union

import numpy as np

from lf2i.utils.confidence_regions import (
    preprocess_neyman_inversion,
    preprocess_confidence_curves,
    NonrectangularEvaluationGrid,
)
from lf2i.utils.calibration_diagnostics_inputs import preprocess_predict_p_values
from lf2i.utils.miscellanea import to_np_if_torch


def compute_confidence_regions(
    calibration_method: str,
    test_statistic: Optional[np.ndarray],
    parameter_grid: np.ndarray,
    critical_values: Optional[np.ndarray],
    p_values: Optional[np.ndarray],
    alpha: Optional[float],
    acceptance_region: Optional[str],
    poi_dim: int,
    return_indices: bool = False,
) -> List[np.ndarray]:
    """Compute LF2I confidence regions via Neyman inversion of hypothesis tests.

    Parameters
    ----------
    calibration_method : str
        Either `critical-values` or `p-values`.
    test_statistic : np.ndarray
        Test statistic evaluated at all values in the parameter grid *for each* observation. Should have dimensions `(num_observations, parameter_grid_size)`.
        Only used if `calibration_method = 'critical-values`.
    parameter_grid : np.ndarray
        Grid over the parameter space which contains the evaluation points that will or will not be included in the confidence region.
    critical_values : np.ndarray
        Critical values evaluated at all values in the parameter grid. Only used if `calibration_method = 'critical-values`.
    p_values : np.ndarray, optional
        Array of p-values evaluated at all values in the parameter grid *for each* observation, against which to compare the provided level :math:`\alpha`.
        Only used if `calibration_method = 'p-values`. Should have dimensions `(num_observations, parameter_grid_size)`.
    acceptance_region : str
        Whether the acceptance region for the test statistic is defined to be on the right or on the left of the critical value.
        Must be either `left` or `right`. Only used if `calibration_method = 'critical-values`.
    alpha: float, optional
        If `calibration_method = 'p-values`, used to decide whether the test rejects or not, otherwise ignored.
    poi_dim : int
        Dimensionality (number) of the parameter of interest.

    Returns
    -------
    List[np.ndarray]
        Sequence whose i-th element is the confidence region for the i-th observation used to evaluate the test statistic.

    Raises
    ------
    ValueError
        Acceptance region must be either `left` or `right`.
    """
    num_obs, test_statistic, critical_values, p_values, parameter_grid = \
        preprocess_neyman_inversion(test_statistic, critical_values, p_values, parameter_grid, poi_dim)

    if calibration_method == 'critical-values':
        if acceptance_region == 'left':
            which_parameters = test_statistic <= critical_values
        elif acceptance_region == 'right':
            which_parameters = test_statistic >= critical_values
        else:
            raise ValueError(f"Acceptance region must be either `left` or `right`, got {acceptance_region}")
    else:
        which_parameters = p_values >= alpha

    if return_indices:
        return [which_parameters[idx, :].nonzero()[0] for idx in range(num_obs)]
    else:
        return [parameter_grid[which_parameters[idx, :].reshape(-1, ), :] for idx in range(num_obs)]


def compute_point_estimates(
    p_values: np.ndarray,
    evaluation_grid: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """Compute Focal point estimates as argmax of p-values over the evaluation grid.

    Parameters
    ----------
    p_values : np.ndarray, shape (n_obs * grid_size,) or (n_obs, grid_size)
        P-values evaluated at each (observation, grid point) pair.
    evaluation_grid : np.ndarray, shape (grid_size, param_dim)
        Parameter grid used during Neyman inversion.
    n_obs : int
        Number of observations.

    Returns
    -------
    np.ndarray, shape (n_obs, param_dim)
        Point estimate θ^Focal = argmax_θ p̂(θ | x) for each observation.
    """
    evaluation_grid_np = to_np_if_torch(evaluation_grid)
    grid_size = len(evaluation_grid_np)
    p_values_matrix = np.asarray(p_values).reshape(n_obs, grid_size)
    pe_idx = np.argmax(p_values_matrix, axis=1)
    return evaluation_grid_np[pe_idx]


def compute_confidence_curves(
    test_statistic_obj,
    calibration_model: dict,
    calib_dict_key: str,
    x: np.ndarray,
    point_estimates: np.ndarray,
    alpha: float,
    acceptance_region: str,
    param_dim: int,
    parameters_calib: Optional[np.ndarray] = None,
    grid_size: int = 200,
    grid_bounds: Optional[np.ndarray] = None,
    evaluation_grid: Optional[np.ndarray] = None,
    slice_dims: Optional[List[int]] = None,
    nonrect_eval_grid: Optional[NonrectangularEvaluationGrid] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], np.ndarray],
]:
    """Compute 1D confidence curves (p-value sweeps) via OAT parameter variation.

    Steps:
        1. Preprocess point_estimates and evaluation_grid.
        2. Preprocess the 1D or slice sweep grid.
        3. Compute curves and return.

    Parameters
    ----------
    test_statistic_obj :
        Trained TestStatistic object with an ``evaluate`` method.
    calibration_model : dict
        Mapping from calib_dict_key to a fitted calibration model.
    calib_dict_key : str
        Key into ``calibration_model`` for the model to use.
    x : np.ndarray, shape (n_obs, data_dim)
        Observed samples.
    point_estimates : np.ndarray, shape (n_obs, param_dim)
        Focal point estimates θ^Focal; other dims are held fixed during the sweep.
    alpha : float
        Rejection threshold; p-value < alpha → rejected.
    acceptance_region : str
        ``'left'`` or ``'right'``.
    param_dim : int
        Number of parameter dimensions.
    grid_size : int, optional
        Points along each sweep axis. Default 200.
    grid_bounds : np.ndarray, shape (param_dim, 2), optional
        Per-dimension [lo, hi]. Derived from ``evaluation_grid`` or calibration data if None.
    evaluation_grid : np.ndarray, optional
        Optional dense evaluation grid used to derive local bounds and identify in-support
        points. When provided it is wrapped in a ``NonrectangularEvaluationGrid`` unless
        ``nonrect_eval_grid`` is already supplied.
    slice_dims : list of int, optional
        Dimensions to vary jointly.  None → classic OAT (one dimension at a time).
    nonrect_eval_grid : NonrectangularEvaluationGrid, optional
        Pre-built support checker.  Constructed from ``evaluation_grid`` when None.

    Returns
    -------
    When ``slice_dims`` is None (default OAT):
        intervals : np.ndarray, shape (n_obs, param_dim, 2)
            Lower/upper endpoints per observation and dimension.  NaN when no accepted point.
        pvalues : np.ndarray, shape (n_obs, param_dim, grid_size)
            P-values along each 1D sweep.
        grid : np.ndarray, shape (param_dim, grid_size)
            Sweep grid coordinates (from obs 0).
    When ``slice_dims`` is provided:
        accepted_list : List[np.ndarray]
            Accepted parameter vectors per observation.
        pvalues_list : List[np.ndarray]
            P-values over the slice grid per observation.
        slice_grid : np.ndarray, shape (n_grid_points, len(slice_dims))
            Slice-dim coordinates of each evaluated grid point.
    """
    # --- Step 1: preprocess point_estimates and evaluation_grid ---
    user_grid_bounds_given = grid_bounds is not None
    point_estimates, eg_np, grid_bounds, built_nonrect = preprocess_confidence_curves(
        point_estimates=point_estimates,
        evaluation_grid=evaluation_grid,
        parameters_calib=parameters_calib,
        grid_bounds=grid_bounds,
    )
    if nonrect_eval_grid is None:
        nonrect_eval_grid = built_nonrect

    x_np = to_np_if_torch(x)
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    n_obs = point_estimates.shape[0]

    _pv_col = 0 if acceptance_region == 'left' else 1

    # --- Step 2 & 3: sweep and compute ---

    if slice_dims is not None:
        return _compute_curves_slice(
            test_statistic_obj=test_statistic_obj,
            calibration_model=calibration_model,
            calib_dict_key=calib_dict_key,
            x_np=x_np,
            point_estimates=point_estimates,
            alpha=alpha,
            pv_col=_pv_col,
            param_dim=param_dim,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
            eg_np=eg_np,
            slice_dims=list(slice_dims),
            nonrect_eval_grid=nonrect_eval_grid,
        )
    else:
        return _compute_curves_oat(
            test_statistic_obj=test_statistic_obj,
            calibration_model=calibration_model,
            calib_dict_key=calib_dict_key,
            x_np=x_np,
            point_estimates=point_estimates,
            alpha=alpha,
            pv_col=_pv_col,
            param_dim=param_dim,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
            eg_np=eg_np,
            nonrect_eval_grid=nonrect_eval_grid,
            user_grid_bounds_given=user_grid_bounds_given,
        )


def compute_confidence_intervals(
    test_statistic_obj,
    calibration_model: dict,
    calib_dict_key: str,
    x: np.ndarray,
    point_estimates: np.ndarray,
    alpha: float,
    acceptance_region: str,
    param_dim: int,
    parameters_calib: Optional[np.ndarray] = None,
    grid_size: int = 200,
    grid_bounds: Optional[np.ndarray] = None,
    evaluation_grid: Optional[np.ndarray] = None,
    slice_dims: Optional[List[int]] = None,
    nonrect_eval_grid: Optional[NonrectangularEvaluationGrid] = None,
) -> np.ndarray:
    """Compute 1D OAT confidence intervals by returning only the endpoints of
    :func:`compute_confidence_curves`.

    Returns
    -------
    np.ndarray, shape (n_obs, param_dim, 2)
        Lower/upper endpoints per observation and dimension.  NaN when no grid
        point was accepted.  When ``slice_dims`` is provided, returns
        ``List[np.ndarray]`` of accepted parameter vectors per observation.
    """
    result = compute_confidence_curves(
        test_statistic_obj=test_statistic_obj,
        calibration_model=calibration_model,
        calib_dict_key=calib_dict_key,
        x=x,
        point_estimates=point_estimates,
        alpha=alpha,
        acceptance_region=acceptance_region,
        param_dim=param_dim,
        parameters_calib=parameters_calib,
        grid_size=grid_size,
        grid_bounds=grid_bounds,
        evaluation_grid=evaluation_grid,
        slice_dims=slice_dims,
        nonrect_eval_grid=nonrect_eval_grid,
    )
    intervals = result[0]
    return intervals


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _eval_pvalues(test_statistic_obj, calibration_model, calib_dict_key, grid_nd, xi, pv_col):
    """Evaluate p-values for one observation xi over grid_nd."""
    ts = test_statistic_obj.evaluate(grid_nd, xi.astype(np.float32), mode='confidence_sets')
    p_vals = calibration_model[calib_dict_key].predict_proba(
        X=preprocess_predict_p_values('confidence_sets', ts, grid_nd, calibration_model[calib_dict_key])
    )[:, pv_col]
    return p_vals


def _compute_curves_oat(
    test_statistic_obj, calibration_model, calib_dict_key, x_np, point_estimates,
    alpha, pv_col, param_dim, grid_size, grid_bounds, eg_np, nonrect_eval_grid,
    user_grid_bounds_given=False,
):
    """Classic OAT sweep: vary one dimension at a time."""
    n_obs = point_estimates.shape[0]
    intervals = np.full((n_obs, param_dim, 2), np.nan)
    all_pvalues = np.full((n_obs, param_dim, grid_size), np.nan)
    grid_values = np.empty((param_dim, grid_size), dtype=np.float32)

    _knn_k = max(50, int(0.05 * len(eg_np))) if eg_np is not None else None

    for i in range(n_obs):
        pe = point_estimates[i]
        xi = x_np[i:i + 1]

        neighbors = None
        if eg_np is not None and not user_grid_bounds_given:
            neighbors = nonrect_eval_grid.neighbors(pe, k=_knn_k)

        for d in range(param_dim):
            if neighbors is not None:
                lo_d = float(neighbors[:, d].min())
                hi_d = float(neighbors[:, d].max())
            else:
                lo_d, hi_d = float(grid_bounds[d, 0]), float(grid_bounds[d, 1])

            sweep_d = np.linspace(lo_d, hi_d, grid_size).astype(np.float32)
            grid_1d = np.tile(pe, (grid_size, 1)).astype(np.float32)
            grid_1d[:, d] = sweep_d

            # Mask points outside eval-grid support
            if nonrect_eval_grid is not None:
                in_support = nonrect_eval_grid.contains(grid_1d)
            else:
                in_support = np.ones(grid_size, dtype=bool)

            p_vals = np.zeros(grid_size)
            if in_support.any():
                p_vals[in_support] = _eval_pvalues(
                    test_statistic_obj, calibration_model, calib_dict_key,
                    grid_1d[in_support], xi, pv_col,
                )

            accepted = grid_1d[p_vals >= alpha, d]
            if len(accepted) > 0:
                intervals[i, d, 0] = accepted.min()
                intervals[i, d, 1] = accepted.max()

            all_pvalues[i, d] = p_vals
            if i == 0:
                grid_values[d] = sweep_d

    return intervals, all_pvalues, grid_values


def _compute_curves_slice(
    test_statistic_obj, calibration_model, calib_dict_key, x_np, point_estimates,
    alpha, pv_col, param_dim, grid_size, grid_bounds, eg_np, slice_dims, nonrect_eval_grid,
):
    """Slice-dims path: evaluate a product grid over selected dimensions."""
    n_obs = point_estimates.shape[0]
    non_slice_dims = [k for k in range(param_dim) if k not in slice_dims]
    _knn_k = max(50, int(0.05 * len(eg_np))) if eg_np is not None else None

    if eg_np is None:
        axes_1d = [np.linspace(grid_bounds[d, 0], grid_bounds[d, 1], grid_size) for d in slice_dims]
        if len(slice_dims) == 1:
            coords = axes_1d[0].reshape(-1, 1)
        else:
            mesh = np.meshgrid(*axes_1d, indexing='ij')
            coords = np.stack([m.ravel() for m in mesh], axis=1)
        slice_grid_fallback = coords.astype(np.float32)
    else:
        slice_grid_fallback = None

    accepted_list: List[np.ndarray] = []
    all_pvalues_list: List[np.ndarray] = []
    slice_grid_out: Optional[np.ndarray] = None

    for i in range(n_obs):
        pe = point_estimates[i]
        xi = x_np[i:i + 1]

        if eg_np is not None:
            if non_slice_dims:
                # Find grid points whose non-slice dims are close to the focal point,
                # letting the slice dims vary freely.
                grid_nd = nonrect_eval_grid.neighbors_by_dims(
                    pe, non_slice_dims, k=_knn_k
                ).astype(np.float32)
            else:
                grid_nd = eg_np.astype(np.float32)
            slice_grid_i = grid_nd[:, slice_dims]
        else:
            n_pts = len(slice_grid_fallback)
            grid_nd = np.tile(pe, (n_pts, 1)).astype(np.float32)
            for idx_s, d in enumerate(slice_dims):
                grid_nd[:, d] = slice_grid_fallback[:, idx_s]
            slice_grid_i = slice_grid_fallback

        if i == 0:
            slice_grid_out = slice_grid_i

        # Mask points outside eval-grid support
        if nonrect_eval_grid is not None:
            in_support = nonrect_eval_grid.contains(grid_nd)
        else:
            in_support = np.ones(len(grid_nd), dtype=bool)

        p_vals = np.zeros(len(grid_nd))
        if in_support.any():
            p_vals[in_support] = _eval_pvalues(
                test_statistic_obj, calibration_model, calib_dict_key,
                grid_nd[in_support], xi, pv_col,
            )

        accepted_list.append(grid_nd[p_vals >= alpha])
        all_pvalues_list.append(p_vals)

    return accepted_list, all_pvalues_list, slice_grid_out
