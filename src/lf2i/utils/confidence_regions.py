from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from lf2i.utils.miscellanea import check_for_nans, to_np_if_torch


class NonrectangularEvaluationGrid:
    """Wraps an evaluation grid and provides in-support membership queries via KDTree.

    Used to mask points that lie outside the support of the calibration data when
    sweeping 1D confidence curves or OAT intervals.
    """

    def __init__(self, grid: np.ndarray, support_quantile: float = 0.95, seed: int = 0):
        grid = np.asarray(grid, dtype=float)
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
        self.grid = grid
        self._lo = grid.min(axis=0)
        self._rng = grid.max(axis=0) - self._lo
        self._rng[self._rng == 0] = 1.0
        self._scaled = (grid - self._lo) / self._rng
        self._tree = cKDTree(self._scaled)

        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(grid), min(500, len(grid)), replace=False)
        nn_dists, _ = self._tree.query(self._scaled[sample_idx], k=min(2, len(grid)))
        nn_dists = nn_dists[:, 1] if nn_dists.ndim == 2 else nn_dists
        self._threshold = float(np.quantile(nn_dists, support_quantile))

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return boolean array of length n: True if point is within grid support."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        points_scaled = (points - self._lo) / self._rng
        dists, _ = self._tree.query(points_scaled, k=1)
        return dists <= self._threshold

    def neighbors(self, point: np.ndarray, k: int = 50) -> np.ndarray:
        """Return the k nearest grid points to `point`."""
        point = np.asarray(point, dtype=float).ravel()
        point_scaled = (point - self._lo) / self._rng
        _, idx = self._tree.query(point_scaled, k=min(k, len(self.grid)))
        return self.grid[idx]

    def neighbors_by_dims(self, point: np.ndarray, dims: List[int], k: int = 50) -> np.ndarray:
        """Return k nearest grid points using only the specified dimensions for distance.

        Useful when sweeping a subset of parameter dimensions while holding others
        fixed: finds grid points whose ``dims`` coordinates are close to ``point``,
        allowing the remaining dimensions to vary freely.

        The KDTree for each unique ``dims`` subset is built lazily and cached.
        """
        dims_key = tuple(dims)
        if not hasattr(self, '_subspace_trees'):
            self._subspace_trees: Dict[tuple, cKDTree] = {}
        if dims_key not in self._subspace_trees:
            self._subspace_trees[dims_key] = cKDTree(self._scaled[:, list(dims)])
        point = np.asarray(point, dtype=float).ravel()
        point_scaled = ((point - self._lo) / self._rng)[list(dims)]
        _, idx = self._subspace_trees[dims_key].query(point_scaled, k=min(k, len(self.grid)))
        return self.grid[idx]


def preprocess_neyman_inversion(
    test_statistics: Optional[Union[np.ndarray, "torch.Tensor"]],
    critical_values: Optional[Union[np.ndarray, "torch.Tensor"]],
    p_values: Optional[Union[np.ndarray, "torch.Tensor"]],
    parameter_grid: Union[np.ndarray, "torch.Tensor"],
    param_dim: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    check_for_nans(parameter_grid)
    parameter_grid = parameter_grid.reshape(-1, param_dim)
    parameter_grid = to_np_if_torch(parameter_grid)

    num_obs = None
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
        parameter_grid,
    )


def preprocess_confidence_curves(
    point_estimates: np.ndarray,
    evaluation_grid: Optional[np.ndarray],
    parameters_calib: np.ndarray,
    grid_bounds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[NonrectangularEvaluationGrid]]:
    """Preprocess inputs shared by confidence curve and interval computation.

    Returns (point_estimates, eg_np, grid_bounds, nonrect_eval_grid).
    ``nonrect_eval_grid`` is None when ``evaluation_grid`` is None.
    """
    point_estimates = to_np_if_torch(point_estimates)
    point_estimates = np.asarray(point_estimates, dtype=float)
    if point_estimates.ndim == 1:
        point_estimates = point_estimates.reshape(1, -1)

    eg_np: Optional[np.ndarray] = None
    nonrect_grid: Optional[NonrectangularEvaluationGrid] = None
    if evaluation_grid is not None:
        eg_np = to_np_if_torch(evaluation_grid)
        eg_np = np.asarray(eg_np, dtype=float)
        if eg_np.ndim == 1:
            eg_np = eg_np.reshape(-1, 1)
        nonrect_grid = NonrectangularEvaluationGrid(eg_np)

    if grid_bounds is None:
        if eg_np is not None:
            grid_bounds = np.stack([eg_np.min(axis=0), eg_np.max(axis=0)], axis=1)
        else:
            params_np = to_np_if_torch(parameters_calib)
            params_np = np.asarray(params_np, dtype=float)
            if params_np.ndim == 1:
                params_np = params_np.reshape(-1, 1)
            grid_bounds = np.stack([params_np.min(axis=0), params_np.max(axis=0)], axis=1)

    return point_estimates, eg_np, grid_bounds, nonrect_grid
