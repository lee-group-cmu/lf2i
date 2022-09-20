from typing import Tuple

import numpy as np


def preprocess_neyman_inversion(
    test_statistic: np.ndarray,
    critical_values: np.ndarray,
    parameter_grid: np.ndarray,
    param_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameter_grid = parameter_grid.reshape(-1, param_dim)
    return test_statistic.reshape(-1, parameter_grid.shape[0]), critical_values.reshape(1, parameter_grid.shape[0]), parameter_grid