from typing import List

import numpy as np
from lf2i.utils.lf2i_inputs import preprocess_neyman_inversion


def compute_confidence_regions(
    test_statistic: np.ndarray,
    critical_values: np.ndarray,
    parameter_grid: np.ndarray,
    acceptance_region: str,
    param_dim: int
) -> List[np.ndarray]:
    test_statistic, critical_values, parameter_grid = \
        preprocess_neyman_inversion(test_statistic, critical_values, parameter_grid, param_dim)

    if acceptance_region == 'left':
        which_parameters = test_statistic <= critical_values
    elif acceptance_region == 'right':
        which_parameters = test_statistic >= critical_values
    else:
        raise ValueError(f"Acceptance region must be either `left` or `right`, got {acceptance_region}")
    return [parameter_grid[which_parameters[idx, :].reshape(-1, ), :] for idx in range(test_statistic.shape[0])]
