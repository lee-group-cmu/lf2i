from typing import List

import numpy as np
from lf2i.utils.calibration_diagnostics_inputs import preprocess_neyman_inversion


def compute_confidence_regions(
    test_statistic: np.ndarray,
    critical_values: np.ndarray,
    parameter_grid: np.ndarray,
    acceptance_region: str,
    poi_dim: int
) -> List[np.ndarray]:
    """Compute LF2I confidence regions via Neyman inversion of hypothesis tests.

    Parameters
    ----------
    test_statistic : np.ndarray
        Test statistic evaluated at all values in the parameter grid *for each* observation. Should have dimensions `(num_observations, parameter_grid_size)`.
    critical_values : np.ndarray
        Critical values evaluated at all values in the parameter grid. 
    parameter_grid : np.ndarray
        Grid over the parameter space which contains the evaluation points that will or will not be included in the confidence region.
    acceptance_region : str
        Whether the acceptance region for the test statistic is defined to be on the right or on the left of the critical value. 
        Must be either `left` or `right`. 
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
    test_statistic, critical_values, parameter_grid = \
        preprocess_neyman_inversion(test_statistic, critical_values, parameter_grid, poi_dim)

    if acceptance_region == 'left':
        which_parameters = test_statistic <= critical_values
    elif acceptance_region == 'right':
        which_parameters = test_statistic >= critical_values
    else:
        raise ValueError(f"Acceptance region must be either `left` or `right`, got {acceptance_region}")
    return [parameter_grid[which_parameters[idx, :].reshape(-1, ), :] for idx in range(test_statistic.shape[0])]
