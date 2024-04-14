from typing import List, Optional

import numpy as np
from lf2i.utils.calibration_diagnostics_inputs import preprocess_neyman_inversion


def compute_confidence_regions(
    calibration_method: str,
    test_statistic: np.ndarray,
    parameter_grid: np.ndarray,
    critical_values: Optional[np.ndarray],
    p_values: Optional[np.ndarray],
    alpha: Optional[float],
    acceptance_region: Optional[str],
    poi_dim: int
) -> List[np.ndarray]:
    """Compute LF2I confidence regions via Neyman inversion of hypothesis tests.

    Parameters
    ----------
    calibration_method : str
        Either `critical-values` or `p-values`.
    test_statistic : np.ndarray
        Test statistic evaluated at all values in the parameter grid *for each* observation. Should have dimensions `(num_observations, parameter_grid_size)`.
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
    test_statistic, critical_values, parameter_grid = \
        preprocess_neyman_inversion(test_statistic, critical_values, parameter_grid, poi_dim)

    if calibration_method == 'critical-values':
        if acceptance_region == 'left':
            which_parameters = test_statistic <= critical_values
        elif acceptance_region == 'right':
            which_parameters = test_statistic >= critical_values
        else:
            raise ValueError(f"Acceptance region must be either `left` or `right`, got {acceptance_region}")
    else:
        which_parameters = p_values > alpha
    return [parameter_grid[which_parameters[idx, :].reshape(-1, ), :] for idx in range(test_statistic.shape[0])]
