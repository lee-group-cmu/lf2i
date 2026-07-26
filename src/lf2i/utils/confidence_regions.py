from typing import Optional, Tuple, Union

import numpy as np

from lf2i.utils.miscellanea import check_for_nans, to_np_if_torch


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


