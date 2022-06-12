from tkinter import N
from typing import Union, Callable
from functools import partial

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

from nn_qr_algorithm import QuantileLoss, QuantileNN, Learner


def train_qr_algorithm(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    algorithm: Union[str, Callable],
    alpha: float,
    prediction_grid: Union[np.ndarray, torch.Tensor],
    algorithm_kwargs: dict = {},
) -> tuple:
    """Dispatcher to train different quantile regressors.

    Parameters
    ----------
    test_statistics : Union[np.ndarray, torch.Tensor]
        Test statistics used to train the quantile regressor (i.e., outputs). Must be a Tensor if algorithm == 'nn'.
    parameters : Union[np.ndarray, torch.Tensor]
        Parameters used to train the quantile regressor (i.e., inputs). Must be a Tensor if algorithm == 'nn'.
    algorithm : Union[str, Callable]
        One of 'gb' for Gradient Boosted Trees, 'nn' for Quantile Neural Network, or a custom algorithm (Callable).
    alpha : float
        The alpha quantile of the test statistic will be estimated. 
        E.g., for 90% confidence intervals, it should be 0.1. Must be in the range `[0, 1]`.
    prediction_grid : Union[np.ndarray, torch.Tensor]
        Parameters for which to predict quantiles of the test statistic.
    algorithm_kwargs : dict, optional
        Keywork arguments for the desired algorithm, by default {}. 
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.

    Returns
    -------
    tuple
        Fitted Quantile Regressor and predicted quantiles for prediction_grid.

    Raises
    ------
    ValueError
        Only one of 'gb', 'nn' or custom algorithm (Callable) is accepted as algorithm.
    """
    if algorithm == "gb":
        algorithm = GradientBoostingRegressor(loss='quantile', alpha=alpha, **algorithm_kwargs)
        algorithm.fit(X=parameters, y=test_statistics)
        predicted_quantiles = algorithm.predict(X=prediction_grid)
    elif algorithm == 'nn':
        nn_kwargs = {arg: algorithm_kwargs[arg] for arg in ['hidden_activation', 'dropout_p'] if arg in algorithm_kwargs}
        quantile_nn = QuantileNN(quantiles=[alpha], input_d=parameters.size(dim=1), 
                                 hidden_layer_shapes=algorithm_kwargs['hidden_layer_shapes'], **nn_kwargs)
        algorithm = Learner(model=quantile_nn, optimizer=partial(torch.optim.Adam, weight_decay=1e-6), loss=QuantileLoss, 
                            device="cuda" if torch.cuda.is_available() else 'cpu')
        algorithm_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
        algorithm.fit(X=parameters, y=test_statistics, **algorithm_kwargs)
        predicted_quantiles = algorithm.predict(X=prediction_grid)
    elif isinstance(algorithm, Callable):
        algorithm.fit(X=parameters, y=test_statistics)
        predicted_quantiles = algorithm.predict(X=prediction_grid)
    else:
        raise ValueError(f"Only 'gb', 'nn' or custom algorithm (Callable) are currently supported, got {algorithm}")

    return algorithm, predicted_quantiles
