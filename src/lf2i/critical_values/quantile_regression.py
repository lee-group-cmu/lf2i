from cgi import test
from typing import Union, Callable, Dict, Tuple, Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

from lf2i.critical_values.nn_qr_algorithm import QuantileLoss, QuantileNN, Learner
from lf2i.utils.lf2i_inputs import preprocess_quantile_regression


def train_qr_algorithm(
    test_statistics: np.ndarray,
    parameters: np.ndarray,
    algorithm: Union[str, Callable],
    prediction_grid: np.ndarray,
    alpha: float,
    param_dim: int,
    algorithm_kwargs: Dict = {}
) -> Tuple[Any, np.ndarray]:
    """Dispatcher to train different quantile regressors.

    Parameters
    ----------
    test_statistics : np.ndarray
        Test statistics used to train the quantile regressor (i.e., outputs). Must be a Tensor if algorithm == 'nn'.
    parameters : np.ndarray
        Parameters used to train the quantile regressor (i.e., inputs). Must be a Tensor if algorithm == 'nn'.
    algorithm : Union[str, Any]
        Either 'gb' for Gradient Boosted Trees, 'nn' for Neural Networks, or a custom algorithm (Any).
        The latter must have `fit` and `predict` methods.
    alpha : float
        The alpha quantile of the test statistic will be estimated. 
        E.g., for 90% confidence intervals, it should be 0.1. Must be in the range `(0, 1)`.
    prediction_grid : np.ndarray
        Parameters for which to predict quantiles of the test statistic.
    param_dim: int
        Dimensionality of the parameter.
    algorithm_kwargs : Dict, optional
        Keywork arguments for the desired algorithm, by default {}. 
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.

    Returns
    -------
    Tuple[Any, np.ndarray]
        Fitted Quantile Regressor and predicted quantiles for each value in prediction_grid.

    Raises
    ------
    ValueError
        Only one of 'gb', 'nn' or custom algorithm (Any) is accepted as algorithm.
    """
    test_statistics, parameters, prediction_grid = preprocess_quantile_regression(test_statistics, parameters, prediction_grid, param_dim)

    if algorithm == "gb":
        algorithm = GradientBoostingRegressor(
            loss='quantile', 
            alpha=alpha, 
            **algorithm_kwargs
        )
        algorithm.fit(X=parameters, y=test_statistics)
        predicted_quantiles = algorithm.predict(X=prediction_grid)
    elif algorithm == 'nn':
        nn_kwargs = {arg: algorithm_kwargs[arg] for arg in ['hidden_activation', 'dropout_p'] if arg in algorithm_kwargs}
        quantile_nn = QuantileNN(
            n_quantiles=len(list(alpha)), 
            input_d=parameters.shape[1], 
            hidden_layer_shapes=algorithm_kwargs['hidden_layer_shapes'], 
            **nn_kwargs
        )
        algorithm = Learner(
            model=quantile_nn, 
            optimizer=lambda params: torch.optim.Adam(params=params, weight_decay=1e-6), 
            loss=QuantileLoss(quantiles=list(alpha)), 
            device="cuda" if torch.cuda.is_available() else 'cpu'
        )
        learner_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
        algorithm.fit(X=parameters, y=test_statistics, **learner_kwargs)
        predicted_quantiles = algorithm.predict(X=prediction_grid)
    elif isinstance(algorithm, Any):
        algorithm.fit(X=parameters, y=test_statistics)
        predicted_quantiles = algorithm.predict(X=prediction_grid)
    else:
        raise ValueError(f"Only 'gb', 'nn' or custom algorithm (Any) are currently supported, got {algorithm}")

    return algorithm, predicted_quantiles
