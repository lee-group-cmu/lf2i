from typing import Union, Callable, Dict, Tuple, Any, List
import os

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor  # TODO: is tree split done according to quantile loss? Default is 'friedman_mse'
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss

from lf2i.critical_values.nn_qr_algorithm import QuantileLoss, QuantileNN, Learner
from lf2i.utils.lf2i_inputs import preprocess_quantile_regression


def train_qr_algorithm(
    test_statistics: np.ndarray,
    parameters: np.ndarray,
    algorithm: Union[str, Callable],
    alpha: Union[float, List[float], Tuple[float]],
    param_dim: int,
    algorithm_kwargs: Union[Dict, str] = {}
) -> Any:
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
    alpha : float  # TODO: update this to include sequences
        The alpha quantile of the test statistic will be estimated. 
        E.g., for 90% confidence intervals, it should be 0.1. Must be in the range `(0, 1)`.
    param_dim: int
        Dimensionality of the parameter.
    algorithm_kwargs : Union[Dict, str], optional
        Keywork arguments for the desired algorithm, by default {}.
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.
        If algorithm == 'gb', pass 'cv' to do a randomized search over `max_depth` and `n_estimators` via 5-fold cross validation.

    Returns
    -------
    Any
        Fitted Quantile Regressor.

    Raises
    ------
    ValueError
        Only one of 'gb', 'nn' or custom algorithm (Any) is currently accepted as algorithm.
    """
    test_statistics, parameters = preprocess_quantile_regression(test_statistics, parameters, param_dim)

    if algorithm == "gb":
        if algorithm_kwargs == 'cv':
            algorithm = RandomizedSearchCV(
                estimator=GradientBoostingRegressor(
                    loss='quantile', 
                    alpha=alpha
                ),
                param_distributions={
                    'max_depth': [1, 3, 5, 7, 10, 15],
                    'n_estimators': [100, 300, 500, 1000]
                },
                n_iter=20,  # default
                scoring=make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False),
                n_jobs=os.cpu_count()-2,
                refit=True,
                cv=5,
                verbose=1
            )
            algorithm.fit(X=parameters, y=test_statistics)
        else:
            algorithm = GradientBoostingRegressor(
                loss='quantile', 
                alpha=alpha, 
                **algorithm_kwargs
            )
            algorithm.fit(X=parameters, y=test_statistics)
    elif algorithm == 'nn':
        quantiles = [alpha] if isinstance(alpha, float) else alpha
        nn_kwargs = {arg: algorithm_kwargs[arg] for arg in ['hidden_activation', 'dropout_p'] if arg in algorithm_kwargs}
        quantile_nn = QuantileNN(
            n_quantiles=len(quantiles), 
            input_d=parameters.shape[1], 
            hidden_layer_shapes=algorithm_kwargs['hidden_layer_shapes'], 
            **nn_kwargs
        )
        algorithm = Learner(
            model=quantile_nn, 
            optimizer=lambda params: torch.optim.Adam(params=params, weight_decay=1e-6), 
            loss=QuantileLoss(quantiles=quantiles), 
            device="cuda" if torch.cuda.is_available() else 'cpu'
        )
        learner_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
        algorithm.fit(X=parameters, y=test_statistics, **learner_kwargs)
    elif isinstance(algorithm, Any):
        algorithm.fit(X=parameters, y=test_statistics)
    else:
        raise ValueError(f"Only 'gb', 'nn' or custom algorithm (Any) are currently supported, got {algorithm}")

    return algorithm
