from typing import Union, Callable, Dict, Any, Sequence
import os

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor  # TODO: is tree split done according to quantile loss? Default is 'friedman_mse'
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss
from catboost import CatBoostRegressor

from lf2i.critical_values.nn_qr_algorithm import QuantileLoss, QuantileNN, Learner
from lf2i.utils.calibration_diagnostics_inputs import preprocess_train_quantile_regression


def train_qr_algorithm(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    algorithm: Union[str, Callable],
    alpha: Union[float, Sequence[float]],
    param_dim: int,
    algorithm_kwargs: Union[Dict[str, Any], Dict[str, Dict[str, Any]]] = {},
    n_jobs: int = -2
) -> Any:
    """Dispatcher to train different quantile regressors.

    Parameters
    ----------
    test_statistics : Union[np.ndarray, torch.Tensor]
        Test statistics used to train the quantile regressor (i.e., outputs).
    parameters : Union[np.ndarray, torch.Tensor]
        Parameters used to train the quantile regressor (i.e., inputs).
    algorithm : Union[str, Any]
        Either 'sk-gb' or 'cat-gb' for gradient boosted trees (Scikit-Learn or CatBoost), 'nn' for a feed-forward neural network, or a custom algorithm (Any).
        The latter must implement a `fit(X=..., y=...)` method.
    alpha : Union[float, Sequence[float]]  # TODO: update this to include sequences, but only with models monotonic in inputs.
        The alpha quantile of the test statistic to be estimated. 
        E.g., for 90% confidence intervals, it should be 0.1 if the acceptance region of the test statistic is on the right of the critical value. 
        Must be in the range `(0, 1)`.
    param_dim: int
        Dimensionality of the parameter.
    algorithm_kwargs : Union[Dict[str, Any], Dict[str, Dict[str, Any]]], optional
        Keywork arguments for the desired algorithm, by default {}.
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.
        
        If algorithm == 'gb', pass {'cv': hp_dist} to do a randomized search over the hyperparameters in hp_dist (a `Dict`) via 5-fold cross validation. 
        Include 'n_iter' as a key to decide how many hyperparameter setting to sample for randomized search. Defaults to 10.
    n_jobs : int, optional
        Number of workers to use when doing random search with 5-fold CV. By default -2, which uses all cores minus one. If -1, use all cores.
        `n_jobs == -1` uses all cores. If `n_jobs < -1`, then `n_jobs = os.cpu_count()+1+n_jobs`.

    Returns
    -------
    Any
        Fitted quantile regressor.

    Raises
    ------
    ValueError
        Only one of 'gb', 'nn' or an instantiated custom quantile regressor (Any) is currently accepted as algorithm.
    """
    assert not isinstance(alpha, Sequence)
    if n_jobs < -1:
        n_jobs = max(1, os.cpu_count()+1+n_jobs)
    elif n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs == 0:
        raise ValueError('n_jobs must be greater than 0')
    else:
        n_jobs = n_jobs
    if algorithm == "sk-gb":
        if 'cv' in algorithm_kwargs:
            algorithm = RandomizedSearchCV(
                estimator=GradientBoostingRegressor(
                    loss='quantile', 
                    alpha=alpha
                ),
                param_distributions=algorithm_kwargs['cv'],
                n_iter=20 if 'n_iter' not in algorithm_kwargs else algorithm_kwargs['n_iter'],
                scoring=make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False),
                n_jobs=n_jobs,
                refit=True,
                cv=5,
                verbose=1
            )
        else:
            algorithm = GradientBoostingRegressor(
                loss='quantile', 
                alpha=alpha, 
                **algorithm_kwargs
            )
        test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
        algorithm.fit(X=parameters, y=test_statistics)
    elif algorithm == "cat-gb":
        if 'cv' in algorithm_kwargs:
            algorithm = RandomizedSearchCV(
                estimator=CatBoostRegressor(
                    loss_function=f'Quantile:alpha={alpha}',
                    silent=True
                ),
                param_distributions=algorithm_kwargs['cv'],
                n_iter=20 if 'n_iter' not in algorithm_kwargs else algorithm_kwargs['n_iter'],
                scoring=make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False),
                n_jobs=n_jobs,
                refit=True,
                cv=5,
                verbose=1
            )
        else:
            algorithm = CatBoostRegressor(
                loss_function=f'Quantile:alpha={alpha}',
                **algorithm_kwargs
            )
        test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
        algorithm.fit(X=parameters, y=test_statistics)
    elif algorithm == 'nn':
        # TODO: remove possibility of using a sequence of quantiles -> could incur in quantile crossings. Need to implement monotonicity constraints.
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
        test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
        learner_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
        algorithm.fit(X=parameters, y=test_statistics, **learner_kwargs)
    elif isinstance(algorithm, Any):
        test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
        algorithm.fit(X=parameters, y=test_statistics)
    else:
        raise ValueError(f"Only 'gb', 'nn' or custom algorithm (Any) are currently supported, got {algorithm}")

    return algorithm
