from typing import Union, Callable, Dict, Any, Sequence
import os

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor  # TODO: is tree split done according to quantile loss? Default is 'friedman_mse'
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss

from lf2i.critical_values.nn_qr_algorithm import QuantileLoss, QuantileNN, Learner
from lf2i.utils.calibration_diagnostics_inputs import preprocess_train_quantile_regression


def train_qr_algorithm(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    algorithm: Union[str, Callable],
    alpha: Union[float, Sequence[float]],
    param_dim: int,
    algorithm_kwargs: Union[Dict[str, Any], Dict[str, Dict[str, Any]]] = {}
) -> Any:
    """Dispatcher to train different quantile regressors.

    Parameters
    ----------
    test_statistics : Union[np.ndarray, torch.Tensor]
        Test statistics used to train the quantile regressor (i.e., outputs).
    parameters : Union[np.ndarray, torch.Tensor]
        Parameters used to train the quantile regressor (i.e., inputs).
    algorithm : Union[str, Any]
        Either 'gb' for gradient boosted trees, 'nn' for a feed-forward neural network, or a custom algorithm (Any).
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
    test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim)
    if algorithm == "gb":
        if 'cv' in algorithm_kwargs:
            algorithm = RandomizedSearchCV(
                estimator=GradientBoostingRegressor(
                    loss='quantile', 
                    alpha=alpha
                ),
                param_distributions=algorithm_kwargs['cv'],
                n_iter=10 if 'n_iter' not in algorithm_kwargs else algorithm_kwargs['n_iter'],
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
        learner_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
        algorithm.fit(X=parameters, y=test_statistics, **learner_kwargs)
    elif isinstance(algorithm, Any):
        algorithm.fit(X=parameters, y=test_statistics)
    else:
        raise ValueError(f"Only 'gb', 'nn' or custom algorithm (Any) are currently supported, got {algorithm}")

    return algorithm
