from typing import Union, Callable, Dict, Any, Sequence
import os

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor  # TODO: is tree split done according to quantile loss? Default is 'friedman_mse'
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss
from catboost import CatBoostRegressor

from lf2i.calibration.torch_utils import QuantileLoss, FeedForwardNN, LearnerRegression
from lf2i.utils.calibration_diagnostics_inputs import preprocess_train_quantile_regression
from lf2i.utils.miscellanea import select_n_jobs


def train_qr_algorithm(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    algorithm: Union[str, Any],
    alpha: Union[float, Sequence[float]],
    param_dim: int,
    algorithm_kwargs: Union[Dict[str, Any], Dict[str, Dict[str, Any]]] = {},
    verbose: bool = True,
    n_jobs: int = -2
) -> Any:
    """Dispatcher to train different quantile regressors and estimate critical values.

    Parameters
    ----------
    test_statistics : Union[np.ndarray, torch.Tensor]
        The i-th element is the test statistics evaluated on the i-th element of `poi` (i.e., :math:`\theta_i`) and on :math:`x \sim F_{\theta_i}`.
    parameters : Union[np.ndarray, torch.Tensor]
        Parameters of interest in the calibration set.
    algorithm : Union[str, Any]
        Either 'cat-gb' for gradient boosted trees, 'nn' for a feed-forward neural network, or a custom algorithm (Any).
        The latter must implement the `fit(X=..., y=...)` method.
    alpha : Union[float, Sequence[float]]
        The alpha quantile of the test statistic to be estimated. 
        E.g., for 90% confidence intervals, it should be 0.9 if the acceptance region of the test statistic is on the left of the critical value. 
        Similarly, it should be 0.1 if the acceptance region of the test statistic is on the right of the critical value. 
        Must be in the range `(0, 1)`.
    param_dim: int
        Dimensionality of the parameter.
    algorithm_kwargs : Union[Dict[str, Any], Dict[str, Dict[str, Any]]], optional
        Keyword arguments for the desired algorithm, by default {}.
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.
        If algorithm == 'cat-gb', pass {'cv': hp_dist} to do a randomized search over the hyperparameters in hp_dist (a `Dict`) via 5-fold cross validation. 
        Include 'n_iter' as a key to decide how many hyperparameter setting to sample for randomized search. Defaults to 25.
    verbose: bool, optional
        Whether to print information on the hyper-parameter search for quantile regression, by default True.
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
        Only one of 'cat-gb', 'nn' or an instantiated custom quantile regressor (Any) is currently accepted as algorithm.
    """
    assert not isinstance(alpha, Sequence)
    n_jobs = select_n_jobs(n_jobs)

    if isinstance(algorithm, str):
        if algorithm == "cat-gb":
            if 'cv' in algorithm_kwargs:
                algorithm = RandomizedSearchCV(
                    estimator=CatBoostRegressor(
                        loss_function=f'Quantile:alpha={alpha}',
                        silent=True
                    ),
                    param_distributions=algorithm_kwargs['cv'],
                    n_iter=25 if 'n_iter' not in algorithm_kwargs else algorithm_kwargs['n_iter'],
                    scoring=make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False),
                    n_jobs=n_jobs,
                    refit=True,
                    cv=5,
                    verbose=1 if verbose else 0
                )
            else:
                algorithm = CatBoostRegressor(
                    loss_function=f'Quantile:alpha={alpha}',
                    **algorithm_kwargs
                )
            test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
            algorithm.fit(X=parameters, y=test_statistics)
        elif algorithm == 'nn':
            # TODO: implement some form of hyperparameter tuning
            quantiles = [alpha] if isinstance(alpha, float) else alpha
            nn_kwargs = {arg: algorithm_kwargs[arg] for arg in ['hidden_activation', 'dropout_p', 'batch_norm'] if arg in algorithm_kwargs}
            feedforward_nn = FeedForwardNN(
                input_d=parameters.shape[1], 
                output_d=len(quantiles),
                hidden_layer_shapes=algorithm_kwargs['hidden_layer_shapes'], 
                **nn_kwargs
            )
            algorithm = LearnerRegression(
                model=feedforward_nn, 
                optimizer=torch.optim.Adam, 
                loss=QuantileLoss(quantiles=quantiles), 
                device="cuda" if torch.cuda.is_available() else 'cpu',
                verbose=verbose
            )
            test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
            learner_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
            algorithm.fit(X=parameters, y=test_statistics, **learner_kwargs)
        else:
            raise ValueError(f"Only 'cat-gb', 'nn' or custom algorithm (Any) are currently supported, got {algorithm}")
    else:
        test_statistics, parameters = preprocess_train_quantile_regression(test_statistics, parameters, param_dim, algorithm)
        algorithm.fit(X=parameters, y=test_statistics)

    return algorithm
