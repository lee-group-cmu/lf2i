from typing import Union, Tuple, Any, Optional, List, Dict

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier

from lf2i.calibration.torch_utils import QuantileLoss, FeedForwardNN, LearnerClassification
from lf2i.utils.calibration_diagnostics_inputs import preprocess_fit_p_values
from lf2i.utils.miscellanea import select_n_jobs


def augment_calibration_set(
    test_statistics: Union[np.ndarray, torch.Tensor],
    poi: Union[np.ndarray, torch.Tensor],
    num_augment: int,
    acceptance_region: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment the calibration set by resampling cutoffs from the empirical distribution of the test statistics. 
    This allows to estimate p-values that are amortized with respect to all levels :math:`\alpha`.

    Parameters
    ----------
    test_statistics : Union[np.ndarray, torch.Tensor]
        The i-th element is the test statistics evaluated on the i-th element of `poi` (i.e., :math:`\theta_i`) and on :math:`x \sim F_{\theta_i}`.
    poi : Union[np.ndarray, torch.Tensor]
        Parameters of interest in the calibration set.
    num_augment : int
        Number of cutoffs to resample for each value in `test_statistics`. The augmented calibration set will be of size `num_augment` :math:`\times B^\prime`, 
        where :math:`B^\prime` is the size of the original calibration set.
    acceptance_region : str
        Whether the acceptance region for the test statistic is defined to be on the right or on the left of the cutoff. 
        Must be either `left` or `right`. 

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Augmented inputs (cutoffs and POIs) and outputs (rejection indicators) to estimate amortized p-values.

    Raises
    ------
    ValueError
        If `acceptance_region` is not one of `left` or `right`. 
    """
    assert test_statistics.shape[0] == poi.shape[0], 'Shape mismatch between test statistics and POIs'
    if isinstance(test_statistics, torch.Tensor):
        test_statistics = test_statistics.numpy()
    if isinstance(poi, torch.Tensor):
        poi = poi.numpy()
    
    # sample cutoffs from empirical distribution and repeat poi/ts to match size
    if poi.ndim == 1:
        poi = np.expand_dims(poi, axis=1)
    rep_poi = np.repeat(poi, repeats=num_augment, axis=0)
    resampled_cutoffs = np.random.choice(a=test_statistics.reshape(-1, ), size=num_augment*poi.shape[0], replace=True).reshape(-1, 1)
    rep_test_statistics = np.repeat(test_statistics.reshape(-1, ), repeats=num_augment).reshape(-1, 1)
    
    # compute rejection indicators
    if acceptance_region == 'left':
        rejection_indicators = (rep_test_statistics >= resampled_cutoffs).astype(int).reshape(-1, )  # output of probs classifier usually 1-dim
    elif acceptance_region == 'right':
        rejection_indicators = (rep_test_statistics <= resampled_cutoffs).astype(int).reshape(-1, )
    else:
        raise ValueError(f'Acceptance region must be either `left` or `right`, got {acceptance_region}.')
    assert resampled_cutoffs.shape[0] == rep_test_statistics.shape[0] == rejection_indicators.shape[0] == rep_poi.shape[0] == num_augment*poi.shape[0]
    
    shuffle_idx = np.random.choice(range(l:=(num_augment*poi.shape[0])), size=l, replace=False)
    return np.hstack((resampled_cutoffs, rep_poi))[shuffle_idx, :], rejection_indicators[shuffle_idx]


def estimate_rejection_proba(
    inputs: np.ndarray, 
    rejection_indicators: np.ndarray, 
    algorithm: Union[str, Any],
    algorithm_kwargs: Union[Dict[str, Any], Dict[str, Dict[str, Any]]] = {},
    cat_poi_idxs: Optional[List[int]] = None,
    verbose: bool = True,
    n_jobs: int = -2
) -> Any:
    """Dispatcher to train different probabilistic classifiers and estimate p-values.

    Parameters
    ----------
    inputs : np.ndarray
        Augmented calibration inputs as provided by `lf2i.calibration.p_values.augment_calibration_set`.
    rejection_indicators : np.ndarray
        Rejection indicators as provided by `lf2i.calibration.p_values.augment_calibration_set`.
    algorithm : str
        Either 'cat-gb' for gradient boosted trees, 'nn' for a feed-forward neural network, or a custom algorithm (Any).
        The latter must implement the `fit(X=..., y=...)` method.
    algorithm_kwargs : Union[Dict[str, Any], Dict[str, Dict[str, Any]]], optional
        Keyword arguments for the desired algorithm, by default {}.
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.
        If algorithm == 'cat-gb', pass {'cv': hp_dist} to do a randomized search over the hyperparameters in hp_dist (a `Dict`) via 5-fold cross validation. 
        Include 'n_iter' as a key to decide how many hyperparameter setting to sample for randomized search. Defaults to 25.
    cat_poi_idxs : Optional[List[int]], optional
        If `algorithm == 'cat-gb'`, sequence of indexes to indicate the columns of `inputs` containing categorical POIs, by default None.
        Note that the first column of `inputs` is always the resampled cutoffs, hence this should be treated as a 1-indexed array (i.e. col 0 of POIs has index 1).
    verbose : bool, optional
        Whether to print information on the hyper-parameter search for quantile regression, by default True.
    n_jobs : int, optional
        Number of workers to use when doing random search with 5-fold CV. By default -2, which uses all cores minus one. If -1, use all cores.
        `n_jobs == -1` uses all cores. If `n_jobs < -1`, then `n_jobs = os.cpu_count()+1+n_jobs`.

    Returns
    -------
    Any
        Fitted probabilistic classifier.
    """
    n_jobs = select_n_jobs(n_jobs)
    if isinstance(algorithm, str):
        if algorithm == 'cat-gb':
            if ('cv' in algorithm_kwargs) or algorithm_kwargs is None:
                algorithm = RandomizedSearchCV(
                    estimator=CatBoostClassifier(
                        loss_function='CrossEntropy',
                        silent=True,
                        monotone_constraints="0:1",  # 1 means non-decreasing function of cutoffs (always 0-th column of inputs),
                    ),
                    param_distributions=algorithm_kwargs['cv'],
                    n_iter=25 if 'n_iter' not in algorithm_kwargs else algorithm_kwargs['n_iter'],
                    n_jobs=n_jobs,
                    refit=True,
                    cv=5,
                    verbose=1 if verbose else 0
                )
            else:
                algorithm = CatBoostClassifier(
                    loss_function='CrossEntropy',
                    silent=True,
                    monotone_constraints="0:1",
                    **algorithm_kwargs
                )
            inputs, rejection_indicators = preprocess_fit_p_values(inputs, algorithm), preprocess_fit_p_values(rejection_indicators, algorithm)
            algorithm.fit(X=inputs, y=rejection_indicators, cat_features=cat_poi_idxs)
        elif algorithm == 'nn':
            # TODO: implement some form of hyperparameter tuning
            nn_kwargs = {arg: algorithm_kwargs[arg] for arg in ['hidden_activation', 'dropout_p', 'batch_norm'] if arg in algorithm_kwargs}
            feedforward_nn = FeedForwardNN(
                input_d=inputs.shape[1], 
                output_d=1,
                hidden_layer_shapes=algorithm_kwargs['hidden_layer_shapes'], 
                **nn_kwargs
            )
            algorithm = LearnerClassification(
                model=feedforward_nn, 
                optimizer=torch.optim.Adam, 
                loss=BCEWithLogitsLoss(), 
                device="cuda" if torch.cuda.is_available() else 'cpu',
                verbose=verbose
            )
            inputs, rejection_indicators = preprocess_fit_p_values(inputs, algorithm), preprocess_fit_p_values(rejection_indicators, algorithm)
            learner_kwargs = {arg: algorithm_kwargs[arg] for arg in ['epochs', 'batch_size']}
            algorithm.fit(X=inputs, y=rejection_indicators, **learner_kwargs)
        else:
            raise ValueError(f"Only 'cat-gb', 'nn' or custom algorithm (Any) are currently supported, got {algorithm}")
    else:
        inputs, rejection_indicators = preprocess_fit_p_values(inputs, algorithm), preprocess_fit_p_values(rejection_indicators, algorithm)
        algorithm.fit(X=inputs, y=rejection_indicators)
    return algorithm
