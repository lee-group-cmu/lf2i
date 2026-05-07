from typing import Union, Tuple, Any, Optional, List, Dict

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier

try:
    from tabicl import TabICLClassifier  # optional convenience import
    HAVE_TABICL = True
except Exception:
    TabICLClassifier = None
    HAVE_TABICL = False

from lf2i.calibration.torch_utils import FeedForwardNN, LearnerClassification
from lf2i.utils.calibration_diagnostics_inputs import preprocess_fit_p_values
from lf2i.utils.miscellanea import select_n_jobs, to_np_if_torch
from lf2i.calibration.parametric_cd import ParametricCDFEstimator
from lf2i.calibration.isplines import ISplineClassifier


def conditional_sampling(
    poi: np.ndarray,
    test_statistics: np.ndarray,
    num_augment: int, 
    min_points_per_bin: int = 50
) -> np.ndarray:
    """
    Perform conditional sampling of test statistics based on the parameters of interest (POI). 
    This method divides the POI space into multidimensional bins, associates each bin with the 
    corresponding test statistics, and resamples conditionally from the empirical distribution 
    of test statistics within each bin.

    Parameters
    ----------
    poi : np.ndarray
        A 2D array where each row represents a parameter of interest (POI) and each column corresponds 
        to a dimension in the parameter space. Shape: (num_samples, num_dimensions).
    test_statistics : np.ndarray
        A 1D array of test statistics evaluated for each parameter of interest. Shape: (num_samples,).
    num_augment : int
        Number of samples to draw from the conditional distribution of test statistics for each POI bin.
    min_points_per_bin : int, optional
        Minimum number of points required per bin for constructing the POI bins. The POI space will 
        be divided into bins such that each bin contains at least this number of points. Default is 50.

    Returns
    -------
    np.ndarray
        A 2D array of resampled test statistics. Shape: (num_samples, num_augment), where `num_samples` 
        corresponds to the number of rows in `poi`.

    Raises
    ------
    AssertionError
        If bin assignments fail or there is no data available for a specific bin.
    """
    # Define bins for each dimension of POI
    def equal_size_bin_edges(poi_onedim, min_points_per_bin):
        n_bins = max(1, len(poi_onedim) // min_points_per_bin)
        return np.percentile(poi_onedim, np.linspace(0, 100, n_bins + 1))  # there are n_bins+1 edges
    poi_bin_edges = [equal_size_bin_edges(poi[:, dim], min_points_per_bin=min_points_per_bin) for dim in range(poi.shape[1])]

    # Assign each poi to a multidimensional bin
    poi_bin_indices = np.stack([np.digitize(poi[:, dim], poi_bin_edges[dim], right=True) - 1 for dim in range(poi.shape[1])], axis=1)
    poi_bin_indices = np.clip(poi_bin_indices, 0, [len(poi_bin_edges[dim]) - 2 for dim in range(poi.shape[1])])  # Clip to valid ranges
    assert (poi_bin_indices.min() >= 0) and (poi_bin_indices.max() < len(poi_bin_edges[0]) - 1), "Bin assignment failed"  # Ensure no alignment issues
    
    # Vectorized sampling from p(ts|poi)
    unique_bins = np.unique(poi_bin_indices, axis=0)
    samples = []
    for bin_idx in unique_bins:
        mask = np.all(poi_bin_indices == bin_idx, axis=1)
        ts_in_bin = test_statistics[mask]
        assert len(ts_in_bin) > 0, f"No data available for b in bin {bin_idx}."
        samples.extend(np.random.choice(ts_in_bin, size=num_augment * mask.sum(), replace=True))

    return np.array(samples).reshape(len(poi), num_augment)


def augment_calibration_set(
    test_statistics: Union[np.ndarray, torch.Tensor],
    poi: Union[np.ndarray, torch.Tensor],
    num_augment: int,
    acceptance_region: str,
    conditional_resampling: bool = True,
    min_points_per_bin: int = 50
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
    conditional_resampling: bool, optional
        Whether to re-sample cutoffs for augmentation from :math:`p(\tau \mid \theta)` or from the marginal :math:`p(\tau)`. Default is True. 
        Conditional sampling should yield better estimates of p-values since it is designed to better represent the tails of each conditional distribution, but
        it could be impractical with a high-dimensional parameter.
    min_points_per_bin : int, optional
        Minimum number of points required per bin for constructing the POI bins. The POI space will 
        be divided into bins such that each bin contains at least this number of points. Default is 50.

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
        test_statistics = to_np_if_torch(test_statistics)
    if isinstance(poi, torch.Tensor):
        poi = to_np_if_torch(poi)
    
    # sample cutoffs from empirical distribution and repeat poi/ts to match size
    if poi.ndim == 1:
        poi = np.expand_dims(poi, axis=1)
    rep_poi = np.repeat(poi, repeats=num_augment, axis=0)
    if conditional_resampling:
        resampled_cutoffs = conditional_sampling(poi, test_statistics, num_augment, min_points_per_bin).reshape(-1, 1)
    else:
        resampled_cutoffs = np.random.choice(a=test_statistics.reshape(-1, ), size=num_augment*poi.shape[0], replace=True).reshape(-1, 1)
    rep_test_statistics = np.repeat(test_statistics.reshape(-1, ), repeats=num_augment).reshape(-1, 1)
    
    # compute rejection indicators
    if acceptance_region == 'left':
        rejection_indicators = (rep_test_statistics >= resampled_cutoffs).astype(int).reshape(-1, )  # output of probs classifier usually expected to be 1-dim
    elif acceptance_region == 'right':
        rejection_indicators = (rep_test_statistics <= resampled_cutoffs).astype(int).reshape(-1, )
    else:
        raise ValueError(f'Acceptance region must be either `left` or `right`, got {acceptance_region}.')
    assert resampled_cutoffs.shape[0] == rep_test_statistics.shape[0] == rejection_indicators.shape[0] == rep_poi.shape[0] == num_augment*poi.shape[0]
    
    shuffle_idx = np.random.permutation(num_augment * poi.shape[0])
    return np.hstack((resampled_cutoffs, rep_poi))[shuffle_idx, :], rejection_indicators[shuffle_idx]


def estimate_rejection_proba(
    inputs: np.ndarray, 
    rejection_indicators: np.ndarray, 
    algorithm: Union[str, Any],
    acceptance_region: str,
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
        Either 'cat-gb' for gradient boosted trees, 'nn' for a feed-forward neural network, 
        'logistic' for logistic regression, 'gam' for a radially symmetric generalized additive model, 
        or a custom algorithm (Any). The latter must implement the `fit(X=..., y=...)` method.
    acceptance_region : str
        Whether the acceptance region for the test statistic is defined to be on the right or on the left of the cutoff. 
        Must be either `left` or `right`. 
    algorithm_kwargs : Union[Dict[str, Any], Dict[str, Dict[str, Any]]], optional
        Keyword arguments for the desired algorithm, by default {}.
        If algorithm == 'nn', then 'hidden_layer_shapes', 'epochs' and 'batch_size' must be present.
        If algorithm == 'cat-gb', pass {'cv': hp_dist} to do a randomized search over the hyperparameters in hp_dist (a `Dict`) via 5-fold cross validation. 
        Include 'n_iter' as a key to decide how many hyperparameter setting to sample for randomized search. Defaults to 10.
        If algorithm == 'logistic', any valid LogisticRegression parameters can be passed.
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
    inputs, rejection_indicators = preprocess_fit_p_values(inputs, algorithm), preprocess_fit_p_values(rejection_indicators, algorithm).reshape(-1, )

    if isinstance(algorithm, str):
        if algorithm == 'cat-gb':
            if 'cv' in algorithm_kwargs:
                algorithm = RandomizedSearchCV(
                    estimator=CatBoostClassifier(
                        loss_function='CrossEntropy',
                        silent=True,
                        monotone_constraints="0:1",  # 1 means non-decreasing function of cutoffs (always 0-th column of inputs),
                    ),
                    param_distributions=algorithm_kwargs['cv'],
                    n_iter=10 if 'n_iter' not in algorithm_kwargs else algorithm_kwargs['n_iter'],
                    n_jobs=n_jobs,
                    refit=False,
                    cv=5,
                    verbose=1 if verbose else 0
                )
                algorithm.fit(X=inputs, y=rejection_indicators, cat_features=cat_poi_idxs)

                # Retrain with CV-selected hyperparameters and isotonic calibration
                algorithm = CalibratedClassifierCV(
                    estimator=CatBoostClassifier(
                        loss_function='CrossEntropy',
                        silent=True,
                        # 1 (-1) means non-decreasing (non-increasing) function of cutoffs (always 0-th column of inputs)
                        monotone_constraints="0:1" if acceptance_region == 'right' else "0:-1",
                        **(algorithm.best_params_ if 'cv' in algorithm_kwargs else algorithm_kwargs)
                    ),
                    method='isotonic',
                    cv=5,
                    n_jobs=n_jobs
                )
            else:
                algorithm = CatBoostClassifier(
                    loss_function='CrossEntropy',
                    silent=True,
                    monotone_constraints="0:1" if acceptance_region == 'right' else "0:-1",
                )
            algorithm.fit(X=inputs, y=rejection_indicators, cat_features=cat_poi_idxs)
        elif algorithm == 'logistic':
            algorithm = CalibratedClassifierCV(
                estimator=LogisticRegression(
                    max_iter=1000,
                    n_jobs=n_jobs,
                    **algorithm_kwargs
                ),
                method='sigmoid',
                cv=5,
                n_jobs=n_jobs
            )
            algorithm.fit(X=inputs, y=rejection_indicators)
        elif algorithm == 'tfm':
            assert HAVE_TABICL, "TabICL is not installed. Please install it to use the 'tfm' algorithm."
            tfm_kwargs = algorithm_kwargs if algorithm_kwargs else {
                'n_estimators': 1,
                'feat_shuffle_method': 'none',
                'class_shuffle_method': 'none',
                'outlier_threshold': 3.0,
                'support_many_classes': False,
                'n_jobs': n_jobs
            }
            algorithm = TabICLClassifier(**tfm_kwargs)
            algorithm.fit(X=inputs, y=rejection_indicators)
        elif algorithm == 'parametric-nn':
            nn_kwargs  = algorithm_kwargs if algorithm_kwargs else {}
            algorithm  = ParametricCDFEstimator(
                acceptance_region = acceptance_region,
                **nn_kwargs
            )
            algorithm.fit(test_statistics = inputs[:, 0],
                          poi = inputs[:, 1:])
        elif algorithm == 'spline':
            algorithm = ISplineClassifier(
                monotone_constraints=1 if acceptance_region == 'right' else -1,
                **algorithm_kwargs,
            )
            algorithm.fit(X=inputs, y=rejection_indicators)
        else:
            raise ValueError(f"Only 'cat-gb', 'logistic', 'tfm', 'parametric-nn', 'spline', or a custom algorithm (Any) are supported, got {algorithm}")
    else:
        inputs, rejection_indicators = preprocess_fit_p_values(inputs, algorithm), preprocess_fit_p_values(rejection_indicators, algorithm)
        algorithm.fit(X=inputs, y=rejection_indicators)
    return algorithm
