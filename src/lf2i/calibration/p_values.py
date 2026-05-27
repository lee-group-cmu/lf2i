from typing import Union, Tuple, Any, Optional, Dict
import inspect
import warnings

import numpy as np
import torch

from lf2i.utils.calibration_diagnostics_inputs import preprocess_fit_p_values
from lf2i.utils.miscellanea import to_np_if_torch
from lf2i.estimators.base_cdf import AbstractCDFEstimator
from lf2i.estimators.base_probabilistic_classifier import AbstractProbabilisticClassifier


def estimate_rejection_proba(
    test_statistics: Union[np.ndarray, torch.Tensor],
    parameters: Union[np.ndarray, torch.Tensor],
    algorithm: Any,
    augment_kwargs: Optional[Dict[str, Any]] = None,
    acceptance_region: Optional[str] = None,
    verbose: bool = True,
) -> Any:
    """Fit a calibration model to estimate rejection probabilities (p-values).

    Handles augmentation internally: CDF estimators receive raw ``(T, θ)`` pairs
    and model ``p(T | θ)`` directly; probabilistic classifiers receive augmented
    ``(τ, θ)`` pairs with ``1[T ≤ τ]`` labels produced by
    :func:`augment_calibration_set`.

    Parameters
    ----------
    test_statistics : array-like of shape (N,)
        Test statistics ``T_i`` evaluated at the i-th calibration parameter
        ``θ_i`` and corresponding sample ``x_i ~ F_{θ_i}``.
    parameters : array-like of shape (N,) or (N, d)
        Parameters of interest ``θ_i`` for each calibration sample.
        Any width ``d ≥ 1`` is accepted; 1-D input is promoted to ``(N, 1)``.
    algorithm : AbstractCDFEstimator or AbstractProbabilisticClassifier
        - **CDF estimator** (``fit(X)`` with no ``y`` argument): receives
          ``(T, θ)`` stacked as ``X`` and learns ``p(T | θ)`` directly.
          ``augment_kwargs`` are ignored for this branch.
        - **Probabilistic classifier** (``fit(X, y)``): receives augmented
          ``(τ, θ)`` inputs and ``1[T ≤ τ]`` labels produced internally by
          :func:`augment_calibration_set`.
    augment_kwargs : dict, optional
        Forwarded to :func:`augment_calibration_set` for probabilistic
        classifiers.  Recognised keys (with defaults):

        - ``num_augment`` (int, default 1): cutoffs resampled per observation.
        - ``conditional_resampling`` (bool, default True): resample from
          ``p(τ | θ)`` rather than the marginal.
        - ``min_points_per_bin`` (int, default 50): minimum bin occupancy for
          conditional resampling.

        Silently ignored (with a verbose note) when ``algorithm`` is a CDF
        estimator.
    acceptance_region : str, optional
        **Deprecated and ignored.**  Directionality is the caller's
        responsibility.  Passing a value emits a :class:`DeprecationWarning`.
    verbose : bool, default True
        Print a one-line summary of the preprocessing performed.

    Returns
    -------
    Any
        The fitted ``algorithm`` object.
    """
    if acceptance_region is not None:
        warnings.warn(
            "The `acceptance_region` argument to `estimate_rejection_proba` is deprecated and "
            "will be removed in a future version. Directionality is now handled by the caller "
            "(e.g. LF2I.inference). The argument is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    assert isinstance(algorithm, (AbstractCDFEstimator, AbstractProbabilisticClassifier)), (
        f"algorithm must implement AbstractCDFEstimator or AbstractProbabilisticClassifier, "
        f"got {type(algorithm)}"
    )

    # Normalise to numpy
    test_statistics = to_np_if_torch(test_statistics).reshape(-1)
    parameters = to_np_if_torch(parameters)
    if parameters.ndim == 1:
        parameters = parameters.reshape(-1, 1)
    assert len(test_statistics) == len(parameters), (
        f"test_statistics and parameters must have the same number of rows, "
        f"got {len(test_statistics)} and {len(parameters)}"
    )

    if 'y' not in inspect.signature(algorithm.fit).parameters:
        # CDF estimator: learn p(T | θ) directly from raw (T, θ) pairs
        if verbose and augment_kwargs:
            print("  [calibration] CDF estimator — augment_kwargs ignored.")
        inputs = preprocess_fit_p_values(
            np.hstack([test_statistics[:, None], parameters]), algorithm
        )
        if verbose:
            print(f"  [calibration] CDF estimator: fitting on {len(inputs)} raw (T, θ) pairs.")
        algorithm.fit(X=inputs)
    else:
        # Probabilistic classifier: augment calibration set first
        _kw: Dict[str, Any] = {
            'num_augment': 1,
            'conditional_resampling': True,
            'min_points_per_bin': 50,
        }
        if augment_kwargs:
            _kw.update(augment_kwargs)
        if verbose:
            n, m = len(test_statistics), _kw['num_augment']
            print(
                f"  [calibration] Classifier: augmenting {n} pairs "
                f"× {m} cutoffs → {n * m} rows "
                f"(conditional_resampling={_kw['conditional_resampling']})."
            )
        inputs, rejection_indicators = augment_calibration_set(
            test_statistics=test_statistics,
            poi=parameters,
            **_kw,
        )
        inputs = preprocess_fit_p_values(inputs, algorithm)
        rejection_indicators = preprocess_fit_p_values(rejection_indicators, algorithm).reshape(-1,)
        algorithm.fit(X=inputs, y=rejection_indicators)

    return algorithm

def augment_calibration_set(
    test_statistics: Union[np.ndarray, torch.Tensor],
    poi: Union[np.ndarray, torch.Tensor],
    num_augment: int,
    conditional_resampling: bool = True,
    min_points_per_bin: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment the calibration set by resampling cutoffs from the empirical distribution of the test statistics.
    This allows to estimate p-values that are amortized with respect to all levels :math:`\alpha`.

    The rejection indicator is always defined as :math:`\\mathbb{1}[T \\le \\tau]`, so the trained
    classifier estimates the CDF :math:`F(\\tau \\mid \\theta) = P(T \\le \\tau \\mid \\theta)`.
    This is monotone increasing and produces ``predict_proba`` output with columns ``[1-CDF, CDF]``.
    Directional p-value selection (based on the test statistic's acceptance region) is the
    responsibility of the caller (e.g. :class:`lf2i.inference.lf2i.LF2I`).

    Parameters
    ----------
    test_statistics : Union[np.ndarray, torch.Tensor]
        The i-th element is the test statistics evaluated on the i-th element of `poi` (i.e., :math:`\theta_i`) and on :math:`x \sim F_{\theta_i}`.
    poi : Union[np.ndarray, torch.Tensor]
        Parameters of interest in the calibration set.
    num_augment : int
        Number of cutoffs to resample for each value in `test_statistics`. The augmented calibration set will be of size `num_augment` :math:`\times B^\prime`,
        where :math:`B^\prime` is the size of the original calibration set.
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
        Augmented inputs (cutoffs and POIs) and outputs (CDF indicators) to estimate amortized p-values.
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
    
    # CDF direction: class=1 means T <= cutoff, so P(class=1 | cutoff, θ) = F(cutoff | θ)
    rejection_indicators = (rep_test_statistics <= resampled_cutoffs).astype(int).reshape(-1, )
    assert resampled_cutoffs.shape[0] == rep_test_statistics.shape[0] == rejection_indicators.shape[0] == rep_poi.shape[0] == num_augment*poi.shape[0]
    
    shuffle_idx = np.random.permutation(num_augment * poi.shape[0])
    return np.hstack((resampled_cutoffs, rep_poi))[shuffle_idx, :], rejection_indicators[shuffle_idx]

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