"""NGBoost-based conditional CDF estimator satisfying AbstractCDFEstimator.

This module provides ``NGBoostCDFEstimator``, which inherits from both
``NGBoost`` (ngboost/ngboost/ngboost.py) and the ``AbstractCDFEstimator``
Protocol.

It models the conditional CDF  F(λ | θ)  of a test statistic λ given
parameters of interest θ, using NGBoost's probabilistic gradient boosting.

Requires the ``ngboost`` package (``pip install ngboost``).

Usage
-----
    import numpy as np

    # X[:, 0]  = test statistics λ
    # X[:, 1:] = parameters of interest θ  (shape n × poi_dim)
    X_cal = np.column_stack([lambda_vals, theta_vals])

    estimator = NGBoostCDFEstimator(n_estimators=200)
    estimator.fit(X_cal)

    X_test = np.column_stack([lambda_test, theta_test])
    proba = estimator.predict_proba(X_test)   # shape (n, 2); rows sum to 1
    # proba[:, 0]  =  F(λ | θ)          P(reject = 0)
    # proba[:, 1]  =  1 − F(λ | θ)      P(reject = 1)  ← the p-value
"""

from typing import Union

import numpy as np
import torch

try:
    from ngboost.api import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore
    from ngboost.learners import default_tree_learner
    HAVE_NGBOOST = True
except ImportError:
    NGBoost = object          # fallback so the class body parses without ngboost
    Normal = None
    LogScore = None
    default_tree_learner = None
    HAVE_NGBOOST = False

from lf2i.estimators.base_cdf import AbstractCDFEstimator
from lf2i.estimators.ngboost_utils.distns.gamma_loc import GammaLoc
from sklearn.model_selection import train_test_split


class NGBoostCDFEstimator(NGBRegressor, AbstractCDFEstimator):
    """Conditional CDF estimator combining NGBoost with the AbstractCDFEstimator interface.

    Models  F(λ | θ)  where λ is a test statistic and θ are parameters of
    interest.  The conditional distribution of λ given θ is learned via
    NGBoost probabilistic gradient boosting: θ are the predictors and λ is
    the response.

    After fitting, ``predict_proba`` evaluates the fitted conditional CDF at
    new (λ, θ) pairs, returning rejection probabilities in the two-column
    format expected by downstream p-value computation in lf2i.

    Requires the ``ngboost`` package (``pip install ngboost``).

    Parameters
    ----------
    Dist : NGBoost distribution class, default ``Normal``
        Distributional family for the conditional distribution of λ | θ.
    Score : NGBoost score class, default ``LogScore``
    Base : sklearn regressor instance, default ``default_tree_learner``
    natural_gradient : bool, default True
    n_estimators : int, default 500
        Upper bound on boosting rounds.  Training stops earlier if
        ``early_stopping_rounds`` is set and the validation loss stagnates.
    learning_rate : float, default 0.01
    minibatch_frac : float, default 1.0
    col_sample : float, default 1.0
    verbose : bool, default True
    verbose_eval : int, default 100
    tol : float, default 1e-4
    random_state : int or None, default None
    validation_fraction : float, default 0.1
        Fraction of training data held out for early-stopping validation.
    early_stopping_rounds : int, default 20
        Stop training if the validation loss does not improve for this many
        consecutive rounds; the model reverts to the best checkpoint.
        Set to ``None`` to disable early stopping and always run all
        ``n_estimators`` rounds.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=None,
        Score=None,
        Base=None,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=20,
    ):
        if not HAVE_NGBOOST:
            raise ImportError(
                "NGBoostCDFEstimator requires the 'ngboost' package. "
                "Install it with: pip install ngboost"
            )
        NGBRegressor.__init__(
            self,
            Dist=Dist if Dist is not None else Normal,
            Score=Score if Score is not None else LogScore,
            Base=Base if Base is not None else default_tree_learner,
            natural_gradient=natural_gradient,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            verbose=verbose,
            verbose_eval=verbose_eval,
            tol=tol,
            random_state=random_state,
            validation_fraction=validation_fraction,
            early_stopping_rounds=early_stopping_rounds,
        )

    # ------------------------------------------------------------------
    # AbstractCDFEstimator interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> "NGBoostCDFEstimator":
        """Fit the conditional CDF estimator to calibration data.

        Parameters
        ----------
        X : array-like of shape (n, 1 + poi_dim)
            Column 0 holds test statistics λ(x_i; θ_i).
            Columns 1: hold the corresponding parameters of interest θ_i.
        **kwargs
            Forwarded to ``NGBRegressor.fit`` (e.g. ``sample_weight``,
            ``X_val`` / ``Y_val`` for early stopping, etc.).

        Returns
        -------
        self
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError(
                "X must be 2-D with shape (n, 1 + poi_dim) where poi_dim ≥ 1. "
                f"Received shape {X.shape}."
            )

        X_train, X_val, Y_train, Y_val = train_test_split(X[:, 1:], X[:, 0], test_size=0.2, random_state=42)
        NGBRegressor.fit(self, X_train, Y_train, X_val=X_val, Y_val=Y_val, **kwargs)
        return self

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """Compute p-values from the fitted conditional CDF F(λ | θ).

        Parameters
        ----------
        X : array-like of shape (n, 1 + poi_dim)
            Column 0 holds test statistics λ.
            Columns 1: hold the parameters of interest θ.
        **kwargs
            Forwarded to ``NGBRegressor.pred_dist`` (e.g. ``max_iter``).

        Returns
        -------
        np.ndarray of shape (n, 2)
            - ``[:, 0]`` = P(reject = 0 | λ, θ) = F(λ | θ)
            - ``[:, 1]`` = P(reject = 1 | λ, θ) = 1 − F(λ | θ)
            Rows sum to 1.
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X = np.asarray(X, dtype=float)

        lambda_vals = X[:, 0]
        features = X[:, 1:]

        max_iter = kwargs.pop("max_iter", None)
        pred = self.pred_dist(features, max_iter=max_iter)
        cdf_vals = pred.cdf(lambda_vals)

        return np.column_stack([1.0 - cdf_vals, cdf_vals])
