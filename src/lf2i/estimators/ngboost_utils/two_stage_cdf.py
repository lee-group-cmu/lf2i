"""Two-stage conditional-CDF calibration with NGBoost estimators.

Both stages are instances of the NGBoost-based ``NGBoostCDFEstimator`` defined
in ``cdf.py``.  By default the two stages use distributions tailored to the
two-stage scheme:

    Stage 1 (global CDF F1(lambda ; theta))  ->  ``GammaLoc``  (3-parameter,
        location-shifted Gamma from ``gamma_loc.py``; suits non-negative,
        non-zero-anchored test statistics lambda).
    Stage 2 (PIT residual correction F2(u ; theta))  ->  ``Kumaraswamy``
        (single-component Kumaraswamy from ``kumaraswamy_mixture.py``; a
        closed-form (0, 1) law, the natural model for PIT values).

Either default can be overridden -- pass your own ``NGBoostCDFEstimator``
instances, or kwargs (including a different ``Dist``, e.g. a multi-component
``make_kumaraswamy_mixture(K)`` for stage 2).

Algorithm
---------
Both stages model a conditional CDF of a scalar response given theta in the
``(n, 1 + poi_dim)`` layout ``NGBoostCDFEstimator`` expects (column 0 = the
response, columns 1: = theta).

Stage 1 -- global fit
    F1(lambda ; theta) is fit with lambda as response and theta as predictors.
    Its CDF evaluated at the training points gives the conditional
    probability-integral-transform

        u = F1(lambda ; theta)   ~= Uniform(0, 1)  if stage 1 is well calibrated.

Stage 2 -- residual correction
    A second estimator models F2(u ; theta), absorbing any departure of u from
    uniformity (the residual miscalibration left by stage 1).

Combined estimator
    F(lambda ; theta) = F2( F1(lambda ; theta) ; theta ).

    If stage 1 were perfect, u would be exactly Uniform(0, 1), F2 the identity,
    and the composition would reduce to F1.

Column convention
-----------------
``NGBoostCDFEstimator.predict_proba`` returns, per row, ``[1 - F, F]`` (the CDF
is column 1, matching the implementation in ``cdf.py``).  This module follows
that convention end to end, so ``predict_proba(X)[:, 1]`` is the combined CDF
F(lambda ; theta) and column 0 is its complement (the right-tail p-value).

Integration notes (unchanged from the original)
------------------------------------------------
``fit(X)`` takes no ``y`` parameter, so ``estimate_rejection_proba`` treats
this as a CDF estimator and calls ``algorithm.fit(X=inputs)``
(``p_values.py:285``).

Requires the ``ngboost`` package (``pip install ngboost``), pulled in through
``NGBoostCDFEstimator``, ``GammaLoc`` and ``Kumaraswamy``.
"""

import copy

import numpy as np

try:  # optional, only used to coerce tensor inputs
    import torch
except ImportError:  # pragma: no cover
    torch = None

# These three siblings are expected to live next to this module.
# Adjust the import paths if they live elsewhere in your package.
try:
    from lf2i.estimators import NGBoostCDFEstimator
except ImportError:  # pragma: no cover
    from cdf import NGBoostCDFEstimator

try:
    from lf2i.estimators import GammaLoc, Kumaraswamy
except ImportError:  # pragma: no cover
    GammaLoc, Kumaraswamy = None, None


class TwoStageCalibrationModel:
    """Two-stage conditional-CDF calibration with two NGBoost estimators.

    Stage 1: ``NGBoostCDFEstimator`` (default ``Dist=GammaLoc``) learns the
             global CDF F1(lambda ; theta).
    Stage 2: ``NGBoostCDFEstimator`` (default ``Dist=Kumaraswamy``) corrects the
             PIT residuals u = F1(lambda ; theta) ~= Uniform(0, 1) by modelling
             F2(u ; theta).
    Combined: F(lambda ; theta) = F2(F1(lambda ; theta) ; theta).

    Each stage may be supplied either as a ready-made ``NGBoostCDFEstimator``
    instance (``stage1`` / ``stage2``) or as a dict of constructor keyword
    arguments (``stage1_kwargs`` / ``stage2_kwargs``).  Supplied instances /
    kwargs are treated as templates and deep-copied at ``fit`` time, so the
    originals are never mutated and the model can be refit safely.

    Parameters
    ----------
    stage1 : NGBoostCDFEstimator, optional
        Estimator for the global stage-1 CDF F1(lambda ; theta).
    stage2 : NGBoostCDFEstimator, optional
        Estimator for the stage-2 residual correction F2(u ; theta).
    stage1_kwargs : dict, optional
        Constructor kwargs for stage 1 (used only if ``stage1`` is ``None``).
        ``Dist`` defaults to ``GammaLoc`` unless overridden here.
    stage2_kwargs : dict, optional
        Constructor kwargs for stage 2 (used only if ``stage2`` is ``None``).
        ``Dist`` defaults to ``Kumaraswamy`` unless overridden here -- pass e.g.
        ``{'Dist': make_kumaraswamy_mixture(3)}`` for a 3-component mixture.
    clip_eps : float, default 1e-6
        The PIT values u feeding stage 2 are clipped to ``[clip_eps,
        1 - clip_eps]`` so the (0, 1) stage-2 response never lands exactly on a
        boundary (where Kumaraswamy's log-density is singular).  Set to 0 to
        disable clipping.
    """

    def __init__(
        self,
        stage1=None,
        stage2=None,
        *,
        stage1_kwargs=None,
        stage2_kwargs=None,
        clip_eps=1e-6,
    ):
        if stage1 is not None and stage1_kwargs is not None:
            raise ValueError("Pass either `stage1` or `stage1_kwargs`, not both.")
        if stage2 is not None and stage2_kwargs is not None:
            raise ValueError("Pass either `stage2` or `stage2_kwargs`, not both.")
        if not 0.0 <= clip_eps < 0.5:
            raise ValueError(f"clip_eps must be in [0, 0.5); received {clip_eps}.")

        if stage1 is not None:
            self._stage1_template = stage1
        else:
            kw = dict(stage1_kwargs or {})
            kw.setdefault("Dist", GammaLoc)
            self._stage1_template = NGBoostCDFEstimator(**kw)

        if stage2 is not None:
            self._stage2_template = stage2
        else:
            kw = dict(stage2_kwargs or {})
            kw.setdefault("Dist", Kumaraswamy)
            self._stage2_template = NGBoostCDFEstimator(**kw)

        self.clip_eps = clip_eps

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X):
        if torch is not None and isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return np.asarray(X, dtype=float)

    def _clip_pit(self, u):
        if self.clip_eps <= 0.0:
            return u
        return np.clip(u, self.clip_eps, 1.0 - self.clip_eps)

    # ------------------------------------------------------------------
    # estimator interface
    # ------------------------------------------------------------------

    def fit(self, X):
        """Fit the two-stage model.

        Parameters
        ----------
        X : array-like of shape (n, 1 + poi_dim)
            Column 0 holds the test statistics lambda; columns 1: hold theta.
        """
        X = self._to_numpy(X)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError(
                "X must be 2-D with shape (n, 1 + poi_dim) where poi_dim >= 1. "
                f"Received shape {X.shape}."
            )
        theta = X[:, 1:]

        # Stage 1: global CDF F1(lambda ; theta).
        self.stage1_ = copy.deepcopy(self._stage1_template)
        self.stage1_.fit(X)

        # PIT values u = F1(lambda ; theta) ~= Uniform(0, 1); stage-2 response.
        u = self._clip_pit(self.stage1_.predict_proba(X)[:, 1])
        X_stage2 = np.hstack([u.reshape(-1, 1), theta])

        # Stage 2: residual correction F2(u ; theta).
        self.stage2_ = copy.deepcopy(self._stage2_template)
        self.stage2_.fit(X_stage2)
        return self

    def predict_proba(self, X):
        """Evaluate the combined CDF F(lambda ; theta) = F2(F1(lambda ; theta) ; theta).

        Parameters
        ----------
        X : array-like of shape (n, 1 + poi_dim)
            Column 0 holds lambda; columns 1: hold theta.

        Returns
        -------
        np.ndarray of shape (n, 2)
            ``[:, 0]`` = 1 - F(lambda ; theta)   (right-tail p-value)
            ``[:, 1]`` =     F(lambda ; theta)   (combined conditional CDF)
            Rows sum to 1.
        """
        X = self._to_numpy(X)
        u = self._clip_pit(self.stage1_.predict_proba(X)[:, 1])
        X_stage2 = np.hstack([u.reshape(-1, 1), X[:, 1:]])
        return self.stage2_.predict_proba(X_stage2)  # [1 - F_corrected, F_corrected]

    def predict_cdf(self, X):
        """Convenience wrapper returning the combined CDF F(lambda ; theta)."""
        return self.predict_proba(X)[:, 1]
