import numpy as np

try:
    from flexcode import FlexCodeModel
    from flexcode.regression_models import XGBoost as _FCXGBoost, RandomForest as _FCRandomForest
    HAVE_FLEXCODE = True
except ImportError:
    FlexCodeModel = None
    _FCXGBoost = None
    _FCRandomForest = None
    HAVE_FLEXCODE = False

from lf2i.estimators.flexcode_utils.cdf import FlexCodeCDFEstimator
from lf2i.estimators.torch_utils.cdf import ParametricCDFEstimator

_FLEXCODE_MODELS = {'XGBoost': _FCXGBoost, 'RandomForest': _FCRandomForest}


class TwoStageCalibrationModel:
    """
    Stage 1: ParametricCDFEstimator (sigmoid) learns global CDF F₁(λ; θ).
    Stage 2: FlexCodeCDFEstimator corrects residuals u = F₁(λ; θ) ≈ Uniform(0,1).
    Combined: F(λ; θ) = F₂(F₁(λ; θ); θ).

    fit(X) has no y parameter → estimate_rejection_proba treats this as a
    CDF estimator and calls algorithm.fit(X=inputs) (p_values.py:285).
    Always called in acceptance_region='right' context (NACS flips first).

    Requires the ``flexcode`` package (``pip install flexcode``).
    """

    def __init__(self, stage1_kwargs=None, stage2_kwargs=None):
        if not HAVE_FLEXCODE:
            raise ImportError(
                "TwoStageCalibrationModel requires the 'flexcode' package. "
                "Install it with: pip install flexcode"
            )
        self.stage1_kwargs = stage1_kwargs or {}
        self.stage2_kwargs = stage2_kwargs or {}

    def fit(self, X):
        # X: (n_augmented, 1 + poi_dim), X[:,0]=λ, X[:,1:]=θ
        self.stage1_ = ParametricCDFEstimator(**self.stage1_kwargs)
        self.stage1_.fit(X=X)

        u = self.stage1_.predict_proba(X)[:, 1]  # CDF values, ≈ Uniform(0,1)
        X_stage2 = np.hstack([u.reshape(-1, 1), X[:, 1:]])

        regression_cls = _FLEXCODE_MODELS[self.stage2_kwargs.get('regression_model', 'XGBoost')]
        max_basis = self.stage2_kwargs.get('max_basis', 31)
        self.stage2_ = FlexCodeCDFEstimator(FlexCodeModel(
            model=regression_cls,
            max_basis=max_basis,
            basis_system='db4',
            z_min=0.0,
            z_max=1.0,
        ))
        self.stage2_.fit(X_stage2)
        return self

    def predict_proba(self, X):
        u = self.stage1_.predict_proba(X)[:, 1]
        X_stage2 = np.hstack([u.reshape(-1, 1), X[:, 1:]])
        return self.stage2_.predict_proba(X_stage2)  # [1−F_corrected, F_corrected]
