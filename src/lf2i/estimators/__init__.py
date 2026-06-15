# ── Protocols / ABCs (always available) ──────────────────────────────────────
from lf2i.estimators.base_probabilistic_classifier import AbstractProbabilisticClassifier
from lf2i.estimators.base_cdf import AbstractCDFEstimator
from lf2i.estimators.base_quantile_regressor import AbstractQuantileRegressor

# ── Torch-based implementations (always available) ───────────────────────────
from lf2i.estimators.torch_utils.cdf import (
    ParametricCDFEstimator,
    BrierScoreLoss,
    QuantileWeightedCRPSLoss,
    SigmoidCDF,
    BetaNetwork,
)

# ── FlexCode-based implementations (optional: requires flexcode) ──────────────
try:
    from lf2i.estimators.flexcode_utils.cdf import FlexCodeCDFEstimator
    from lf2i.estimators.flexcode_utils.two_stage_cdf import TwoStageCalibrationModel
    HAVE_FLEXCODE = True
except ImportError:
    HAVE_FLEXCODE = False

# ── NGBoost-based implementations (optional: requires ngboost) ────────────────
try:
    from lf2i.estimators.ngboost_utils.cdf import NGBoostCDFEstimator
    from lf2i.estimators.ngboost_utils.two_stage_cdf import TwoStageCalibrationModel as NGBoostTwoStageCalibrationModel
    from lf2i.estimators.ngboost_utils.distns.gamma_loc import GammaLoc, GammaLocLogScore
    from lf2i.estimators.ngboost_utils.distns.kumaraswamy_mixture import Kumaraswamy
    HAVE_NGBOOST = True
except ImportError:
    HAVE_NGBOOST = False
