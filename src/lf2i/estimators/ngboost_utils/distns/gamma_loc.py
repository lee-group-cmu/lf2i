"""Location-varying (3-parameter) Gamma distribution for NGBoost.

Extends the standard 2-parameter Gamma distribution in ngboost/distns/gamma.py
by adding a free location (shift) parameter, making it suitable for data that
is not anchored at zero.

The distribution models Y ~ GammaLoc(alpha, beta, loc), valid for Y > loc:

    f(y; α, β, loc) = β^α / Γ(α) · (y − loc)^(α−1) · exp(−β(y − loc))

Parameters are stored in raw (unconstrained) form:
    params[0]  →  loc   (free, unbounded location/shift)
    params[1]  →  log(α)  →  α = exp(params[1]) > 0  (shape)
    params[2]  →  log(β)  →  β = exp(params[2]) > 0  (rate; scale = 1/β)
"""

import numpy as np
import scipy as sp
from scipy.stats import gamma

try:
    from ngboost.distns.distn import RegressionDistn
    from ngboost.scores import LogScore
    HAVE_NGBOOST = True
except ImportError:
    RegressionDistn = object   # fallback so class body parses without ngboost
    LogScore = object
    HAVE_NGBOOST = False


class GammaLocLogScore(LogScore):
    """Log score (negative log-likelihood) for the location-varying Gamma.

    Analytical gradients w.r.t. the three raw parameters (loc, log α, log β)
    are provided in ``d_score``.  The Fisher-information metric falls back to
    the Monte-Carlo estimator inherited from ``LogScore.metric()``, which
    avoids the complexity of the full 3×3 analytical FIM for the shifted Gamma.
    """

    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        """Gradient of −log f w.r.t. each raw parameter.

        Let y_s = Y − loc  (clamped away from zero for numerical safety).

        Derivation from  −log f = −α log β + log Γ(α) − (α−1) log y_s + β y_s:

            ∂(−log f)/∂loc        = (α−1)/y_s  −  β          [chain rule: ∂y_s/∂loc = −1]
            ∂(−log f)/∂(log α)    = α · (ψ(α) − log(β · y_s))
            ∂(−log f)/∂(log β)    = β · y_s  −  α

        where ψ denotes the digamma function.
        """
        D = np.zeros((len(Y), 3))
        y_s = np.maximum(Y - self.loc, self.eps)

        # ∂(−log f) / ∂ loc
        D[:, 0] = (self.alpha - 1.0) / y_s - self.beta

        # ∂(−log f) / ∂ (log α)
        D[:, 1] = self.alpha * (
            sp.special.digamma(self.alpha) - np.log(self.eps + self.beta * y_s)
        )

        # ∂(−log f) / ∂ (log β)
        D[:, 2] = self.beta * y_s - self.alpha

        return D

    # metric() is intentionally inherited from LogScore (Monte-Carlo estimation).


class GammaLoc(RegressionDistn):
    """Location-varying (3-parameter) Gamma distribution for NGBoost.

    Parameters
    ----------
    params : array-like of shape (3, n_samples)
        Raw (unconstrained) parameter vectors passed by the NGBoost engine:
          params[0]  —  loc    (free location shift)
          params[1]  —  log α  (log shape;  α = exp(params[1]) > 0)
          params[2]  —  log β  (log rate;   β = exp(params[2]) > 0)

    Notes
    -----
    The distribution is valid for Y > loc.  Unlike the base ``Gamma``
    distribution (which fixes loc = 0), this variant allows the support to
    shift freely during gradient boosting.

    Only ``LogScore`` is implemented.  To use this distribution::

        from ngboost import NGBRegressor
        from REFERENCE.gamma_loc import GammaLoc

        model = NGBRegressor(Dist=GammaLoc)
        model.fit(X_train, Y_train)
    """

    n_params = 3
    scores = [GammaLocLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.alpha = np.exp(params[1])
        self.beta = np.exp(params[2])
        self.dist = gamma(
            a=self.alpha,
            loc=self.loc,
            scale=1.0 / self.beta,
        )
        self.eps = 1e-10

    @staticmethod
    def fit(Y):
        """Fit initial parameters to Y using unconstrained scipy MLE.

        Unlike the base ``Gamma.fit``, the location is not fixed at 0.

        Returns
        -------
        np.ndarray of shape (3,)
            [loc, log(α), log(β)]
        """
        a, loc, scale = gamma.fit(Y)
        return np.array([loc, np.log(a), np.log(1.0 / scale)])

    def sample(self, m):
        return np.array([self.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "alpha": self.alpha, "beta": self.beta}


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from ngboost import NGBRegressor

    rng = np.random.default_rng(0)
    n, p = 300, 4
    X = rng.standard_normal((n, p))
    # Shifted Gamma: shape=2, rate=1, loc=3  →  Y in (3, ∞)
    Y = rng.gamma(shape=2.0, scale=1.0, size=n) + 3.0

    model = NGBRegressor(Dist=GammaLoc, n_estimators=50, verbose=False)
    model.fit(X, Y)

    preds = model.predict(X[:5])
    print("GammaLoc smoke test — point predictions (should be ~5):", preds)

    dist_pred = model.pred_dist(X[:3])
    cdf_vals = dist_pred.cdf(Y[:3])
    print("GammaLoc CDF at training points:", cdf_vals)
    print("GammaLoc smoke test PASSED.")
