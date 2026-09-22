"""General mixture of Kumaraswamy distributions for NGBoost.

This module builds an NGBoost ``RegressionDistn`` for a ``K``-component mixture
of Kumaraswamy distributions on the open interval ``(0, 1)``.  It follows the
same SymPy-factory pattern used in ``example_mixture_normal.ipynb`` (the
mixture PDF and its negative log-likelihood are written symbolically and handed
to ``ngboost.distns.sympy_utils.make_distribution``, which auto-generates the
score, gradients, Fisher-information metric, ``fit`` and ``sample``).  It
defaults to a single component (``n_components=1``), i.e. a plain Kumaraswamy.

Why Kumaraswamy
---------------
The Kumaraswamy distribution is a Beta-like law on ``(0, 1)`` whose CDF (and
inverse CDF) are available in closed form, which makes it a natural conditional
model for probability-integral-transform (PIT) values -- exactly the kind of
``(0, 1)``-valued response that arises in the residual-correction stage of
two-stage CDF calibration.

Single component ``Kumaraswamy(a, b)`` on ``(0, 1)``:

    pdf:  f(y; a, b) = a · b · y^(a−1) · (1 − y^a)^(b−1)
    cdf:  F(y; a, b) = 1 − (1 − y^a)^b
    ppf:  F⁻¹(q)     = (1 − (1 − q)^(1/b))^(1/a)
    mean: E[Y]       = b · B(1 + 1/a, b)

``K``-component mixture with weights ``w_k`` (softmax, last logit fixed at 0):

    f(y) = Σ_k w_k · f(y; a_k, b_k)
    F(y) = Σ_k w_k · [1 − (1 − y^{a_k})^{b_k}]

Parameterisation (internal / unconstrained)
--------------------------------------------
Per component ``k`` the shape parameters use a log link (so they stay
positive):  ``a_k = exp(log_a_k)``, ``b_k = exp(log_b_k)``.  The ``K − 1``
mixture logits use an identity link and are mapped to weights by a softmax
whose ``K``-th logit is pinned at 0:

    Z = Σ_{j<K} exp(logit_w_j) + 1
    w_j = exp(logit_w_j) / Z   (j < K),     w_K = 1 / Z

For ``K = 1`` there are no logits and ``w_1 = 1``.

Why the subclass
----------------
``make_distribution`` only exposes a ``.cdf`` method when it is given a
``scipy.stats`` class to delegate to.  A mixture has no such scipy class, so
the factory-generated base class is subclassed here to add the analytic
``cdf`` / ``pdf`` / ``ppf`` (vectorised over the batch).  The ``cdf`` method is
what downstream CDF-estimator code (e.g. ``NGBoostCDFEstimator.predict_proba``,
which calls ``pred.cdf(...)``) relies on.

Usage
-----
    from ngboost import NGBRegressor
    from kumaraswamy_mixture import Kumaraswamy, make_kumaraswamy_mixture

    # single-component default
    ngb = NGBRegressor(Dist=Kumaraswamy)

    # three-component mixture
    Mix3 = make_kumaraswamy_mixture(n_components=3)
    ngb = NGBRegressor(Dist=Mix3)

Requires the ``ngboost`` package (``pip install ngboost``).
"""

import numpy as np
import sympy as sp
from scipy.special import beta as _beta_fn

try:
    from ngboost.distns.sympy_utils import make_distribution
    HAVE_NGBOOST = True
except ImportError:  # fallback so the module imports without ngboost
    make_distribution = None
    HAVE_NGBOOST = False


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _kumaraswamy_pdf(y, a, b):
    """Symbolic / numeric Kumaraswamy density a·b·y^(a−1)·(1−y^a)^(b−1)."""
    return a * b * y ** (a - 1) * (1 - y ** a) ** (b - 1)


def _softmax_weights(logit_arrays, n_components, ref_shape):
    """Mixture weights from ``K − 1`` logit arrays (last logit fixed at 0).

    Returns a list of ``n_components`` weight arrays, each broadcastable to
    ``ref_shape``.  For a single component this is just ``[ones(ref_shape)]``.
    """
    if n_components == 1:
        return [np.ones(ref_shape)]
    exps = [np.exp(np.clip(np.asarray(l, float), -700, 700)) for l in logit_arrays]
    Z = np.ones(ref_shape)
    for e in exps:
        Z = Z + e
    weights = [e / Z for e in exps]
    weights.append(np.ones(ref_shape) / Z)
    return weights


# ----------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------

def make_kumaraswamy_mixture(n_components=1, name=None):
    """Create an NGBoost distribution for a ``K``-component Kumaraswamy mixture.

    Parameters
    ----------
    n_components : int, default 1
        Number of Kumaraswamy components ``K``.  ``K = 1`` yields a plain
        Kumaraswamy distribution.
    name : str or None
        Class name for the generated distribution.  Defaults to
        ``"Kumaraswamy"`` for ``K = 1`` and ``"KumaraswamyMixture{K}"``
        otherwise.

    Returns
    -------
    type
        A ``RegressionDistn`` subclass ready for ``NGBRegressor(Dist=...)``,
        carrying analytic ``cdf`` / ``pdf`` / ``ppf`` / ``mean`` / ``sample``
        methods and a ``n_components`` attribute.
    """
    if not HAVE_NGBOOST:
        raise ImportError(
            "make_kumaraswamy_mixture requires the 'ngboost' package. "
            "Install it with: pip install ngboost"
        )
    K = int(n_components)
    if K < 1:
        raise ValueError(f"n_components must be >= 1; received {n_components}.")
    name = name or (f"KumaraswamyMixture{K}" if K > 1 else "Kumaraswamy")

    # ---- symbols ----
    a_syms = [sp.symbols(f"a{k + 1}", positive=True) for k in range(K)]
    b_syms = [sp.symbols(f"b{k + 1}", positive=True) for k in range(K)]
    logit_syms = [sp.symbols(f"logit_w{j + 1}") for j in range(K - 1)]
    y = sp.symbols("y", positive=True)

    # ---- symbolic mixture weights (softmax, K-th logit fixed at 0) ----
    if K == 1:
        weights = [sp.Integer(1)]
    else:
        exps = [sp.exp(l) for l in logit_syms]
        Z = sum(exps) + 1
        weights = [e / Z for e in exps] + [1 / Z]

    # ---- symbolic mixture PDF and negative log-likelihood ----
    mixture_pdf = sum(
        weights[k] * _kumaraswamy_pdf(y, a_syms[k], b_syms[k]) for k in range(K)
    )
    score = -sp.log(mixture_pdf)
    # For a single component, expand log(product) into a sum of logs so that
    # y^(a-1) is never evaluated numerically (it overflows for a < 1, y ~ 0).
    # For mixtures the factory's logsumexp path expands each term internally,
    # so the score is left as -log(Σ_k ...) to keep that detection working.
    score_expr = sp.expand_log(score, force=True) if K == 1 else score

    # ---- parameter list (order fixes the internal-parameter layout) ----
    #   [log a_1 ... log a_K, log b_1 ... log b_K, logit_w_1 ... logit_w_{K-1}]
    params = (
        [(a, True) for a in a_syms]
        + [(b, True) for b in b_syms]
        + [(l, False) for l in logit_syms]
    )

    # ---- domain-aware initialisation (quantile split; a_k = 1 → Beta(1, b)) ----
    def fit_fn(Y):
        Y = np.clip(np.asarray(Y, float), 1e-6, 1 - 1e-6)
        sorted_Y = np.sort(Y)
        n = len(sorted_Y)
        log_a, log_b = [], []
        for k in range(K):
            grp = sorted_Y[k * n // K:(k + 1) * n // K]
            if grp.size == 0:
                grp = sorted_Y
            m = np.clip(np.mean(grp), 1e-3, 1 - 1e-3)
            # With a = 1, Kumaraswamy(1, b) = Beta(1, b) has mean 1/(1+b),
            # so b = (1 - mean) / mean gives a stable, data-adaptive start.
            log_a.append(0.0)
            log_b.append(np.log(max((1 - m) / m, 1e-6)))
        logits = [0.0] * (K - 1)
        return np.array(log_a + log_b + logits)

    # ---- sampling: pick a component, then invert the Kumaraswamy CDF ----
    def sample_fn(self, m):
        a, b, w = _gather(self)
        n_obs = a[0].shape[0]
        A = np.stack([np.broadcast_to(ai, (n_obs,)) for ai in a], axis=-1)
        B = np.stack([np.broadcast_to(bi, (n_obs,)) for bi in b], axis=-1)
        W = np.stack([np.broadcast_to(wi, (n_obs,)) for wi in w], axis=-1)
        cum = np.cumsum(W, axis=1)
        idx = np.arange(n_obs)
        out = np.zeros((m, n_obs))
        for s in range(m):
            comp = (np.random.random(n_obs)[:, None] < cum).argmax(axis=1)
            a_sel, b_sel = A[idx, comp], B[idx, comp]
            ru = np.random.random(n_obs)
            out[s] = (1 - (1 - ru) ** (1 / b_sel)) ** (1 / a_sel)
        return out

    # ---- analytic mean: Σ_k w_k · b_k · B(1 + 1/a_k, b_k) ----
    def mean_fn(self):
        a, b, w = _gather(self)
        out = np.zeros(a[0].shape)
        for k in range(K):
            out = out + w[k] * b[k] * _beta_fn(1 + 1 / a[k], b[k])
        return out

    Base = make_distribution(
        params=params,
        y=y,
        score_expr=score_expr,
        fit_fn=fit_fn,
        sample_fn=sample_fn,
        mean_fn=mean_fn,
        name=f"{name}Base",
    )

    # ---- analytic cdf / pdf / ppf added via subclass ----
    def cdf(self, y):
        yv = np.clip(np.asarray(y, float), 0.0, 1.0)
        a, b, w = _gather(self)
        out = np.zeros(np.broadcast_shapes(yv.shape, a[0].shape))
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in range(self.n_components):
                Fk = 1.0 - np.power(1.0 - np.power(yv, a[k]), b[k])
                out = out + w[k] * Fk
        return np.clip(out, 0.0, 1.0)

    def pdf(self, y):
        yv = np.asarray(y, float)
        a, b, w = _gather(self)
        out = np.zeros(np.broadcast_shapes(yv.shape, a[0].shape))
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            for k in range(self.n_components):
                fk = (a[k] * b[k] * np.power(yv, a[k] - 1)
                      * np.power(1 - np.power(yv, a[k]), b[k] - 1))
                out = out + w[k] * fk
        return out

    def ppf(self, q):
        # Closed form for K = 1; vectorised bisection on the analytic CDF
        # for mixtures (no closed-form inverse).
        a, b, _ = _gather(self)
        shape = a[0].shape
        qv = np.broadcast_to(np.asarray(q, float), shape).astype(float)
        if self.n_components == 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                return (1 - (1 - qv) ** (1 / b[0])) ** (1 / a[0])
        lo = np.zeros(shape)
        hi = np.ones(shape)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            Fm = self.cdf(mid)
            hi = np.where(Fm >= qv, mid, hi)
            lo = np.where(Fm < qv, mid, lo)
        return 0.5 * (lo + hi)

    cls = type(name, (Base,), {
        "n_components": K,
        "cdf": cdf,
        "pdf": pdf,
        "ppf": ppf,
    })
    return cls


def _gather(self):
    """Collect per-component (a, b) arrays and mixture weights from ``self``."""
    K = self.n_components
    a = [np.atleast_1d(np.asarray(getattr(self, f"a{k + 1}"), float)) for k in range(K)]
    b = [np.atleast_1d(np.asarray(getattr(self, f"b{k + 1}"), float)) for k in range(K)]
    logits = [np.atleast_1d(np.asarray(getattr(self, f"logit_w{j + 1}"), float))
              for j in range(K - 1)]
    w = _softmax_weights(logits, K, a[0].shape)
    return a, b, w


# Single-component default (plain Kumaraswamy on (0, 1)).
if HAVE_NGBOOST:
    Kumaraswamy = make_kumaraswamy_mixture(1, name="Kumaraswamy")
else:  # keep the name importable without ngboost
    Kumaraswamy = None
