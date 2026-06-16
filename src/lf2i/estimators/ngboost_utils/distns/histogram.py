"""Fixed-bin histogram distribution on [0, 1] for NGBoost.

Bins are equally spaced on [0, 1] with width ``1/n_bins``.  The ``n_bins``
bin heights are learnable but fixed in count: only the heights (probabilities)
are updated during boosting, not the bin positions.

Parameterisation
----------------
``n_bins - 1`` unconstrained logit parameters.  A softmax with the last logit
pinned at 0 maps them to the ``n_bins`` bin probabilities ``h_k``:

    Z = Σ_{j < K-1} exp(logit_j) + 1
    h_j = exp(logit_j) / Z   (j < K-1),    h_{K-1} = 1 / Z

The density inside bin k = ⌊y · K⌋ is

    f(y; h) = h_k · K,    y ∈ [k/K, (k+1)/K)

Negative log-likelihood and its gradient are computed analytically:

    NLL(y)         = −log(h_{bin(y)}) − log K
    ∂NLL/∂logit_j = h_j − 𝟙[bin(y) == j]

The Fisher-information metric is inherited from ``LogScore`` (Monte-Carlo
approximation), avoiding the need to implement the full (K−1)×(K−1) FIM.

Usage
-----
    from ngboost import NGBRegressor
    from histogram import make_histogram_distn, Histogram10

    ngb = NGBRegressor(Dist=Histogram10)

    # custom bin count
    H20 = make_histogram_distn(n_bins=20)
    ngb = NGBRegressor(Dist=H20)
"""

import numpy as np

try:
    from ngboost.distns.distn import RegressionDistn
    from ngboost.scores import LogScore
    HAVE_NGBOOST = True
except ImportError:
    RegressionDistn = object
    LogScore = object
    HAVE_NGBOOST = False


def make_histogram_distn(n_bins=20, name=None):
    """Create a fixed-bin histogram NGBoost distribution on [0, 1].

    Parameters
    ----------
    n_bins : int, default 10
        Number of equally-spaced bins on [0, 1].  Must be >= 2.
    name : str or None
        Class name for the returned distribution.  Defaults to
        ``"Histogram{n_bins}"``.

    Returns
    -------
    type
        A ``RegressionDistn`` subclass with ``n_bins - 1`` learnable logit
        parameters, ready for ``NGBRegressor(Dist=...)``.
    """
    if not HAVE_NGBOOST:
        raise ImportError(
            "make_histogram_distn requires the 'ngboost' package. "
            "Install it with: pip install ngboost"
        )
    K = int(n_bins)
    if K < 2:
        raise ValueError(f"n_bins must be >= 2; received {n_bins}.")
    n_p = K - 1
    cls_name = name or f"Histogram{K}"

    class HistogramLogScore(LogScore):
        def score(self, Y):
            """NLL per observation, shape (n_obs,)."""
            return -self._hist_logpdf(Y)

        def d_score(self, Y):
            """Gradient of NLL w.r.t. logits, shape (n_obs, n_params)."""
            Y = np.asarray(Y, float)
            w = self._softmax_weights()            # (K, n_obs)
            bin_idx = np.clip((Y * K).astype(int), 0, K - 1)   # (n_obs,)
            # indicator[i, j] = 1 if bin_of(Y[i]) == j, else 0
            indicator = (bin_idx[:, None] == np.arange(n_p)[None, :]).astype(float)
            # gradient[i, j] = h_j[i] - indicator[i, j]
            return w[:n_p].T - indicator                         # (n_obs, n_p)

        # metric() inherited from LogScore (Monte-Carlo Fisher approximation)

    class HistogramDistn(RegressionDistn):
        n_params = n_p
        scores = [HistogramLogScore]

        def __init__(self, params):
            # params: (n_params, n_obs)
            super().__init__(params)

        def _softmax_weights(self):
            """Bin probabilities via softmax; last logit pinned at 0."""
            logits = np.clip(self._params, -700, 700)   # (n_p, n_obs)
            exps = np.exp(logits)
            Z = exps.sum(axis=0, keepdims=True) + 1.0
            return np.vstack([exps / Z, 1.0 / Z])       # (K, n_obs)

        def _hist_logpdf(self, Y):
            Y = np.asarray(Y, float)
            w = self._softmax_weights()                  # (K, n_obs)
            n_obs = w.shape[1]
            bin_idx = np.clip((Y * K).astype(int), 0, K - 1)
            h = w[bin_idx, np.arange(n_obs)]            # (n_obs,)
            return np.log(np.maximum(h * K, 1e-300))

        def cdf(self, y):
            yv = np.clip(np.atleast_1d(np.asarray(y, float)), 0.0, 1.0)
            w = self._softmax_weights()                  # (K, n_obs)
            n_obs = w.shape[1]
            # cumw[k, :] = sum of the first k bin probabilities
            cumw = np.vstack([np.zeros((1, n_obs)), np.cumsum(w, axis=0)])
            bin_idx = np.clip((yv * K).astype(int), 0, K - 1)   # (n_y,)
            frac = np.clip((yv - bin_idx / K) * K, 0.0, 1.0)    # fractional position within bin
            if len(yv) == n_obs:
                idx = np.arange(n_obs)
                return np.clip(cumw[bin_idx, idx] + frac * w[bin_idx, idx], 0.0, 1.0)
            # general (n_y, n_obs) broadcast
            out = cumw[bin_idx, :] + frac[:, None] * w[bin_idx, :]
            return np.clip(out, 0.0, 1.0)

        def pdf(self, y):
            yv = np.atleast_1d(np.asarray(y, float))
            w = self._softmax_weights()                  # (K, n_obs)
            bin_idx = np.clip((yv * K).astype(int), 0, K - 1)
            return w[bin_idx] * K

        def mean(self):
            w = self._softmax_weights()                  # (K, n_obs)
            mids = (np.arange(K) + 0.5) / K             # bin midpoints
            return (mids[:, None] * w).sum(axis=0)

        def sample(self, m):
            w = self._softmax_weights()                  # (K, n_obs)
            n_obs = w.shape[1]
            cumw = np.cumsum(w, axis=0)                  # (K, n_obs)
            out = np.zeros((m, n_obs))
            for s in range(m):
                u = np.random.random(n_obs)
                bin_k = np.clip((u[None, :] >= cumw).sum(axis=0), 0, K - 1)
                u2 = np.random.random(n_obs)
                out[s] = (bin_k + u2) / K
            return out

        @staticmethod
        def fit(Y):
            """Initialise logits from empirical bin counts of Y."""
            Y = np.asarray(Y, float)
            bin_idx = np.clip((Y * K).astype(int), 0, K - 1)
            counts = np.bincount(bin_idx, minlength=K).astype(float)
            counts = np.maximum(counts, 1.0)             # Laplace smoothing
            log_p = np.log(counts / counts.sum())
            # logit_j = log(h_j) - log(h_{K-1})
            return log_p[:n_p] - log_p[-1]

    HistogramDistn.__name__ = cls_name
    HistogramDistn.__qualname__ = cls_name
    HistogramLogScore.__name__ = f"{cls_name}LogScore"
    HistogramLogScore.__qualname__ = f"{cls_name}LogScore"

    return HistogramDistn


if HAVE_NGBOOST:
    Histogram20 = make_histogram_distn(20, name="Histogram20")
else:
    Histogram20 = None
