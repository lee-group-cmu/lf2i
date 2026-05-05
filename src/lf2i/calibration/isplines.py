"""
Monotone I-spline CDF classifier for p-value estimation.

Fits P(reject | cutoff, poi) where the output is guaranteed non-decreasing
in the cutoff dimension (column 0 of X) via an I-spline basis.

Sklearn-compatible interface: .fit(X, y) / .predict_proba(X).
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


# ---------------------------------------------------------------------------
# I-spline basis
# ---------------------------------------------------------------------------

def _bspline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Compute B-spline basis matrix via the Cox-de Boor recursion.

    Parameters
    ----------
    x      : (N,) query points, assumed in [knots[0], knots[-1]]
    knots  : (K,) knot sequence (including boundary repeats)
    degree : spline degree (3 = cubic)

    Returns
    -------
    B : (N, n_basis) where n_basis = len(knots) - degree - 1
    """
    n_basis = len(knots) - degree - 1
    N = len(x)

    # Degree-0 indicator basis
    B = torch.zeros(N, len(knots) - 1, device=x.device, dtype=x.dtype)
    for i in range(len(knots) - 1):
        if i < len(knots) - 2:
            B[:, i] = ((x >= knots[i]) & (x < knots[i + 1])).float()
        else:  # clamp rightmost point into last interval
            B[:, i] = ((x >= knots[i]) & (x <= knots[i + 1])).float()

    # Recurse up to target degree
    for d in range(1, degree + 1):
        B_new = torch.zeros(N, len(knots) - d - 1, device=x.device, dtype=x.dtype)
        for i in range(len(knots) - d - 1):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            t1 = (x - knots[i]) / denom1 * B[:, i] if denom1 > 0 else torch.zeros(N, device=x.device)
            t2 = (knots[i + d + 1] - x) / denom2 * B[:, i + 1] if denom2 > 0 else torch.zeros(N, device=x.device)
            B_new[:, i] = t1 + t2
        B = B_new

    return B[:, :n_basis]


def _ispline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    I-spline basis = cumulative sum of B-spline basis (Ramsay 1988).
    Each column is non-decreasing in x; all values in [0, 1].

    Returns
    -------
    I : (N, n_basis)
    """
    B = _bspline_basis(x, knots, degree)          # (N, n_basis)
    # Cumulative sum across basis functions (left to right = increasing x support)
    I = torch.cumsum(B, dim=1)
    # Normalize so each column reaches 1 at the right boundary
    I = I / (I.max(dim=0).values.clamp(min=1e-8))
    return I


def _make_knots(x: torch.Tensor, n_interior: int, degree: int) -> torch.Tensor:
    """Quantile-spaced interior knots with `degree`-fold boundary repeats."""
    quantiles = torch.linspace(0, 100, n_interior + 2)[1:-1].tolist()
    interior = torch.tensor(
        np.percentile(x.cpu().numpy(), quantiles), dtype=x.dtype, device=x.device
    )
    lo, hi = x.min().unsqueeze(0), x.max().unsqueeze(0)
    boundary_lo = lo.repeat(degree + 1)
    boundary_hi = hi.repeat(degree + 1)
    return torch.cat([boundary_lo, interior, boundary_hi])


# ---------------------------------------------------------------------------
# Torch module
# ---------------------------------------------------------------------------

class ISplineLogisticModel(nn.Module):
    """
    P(reject=1 | cutoff, poi) = sigmoid( I(cutoff) @ softplus(alpha(poi)) + beta(poi) )

    - I(cutoff) : (N, K) I-spline features — non-decreasing in cutoff
    - alpha(poi): (N, K) non-negative weights via softplus  -> monotone in cutoff
    - beta(poi) : (N,)   intercept

    Both alpha and beta are predicted by a small MLP over the POI features.
    """

    def __init__(
        self,
        n_poi: int,
        n_splines: int = 8,
        degree: int = 3,
        hidden_dim: int = 32,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.n_splines = n_splines
        self.degree = degree
        self.register_buffer("knots", torch.zeros(1))   # set at fit time

        # MLP: poi -> (alpha_raw (K), beta (1))
        layers = [nn.Linear(max(n_poi, 1), hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, n_splines + 1)]  # K alphas + 1 beta
        self.mlp = nn.Sequential(*layers)

    def _set_knots(self, cutoffs: torch.Tensor):
        knots = _make_knots(cutoffs, n_interior=self.n_splines - self.degree - 1, degree=self.degree)
        self.knots = knots.to(cutoffs.device)

    def forward(self, cutoffs: torch.Tensor, poi: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cutoffs : (N,)
        poi     : (N, n_poi)  — pass zeros tensor if no POI features

        Returns
        -------
        logits : (N,)
        """
        I = _ispline_basis(cutoffs, self.knots, self.degree)       # (N, K)
        out = self.mlp(poi)                                         # (N, K+1)
        alpha_raw, beta = out[:, :-1], out[:, -1]                  # (N,K), (N,)
        alpha = torch.nn.functional.softplus(alpha_raw)            # non-negative -> monotone
        logits = (I * alpha).sum(dim=1) + beta                     # (N,)
        return logits


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class ISplineClassifier:
    """
    Sklearn-style wrapper around ISplineLogisticModel.

    Column 0 of X is treated as the cutoff (monotone dimension).
    Remaining columns are POI features passed through the MLP.

    Parameters
    ----------
    n_splines            : number of I-spline basis functions
    degree               : spline degree (3 = cubic)
    hidden_dim           : MLP hidden layer width
    n_hidden             : number of MLP hidden layers
    lr                   : Adam learning rate
    epochs               : training epochs
    batch_size           : mini-batch size
    monotone_constraints : int, either +1 (non-decreasing in cutoff) or -1 (non-increasing).
                           Mirrors CatBoost's monotone_constraints convention.
                           Default is +1.
    device               : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_splines: int = 8,
        degree: int = 3,
        hidden_dim: int = 32,
        n_hidden: int = 2,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 512,
        monotone_constraints: int = 1,
        device: Optional[str] = None,
    ):
        if monotone_constraints not in (1, -1):
            raise ValueError(f"monotone_constraints must be 1 or -1, got {monotone_constraints}")
        self.n_splines = n_splines
        self.degree = degree
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.monotone_constraints = monotone_constraints
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: Optional[ISplineLogisticModel] = None
        self.classes_ = np.array([0, 1])

    def _to_tensors(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        cutoffs = X_t[:, 0]
        poi = X_t[:, 1:] if X_t.shape[1] > 1 else torch.zeros(len(X_t), 1, device=self.device)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
            return cutoffs, poi, y_t
        return cutoffs, poi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ISplineClassifier":
        cutoffs, poi, y_t = self._to_tensors(X, y)
        n_poi = poi.shape[1]

        self.model_ = ISplineLogisticModel(
            n_poi=n_poi,
            n_splines=self.n_splines,
            degree=self.degree,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
        ).to(self.device)

        self.model_._set_knots(cutoffs)
        optimizer = Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        N = len(y_t)
        self.model_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                idx = perm[start: start + self.batch_size]
                logits = self.monotone_constraints * self.model_(cutoffs[idx], poi[idx])
                loss = loss_fn(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None, "Call .fit() first."
        cutoffs, poi = self._to_tensors(X)
        with torch.no_grad():
            logits = self.monotone_constraints * self.model_(cutoffs, poi)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)