"""
Monotone I-spline CDF classifier with POI-conditional knots.

Fits P(reject | cutoff, poi) where:
  - the output is guaranteed non-decreasing in the cutoff dimension
  - the I-spline knot locations themselves vary with the POI

Sklearn-compatible interface: .fit(X, y) / .predict_proba(X).
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Batched I-spline basis (per-sample knot vectors)
# ---------------------------------------------------------------------------

def _bspline_basis_batched(
    x: torch.Tensor,        # (N,)
    knots: torch.Tensor,    # (N, K_total)  — sorted, per-sample
    degree: int,
) -> torch.Tensor:
    """
    Batched Cox-de Boor recursion. Each sample i uses its own knot vector knots[i].

    Returns
    -------
    B : (N, n_basis) where n_basis = K_total - degree - 1
    """
    N, K_total = knots.shape
    n_basis = K_total - degree - 1

    # Degree-0 indicators: B[:, i] = 1 if knots[:, i] <= x < knots[:, i+1]
    x_ = x.unsqueeze(1)                                  # (N, 1)
    left = knots[:, :-1]                                 # (N, K_total-1)
    right = knots[:, 1:]                                 # (N, K_total-1)
    B = ((x_ >= left) & (x_ < right)).to(x.dtype)        # (N, K_total-1)
    # Clamp the rightmost endpoint into the last interval
    rightmost_in = (x == knots[:, -1])
    if rightmost_in.any():
        B[rightmost_in, -1] = 1.0

    # Recurse up to target degree
    for d in range(1, degree + 1):
        K_d = K_total - d - 1                            # length after this step
        denom1 = knots[:, d:d + K_d] - knots[:, :K_d]     # (N, K_d)
        denom2 = knots[:, d + 1:d + 1 + K_d] - knots[:, 1:1 + K_d]
        # Avoid divide-by-zero (only happens if knots coincide)
        safe1 = denom1.clamp(min=1e-12)
        safe2 = denom2.clamp(min=1e-12)
        t1 = (x_ - knots[:, :K_d]) / safe1 * B[:, :K_d]
        t2 = (knots[:, d + 1:d + 1 + K_d] - x_) / safe2 * B[:, 1:1 + K_d]
        # Zero contributions where the denominator was actually zero
        t1 = torch.where(denom1 > 0, t1, torch.zeros_like(t1))
        t2 = torch.where(denom2 > 0, t2, torch.zeros_like(t2))
        B = t1 + t2

    return B[:, :n_basis]


def _ispline_basis_batched(
    x: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Batched I-spline basis = cumulative sum of B-spline basis along the basis dim.
    Each row is non-decreasing in x. Values lie in [0, 1] up to a per-row scale.
    """
    B = _bspline_basis_batched(x, knots, degree)         # (N, n_basis)
    I = torch.cumsum(B, dim=1)
    # Per-sample normalization so the rightmost basis function tops out near 1
    I = I / I[:, -1:].clamp(min=1e-8)
    return I


# ---------------------------------------------------------------------------
# Torch module
# ---------------------------------------------------------------------------

class ISplineHyperModel(nn.Module):
    """
    P(reject=1 | cutoff, poi) = sigmoid( I(cutoff; knots(poi)) @ softplus(alpha(poi)) + beta(poi) )

    A single MLP outputs three things from the POI vector:
      - log-gaps between knots (turned into sorted knots via cumulative softplus,
        rescaled to span [cutoff_min, cutoff_max])
      - non-negative spline weights alpha (via softplus)
      - intercept beta

    The cutoff support [cutoff_min, cutoff_max] is fixed at fit time from the data.
    Boundary knots are repeated `degree+1` times at each end (standard).
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

        # Number of *interior* knots needed so that
        #   K_total = 2*(degree+1) + n_interior  and  n_basis = K_total - degree - 1 = n_splines
        # gives n_interior = n_splines - degree - 1.
        self.n_interior = n_splines - degree - 1
        if self.n_interior < 0:
            raise ValueError(f"n_splines must be >= degree+1, got {n_splines} and {degree}")

        # Cutoff support (set at fit time)
        self.register_buffer("cutoff_min", torch.zeros(1))
        self.register_buffer("cutoff_max", torch.ones(1))

        # MLP outputs:
        #   - n_interior + 1 log-gaps  (interior knots produced by cumulative softplus
        #     and then rescaled to the interior region; we need n_interior + 1 gaps so
        #     the cumulative sum ranges from 0 to "full interior span" through n_interior
        #     interior knot positions)
        #   - n_splines alpha values
        #   - 1 beta value
        out_dim = (self.n_interior + 1) + n_splines + 1
        layers = [nn.Linear(max(n_poi, 1), hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def _set_support(self, cutoffs: torch.Tensor):
        self.cutoff_min = cutoffs.min().detach().reshape(1)
        self.cutoff_max = cutoffs.max().detach().reshape(1)

    def _build_knots(self, gap_logits: torch.Tensor) -> torch.Tensor:
        """
        gap_logits : (N, n_interior + 1)  -- raw outputs from MLP

        Returns
        -------
        knots : (N, K_total) sorted per row, with degree+1 boundary repeats
        """
        N = gap_logits.shape[0]
        gaps = F.softplus(gap_logits) + 1e-4              # (N, n_interior+1), strictly positive
        cum = torch.cumsum(gaps, dim=1)                   # (N, n_interior+1)
        # Normalize so that the last cumulative value is 1, then take the first
        # n_interior entries — those become positions in (0, 1).
        normalized = cum / cum[:, -1:].clamp(min=1e-8)
        interior_unit = normalized[:, :-1]                # (N, n_interior), in (0, 1) sorted

        span = (self.cutoff_max - self.cutoff_min)
        interior = self.cutoff_min + interior_unit * span  # (N, n_interior)

        lo = self.cutoff_min.expand(N, self.degree + 1)   # (N, degree+1)
        hi = self.cutoff_max.expand(N, self.degree + 1)   # (N, degree+1)
        return torch.cat([lo, interior, hi], dim=1)        # (N, K_total)

    def forward(self, cutoffs: torch.Tensor, poi: torch.Tensor) -> torch.Tensor:
        out = self.mlp(poi)                                # (N, out_dim)
        n_gap = self.n_interior + 1
        gap_logits = out[:, :n_gap]
        alpha_raw = out[:, n_gap:n_gap + self.n_splines]
        beta = out[:, -1]

        knots = self._build_knots(gap_logits)              # (N, K_total)

        # Clamp cutoffs to the support so the basis is well-defined
        cutoffs_c = cutoffs.clamp(self.cutoff_min.item(), self.cutoff_max.item())
        I = _ispline_basis_batched(cutoffs_c, knots, self.degree)  # (N, n_splines)

        alpha = F.softplus(alpha_raw)                      # (N, n_splines)
        logits = (I * alpha).sum(dim=1) + beta             # (N,)
        return logits


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class ISplineClassifier:
    """
    Sklearn-style wrapper around ISplineHyperModel.

    Column 0 of X is the cutoff (monotone dimension).
    Remaining columns are POI features fed to the hypernetwork.

    monotone_constraints : +1 for non-decreasing in cutoff, -1 for non-increasing.
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
        verbose: bool = True,
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
        self.verbose = verbose
        self.model_: Optional[ISplineHyperModel] = None
        self.classes_ = np.array([0, 1])

    def _to_tensors(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        cutoffs = X_t[:, 0]
        poi = X_t[:, 1:] if X_t.shape[1] > 1 else torch.zeros(len(X_t), 1, device=self.device)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
            return cutoffs, poi, y_t
        return cutoffs, poi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ISplineHyperClassifier":
        cutoffs, poi, y_t = self._to_tensors(X, y)

        self.model_ = ISplineHyperModel(
            n_poi=poi.shape[1],
            n_splines=self.n_splines,
            degree=self.degree,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
        ).to(self.device)
        self.model_._set_support(cutoffs)

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
            if self.verbose:
                with torch.no_grad():
                    train_loss = loss_fn(self.monotone_constraints * self.model_(cutoffs, poi), y_t).item()
                print(f"epoch {epoch + 1}/{self.epochs}  train loss: {train_loss:.4f}")

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
