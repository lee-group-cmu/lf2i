from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ── FNO building blocks (kept for functional/operator use case) ───────────────

class SpectralConv1d(nn.Module):
    """FNO spectral convolution: pointwise complex multiply on low Fourier modes."""

    def __init__(self, channels_in, channels_out, modes):
        super().__init__()
        self.modes = modes
        init_scale = 1.0 / (channels_in * channels_out)
        self.weight = nn.Parameter(
            init_scale * torch.randn(channels_in, channels_out, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        n_points = x.shape[-1]
        spectrum = torch.fft.rfft(x, dim=-1)

        out_spectrum = torch.zeros(
            x.shape[0], self.weight.shape[1], spectrum.shape[-1],
            device=x.device, dtype=torch.cfloat,
        )
        kept = min(self.modes, spectrum.shape[-1])
        out_spectrum[:, :, :kept] = torch.einsum(
            "bim,iom->bom", spectrum[:, :, :kept], self.weight[:, :, :kept],
        )
        return torch.fft.irfft(out_spectrum, n=n_points, dim=-1)


class FNOEncoder(nn.Module):
    """Encode theta(.) on a fixed grid into a latent vector. For gridded/functional theta."""

    def __init__(self, grid, width=64, modes=16, depth=4, latent=128):
        super().__init__()
        self.register_buffer("grid", grid)
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(depth)]
        )
        self.local = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)]
        )
        self.project = nn.Sequential(
            nn.Linear(width, latent), nn.SiLU(), nn.Linear(latent, latent),
        )

    def forward(self, theta):
        positions = self.grid.expand(theta.shape[0], -1, -1)
        x = torch.cat([theta, positions], dim=-1)
        x = self.lift(x).permute(0, 2, 1)
        for spectral_layer, local_layer in zip(self.spectral, self.local):
            x = F.gelu(spectral_layer(x) + local_layer(x))
        return self.project(x.mean(dim=-1))


# ── MLP encoder for flat parameter vectors ────────────────────────────────────

class MLPEncoder(nn.Module):
    """Encode a flat parameter vector θ ∈ R^d into a latent vector of size `latent`."""

    _ACTIVATIONS = {
        'tanh': nn.Tanh,
        'elu':  nn.ELU,
        'silu': nn.SiLU,
        'relu': nn.ReLU,
    }

    def __init__(self, theta_dim: int, hidden_dim: int = 64, n_hidden: int = 2,
                 latent: int = 128, activation: str = 'elu'):
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self._ACTIVATIONS)}, got '{activation}'.")
        act_cls = self._ACTIVATIONS[activation]
        layers = [nn.Linear(theta_dim, hidden_dim), act_cls()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers.append(nn.Linear(hidden_dim, latent))
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:  # [B, theta_dim] → [B, latent]
        return self.net(theta)


# ── Monotone score-quantile operator ─────────────────────────────────────────

class ScoreQuantileOperator(nn.Module):
    """
    Conditional quantile function Q(θ, τ) for a scalar test statistic.

    Architecture: MLPEncoder maps θ → latent, then three heads produce median,
    log-scale, and free monotone shape on tail-dense knots. The quantile curve
    is guaranteed crossing-free by construction.

    Parameters
    ----------
    theta_dim : int
        Dimensionality of the parameter vector θ.
    hidden_dim : int
        Width of each hidden layer in MLPEncoder.
    n_hidden : int
        Number of hidden layers in MLPEncoder.
    activation : str
        Hidden-layer activation ('elu', 'tanh', 'silu', 'relu').
    n_knots : int
        Number of knots for the piecewise-linear quantile shape.
    tau_min, tau_max : float
        Quantile range represented by the model.
    latent : int
        Encoder output size.
    """

    def __init__(self, theta_dim: int, hidden_dim: int = 64, n_hidden: int = 2,
                 activation: str = 'elu', n_knots: int = 32,
                 tau_min: float = 1e-3, tau_max: float = 1 - 1e-3, latent: int = 128):
        super().__init__()
        self.n_knots = n_knots
        self.tau_min = tau_min
        self.tau_max = tau_max

        spaced = torch.sigmoid(torch.linspace(-3.5, 3.5, n_knots))
        spaced = (spaced - spaced.min()) / (spaced.max() - spaced.min())
        self.register_buffer("knots", tau_min + (tau_max - tau_min) * spaced)

        self.encoder = MLPEncoder(theta_dim, hidden_dim, n_hidden, latent, activation)
        self.to_median = nn.Linear(latent, 1)
        self.to_log_scale = nn.Linear(latent, 1)
        self.to_increments = nn.Linear(latent, n_knots - 1)

        self.register_buffer("stat_mean", torch.zeros(1))
        self.register_buffer("stat_std", torch.ones(1))

    def _shape(self, latent: torch.Tensor) -> torch.Tensor:
        """Free monotone shape φ at the knots: normalized positive increments in [0, 1]."""
        widths = F.softplus(self.to_increments(latent)) + 1e-4
        widths = widths / widths.sum(dim=-1, keepdim=True)
        cumulative = torch.cumsum(widths, dim=-1)
        zero = torch.zeros(latent.shape[0], 1, device=latent.device)
        return torch.cat([zero, cumulative], dim=-1)  # [B, n_knots]

    def _interpolate(self, heights: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear interpolation of knot `heights` [B, K] at `tau` [B, M]."""
        knots = self.knots.contiguous()
        right = torch.searchsorted(knots, tau.contiguous())
        idx = torch.clamp(right - 1, 0, self.n_knots - 2)

        left_tau = knots[idx]
        right_tau = knots[idx + 1]
        frac = (tau - left_tau) / (right_tau - left_tau)

        left_h = heights.gather(1, idx)
        right_h = heights.gather(1, idx + 1)
        return left_h + frac * (right_h - left_h)

    def _knot_quantiles(self, theta: torch.Tensor, standardized: bool = False) -> torch.Tensor:
        """Quantile values Q(θ, ·) at the model's τ-knots, shape [B, n_knots]."""
        latent = self.encoder(theta)
        shape = self._shape(latent)
        median = self.to_median(latent)
        scale = F.softplus(self.to_log_scale(latent)) + 1e-3

        center = self._interpolate(shape, torch.full((latent.shape[0], 1), 0.5, device=theta.device))
        knot_quantiles = median + scale * (shape - center)

        if standardized:
            return knot_quantiles
        return knot_quantiles * self.stat_std + self.stat_mean

    def forward(self, theta: torch.Tensor, tau: torch.Tensor,
                standardized: bool = False) -> torch.Tensor:
        """Quantile Q(θ, τ). `tau` is [B, M]; returns [B, M]."""
        knot_quantiles = self._knot_quantiles(theta, standardized=True)
        quantile = self._interpolate(knot_quantiles, tau)
        if standardized:
            return quantile
        return quantile * self.stat_std + self.stat_mean

    @torch.no_grad()
    def quantile_curve(self, theta: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Evaluate the curve at a 1-D grid `taus` [G]; returns [B, G] in statistic units."""
        tau_batched = taus[None, :].expand(theta.shape[0], -1).to(theta.device)
        return self.forward(theta, tau_batched)

    @torch.no_grad()
    def cdf(self, theta: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        """Conditional CDF F_θ(stat). `stat` is [B] or [B, M]; return matches that shape."""
        knot_quantiles = self._knot_quantiles(theta)  # [B, K], statistic units
        single = stat.dim() == 1
        values = stat[:, None] if single else stat     # [B, M]

        below = torch.searchsorted(knot_quantiles.contiguous(), values.contiguous(), right=True)
        idx = torch.clamp(below - 1, 0, self.n_knots - 2)

        q_lo = knot_quantiles.gather(1, idx)
        q_hi = knot_quantiles.gather(1, idx + 1)
        frac = ((values - q_lo) / (q_hi - q_lo + 1e-12)).clamp(0, 1)

        tau = (self.knots[idx] + frac * (self.knots[idx + 1] - self.knots[idx])).clamp(
            self.tau_min, self.tau_max
        )
        return tau.squeeze(1) if single else tau

    @torch.no_grad()
    def critical_value(self, theta: torch.Tensor, alpha: float) -> torch.Tensor:
        """LF2I critical value: the (1 − α) quantile under P_θ."""
        tau = torch.full((theta.shape[0], 1), 1.0 - alpha, device=theta.device)
        return self.forward(theta, tau)


# ── Spectral Lipschitz penalty ────────────────────────────────────────────────

class SpectralPenalty:
    """Sum of squared spectral norms of the given layers' weight matrices."""

    def __init__(self, layers):
        self.weights = [layer.weight for layer in layers]
        self.left_vectors = [torch.randn(layer.weight.shape[0]) for layer in layers]

    def __call__(self, n_iter: int = 1) -> torch.Tensor:
        total = 0.0
        for i, weight in enumerate(self.weights):
            matrix = weight.view(weight.shape[0], -1)
            left = self.left_vectors[i].to(matrix.device)

            with torch.no_grad():
                for _ in range(n_iter):
                    right = torch.mv(matrix.t(), left)
                    right = right / (right.norm() + 1e-12)
                    left = torch.mv(matrix, right)
                    left = left / (left.norm() + 1e-12)
                self.left_vectors[i] = left

            right = torch.mv(matrix.t().detach(), left)
            right = right / (right.norm() + 1e-12)
            sigma_max = torch.dot(left, torch.mv(matrix, right))
            total = total + sigma_max ** 2
        return total


def pinball_loss(target: torch.Tensor, predicted: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Quantile (pinball) loss, averaged over batch and τ levels."""
    if target.dim() == 1:
        target = target.unsqueeze(1)
    error = target - predicted
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


def lipschitz_layers(model: ScoreQuantileOperator) -> list:
    """Linear layers whose Lipschitz constant controls smoothness of Q in θ."""
    layers = [m for m in model.encoder.net if isinstance(m, nn.Linear)]
    layers += [model.to_median, model.to_log_scale, model.to_increments]
    return layers


# ── AbstractCDFEstimator wrapper ──────────────────────────────────────────────

class QuantileOperatorCDFEstimator:
    """
    CDF estimator based on the Score-Quantile Operator.

    Wraps ScoreQuantileOperator in a sklearn-style interface that satisfies
    AbstractCDFEstimator. Theta is treated as a flat parameter vector of fixed
    length, encoded by an MLP before the quantile head.

    Training uses AdamW + pinball loss + spectral Lipschitz penalty. If CUDA is
    available it is used for training; the model is moved to CPU before returning
    from fit() so that predict_proba() does not require a GPU.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden layer in the MLP encoder. Default 64.
    n_hidden : int
        Number of hidden layers. Default 2.
    activation : str
        Hidden-layer activation ('elu', 'tanh', 'silu', 'relu'). Default 'elu'.
    latent : int
        Encoder output / latent size. Default 128.
    n_knots : int
        Knots for the piecewise-linear quantile shape. Default 32.
    tau_min, tau_max : float
        Quantile range represented by the model. Defaults 1e-3 / 1-1e-3.
    epochs : int
        Training epochs. Default 75.
    lr : float
        AdamW learning rate. Default 1e-3.
    batch_size : int
        Mini-batch size. Default 256.
    n_levels : int
        Random τ levels drawn per sample per step. Default 16.
    lambda_spectral : float
        Weight on the spectral Lipschitz penalty. Default 1e-2.
    weight_decay : float
        AdamW weight decay. Default 1e-4.
    val_fraction : float
        Fraction of data held out for validation / early stopping. Default 0.15.
    device : str, optional
        'cuda' or 'cpu'. Auto-detected if None.
    verbose : bool
        Print progress every 10 epochs. Default True.
    """

    def __init__(
        self,
        hidden_dim:       int   = 64,
        n_hidden:         int   = 2,
        activation:       str   = 'elu',
        latent:           int   = 128,
        n_knots:          int   = 32,
        tau_min:          float = 1e-3,
        tau_max:          float = 1 - 1e-3,
        epochs:           int   = 75,
        lr:               float = 1e-3,
        batch_size:       int   = 256,
        n_levels:         int   = 16,
        lambda_spectral:  float = 1e-2,
        weight_decay:     float = 1e-4,
        val_fraction:     float = 0.15,
        device:           Optional[str] = None,
        verbose:          bool  = True,
    ):
        self.hidden_dim      = hidden_dim
        self.n_hidden        = n_hidden
        self.activation      = activation
        self.latent          = latent
        self.n_knots         = n_knots
        self.tau_min         = tau_min
        self.tau_max         = tau_max
        self.epochs          = epochs
        self.lr              = lr
        self.batch_size      = batch_size
        self.n_levels        = n_levels
        self.lambda_spectral = lambda_spectral
        self.weight_decay    = weight_decay
        self.val_fraction    = val_fraction
        self.verbose         = verbose

        self._device = torch.device(
            device if device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.model_:   Optional[ScoreQuantileOperator] = None
        self.history_: Optional[dict]                  = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> 'QuantileOperatorCDFEstimator':
        """
        Fit the quantile operator to calibration data.

        Parameters
        ----------
        X : array-like, shape (n, 1 + theta_dim)
            Column 0 is the test statistic λ; columns 1: are the parameters θ.

        Returns
        -------
        self
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        test_statistics = X[:, 0].astype(np.float32)
        poi = X[:, 1:].astype(np.float32)
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)
        theta_dim = poi.shape[1]
        n = len(test_statistics)

        # ── train / val split ─────────────────────────────────────────────────
        n_val = max(1, int(self.val_fraction * n))
        perm = torch.randperm(n)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        stat_t  = torch.from_numpy(test_statistics)
        theta_t = torch.from_numpy(poi)

        # ── build model ───────────────────────────────────────────────────────
        self.model_ = ScoreQuantileOperator(
            theta_dim  = theta_dim,
            hidden_dim = self.hidden_dim,
            n_hidden   = self.n_hidden,
            activation = self.activation,
            n_knots    = self.n_knots,
            tau_min    = self.tau_min,
            tau_max    = self.tau_max,
            latent     = self.latent,
        ).to(self._device)

        # ── standardize statistic (training split only) ───────────────────────
        stat_train = stat_t[train_idx]
        self.model_.stat_mean.copy_(stat_train.mean())
        self.model_.stat_std.copy_(stat_train.std() + 1e-8)
        stat_std = (stat_t - self.model_.stat_mean) / self.model_.stat_std

        # ── spectral penalty and optimizer ────────────────────────────────────
        spectral_penalty = SpectralPenalty(lipschitz_layers(self.model_))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # ── data loaders ──────────────────────────────────────────────────────
        train_loader = DataLoader(
            TensorDataset(theta_t[train_idx], stat_std[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        theta_val    = theta_t[val_idx].to(self._device)
        stat_val_std = stat_std[val_idx].to(self._device)

        # ── training loop ─────────────────────────────────────────────────────
        train_history: list = []
        val_history:   list = []
        best_val   = float('inf')
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            running = 0.0

            for theta_batch, stat_batch in train_loader:
                theta_batch = theta_batch.to(self._device)
                stat_batch  = stat_batch.to(self._device)

                tau = self.model_.tau_min + (self.model_.tau_max - self.model_.tau_min) * torch.rand(
                    theta_batch.shape[0], self.n_levels, device=self._device
                )
                predicted = self.model_(theta_batch, tau, standardized=True)
                pinball = pinball_loss(stat_batch, predicted, tau)
                loss = pinball + self.lambda_spectral * spectral_penalty()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += pinball.item() * theta_batch.shape[0]

            train_history.append(running / train_idx.numel())

            self.model_.eval()
            with torch.no_grad():
                tau_val = self.model_.tau_min + (self.model_.tau_max - self.model_.tau_min) * torch.rand(
                    theta_val.shape[0], self.n_levels, device=self._device
                )
                val_loss = pinball_loss(
                    stat_val_std, self.model_(theta_val, tau_val, standardized=True), tau_val
                )

            val_history.append(float(val_loss))

            if val_loss < best_val:
                best_val   = float(val_loss)
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d}/{self.epochs}  |  "
                    f"train: {train_history[-1]:.5f}  |  val: {val_history[-1]:.5f}",
                    flush=True,
                )

        # ── restore best checkpoint and move to CPU ───────────────────────────
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.model_.to('cpu')

        self.history_ = {'train': train_history, 'val': val_history}

        if self.verbose:
            print(f"Best val pinball: {best_val:.4f}")

        return self

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """
        Compute p-values from the fitted quantile operator.

        Parameters
        ----------
        X : array-like, shape (n, 1 + theta_dim)
            Column 0 is the test statistic λ; columns 1: are the parameters θ.

        Returns
        -------
        np.ndarray, shape (n, 2)
            Column 0 = 1 − F(λ | θ), column 1 = F(λ | θ). Rows sum to 1.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        if isinstance(X, torch.Tensor):
            X = X.numpy()

        poi = X[:, 1:].astype(np.float32)
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)

        theta_t = torch.from_numpy(poi)
        stat_t  = torch.from_numpy(X[:, 0].astype(np.float32))

        with torch.no_grad():
            cdf_vals = self.model_.cdf(theta_t, stat_t)

        cdf_np = cdf_vals.numpy()
        return np.column_stack([1.0 - cdf_np, cdf_np])
