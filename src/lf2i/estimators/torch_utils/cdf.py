from typing import Optional, Tuple, Union
import warnings

import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class BrierScoreLoss(nn.Module):
    """Cross-product Brier score, diagonal excluded to remove systematic CDF-high bias."""
    def forward(
            self,
            cdf_vals: torch.Tensor,
            lambda_obs: torch.Tensor
    ) -> torch.Tensor:
        n = lambda_obs.shape[0]
        indicators = (lambda_obs.unsqueeze(1) <= lambda_obs.unsqueeze(0)).float()
        mask = ~torch.eye(n, dtype=torch.bool, device=lambda_obs.device)
        return ((cdf_vals - indicators)**2)[mask].mean()

class QuantileWeightedCRPSLoss(nn.Module):
    """
    Pinball loss integrated over α ∈ (0, 1) with a smooth weight function w(α):

        L = 2 · mean_{i} ∫ w(α) · ρ_α(λ_i − F̃⁻¹(α; β(θ_i))) dα

    The weight function controls which quantile levels the fit prioritises.
    Built-in options (via `weight_fn`):

        'uniform'   : w(α) = 1  — standard unweighted CRPS / pinball
        'gaussian'  : w(α) ∝ N(α; center, bandwidth²) — smooth emphasis
                      around a target α level
        'beta'      : w(α) ∝ Beta(α; a, b) — flexible skewed weighting
        callable    : any user-supplied function w(alpha_grid) → Tensor

    Parameters
    ----------
    n_alpha : int
        Number of quadrature points over (0, 1). Default 500.
    weight_fn : str or callable
        Weight function. One of 'uniform', 'gaussian', 'beta', or a callable
        taking a 1-D Tensor of α values and returning a same-shaped Tensor of
        non-negative weights.
    center : float
        Centre of the weight mass for 'gaussian'. Typically set to the target
        α level, e.g. 0.1 for a 90% confidence set. Default 0.1.
    bandwidth : float
        Standard deviation of the Gaussian weight. Smaller = more concentrated.
        Default 0.1.
    beta_a : float
        α parameter of Beta weight. Default 2.0.
    beta_b : float
        β parameter of Beta weight. Default 5.0.
    """

    def __init__(
        self,
        n_alpha:    int             = 1000,
        weight_fn:  Union[str, callable] = 'gaussian',
        center:     float           = 0.1,
        bandwidth:  float           = 0.1,
        beta_a:     float           = 2.0,
        beta_b:     float           = 5.0,
    ):
        super().__init__()
        self.n_alpha = n_alpha

        alpha_grid = torch.linspace(1e-4, 1 - 1e-4, n_alpha) # (1-2e-4)*torch.rand(n_alpha) + 1e-4    # (M,)
        self.register_buffer('alpha_grid', alpha_grid)

        # ── build weight vector ───────────────────────────────────────────────
        if callable(weight_fn) and not isinstance(weight_fn, str):
            weights = weight_fn(alpha_grid)
        elif weight_fn == 'uniform':
            weights = torch.ones(n_alpha)
        elif weight_fn == 'gaussian':
            weights = torch.exp(
                -0.5 * ((alpha_grid - center) / bandwidth) ** 2
            )
        elif weight_fn == 'beta':
            # Beta(a, b) density (unnormalised) — no scipy dependency
            weights = (alpha_grid ** (beta_a - 1)) * ((1 - alpha_grid) ** (beta_b - 1))
        else:
            raise ValueError(
                f"weight_fn must be 'uniform', 'gaussian', 'beta', or a callable. "
                f"Got '{weight_fn}'."
            )

        # normalise so weights sum to 1 — keeps loss scale stable
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)

    def forward(
        self,
        predicted_quantiles: torch.Tensor,   # (n, M) — F̃⁻¹(α; β(θ_i))
        lambda_obs:          torch.Tensor,   # (n,)
    ) -> torch.Tensor:
        residuals = lambda_obs.unsqueeze(1) - predicted_quantiles   # (n, M)
        alpha_row = self.alpha_grid.unsqueeze(0)                    # (1, M)
        pinball   = torch.where(
            residuals >= 0,
            alpha_row * residuals,
            -(1 - alpha_row) * residuals,
        )
        # weight each α level, then average over samples
        return 2 * (pinball * self.weights.unsqueeze(0)).sum(dim=1).mean()

class SigmoidCDF(nn.Module):
    """
    Sigmoid CDF for modeling test statistic at fixed theta.

    Parameterized by a parameter Beta, with location component mu (ED50)
    and slope component kappa.

    Notes
    ----
    - Using `log_kappa` to control values of kappa to (0, infty] for optimization.
    """
    def forward(self,
                lambda_vals: torch.Tensor, # (n, ) or (n, 1)
                mu: torch.Tensor, # (n, 1)
                log_kappa: torch.Tensor # (n, 1)
                ) -> torch.Tensor:
        kappa = torch.exp(log_kappa)
        return torch.sigmoid(kappa * (lambda_vals - mu))

    def quantile(self,
                 alpha: torch.Tensor, # (1, M)
                 mu: torch.Tensor, # (n, 1)
                 log_kappa: torch.Tensor # n, 1)
                 ) -> torch.Tensor: # (n, M)
        """Inverse CDF of the sigmoid."""
        kappa = torch.exp(log_kappa)
        return mu + (1/kappa) * torch.log(alpha / (1 - alpha))

class BetaNetwork(nn.Module):
    """
    Shallow feed-forward network mapping parameters of interest θ to the
    logistic CDF parameters β = (μ, log s).

    Parameters
    ----------
    theta_dim : int
        Dimensionality of the parameter of interest.
    hidden_dim : int
        Width of each hidden layer.
    n_hidden : int
        Number of hidden layers.
    activation : str
        Activation function for hidden layers. One of:
        'tanh', 'elu', 'silu', 'relu'. Default 'elu'.
    """

    _ACTIVATIONS = {
        'tanh': nn.Tanh,
        'elu':  nn.ELU,
        'silu': nn.SiLU,
        'relu': nn.ReLU,
    }

    def __init__(
        self,
        theta_dim:  int,
        hidden_dim: int = 64,
        n_hidden:   int = 2,
        activation: str = 'elu',
    ):
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS)}, got '{activation}'."
            )
        act_cls = self._ACTIVATIONS[activation]
        layers  = [nn.Linear(theta_dim, hidden_dim), act_cls()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers += [nn.Linear(hidden_dim, 2)]   # outputs: (μ, log s) # TODO: add dropout layers for regularization
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:   # (n, 2)
        return self.net(theta)

class ParametricCDFEstimator:
    """
    Parametric CDF estimator for likelihood-free frequentist inference.

    Fits a logistic CDF F̃(λ; β(θ)) to the calibration test statistic
    distribution, where β(θ) = (μ(θ), log kappa(θ)) are the outputs of a
    shallow neural network trained end-to-end.

    Follows a sklearn-style interface:
        estimator.fit(test_statistics, poi)
        estimator.predict_proba(X)   # X[:, 0] = λ, X[:, 1:] = θ

    This class is used directly as the calibration model stored in
    lf2i.calibration_model, replacing ParametricCDFPredictor.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden layer in BetaNetwork. Default 64.
    n_hidden : int
        Number of hidden layers. Default 2.
    activation : str
        Hidden layer activation. One of 'elu', 'tanh', 'silu', 'relu'.
        Default 'tanh'.
    loss : str
        Loss function. One of 'brier', 'weighted'. Default 'brier'.
    n_alpha : int
        Quadrature points for 'weighted'. Default 500.
    weight_fn : str or callable
        Weight function for 'weighted'. One of 'uniform', 'gaussian', 'beta',
        or a callable taking a 1-D alpha Tensor and returning non-negative
        weights. Default 'gaussian'.
    center : float
        Centre of Gaussian weight mass. Set to target α level, e.g. 0.1 for
        a 90% confidence set. Default 0.1.
    bandwidth : float
        Standard deviation of Gaussian weight. Default 0.1.
    beta_a : float
        α parameter of Beta weight. Default 2.0.
    beta_b : float
        β parameter of Beta weight. Default 5.0.
    epochs : int
        Training epochs. Default 500.
    lr : float
        Adam learning rate. Default 1e-3.
    batch_size : int
        Mini-batch size. Default 512.
    device : str, optional
        'cuda' or 'cpu'. Auto-detected if None.
    verbose : bool
        Print loss every 100 epochs. Default True.
    """

    def __init__(
        self,
        # architecture
        hidden_dim:  int   = 64,
        n_hidden:    int   = 2,
        activation:  str   = 'elu',
        # loss
        loss:        str   = 'brier',
        cdf_model:   str   = 'sigmoid',   # kept for backwards compat; 'ispline' emits DeprecationWarning + raises
        n_alpha:     int   = 500,
        weight_fn:   Union[str, callable] = 'gaussian',
        center:      float = 0.1,
        bandwidth:   float = 0.1,
        beta_a:      float = 2.0,
        beta_b:      float = 5.0,
        # Normalization
        normalize_ts:   str   = 'none',
        normalize_theta: str = 'none',
        # optimisation
        epochs:      int   = 500,
        lr:          float = 1e-3,
        batch_size:  int   = 512,
        smooth_reg:  float = 0.0,    # weight on max-entropy regularisation (log κ)
        knn_k:       int   = 30,     # moment loss: KNN neighbours for conditional moment targets
        lk_weight:   float = 1.0,    # moment loss: weight on log_kappa MSE term
        warmup:      int   = 0,      # epochs with flat LR before cosine decay (useful for moment loss)
        device:      Optional[str] = None,
        verbose:     bool  = True,
    ):
        if loss not in ('brier', 'weighted_brier', 'weighted', 'moment'):
            raise ValueError(
                f"loss must be 'brier', 'weighted_brier', 'weighted', or 'moment', got '{loss}'."
            )

        if normalize_ts not in ('none', 'mean-std', 'min-max', 'percentiles'):
            raise ValueError(
                f"normalize_ts must be one of 'none', 'mean-std', 'min-max', 'percentiles'. "
                f"Got '{normalize_ts}'."
            )

        if normalize_theta not in ('none', 'mean-std', 'min-max'):
            raise ValueError(
                f"normalize_theta must be one of 'none', 'mean-std', 'min-max'. "
                f"Got '{normalize_theta}'."
            )

        self.normalize_ts    = normalize_ts
        self.normalize_theta = normalize_theta

        self.hidden_dim        = hidden_dim
        self.n_hidden          = n_hidden
        self.activation        = activation
        self.loss              = loss
        self.n_alpha           = n_alpha
        self.weight_fn         = weight_fn
        self.center            = center
        self.bandwidth         = bandwidth
        self.beta_a            = beta_a
        self.beta_b            = beta_b
        self.epochs            = epochs
        self.lr                = lr
        self.batch_size        = batch_size
        self.smooth_reg        = smooth_reg
        self.knn_k             = knn_k
        self.lk_weight         = lk_weight
        self.warmup            = warmup
        self.verbose           = verbose

        self.device = torch.device(
            device if device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # set after fit()
        self.beta_net_:    Optional[BetaNetwork] = None
        self.cdf_model_:   Optional[SigmoidCDF]  = None
        self.lambda_min_:  Optional[float]        = None
        self.lambda_max_:  Optional[float]        = None
        self.lambda_lo_:    Optional[float]                = None
        self.lambda_hi_:    Optional[float]                = None
        self.lambda_mean_:  Optional[float]                = None
        self.lambda_std_:   Optional[float]                = None
        self.theta_mean_:   Optional[np.ndarray]           = None
        self.theta_std_:    Optional[np.ndarray]           = None
        self.theta_lo_:     Optional[np.ndarray]           = None
        self.theta_hi_:     Optional[np.ndarray]           = None
        self.history_:      Optional[list]                 = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize_ts(self, ts: np.ndarray) -> np.ndarray:
        if self.normalize_ts == 'none':
            return ts
        elif self.normalize_ts == 'mean-std':
            return (ts - self.lambda_mean_) / self.lambda_std_
        else:  # 'min-max' or 'percentiles'
            return (ts - self.lambda_lo_) / (self.lambda_hi_ - self.lambda_lo_ + 1e-8)

    def _normalize_theta(self, theta: np.ndarray) -> np.ndarray:
        if self.normalize_theta == 'none':
            return theta
        elif self.normalize_theta == 'mean-std':
            return (theta - self.theta_mean_) / self.theta_std_
        else:  # 'min-max'
            return (theta - self.theta_lo_) / (self.theta_hi_ - self.theta_lo_ + 1e-8)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,   # (n, 1 + poi_dim): column 0 = λ, columns 1: = θ
    ) -> 'ParametricCDFEstimator':
        """
        Fit the parametric CDF to calibration data.

        Parameters
        ----------
        X : np.ndarray
            Shape (n, 1 + poi_dim). Column 0 is the test statistic λ(x_i; θ_i),
            columns 1: are the corresponding parameters of interest θ_i.

        Returns
        -------
        self
        """
        test_statistics = X[:, 0]
        poi = X[:, 1:]
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)
        theta_dim = poi.shape[1]

        # ── initialise model ──────────────────────────────────────────────────
        if self.normalize_ts == 'mean-std':
            self.lambda_mean_ = float(test_statistics.mean())
            self.lambda_std_  = float(test_statistics.std()) + 1e-8
        elif self.normalize_ts == 'min-max':
            self.lambda_lo_ = float(test_statistics.min())
            self.lambda_hi_ = float(test_statistics.max())
        elif self.normalize_ts == 'percentiles':
            self.lambda_lo_ = float(np.percentile(test_statistics, 0.01))
            self.lambda_hi_ = float(np.percentile(test_statistics, 99.99))

        if self.normalize_theta == 'mean-std':
            self.theta_mean_ = poi.mean(axis=0)
            self.theta_std_  = poi.std(axis=0) + 1e-8
        elif self.normalize_theta == 'min-max':
            self.theta_lo_ = poi.min(axis=0)
            self.theta_hi_ = poi.max(axis=0)

        self.beta_net_  = BetaNetwork(
            theta_dim, self.hidden_dim, self.n_hidden, self.activation
        ).to(self.device)
        self.cdf_model_ = SigmoidCDF().to(self.device)
        model_params    = self.beta_net_.parameters()

        optimizer = torch.optim.Adam(model_params, lr=self.lr)
        if self.warmup > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer, factor=1.0, total_iters=self.warmup
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(self.epochs - self.warmup, 1)
                    ),
                ],
                milestones=[self.warmup],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )

        # ── criterion ─────────────────────────────────────────────────────────
        if self.loss == 'brier':
            criterion = BrierScoreLoss()
        else:
            criterion = WeightedPinballLoss(
                n_alpha   = self.n_alpha,
                weight_fn = self.weight_fn,
                center    = self.center,
                bandwidth = self.bandwidth,
                beta_a    = self.beta_a,
                beta_b    = self.beta_b,
            ).to(self.device)

        # ── data ──────────────────────────────────────────────────────────────
        ts_input  = self._normalize_ts(test_statistics)
        poi_input = self._normalize_theta(poi)
        lambda_t  = torch.FloatTensor(ts_input).to(self.device)
        theta_t   = torch.FloatTensor(poi_input).to(self.device)
        loader   = DataLoader(
            TensorDataset(lambda_t, theta_t),
            batch_size = self.batch_size,
            shuffle    = True,
        )
        if self.verbose:
            print(f"Raw λ range: [{test_statistics.min():.3f}, {test_statistics.max():.3f}]")
            print(f"Normalized λ range: [{ts_input.min():.3f}, {ts_input.max():.3f}]")
            print(f"Raw θ range: [{poi.min(axis=0)}, {poi.max(axis=0)}]")
            print(f"Normalized θ range: [{poi_input.min(axis=0)}, {poi_input.max(axis=0)}]")

        # ── training loop ──────────────────────────────────────────────────────
        self.history_ = []
        self.beta_net_.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for lam_b, theta_b in loader:
                optimizer.zero_grad()

                beta      = self.beta_net_(theta_b)
                mu        = beta[:, 0:1]
                log_kappa = beta[:, 1:2]
                if isinstance(criterion, WeightedPinballLoss):
                    alpha_row           = criterion.alpha_grid.unsqueeze(0).to(lam_b.device)
                    predicted_quantiles = self.cdf_model_.quantile(alpha_row, mu, log_kappa)
                    batch_loss          = criterion(predicted_quantiles, lam_b) + self.smooth_reg * log_kappa.mean()
                else:
                    cdf_vals   = self.cdf_model_(lam_b.unsqueeze(0), mu, log_kappa)
                    batch_loss = criterion(cdf_vals, lam_b)

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            scheduler.step()
            avg = epoch_loss / len(loader)
            self.history_.append(avg)
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs}  |  loss: {avg:.5f}", flush=True)

        self.beta_net_.eval()
        return self

    # ── predict_proba ─────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Compute p-values from the fitted parametric CDF.

        Follows the sklearn predict_proba convention expected by lf2i:
        accepts X as a keyword argument and returns a two-column matrix.

        Parameters
        ----------
        X : np.ndarray
            Shape (n, 1 + poi_dim).
            X[:, 0]  — test statistic values λ
            X[:, 1:] — parameters of interest θ

        Returns
        -------
        np.ndarray
            Shape (n, 2).
            Column 0 = 1 − CDF(λ | θ), column 1 = CDF(λ | θ).
            The directionality of p-values is resolved by the caller
            (e.g. `lf2i.inference`) based on the test statistic's acceptance region.
        """
        if self.beta_net_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if X is None:
            raise ValueError("X must be provided.")

        poi = X[:, 1:]
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)

        with torch.no_grad():
            ts_input  = self._normalize_ts(X[:, 0])
            poi_input = self._normalize_theta(poi)
            lambda_t  = torch.FloatTensor(ts_input).to(self.device)
            theta_t   = torch.FloatTensor(poi_input).to(self.device)

            beta      = self.beta_net_(theta_t)
            mu        = beta[:, 0:1]
            log_kappa = beta[:, 1:2]
            cdf_vals  = self.cdf_model_(
                lambda_t.unsqueeze(1), mu, log_kappa
            ).squeeze(1)

        cdf_vals_np = cdf_vals.cpu().numpy()
        return np.column_stack([1.0 - cdf_vals_np, cdf_vals_np])
