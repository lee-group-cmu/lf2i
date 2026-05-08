from typing import Optional, Tuple, Union
import warnings

import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ── I-spline support (optional) ───────────────────────────────────────────────
try:
    from lf2i.calibration.isplines import Isplines   # copy isplines.py here
    _ISPLINES_AVAILABLE = True
except ImportError:
    _ISPLINES_AVAILABLE = False


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
    
class WeightedBrierScoreLoss(nn.Module):
    """
    Brier score with per-column Gaussian weights that emphasise a target CDF level.

    Up-weights λ_j values whose empirical rank in the mini-batch is near `center`,
    focusing CDF accuracy where p-values are computed (near the critical α level).

    Parameters
    ----------
    center : float
        Target CDF level to emphasise (e.g. 0.9 for a 90% confidence set).
    bandwidth : float
        Width of the Gaussian emphasis window. Default 0.1.
    """
    def __init__(self, center: float = 0.9, bandwidth: float = 0.1):
        super().__init__()
        self.center    = center
        self.bandwidth = bandwidth

    def forward(
            self,
            cdf_vals:   torch.Tensor,   # (n, n)
            lambda_obs: torch.Tensor,   # (n,)
    ) -> torch.Tensor:
        n = lambda_obs.shape[0]
        indicators = (lambda_obs.unsqueeze(1) <= lambda_obs.unsqueeze(0)).float()
        mask = ~torch.eye(n, dtype=torch.bool, device=lambda_obs.device)
        # empirical rank of each λ_j as a proxy for its CDF level
        ranks = torch.argsort(torch.argsort(lambda_obs)).float() / max(n - 1, 1)  # (n,) in [0,1]
        col_weights = torch.exp(-0.5 * ((ranks - self.center) / self.bandwidth) ** 2)
        col_weights = col_weights / col_weights.sum()
        sq_err = (cdf_vals - indicators) ** 2    # (n, n)
        # weight along column axis (j = λ level), exclude diagonal
        return (sq_err * col_weights.unsqueeze(0))[mask].sum()


# TODO: Check the weighting function
class WeightedPinballLoss(nn.Module):
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

        alpha_grid = (1-2e-4)*torch.rand(n_alpha) + 1e-4 # torch.linspace(1e-4, 1 - 1e-4, n_alpha)   # (M,)
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
            (1 - alpha_row) * residuals,
            -alpha_row * residuals,
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
        return mu - (1/kappa) * torch.log(alpha / (1 - alpha))
    
# class BetaNetwork(nn.Module):
#     """
#     Shallow neural network for learning mapping theta to sigmoid parameters beta = (mu, log_kappa).
#     """
#     def __init__(self, 
#                  theta_dim: int,
#                  hidden_dim: int = 32,
#                  n_hidden: int = 2) -> None:
#         super().__init__()
#         layers = [nn.Linear(theta_dim, hidden_dim), nn.Tanh()]
#         for _ in range(n_hidden - 1):
#             layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
#         layers += [nn.Linear(hidden_dim, 2)] # outputs: (mu, log_kappa)
#         self.net = nn.Sequential(*layers)

#     def forward(self, theta: torch.Tensor) -> torch.Tensor:
#         return self.net(theta)

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
        activation: str = 'tanh',
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
    
class ISplineWeightNetwork(nn.Module):
    """
    Feed-forward network mapping θ → softmax weights for an I-spline CDF.

    Output:
      - weights : (n, n_basis) via softmax — non-negative, sum to 1 per row.

    This guarantees F̃(0; θ) = 0 and F̃(1; θ) = 1 by construction (I-spline
    boundary conditions), with no clamping needed and gradients always flowing.
    """
    _ACTIVATIONS = {'tanh': nn.Tanh, 'elu': nn.ELU, 'silu': nn.SiLU, 'relu': nn.ReLU}

    def __init__(
        self,
        theta_dim:  int,
        n_basis:    int,
        hidden_dim: int = 64,
        n_hidden:   int = 2,
        activation: str = 'tanh',
    ):
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self._ACTIVATIONS)}, got '{activation}'.")
        act_cls = self._ACTIVATIONS[activation]
        layers  = [nn.Linear(theta_dim, hidden_dim), act_cls()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers += [nn.Linear(hidden_dim, n_basis)]
        self.net     = nn.Sequential(*layers)
        self.n_basis = n_basis

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        out = self.net(theta)                    # (n, n_basis)
        return torch.softmax(out, dim=1)         # (n, n_basis), sums to 1 per row


class ISplineCDFModel(nn.Module):
    """
    Evaluates a softmax-weighted I-spline sum as a CDF.

    Basis functions are fixed (computed via numpy Isplines — no grad needed).
    Gradients flow only through `weights` from ISplineWeightNetwork.

    Because softmax weights sum to 1 and each I-spline basis I_k satisfies
    I_k(0)=0 and I_k(1)=1, the output is always in [0,1] without any clamping.

    The domain is normalised to [0,1] using the empirical min/max of the
    training test statistics, stored on ParametricCDFEstimator after fit().

    Parameters
    ----------
    n_knots : int
        Number of equally-spaced interior+boundary knots on [0,1].
        Number of basis functions = n_knots - 2 + order.
    order : int
        Spline order = degree + 1.  Default 3 (quadratic pieces, C1 joints).
    """

    def __init__(self, n_knots: int = 6, order: int = 3):
        super().__init__()
        if not _ISPLINES_AVAILABLE:
            raise ImportError(
                "ISplineCDFModel requires Isplines. "
                "Copy isplines.py to lf2i/calibration/isplines.py first."
            )
        self.order   = order
        self.n_knots = n_knots
        self.mesh    = np.linspace(0.0, 1.0, n_knots)
        self.n_basis = n_knots - 2 + order

    # ── basis evaluation (numpy, no autograd) ─────────────────────────────────

    def _eval_basis(self, lambda_norm: np.ndarray) -> np.ndarray:
        """
        Evaluate all n_basis I-spline basis functions at normalised λ values.
        lambda_norm : (n,) float — values outside [0,1] extrapolate naturally.
        returns     : (n, n_basis)
        """
        lam = np.clip(lambda_norm, 0.0, 1.0)   # I-spline domain is [0,1]; clamp to boundary values
        isp = Isplines(self.order, self.mesh, lam)
        return np.stack([isp.I(i + 1) for i in range(self.n_basis)], axis=1)

    # ── forward: element-wise CDF (used in predict_proba) ─────────────────────

    def forward(
        self,
        lambda_norm: np.ndarray,    # (n,) numpy — no grad needed
        weights:     torch.Tensor,  # (n, n_basis) softmax weights
    ) -> torch.Tensor:              # (n,) CDF values in [0, 1]
        B = torch.FloatTensor(self._eval_basis(lambda_norm)).to(weights.device)
        return (B * weights).sum(dim=1)          # always in [0,1], no clamp needed

    # ── forward_brier: full (n×n) cross-CDF matrix (used in Brier loss) ───────

    def forward_brier(
        self,
        lambda_norm: np.ndarray,    # (n,) numpy
        weights:     torch.Tensor,  # (n, n_basis) softmax weights
    ) -> torch.Tensor:              # (n, n)  out[i,j] = F̃(λ_j; β(θ_i))
        B = torch.FloatTensor(self._eval_basis(lambda_norm)).to(weights.device)
        return weights @ B.T                     # (n, n_basis) @ (n_basis, n) = (n, n)

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
    acceptance_region : str
        'left'  — p-value = 1 − F̃(λ; β(θ)). Use when large λ → rejection
                  (e.g. WALDO, LRT).
        'right' — p-value = F̃(λ; β(θ)). Use when small λ → rejection
                  (e.g. log-posterior).
    hidden_dim : int
        Width of each hidden layer in BetaNetwork. Default 64.
    n_hidden : int
        Number of hidden layers. Default 2.
    activation : str
        Hidden layer activation. One of 'elu', 'tanh', 'silu', 'relu'.
        Default 'tanh'.
    loss : str
        Loss function. One of 'brier', 'pinball'. Default 'brier'.
    n_alpha : int
        Quadrature points for 'pinball'. Default 500.
    weight_fn : str or callable
        Weight function for 'pinball'. One of 'uniform', 'gaussian', 'beta',
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
        acceptance_region: str,
        # architecture
        hidden_dim:  int   = 64,
        n_hidden:    int   = 2,
        activation:  str   = 'tanh',
        # loss
        loss:        str   = 'brier',
        # ── CDF model ─────────────────────────────────────────────────────────────
        cdf_model:   str   = 'sigmoid',   # 'sigmoid' or 'ispline'
        n_knots:     int   = 6,           # ispline only: knots on [0,1]
        spline_order:int   = 3,           # ispline only: order (degree+1)
        n_alpha:     int   = 500,
        weight_fn:   Union[str, callable] = 'gaussian',
        center:      float = 0.1,
        bandwidth:   float = 0.1,
        beta_a:      float = 2.0,
        beta_b:      float = 5.0,
        # Normalization
        normalize:   str   = 'none',
        # optimisation
        epochs:      int   = 500,
        lr:          float = 1e-3,
        batch_size:  int   = 512,
        smooth_reg:  float = 0.0,    # ispline only: L2 penalty on consecutive weight differences
        device:      Optional[str] = None,
        verbose:     bool  = True,
    ):
        if acceptance_region not in ('left', 'right'):
            raise ValueError(
                f"acceptance_region must be 'left' or 'right', got '{acceptance_region}'."
            )
        if loss not in ('brier', 'weighted_brier', 'pinball'):
            raise ValueError(
                f"loss must be 'brier', 'weighted_brier', or 'pinball', got '{loss}'."
            )

        if cdf_model not in ('sigmoid', 'ispline'):
            raise ValueError(
                f"cdf_model must be 'sigmoid' or 'ispline', got '{cdf_model}'."
            )
        if cdf_model == 'ispline':
            warnings.warn(
                "cdf_model='ispline' is deprecated and will be removed in a future version. "
                "Use cdf_model='sigmoid' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if cdf_model == 'ispline' and loss == 'pinball':
            raise ValueError(
                "I-spline CDF has no closed-form quantile function. "
                "Use loss='brier' or 'weighted_brier' with cdf_model='ispline'."
            )
        if normalize not in ('none', 'mean-std', 'min-max', 'percentiles'):
            raise ValueError(
                f"normalize must be one of 'none', 'mean-std', 'min-max', 'percentiles'. "
                f"Got '{normalize}'."
            )
        self.normalize    = normalize
        self.cdf_model    = cdf_model
        self.n_knots      = n_knots
        self.spline_order = spline_order

        self.acceptance_region = acceptance_region
        self.hidden_dim        = hidden_dim
        self.n_hidden          = n_hidden
        self.activation        = activation
        self.loss              = loss
        self.n_alpha           = n_alpha
        self.weight_fn         = weight_fn
        self.center            = center if acceptance_region == 'right' else 1 - center   # match target to direction
        self.bandwidth         = bandwidth
        self.beta_a            = beta_a
        self.beta_b            = beta_b
        self.epochs            = epochs
        self.lr                = lr
        self.batch_size        = batch_size
        self.smooth_reg        = smooth_reg
        self.verbose           = verbose

        self.device = torch.device(
            device if device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # set after fit()
        self.beta_net_:     Optional[BetaNetwork]          = None
        self.weight_net_:   Optional[ISplineWeightNetwork] = None
        self.cdf_model_:    Optional[SigmoidCDF]           = None
        self.ispline_model_: Optional[ISplineCDFModel]     = None
        self.lambda_min_:   Optional[float]                = None
        self.lambda_max_:   Optional[float]                = None
        self.lambda_lo_:    Optional[float]                = None
        self.lambda_hi_:    Optional[float]                = None
        self.lambda_mean_:  Optional[float]                = None
        self.lambda_std_:   Optional[float]                = None
        self.history_:      Optional[list]                 = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize_ts(self, ts: np.ndarray) -> np.ndarray:
        if self.normalize == 'none':
            return ts
        elif self.normalize == 'mean-std':
            return (ts - self.lambda_mean_) / self.lambda_std_
        else:  # 'min-max' or 'percentiles'
            return (ts - self.lambda_lo_) / (self.lambda_hi_ - self.lambda_lo_ + 1e-8)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        test_statistics: np.ndarray,   # (n,)
        poi:             np.ndarray,   # (n,) or (n, d)
    ) -> 'ParametricCDFEstimator':
        """
        Fit the parametric CDF to calibration test statistics.

        Parameters
        ----------
        test_statistics : np.ndarray
            Shape (n,). Calibration test statistics λ(x_i; θ_i).
        poi : np.ndarray
            Shape (n,) or (n, d). Corresponding parameters of interest θ_i.

        Returns
        -------
        self
        """
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)
        theta_dim = poi.shape[1]

        # ── initialise model ──────────────────────────────────────────────────
        if self.cdf_model == 'sigmoid':

            if self.normalize == 'mean-std':
                self.lambda_mean_ = float(test_statistics.mean())
                self.lambda_std_  = float(test_statistics.std()) + 1e-8
            elif self.normalize == 'min-max':
                self.lambda_lo_ = float(test_statistics.min())
                self.lambda_hi_ = float(test_statistics.max())
            elif self.normalize == 'percentiles':
                self.lambda_lo_ = float(np.percentile(test_statistics, 0.1))
                self.lambda_hi_ = float(np.percentile(test_statistics, 99.9))

            self.beta_net_  = BetaNetwork(
                theta_dim, self.hidden_dim, self.n_hidden, self.activation
            ).to(self.device)
            self.cdf_model_ = SigmoidCDF().to(self.device)
            model_params    = self.beta_net_.parameters()

        else:   # ispline
            self.ispline_model_ = ISplineCDFModel(
                n_knots=self.n_knots, order=self.spline_order
            ).to(self.device)
            self.weight_net_ = ISplineWeightNetwork(
                theta_dim   = theta_dim,
                n_basis     = self.ispline_model_.n_basis,
                hidden_dim  = self.hidden_dim,
                n_hidden    = self.n_hidden,
                activation  = self.activation,
            ).to(self.device)
            # store normalisation range from training data
            self.lambda_min_ = float(test_statistics.min())
            self.lambda_max_ = float(test_statistics.max())
            model_params     = self.weight_net_.parameters()

        optimizer = torch.optim.Adam(model_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # ── criterion ─────────────────────────────────────────────────────────
        if self.loss == 'brier':
            criterion = BrierScoreLoss()
        elif self.loss == 'weighted_brier':
            criterion = WeightedBrierScoreLoss(
                center    = self.center,
                bandwidth = self.bandwidth,
            )
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
        # lambda_t = torch.FloatTensor(test_statistics).to(self.device)
        ts_input = self._normalize_ts(test_statistics) if self.cdf_model == 'sigmoid' else test_statistics
        lambda_t = torch.FloatTensor(ts_input).to(self.device)
        theta_t  = torch.FloatTensor(poi).to(self.device)
        loader   = DataLoader(
            TensorDataset(lambda_t, theta_t),
            batch_size = self.batch_size,
            shuffle    = True,
        )

        # ── training loop ──────────────────────────────────────────────────────
        self.history_ = []
        if self.cdf_model == 'sigmoid':
            self.beta_net_.train()
        else:
            self.weight_net_.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for lam_b, theta_b in loader:
                optimizer.zero_grad()

                if self.cdf_model == 'sigmoid':
                    # ── sigmoid path (unchanged) ──────────────────────────────
                    beta      = self.beta_net_(theta_b)
                    mu        = beta[:, 0:1]
                    log_kappa = beta[:, 1:2]
                    if isinstance(criterion, WeightedPinballLoss):
                        alpha_row           = criterion.alpha_grid.unsqueeze(0).to(lam_b.device) # (1, M)
                        predicted_quantiles = self.cdf_model_.quantile(alpha_row, mu, log_kappa)
                        batch_loss          = criterion(predicted_quantiles, lam_b) + log_kappa.mean() # <- This amounts to maximum entropy regularization (entropy for logistic is \propto -log(\kappa), so maximize entropy by minimizing \kappa)
                    else:
                        cdf_vals   = self.cdf_model_(lam_b.unsqueeze(0), mu, log_kappa)
                        batch_loss = criterion(cdf_vals, lam_b)

                else:
                    # ── ispline path ──────────────────────────────────────────
                    weights  = self.weight_net_(theta_b)
                    lam_norm = (
                        (lam_b.detach().cpu().numpy() - self.lambda_min_)
                        / (self.lambda_max_ - self.lambda_min_ + 1e-8)
                    )
                    cdf_vals   = self.ispline_model_.forward_brier(lam_norm, weights)
                    batch_loss = criterion(cdf_vals, lam_b)

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            scheduler.step()
            avg = epoch_loss / len(loader)
            self.history_.append(avg)
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs}  |  loss: {avg:.5f}")

        if self.cdf_model == 'sigmoid':
            self.beta_net_.eval()
        else:
            self.weight_net_.eval()
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
            Column 0 = 1 − p-value, column 1 = p-value.
        """
        if self.cdf_model == 'sigmoid' and self.beta_net_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if self.cdf_model == 'ispline' and self.weight_net_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if X is None:
            raise ValueError("X must be provided.")

        poi = X[:, 1:]
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)

        with torch.no_grad():
            # lambda_t = torch.FloatTensor(X[:, 0]).to(self.device)
            ts_input = self._normalize_ts(X[:, 0]) if self.cdf_model == 'sigmoid' else X[:, 0]
            lambda_t = torch.FloatTensor(ts_input).to(self.device)
            theta_t  = torch.FloatTensor(poi).to(self.device)

            if self.cdf_model == 'sigmoid':
                beta      = self.beta_net_(theta_t)
                mu        = beta[:, 0:1]
                log_kappa = beta[:, 1:2]
                cdf_vals  = self.cdf_model_(
                    lambda_t.unsqueeze(1), mu, log_kappa
                ).squeeze(1)

            else:   # ispline
                weights  = self.weight_net_(theta_t)
                lam_norm = (
                    (X[:, 0] - self.lambda_min_)
                    / (self.lambda_max_ - self.lambda_min_ + 1e-8)
                )
                cdf_vals = self.ispline_model_.forward(lam_norm, weights)

        if self.acceptance_region == 'left':
            pvalues = 1.0 - cdf_vals.cpu().numpy()
        else:
            pvalues = cdf_vals.cpu().numpy()

        return np.column_stack([1.0 - pvalues, pvalues])