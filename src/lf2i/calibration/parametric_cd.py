from typing import Optional, Tuple, Union

import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


#TODO: Implement with custom grid for evaluation of the integral
class BrierScoreLoss(nn.Module):
    """Brier Score Loss Function.
    """
    def forward(
            self,
            cdf_vals: torch.Tensor,
            lambda_obs: torch.Tensor
    ) -> torch.Tensor:
        indicators = (lambda_obs.unsqueeze(1) <= lambda_obs.unsqueeze(0)).float()
        return ((cdf_vals - indicators)**2).mean()
    
# TODO: Verify this works properly (generated from Claude)
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
        n_alpha:    int             = 500,
        weight_fn:  Union[str, callable] = 'gaussian',
        center:     float           = 0.1,
        bandwidth:  float           = 0.1,
        beta_a:     float           = 2.0,
        beta_b:     float           = 5.0,
    ):
        super().__init__()
        self.n_alpha = n_alpha

        alpha_grid = torch.linspace(1e-4, 1 - 1e-4, n_alpha)   # (M,)
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
            (alpha_row - 1) * residuals,
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
        layers += [nn.Linear(hidden_dim, 2)]   # outputs: (μ, log s)
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:   # (n, 2)
        return self.net(theta)
    
# TODO: Check that this is proper
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
        n_alpha:     int   = 500,
        weight_fn:   Union[str, callable] = 'gaussian',
        center:      float = 0.1,
        bandwidth:   float = 0.1,
        beta_a:      float = 2.0,
        beta_b:      float = 5.0,
        # optimisation
        epochs:      int   = 500,
        lr:          float = 1e-3,
        batch_size:  int   = 512,
        device:      Optional[str] = None,
        verbose:     bool  = True,
    ):
        if acceptance_region not in ('left', 'right'):
            raise ValueError(
                f"acceptance_region must be 'left' or 'right', got '{acceptance_region}'."
            )
        if loss not in ('brier', 'pinball'):
            raise ValueError(
                f"loss must be 'brier' or 'pinball', got '{loss}'."
            )

        self.acceptance_region = acceptance_region
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
        self.verbose           = verbose

        self.device = torch.device(
            device if device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # set after fit()
        self.beta_net_:  Optional[BetaNetwork] = None
        self.cdf_model_: Optional[SigmoidCDF]  = None
        self.history_:   Optional[list]         = None

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

        self.beta_net_  = BetaNetwork(
            theta_dim, self.hidden_dim, self.n_hidden, self.activation
        ).to(self.device)
        self.cdf_model_ = SigmoidCDF().to(self.device)

        optimizer = torch.optim.Adam(self.beta_net_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # ── criterion ────────────────────────────────────────────────────────
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

        # ── data ─────────────────────────────────────────────────────────────
        lambda_t = torch.FloatTensor(test_statistics).to(self.device)
        theta_t  = torch.FloatTensor(poi).to(self.device)
        loader   = DataLoader(
            TensorDataset(lambda_t, theta_t),
            batch_size = self.batch_size,
            shuffle    = True,
        )

        # ── training loop ─────────────────────────────────────────────────────
        self.history_ = []
        self.beta_net_.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for lam_b, theta_b in loader:
                optimizer.zero_grad()

                beta  = self.beta_net_(theta_b)   # (n, 2)
                mu    = beta[:, 0:1]              # (n, 1)
                log_kappa = beta[:, 1:2]              # (n, 1)

                if isinstance(criterion, WeightedPinballLoss):
                    alpha_row           = criterion.alpha_grid.unsqueeze(0)
                    predicted_quantiles = self.cdf_model_.quantile(alpha_row, mu, log_kappa)
                    batch_loss          = criterion(predicted_quantiles, lam_b)
                else:
                    # brier: cdf_vals[i, j] = F̃(λ_j; β(θ_i))
                    cdf_vals   = self.cdf_model_(lam_b.unsqueeze(0), mu, log_kappa)
                    batch_loss = criterion(cdf_vals, lam_b)

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            scheduler.step()
            avg = epoch_loss / len(loader)
            self.history_.append(avg)

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs}  |  loss: {avg:.5f}")

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
            Column 0 = 1 − p-value, column 1 = p-value.
        """
        if self.beta_net_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if X is None:
            raise ValueError("X must be provided.")

        poi = X[:, 1:]
        if poi.ndim == 1:
            poi = poi.reshape(-1, 1)

        with torch.no_grad():
            lambda_t = torch.FloatTensor(X[:, 0]).to(self.device)
            theta_t  = torch.FloatTensor(poi).to(self.device)

            beta     = self.beta_net_(theta_t)                              # (n, 2)
            mu       = beta[:, 0:1]
            log_kappa    = beta[:, 1:2]
            cdf_vals = self.cdf_model_(
                lambda_t.unsqueeze(1), mu, log_kappa
            ).squeeze(1)                                                    # (n,)

        if self.acceptance_region == 'left':
            pvalues = 1.0 - cdf_vals.cpu().numpy()
        else:
            pvalues = cdf_vals.cpu().numpy()

        return np.column_stack([1.0 - pvalues, pvalues])