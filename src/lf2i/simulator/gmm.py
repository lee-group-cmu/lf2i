from typing import Union, Optional, Dict, Tuple, Callable

import torch
from torch.distributions import (
    MultivariateNormal,
    Uniform,
    MixtureSameFamily,
    Categorical,
)

from lf2i.simulator._base import Simulator

class GaussianMixture(Simulator):
    """Gaussian Mixture Model simulator with fixed covariance structure and mixing weights.
    
    The component means are functions of the parameter of interest, while the covariance 
    matrix and mixing weights are fixed. This allows modeling complex, multimodal 
    likelihoods where the modes move as parameters change.
    
    Supports a symmetric bimodal default for 1D parameters: X | θ ~ 0.5·N(-θ, Σ) + 0.5·N(θ, Σ).
    General k-component mixtures require a custom ``mean_fn``.

    Parameter of interest: component mean locations (via mean_fn).

    Parameters
    ----------
    likelihood_cov : Union[float, torch.Tensor]
        Covariance structure shared across all mixture components.
        - If `float` or `Tensor` with a single value: interpreted as scalar variance 
          (i.e., Σ = cov·I where I is the data_dim × data_dim identity matrix).
        - If `Tensor` with `data_dim` values: interpreted as per-dimension variances 
          along the diagonal (i.e., Σ = diag(cov)).
        - If `Tensor` of shape `(data_dim, data_dim)`: used directly as the covariance matrix.
    prior : str
        Prior distribution for the parameter of interest. Either ``'gaussian'`` or ``'uniform'``.
    poi_space_bounds : Dict[str, float]
        Bounds of the parameter space. Used to construct the parameter grid for confidence 
        region evaluation. Must contain keys ``'low'`` and ``'high'``.
        Currently assumes that each dimension of the parameter has the same bounds.
    poi_grid_size : int
        Approximate number of points in the parameter grid.
        - For `poi_dim == 1`: exactly `poi_grid_size` points.
        - For `poi_dim > 1`: the grid is constructed as a Cartesian product with 
          `⌈poi_grid_size^(1/poi_dim)⌉` points per dimension, resulting in a grid 
          with approximately `poi_grid_size` total points.
        Example: if `poi_grid_size == 1000` and `poi_dim == 2`, the grid will have 
        `32 × 32 = 1024` points.
    poi_dim : int
        Dimensionality of the parameter of interest.
    data_dim : int
        Dimensionality of a single observation X.
    batch_size : int
        Number of samples drawn per parameter configuration. Each simulation produces 
        a batch of shape `(batch_size, data_dim)`.
    n_components : int, optional
        Number of mixture components (default: 2).
    mixture_weights : Optional[torch.Tensor], optional
        Fixed mixing weights of shape `(n_components,)`. If `None`, uniform weights 
        (1/n_components for each component) are used. Values are automatically normalized 
        to sum to 1.
    mean_fn : Optional[Callable], optional
        A callable that maps parameters to component means:
        
            mean_fn(params: torch.Tensor) -> torch.Tensor
            Input:  (size, poi_dim)
            Output: (size, n_components, data_dim)
        
        If `None`, the symmetric bimodal default is used, which requires `poi_dim == 1` 
        and `n_components == 2`:
        
            μ₁(θ) = -θ · 1_{data_dim}
            μ₂(θ) = +θ · 1_{data_dim}
        
        where 1_{data_dim} is a vector of ones in R^{data_dim}.
        
        For custom mixtures, provide a function that defines how component means depend 
        on the parameter. See Examples section.
    prior_kwargs : Optional[Dict[str, Union[float, torch.Tensor]]], optional
        Additional arguments for the prior distribution.
        - If `prior == 'gaussian'`: must contain `'loc'` and `'cov'` (scalars or tensors).
          Example: `{'loc': 0.0, 'cov': 1.0}` for N(0, I).
        - If `prior == 'uniform'`: optionally contains `'low'` and `'high'`.
          If `None`, `poi_space_bounds` is used.

    Examples
    --------
    **Symmetric bimodal (default mean_fn):**
    
    >>> simulator = GaussianMixture(
    ...     likelihood_cov=1.0,
    ...     prior='uniform',
    ...     poi_space_bounds={'low': -3.0, 'high': 3.0},
    ...     poi_grid_size=1000,
    ...     poi_dim=1,
    ...     data_dim=1,
    ...     batch_size=50,
    ... )
    >>> params, samples = simulator.simulate_for_test_statistic(size=512, estimation_method='likelihood')
    >>> params.shape   # (512, 1)
    >>> samples.shape  # (512, 50, 1)

    **Custom three-component mixture (1D parameter, 1D data):**
    
    >>> def my_mean_fn(params):  # params: (size, 1)
    ...     size = params.shape[0]
    ...     theta = params.squeeze(-1)  # (size,)
    ...     mu1 = (-2 * theta).unsqueeze(-1)  # component 1 at -2θ
    ...     mu2 = torch.zeros(size, 1)         # component 2 at 0
    ...     mu3 = (2 * theta).unsqueeze(-1)    # component 3 at +2θ
    ...     return torch.stack([mu1, mu2, mu3], dim=1)  # (size, 3, 1)
    ...
    >>> simulator3 = GaussianMixture(
    ...     likelihood_cov=0.5,
    ...     prior='uniform',
    ...     poi_space_bounds={'low': 0.0, 'high': 5.0},
    ...     poi_grid_size=1000,
    ...     poi_dim=1,
    ...     data_dim=1,
    ...     batch_size=50,
    ...     n_components=3,
    ...     mean_fn=my_mean_fn,
    ...     mixture_weights=torch.tensor([0.2, 0.5, 0.3]),
    ... )

    **2D parameter, 1D data, custom mean function:**
    
    >>> def bivariate_mean_fn(params):  # params: (size, 2)
    ...     theta0 = params[:, 0]  # (size,)
    ...     theta1 = params[:, 1]  # (size,)
    ...     mu1 = theta0.unsqueeze(-1)  # component 1 moves with theta0
    ...     mu2 = theta1.unsqueeze(-1)  # component 2 moves with theta1
    ...     return torch.stack([mu1, mu2], dim=1)  # (size, 2, 1)
    ...
    >>> simulator2d = GaussianMixture(
    ...     likelihood_cov=1.0,
    ...     prior='uniform',
    ...     poi_space_bounds={'low': -2.0, 'high': 2.0},
    ...     poi_grid_size=1000,
    ...     poi_dim=2,
    ...     data_dim=1,
    ...     batch_size=10,
    ...     n_components=2,
    ...     mean_fn=bivariate_mean_fn,
    ... )

    Notes
    -----
    - The covariance matrix is shared across all components and remains fixed.
    - Component means are the only aspect of the mixture that depends on the parameter.
    - The `mean_fn` is the primary extension point for defining custom mixture behaviors.
    - For `poi_dim == 1`, remember to call `.unsqueeze(-1)` on the grid after initialization
      to ensure consistent shape `(grid_size, 1)` for downstream operations.
    """

    def __init__(self,
                 likelihood_cov: Union[float, torch.Tensor],
                 prior: str,
                 poi_space_bounds: Dict[str, float],
                 poi_grid_size: int,
                 poi_dim: int,
                 data_dim: int,
                 batch_size: int,
                 n_components: int = 2,
                 mixture_weights: Optional[torch.Tensor]=None,
                 mean_fn: Optional[Callable]=None,
                 prior_kwargs: Optional[Dict[str, Union[float, torch.Tensor]]]=None
                 ) -> None:
        super().__init__(poi_dim=poi_dim, data_dim=data_dim, batch_size=batch_size, nuisance_dim=0)

        self.poi_space_bounds = poi_space_bounds
        self.n_components = n_components

        # Parameter eval grid (assuming )
        if poi_dim == 1:
            self.poi_grid = torch.linspace(
                start=poi_space_bounds["low"],
                end=poi_space_bounds["high"],
                steps=poi_grid_size
            )
        else:
            steps_per_dim = int(poi_grid_size ** (1 / poi_dim)) + 1
            self.poi_grid = torch.cartesian_prod(
                *[
                    torch.linspace(
                        start=poi_space_bounds["low"],
                        end=poi_space_bounds["high"],
                        steps=steps_per_dim
                    )
                    for _ in range(poi_dim)
                ]
            )
        self.poi_grid_size = self.poi_grid.shape[0]

        # QR/diagnostic sampling
        self.qr_prior = Uniform(
            low=poi_space_bounds["low"] * torch.ones(poi_dim),
            high=poi_space_bounds["high"] * torch.ones(poi_dim)
        )

        # Covariance matrix
        if isinstance(likelihood_cov, (int, float)):
            self.likelihood_cov = torch.eye(data_dim) * likelihood_cov
        elif isinstance(likelihood_cov, torch.Tensor) and likelihood_cov.numel() == 1:
            self.likelihood_cov = torch.eye(data_dim) * likelihood_cov.item()
        elif isinstance(likelihood_cov, torch.Tensor) and likelihood_cov.ndim == 1:
            self.likelihood_cov = torch.diag(likelihood_cov)
        elif isinstance(likelihood_cov, torch.Tensor) and likelihood_cov.ndim > 1 and likelihood_cov.numel() == data_dim:
            self.likelihood_cov = torch.diagflat(likelihood_cov)
        elif isinstance(likelihood_cov, torch.Tensor) and likelihood_cov.shape == torch.Size([data_dim, data_dim]):
            self.likelihood_cov = likelihood_cov
        elif (isinstance(likelihood_cov, torch.Tensor) and likelihood_cov.shape == torch.Size([n_components, data_dim, data_dim])):
            self.likelihood_cov = likelihood_cov
        else:
            raise ValueError(f"'likelihood_cov' must be single variance value, torch.Tensor of with data_dim elements, or torch.Tensor of shape (data_dim, data_dim).")

        # Mixing weights
        if mixture_weights is None:
            self.mixture_weights = torch.ones(n_components) / n_components
        else:
            self.mixture_weights = mixture_weights.float() / mixture_weights.float().sum()

        # Mean function
        if mean_fn is not None:
            self.mean_fn = mean_fn
        else:
            if poi_dim != 1 or n_components !=2:
                raise ValueError(
                    "The default symmetric bimodal mean_fn requires poi_dim == 1 and" \
                    "n_components == 2. Provide a custom mean_fn for other configurations."
                )
            # default symmetric bimodal mean_fn
            _data_dim = data_dim

            def _symmetric_bimodal(params: torch.Tensor) -> torch.Tensor:
                """params: (size, 1) -> means: (size, 2, data_dim)"""
                theta = params
                neg = (-theta).expand(-1, _data_dim)
                pos = theta.expand(-1, _data_dim)
                return torch.stack([neg, pos], dim = 1)
            
            self.mean_fn = _symmetric_bimodal

        # Priors
        if prior == "uniform":
            if prior_kwargs is None:
                prior_kwargs = poi_space_bounds
            self.prior = Uniform(
                low=torch.ones(poi_dim) * prior_kwargs["low"],
                high=torch.ones(poi_dim) * prior_kwargs["high"]
            )
        elif prior == "gaussian":
            self.prior = MultivariateNormal(
                loc=torch.ones(poi_dim) * prior_kwargs["loc"],
                covariance_matrix=torch.eye(poi_dim) * prior_kwargs["cov"]
            )
        else:
            raise NotImplementedError(
                f"Prior '{prior}' is not supported. Select 'uniform' or 'gaussian'."
            )

    def _make_likelihood(self, params: torch.Tensor) -> MixtureSameFamily:
        """Build a batched GMM distribution for a set of parameters.

        Parameters
        ----------
        params : torch.Tensor
            Shape (size, poi_dim).

        Returns
        -------
        MixtureSameFamily
            A batched distribution with batch shape (size,) and event shape (data_dim,).
            Calling .sample((batch_size,)) yields a tensor of shape 
            (batch_size, size, data_dim).
        """

        size = params.shape[0]

        component_means = self.mean_fn(params) # (size, n_components, data_dim)

        if self.likelihood_cov.ndim == 2:
            component_cov = (
                self.likelihood_cov
                .unsqueeze(0).unsqueeze(0)
                .expand(size, self.n_components, self.data_dim, self.data_dim)
                             )
        else:
            component_cov = (
                self.likelihood_cov
                .unsqueeze(0)
                .expand(size, self.n_components, self.data_dim, self.data_dim)
            )

        mix = Categorical(probs=self.mixture_weights.expand(size, -1))
        comp = MultivariateNormal(loc=component_means, covariance_matrix=component_cov)
        return MixtureSameFamily(mix, comp)

    def _simulate(self, params: torch.Tensor) -> torch.Tensor:
        """Draw batch_size samples from the GMM for each parameter in params.

        Parameters
        ----------
        params : torch.Tensor
            Shape (size, poi_dim).

        Returns
        -------
        torch.Tensor
            Shape (size, batch_size, data_dim).
        """

        likelihood = self._make_likelihood(params)
        samples = likelihood.sample(sample_shape = (self.batch_size,)) # (batch_size, size, data_dim)

        return torch.transpose(samples, 0, 1) # (size, batch_size, data_dim)

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        """Simulate data for a given set of parameters.

        Parameters
        ----------
        params : torch.Tensor
            Shape (size, poi_dim).

        Returns
        -------
        torch.Tensor
            Shape (size, batch_size, data_dim).
        """
        return self._simulate(params)

    def simulate_for_test_statistic(self, 
                                    size: int, 
                                    estimation_method: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the prior and simulate data for test-statistic estimation.

        This method generates training data used to fit the neural network that 
        estimates the test statistic (e.g., likelihood ratio, posterior density).

        Parameters
        ----------
        size : int
            Number of parameter draws from the prior.
        estimation_method : str
            One of 'likelihood', 'prediction', or 'posterior'. Determines which 
            type of test statistic is being estimated.

        Returns
        -------
        params : torch.Tensor
            Shape (size, poi_dim). Parameter values drawn from the prior.
        samples : torch.Tensor
            Shape (size, batch_size, data_dim). Simulated observations for each parameter.
        """
        if estimation_method not in ("likelihood", "prediction", "posterior"):
            raise ValueError(
                f"estimation_method must be one of ['likelihood', 'prediction', 'posterior'], "
                f"got '{estimation_method}'."
            )
        params = self.prior.sample(sample_shape=(size,)).reshape(size, self.poi_dim)
        samples = self._simulate(params)
        return params, samples

    def simulate_for_critical_values(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample uniformly from the parameter space and simulate data for critical-value estimation.

        This method generates calibration data used to estimate critical values or p-values
        via quantile regression or other calibration methods.

        Parameters
        ----------
        size : int
            Number of parameter draws from the uniform distribution over poi_space_bounds.

        Returns
        -------
        params : torch.Tensor
            Shape (size, poi_dim). Parameter values drawn uniformly.
        samples : torch.Tensor
            Shape (size, batch_size, data_dim). Simulated observations for each parameter.
        """
        params = self.qr_prior.sample(sample_shape=(size,)).reshape(size, self.poi_dim)
        samples = self._simulate(params)
        return params, samples

    def simulate_for_diagnostics(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate data for conditional coverage diagnostics.

        This method generates validation data used to assess whether the confidence 
        regions achieve the target coverage level across the parameter space.

        Parameters
        ----------
        size : int
            Number of parameter draws.

        Returns
        -------
        params : torch.Tensor
            Shape (size, poi_dim). Parameter values drawn uniformly.
        samples : torch.Tensor
            Shape (size, batch_size, data_dim). Simulated observations for each parameter.
        
        Notes
        -----
        This delegates to simulate_for_critical_values since both use uniform sampling.
        """

        return self.simulate_for_critical_values(size)