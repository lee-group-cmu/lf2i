from typing import Union, Optional, Dict, Tuple

import torch
from torch.distributions import Categorical, MultivariateNormal, Normal, Uniform

from lf2i.simulator._base import Simulator


class GaussianMixtureLocation(Simulator):
    """Gaussian mixture simulator for inference of a location parameter.

    The likelihood is a two-component Gaussian mixture:
        X | theta ~ w_0 * N(theta, sigma_0^2 I) + w_1 * N(theta, sigma_1^2 I)
    where the mixture weights and component scales are fixed.

    The prior over theta is an isotropic Gaussian: theta ~ N(prior_loc, prior_cov * I).

    Parameters
    ----------
    poi_space_bounds : Dict[str, float]
        Bounds for the parameter grid ('low' and 'high'). Each dimension shares the same bounds.
    poi_grid_size : int
        Number of points in the parameter grid.
        For poi_dim > 1 the actual grid size may be slightly larger (rounded up per dimension).
    poi_dim : int
        Dimensionality of the location parameter theta.
    data_dim : int
        Dimensionality of each observation (must equal poi_dim).
    batch_size : int
        Number of observations drawn for each parameter value.
    mixture_weights : torch.Tensor, optional
        Two-element tensor of mixture weights. Defaults to [0.5, 0.5].
    mixture_scales : torch.Tensor, optional
        Two-element tensor of per-component standard deviations. Defaults to [1.0, 0.1].
    prior_kwargs : Dict[str, Union[float, torch.Tensor]], optional
        Must contain 'loc' (scalar or Tensor) and 'cov' (scalar) for the isotropic Gaussian prior.
        Defaults to standard normal: loc=0, cov=1.
    """

    def __init__(
        self,
        poi_space_bounds: Dict[str, float],
        poi_grid_size: int,
        poi_dim: int,
        data_dim: int,
        batch_size: int,
        mixture_weights: Optional[torch.Tensor] = None,
        mixture_scales: Optional[torch.Tensor] = None,
        prior_kwargs: Optional[Dict[str, Union[float, torch.Tensor]]] = None,
    ):
        super().__init__(poi_dim=poi_dim, data_dim=data_dim, batch_size=batch_size, nuisance_dim=0)

        self.poi_space_bounds = poi_space_bounds

        if poi_dim == 1:
            self.poi_grid = torch.linspace(
                start=poi_space_bounds['low'],
                end=poi_space_bounds['high'],
                steps=poi_grid_size,
            )
        else:
            steps = int(poi_grid_size ** (1 / poi_dim)) + 1
            self.poi_grid = torch.cartesian_prod(
                *[
                    torch.linspace(poi_space_bounds['low'], poi_space_bounds['high'], steps)
                    for _ in range(poi_dim)
                ]
            )
        self.poi_grid_size = self.poi_grid.shape[0]

        self.qr_prior = Uniform(
            low=poi_space_bounds['low'] * torch.ones(poi_dim),
            high=poi_space_bounds['high'] * torch.ones(poi_dim),
        )

        self.mixture_weights = mixture_weights if mixture_weights is not None else torch.tensor([0.5, 0.5])
        self.mixture_scales = mixture_scales if mixture_scales is not None else torch.tensor([1.0, 0.1])

        if prior_kwargs is None:
            prior_kwargs = {'loc': 0.0, 'cov': 1.0}
        self.prior = MultivariateNormal(
            loc=torch.ones(poi_dim) * prior_kwargs['loc'],
            covariance_matrix=torch.eye(poi_dim) * prior_kwargs['cov'],
        )

    def likelihood(self, loc: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """Return the mixture likelihood distribution at a given location parameter.

        Parameters
        ----------
        loc : torch.Tensor
            Shape (poi_dim,) — a single parameter value.

        Returns
        -------
        torch.distributions.MixtureSameFamily
            The mixture distribution X | theta = loc.
        """
        mix = Categorical(probs=self.mixture_weights)
        comp = Normal(
            loc=loc.unsqueeze(0).expand(len(self.mixture_weights), self.poi_dim),
            scale=self.mixture_scales.unsqueeze(1).expand(len(self.mixture_weights), self.poi_dim),
        )
        return torch.distributions.MixtureSameFamily(mix, torch.distributions.Independent(comp, 1))

    def _simulate(self, params: torch.Tensor) -> torch.Tensor:
        """Draw `batch_size` observations for each row of `params`.

        Parameters
        ----------
        params : torch.Tensor
            Shape (size, poi_dim).

        Returns
        -------
        torch.Tensor
            Shape (size, batch_size, data_dim).
        """
        size = params.shape[0]
        # Draw mixture component index per (param, observation) pair
        idx = Categorical(probs=self.mixture_weights).sample(sample_shape=(size, self.batch_size))  # (size, batch_size)
        scale = self.mixture_scales[idx]  # (size, batch_size)

        loc = params.unsqueeze(1).expand(size, self.batch_size, self.poi_dim)  # (size, batch_size, poi_dim)
        scale = scale.unsqueeze(-1).expand_as(loc)  # (size, batch_size, poi_dim)

        return Normal(loc=loc, scale=scale).sample()  # (size, batch_size, poi_dim)

    def simulate_for_test_statistic(self, size: int, estimation_method: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if estimation_method not in ('likelihood', 'prediction', 'posterior'):
            raise ValueError(
                f"Only one of ['likelihood', 'prediction', 'posterior'] is supported, got {estimation_method}"
            )
        params = self.prior.sample(sample_shape=(size,)).reshape(size, self.poi_dim)
        samples = self._simulate(params)
        return params, samples  # (size, poi_dim), (size, batch_size, data_dim)

    def simulate_for_critical_values(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.qr_prior.sample(sample_shape=(size,)).reshape(size, self.poi_dim)
        samples = self._simulate(params)
        return params, samples

    def simulate_for_diagnostics(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(size)
