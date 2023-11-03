from typing import Union, Optional, Dict, Tuple

import numpy as np
import torch
from torch.distributions import Distribution, MultivariateNormal, Uniform
from pyro import distributions as pdist
from sbi.utils import MultipleIndependent

from lf2i.simulator._base import Simulator


class GaussianMean(Simulator):
    """Gaussian simulator with fixed covariance structure. Supports any parameter dimensionality and batch size.
    Assumes diagonal covariance matrix.

    Parameter of interest: mean.

    Parameters
    ----------
    likelihood_cov : Union[float, torch.Tensor]
        Covariance structure of the likelihood. 
        If `float` or `Tensor` with only one value, it is interpreted as the (equal) variance for each component. 
        If `Tensor` with `poi_dim` values, the i-th one is the variance of the i-th component.
    prior : str
        Either `gaussian` or `uniform`.
    poi_space_bounds : Dict[str, float]
        Bounds of the space of parameters of interest. Used to construct the parameter grid, which contains the evaluation points for the confidence regions.
        Must contain `low` and `high`. Assumes that each dimension of the parameter has the same bounds.
    poi_grid_size : int
        Number of points in the parameter grid. 
        If `(poi_grid_size)**(1/poi_dim)` is not an integer, the closest larger number is chosen. 
        E.g., if `poi_grid_size == 1000` and `poi_dim == 2`, then the grid will have `32 x 32 = 1024` points.
    poi_dim : int
        Dimensionality of the parameter of interest.
    data_dim : int
        Dimensionality of the data.
    batch_size : int
        Size of each batch of samples generated from a specific parameter value.
    prior_kwargs : Optional[Dict[Union[float, torch.Tensor]]], optional
        If `prior == 'gaussian'`, must contain `loc` and `cov`. These can be scalars or tensors, as specified for `likelihood_cov`.
        If `prior == 'uniform'`, must contain 'low' and 'high'. Assumes that each dimension of the parameter has the same bounds. If None, `parameter_space_bounds` is used.
    """
    
    def __init__(
        self,
        likelihood_cov: Union[float, torch.Tensor], 
        prior: str,
        poi_space_bounds: Dict[str, float], 
        poi_grid_size: int,
        poi_dim: int,
        data_dim: int,
        batch_size: int,
        prior_kwargs: Optional[Dict[str, Union[float, torch.Tensor]]] = None
    ):
        super().__init__(poi_dim=poi_dim, data_dim=data_dim, batch_size=batch_size, nuisance_dim=0) 

        self.poi_space_bounds = poi_space_bounds
        if poi_dim == 1:
            self.poi_grid = torch.linspace(start=poi_space_bounds['low'], end=poi_space_bounds['high'], steps=poi_grid_size)
        else:
            self.poi_grid = torch.cartesian_prod(
                *[torch.linspace(start=poi_space_bounds['low'], end=poi_space_bounds['high'], steps=int(poi_grid_size**(1/poi_dim))+1) for _ in range(poi_dim)]
            )
        self.poi_grid_size = self.poi_grid.shape[0]
        
        # sampling parameters to estimate critical values via quantile regression
        if poi_dim == 1:
            self.qr_prior = Uniform(low=poi_space_bounds['low']*torch.ones(1), high=poi_space_bounds['high']*torch.ones(1))
        else:
            self.qr_prior = MultipleIndependent(dists=[
                pdist.Uniform(low=poi_space_bounds['low']*torch.ones(1), high=poi_space_bounds['high']*torch.ones(1)) for _ in range(poi_dim)
            ])
        
        self.likelihood = lambda loc: MultivariateNormal(loc=loc, covariance_matrix=torch.eye(n=poi_dim)*likelihood_cov)
        if prior == 'gaussian': 
            self.prior = MultivariateNormal(loc=torch.ones(size=(poi_dim, ))*prior_kwargs['loc'], 
                                            covariance_matrix=torch.eye(n=poi_dim)*prior_kwargs['cov'])
        elif prior == 'uniform':
            if prior_kwargs is None:
                prior_kwargs = poi_space_bounds
            if poi_dim == 1:
                self.prior = Uniform(low=prior_kwargs['low']*torch.ones(1), high=prior_kwargs['high']*torch.ones(1))
            else:
                self.prior = MultipleIndependent(dists=[
                    pdist.Uniform(low=prior_kwargs['low']*torch.ones(1), high=prior_kwargs['high']*torch.ones(1)) for _ in range(poi_dim)
                ])
        else: 
            raise NotImplementedError
            
    def simulate_for_test_statistic(self, size: int, estimation_method: str) -> Tuple[torch.Tensor]:
        if estimation_method == 'likelihood':
            raise NotImplementedError
        elif estimation_method in ['prediction', 'posterior']:
            params = self.prior.sample(sample_shape=(size, )).reshape(size, self.poi_dim)
            # shape is interpreted as 'draw `shape` samples for each d-dim element of params'
            samples = self.likelihood(loc=params).sample(sample_shape=(self.batch_size, ))
            return params, torch.transpose(samples, 0, 1)  # (size, batch_size, data_dim)
        else:
            raise ValueError(f"Only one of ['likelihood', 'prediction', 'posterior'] is supported, got {estimation_method}")

    def simulate_for_critical_values(self, size: int) -> Tuple[torch.Tensor]:
        params = self.qr_prior.sample(sample_shape=(size, )).reshape(size, self.poi_dim)
        samples = self.likelihood(loc=params).sample(sample_shape=(self.batch_size, ))
        return params, torch.transpose(samples, 0, 1)  # (size, batch_size, data_dim)
    
    def simulate_for_diagnostics(self, size: int) -> Tuple[torch.Tensor]:
        return self.simulate_for_critical_values(size)
