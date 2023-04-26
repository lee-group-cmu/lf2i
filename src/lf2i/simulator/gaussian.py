from typing import Union, Optional, Dict, Tuple

import numpy as np
import torch
from torch.distributions import Distribution, MultivariateNormal, Uniform
from pyro import distributions as pdist
from sbi.utils import MultipleIndependent

from lf2i.simulator._base import Simulator


class GaussianMean(Simulator):
    """Gaussian simulator with fixed covariance structure. Supports any parameter dimensionality and data sample size.
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
    data_sample_size : int
        Size `n` of each sample generated from a specific parameter.
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
        data_sample_size: int,
        prior_kwargs: Optional[Dict[str, Union[float, torch.Tensor]]] = None
    ):
        super().__init__(poi_dim=poi_dim, data_dim=data_dim, data_sample_size=data_sample_size) 

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
            
    def simulate_for_test_statistic(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.prior.sample(sample_shape=(b, )).reshape(b, self.poi_dim)
        # shape is interpreted as 'draw `shape` samples for each d-dim element of params'
        samples = self.likelihood(loc=params).sample(sample_shape=(self.data_sample_size, ))
        return params, torch.transpose(samples, 0, 1)  # (b, data_sample_size, data_dim)
    
    def simulate_for_critical_values(self, b_prime: int) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.qr_prior.sample(sample_shape=(b_prime, )).reshape(b_prime, self.poi_dim)
        samples = self.likelihood(loc=params).sample(sample_shape=(self.data_sample_size, ))
        return params, torch.transpose(samples, 0, 1)  # (b_prime, data_sample_size, data_dim)
    
    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)


class GaussianMeanFirstCoord(Simulator):

    def __init__(
        self,
        likelihood_cov: torch.Tensor, 
        poi_prior: Distribution,
        poi_bounds: Dict[str, float], 
        poi_grid_size: int,
        nuisance_prior: Distribution,
        nuisance_bounds: int,
        data_sample_size: int
    ):
        super().__init__(poi_dim=1, data_dim=2, data_sample_size=data_sample_size, nuisance_dim=1)

        self.poi_bounds = poi_bounds
        self.poi_grid = torch.linspace(start=poi_bounds['low'], end=poi_bounds['high'], steps=poi_grid_size)
        self.poi_grid_size = poi_grid_size
        
        # sampling parameters to estimate critical values via quantile regression
        self.poi_qr_prior = Uniform(low=poi_bounds['low']*torch.ones(1), high=poi_bounds['high']*torch.ones(1))
        self.nuisance_qr_prior = Uniform(low=nuisance_bounds['low']*torch.ones(1), high=nuisance_bounds['high']*torch.ones(1))
        
        self.likelihood = lambda poi, nuisance: MultivariateNormal(loc=torch.hstack((poi, nuisance)), covariance_matrix=likelihood_cov)
        self.poi_prior = poi_prior
        self.nuisance_prior = nuisance_prior
    
    def simulate_for_test_statistic(
        self, 
        b: int,
        estimation_method: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if estimation_method == 'likelihood':
            raise NotImplementedError
        elif estimation_method == 'likelihood_bap': # best average power (Heinrich, 2022)
            # sample nulls
            poi_h0 = self.poi_prior.sample(sample_shape=(1, )).reshape(1, 1)
            nuisance_h0 = self.nuisance_prior.sample(sample_shape=(1, )).reshape(1, 1)
            samples_h0 = self.likelihood(poi=poi_h0, nuisance=nuisance_h0).sample(sample_shape=(b//2, ))
            labels_h0 = torch.ones(b//2)

            # sample alternatives
            scale = self.nuisance_prior.sample(sample_shape=(1, )).item()  # lukas samples scale from nuisance range 
            poi_h1_left, poi_h1_right = poi_h0-scale, poi_h0+scale
            samples_h1_left = self.likelihood(poi=poi_h1_left, nuisance=nuisance_h0).sample(sample_shape=(b//4, ))
            samples_h1_right = self.likelihood(poi=poi_h1_right, nuisance=nuisance_h0).sample(sample_shape=(b//4, ))
            labels_h1 = torch.zeros((b//4)*2)
            params_samples = torch.hstack((
                torch.tile(poi_h0, dims=((b//2)+(b//4)*2, 1)),
                torch.vstack((
                    samples_h0.reshape(b//2, self.data_dim), 
                    samples_h1_left.reshape(b//4, self.data_dim), 
                    samples_h1_right.reshape(b//4, self.data_dim)
                ))
            ))

            permuted_index = torch.from_numpy(np.random.choice(
                a=range(params_samples.shape[0]), size=params_samples.shape[0], replace=False
            ))
            return params_samples[permuted_index, :], torch.concat((labels_h0, labels_h1))[permuted_index]
        elif estimation_method in ['prediction', 'posterior']:
            raise NotImplementedError
        else:
            raise ValueError(f"Only one of ['likelihood', 'likelihood_bap', 'prediction', 'posterior'] is supported, got {estimation_method}")
    
    def simulate_for_critical_values(
        self, 
        b_prime: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = self.poi_prior.sample(sample_shape=(b_prime, )).reshape(-1, 1)
        nuisance = self.nuisance_prior.sample(sample_shape=(b_prime, )).reshape(-1, 1)
        samples = self.likelihood(poi=poi, nuisance=nuisance).sample(sample_shape=(self.data_sample_size, ))
        return poi, nuisance, torch.transpose(samples, 0, 1)
    
    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)
