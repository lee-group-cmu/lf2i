from typing import Union, Optional, Dict, Tuple

import torch
from torch.distributions import MultivariateNormal, Uniform
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
        If `Tensor` with `param_dim` values, the i-th one is the variance of the i-th component.
    prior : str
        Either `gaussian` or `uniform`.
    parameter_space_bounds : Dict[str, float]
        Bounds of the parameter space. Used to construct the parameter grid, which contains the evaluation points for the confidence regions.
        Must contain `low` and `high`. Assumes that each dimension of the parameter has the same bounds.
    param_grid_size : int
        Number of points in the parameter grid. 
        If `(param_grid_size)**(1/param_dim)` is not an integer, the closest larger number is chosen. 
        E.g., if `param_grid_size == 1000` and `param_dim == 2`, then the grid will have `32 x 32 = 1024` points.
    param_dim : int
        Dimensionality of the parameter of interest.
    data_dim : int
        Dimensionality of the data.
    data_sample_size : int
        Size `n` of each sample generated from a specific parameter.
    prior_kwargs : Optional[Dict[Union[float, torch.Tensor]]], optional
        If `prior == 'gaussian'`, must contain `loc` and `cov`. These can be scalars or tensors, as specified for `likelihood_cov`.
        If `prior == 'uniform'`, must contain 'low' and 'high'. Assumes that each dimension of the parameter has the same bounds. If None, 
    """
    
    def __init__(
        self,
        likelihood_cov: Union[float, torch.Tensor], 
        prior: str,
        parameter_space_bounds: Dict[str, float], 
        param_grid_size: int,
        param_dim: int,
        data_dim: int,
        data_sample_size: int,
        prior_kwargs: Optional[Dict[str, Union[float, torch.Tensor]]] = None
    ):
        super().__init__(param_dim=param_dim, data_dim=data_dim, data_sample_size=data_sample_size) 

        self.parameter_space_bounds = parameter_space_bounds
        if param_dim == 1:
            self.param_grid = torch.linspace(start=parameter_space_bounds['low'], end=parameter_space_bounds['high'], steps=param_grid_size)
        else:
            self.param_grid = torch.cartesian_prod(
                *[torch.linspace(start=parameter_space_bounds['low'], end=parameter_space_bounds['high'], steps=int(param_grid_size**(1/param_dim))+1) for _ in range(param_dim)]
            ).numpy()
        self.param_grid_size = self.param_grid.shape[0]
        
        self.param_dim = param_dim
        self.data_dim = data_dim
        self.data_sample_size = data_sample_size
        
        # sampling parameters to estimate critical values via quantile regression
        if param_dim == 1:
            self.qr_prior = Uniform(low=parameter_space_bounds['low']*torch.ones(1), high=parameter_space_bounds['high']*torch.ones(1))
        else:
            self.qr_prior = MultipleIndependent(dists=[
                pdist.Uniform(low=parameter_space_bounds['low']*torch.ones(1), high=parameter_space_bounds['high']*torch.ones(1)) for _ in range(param_dim)
            ])
        
        self.likelihood = lambda loc: MultivariateNormal(loc=loc, covariance_matrix=torch.eye(n=param_dim)*likelihood_cov)
        if prior == 'gaussian': 
            self.prior = MultivariateNormal(loc=torch.ones(size=(param_dim, ))*prior_kwargs['loc'], 
                                            covariance_matrix=torch.eye(n=param_dim)*prior_kwargs['cov'])
        elif prior == 'uniform':
            if prior_kwargs is None:
                prior_kwargs = parameter_space_bounds
            if param_dim == 1:
                self.prior = Uniform(low=prior_kwargs['low']*torch.ones(1), high=prior_kwargs['high']*torch.ones(1))
            else:
                self.prior = MultipleIndependent(dists=[
                    pdist.Uniform(low=prior_kwargs['low']*torch.ones(1), high=prior_kwargs['high']*torch.ones(1)) for _ in range(param_dim)
                ])
        else: 
            raise NotImplementedError
            
    def simulate_for_test_statistic(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.prior.sample(sample_shape=(b, )).reshape(b, self.param_dim)
        # shape is interpreted as 'draw `shape` samples for each d-dim element of params'
        samples = self.likelihood(loc=params).sample(sample_shape=(self.data_sample_size, ))
        return params, torch.transpose(samples, 0, 1)  # (b, data_sample_size, data_dim)
    
    def simulate_for_critical_values(self, b_prime: int) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.qr_prior.sample(sample_shape=(b_prime, )).reshape(b_prime, self.param_dim)
        samples = self.likelihood(loc=params).sample(sample_shape=(self.data_sample_size, ))
        return params, torch.transpose(samples, 0, 1)  # (b, data_sample_size, data_dim)
    
    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)
