from typing import Optional, Dict, Tuple, List

import torch
from torch.distributions import (
    Uniform, 
    Categorical, 
    Poisson
)
from sbi.utils import MultipleIndependent
from lf2i.simulator._base import Simulator


class OnOff(Simulator):
    """
    Poisson counting experiment as detailed in https://arxiv.org/abs/2107.03920.

    POI is signal strength mu. Nuisance is background scaling factor nu. 
    In addition, the following are treated as fixed hyperparameters:
        - Nominally expected signal and background counts s and b.
        - Relationship in measurement time between the two processes tau.
    """

    # default settings
    _MU_RANGE = {'low': 0, 'high': 5}
    _NU_RANGE = {'low': 0.6, 'high': 1.4}
    _S = 15
    _B = 70
    _TAU = 1

    def __init__(
        self,
        poi_grid_size: int,
        poi_space_bounds: Optional[Dict[str, float]] = None,
        nuisance_space_bounds: Optional[Dict[str, float]] = None,
        s: Optional[float] = None,
        b: Optional[float] = None,
        tau: Optional[float] = None
    ) -> None:
        super().__init__(poi_dim=1, data_dim=2, batch_size=1, nuisance_dim=1)

        self.s = s or self._S
        self.b = b or self._B
        self.tau = tau or self._TAU

        self.poi_space_bounds = poi_space_bounds or self._MU_RANGE
        self.nuisance_space_bounds = nuisance_space_bounds or self._NU_RANGE
        self.poi_grid = torch.linspace(start=self.poi_space_bounds['low'], end=self.poi_space_bounds['high'], steps=poi_grid_size)
        
        self.poi_prior = Uniform(low=self.poi_space_bounds['low'], high=self.poi_space_bounds['high'])
        self.nuisance_prior = Uniform(low=self.nuisance_space_bounds['low'], high=self.nuisance_space_bounds['high'])
        self.joint_prior = MultipleIndependent(dists=[
            Uniform(low=torch.ones(1)*self.poi_space_bounds['low'], high=torch.ones(1)*self.poi_space_bounds['high']),
            Uniform(low=torch.ones(1)*self.nuisance_space_bounds['low'], high=torch.ones(1)*self.nuisance_space_bounds['high'])
        ])
        self.likelihood = lambda mu, nu: MultipleIndependent(
            # signal, background
            dists=[Poisson(rate=nu*self.b + mu*self.s), Poisson(rate=nu*self.tau*self.b)]
        )

    @property
    def param_space_bounds(self) -> Dict[str, List[float]]:
        return {'mu': list(self.poi_space_bounds.values()), 'nu': list(self.nuisance_space_bounds.values())}

    def __call__(
        self,
        batch_size: int,  # will draw batch_size samples from each param
        param: torch.Tensor  # poi and nuisance together
    ) -> torch.Tensor:
        # poi first, nuisance second
        mu, nu = param[:, :self.poi_dim], param[:, self.poi_dim:]
        return torch.dstack((
            # B x N x D -> N x B x D
            torch.transpose(Poisson(rate=nu*self.b + mu*self.s).sample(sample_shape=(batch_size, )).reshape(batch_size, mu.shape[0], 1), 0, 1),  # signal
            torch.transpose(Poisson(rate=nu*self.tau*self.b).sample(sample_shape=(batch_size, )).reshape(batch_size, nu.shape[0], 1), 0, 1)  # background
        )) 

    def simulate_for_test_statistic(
        self, 
        size: int, 
        estimation_method: str,
        p: float = 0.5
    ) -> Tuple[torch.Tensor]:
        if estimation_method == 'likelihood':
            # class
            Y = Categorical(probs=torch.Tensor([1-p, p])).sample(sample_shape=(size, )).long()
            # parameters
            mu = self.poi_prior.sample(sample_shape=(size, )).reshape(-1, 1)
            nu = self.nuisance_prior.sample(sample_shape=(size, )).reshape(-1, 1)
            mu_marginal = self.poi_prior.sample(sample_shape=(Y[Y == 0].shape[0], )).reshape(-1, 1)
            nu_marginal = self.nuisance_prior.sample(sample_shape=(Y[Y == 0].shape[0], )).reshape(-1, 1)
            # samples
            samples = self(batch_size=1, param=torch.hstack((mu, nu))).reshape(size, self.data_dim)
            samples_marginal = self(batch_size=1, param=torch.hstack((mu_marginal, nu_marginal))).reshape(mu_marginal.shape[0], self.data_dim)
            samples[Y == 0, :] = samples_marginal
            return Y, torch.hstack((mu, nu)), samples
        elif estimation_method in ['prediction', 'posterior']:
            mu = self.poi_prior.sample(sample_shape=(size, )).reshape(-1, 1)
            nu = self.nuisance_prior.sample(sample_shape=(size, )).reshape(-1, 1)
            samples = self(batch_size=1, param=torch.hstack((mu, nu))).reshape(size, self.data_dim)
            return torch.hstack((mu, nu)), samples
        else:
            raise ValueError(f"Only one of ['likelihood', 'prediction', 'posterior'] is supported, got {estimation_method}")
        
    def simulate_for_critical_values(
        self, 
        size: int
    ) -> Tuple[torch.Tensor]:
        # parameters
        mu = self.poi_prior.sample(sample_shape=(size, )).reshape(-1, 1)
        nu = self.nuisance_prior.sample(sample_shape=(size, )).reshape(-1, 1)
        # samples
        samples = self(batch_size=self.batch_size, param=torch.hstack((mu, nu)))
        assert samples.shape == (size, self.batch_size, self.data_dim)
        return torch.hstack((mu, nu)), samples

    def simulate_for_diagnostics(
        self, 
        size: int
    ) -> Tuple[torch.Tensor]:
        return self.simulate_for_critical_values(size)
