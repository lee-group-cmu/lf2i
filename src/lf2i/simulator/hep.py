from typing import Dict, List, Optional, Tuple

import torch
from torch.distributions import Poisson, Uniform

from lf2i.simulator._base import Simulator


class OnOff(Simulator):
    """Poisson counting experiment ("on-off" problem), as described in https://arxiv.org/abs/2107.03920.

    Parameter of interest is signal strength mu. Nuisance is background scaling factor nu.
    In addition, the following are treated as fixed hyperparameters:

    - Nominally expected signal and background counts s and b.
    - Relationship in measurement time between the two processes tau.

    Parameters
    ----------
    poi_grid_size : int
        Number of points in the parameter grid over mu.
    batch_size : int
        Size of each batch of samples generated from a specific parameter value.
    poi_space_bounds : Optional[Dict[str, float]], optional
        Bounds of mu. Must contain 'low' and 'high'. Defaults to {'low': 0, 'high': 5}.
    nuisance_space_bounds : Optional[Dict[str, float]], optional
        Bounds of nu. Must contain 'low' and 'high'. Defaults to {'low': 0.6, 'high': 1.4}.
    s : Optional[float], optional
        Nominally expected signal count. Defaults to 15.
    b : Optional[float], optional
        Nominally expected background count. Defaults to 70.
    tau : Optional[float], optional
        Relationship in measurement time between the two processes. Defaults to 1.
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
        batch_size: int,
        poi_space_bounds: Optional[Dict[str, float]] = None,
        nuisance_space_bounds: Optional[Dict[str, float]] = None,
        s: Optional[float] = None,
        b: Optional[float] = None,
        tau: Optional[float] = None
    ) -> None:
        super().__init__(poi_dim=1, data_dim=2, batch_size=batch_size, nuisance_dim=1)

        self.s = s or self._S
        self.b = b or self._B
        self.tau = tau or self._TAU

        self.poi_space_bounds = poi_space_bounds or self._MU_RANGE
        self.nuisance_space_bounds = nuisance_space_bounds or self._NU_RANGE
        self.poi_grid = torch.linspace(start=self.poi_space_bounds['low'], end=self.poi_space_bounds['high'], steps=poi_grid_size)

        self.poi_prior = Uniform(low=self.poi_space_bounds['low'], high=self.poi_space_bounds['high'])
        self.nuisance_prior = Uniform(low=self.nuisance_space_bounds['low'], high=self.nuisance_space_bounds['high'])

    @property
    def param_space_bounds(self) -> Dict[str, List[float]]:
        return {'mu': list(self.poi_space_bounds.values()), 'nu': list(self.nuisance_space_bounds.values())}

    def __call__(self, param: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        batch_size = batch_size or self.batch_size
        # poi first, nuisance second
        mu, nu = param[:, :self.poi_dim], param[:, self.poi_dim:]
        size = param.shape[0]
        return torch.dstack((
            # (batch_size, size, 1) -> (size, batch_size, 1)
            torch.transpose(Poisson(rate=nu*self.b + mu*self.s).sample(sample_shape=(batch_size,)).reshape(batch_size, size, 1), 0, 1),  # signal region: Ns
            torch.transpose(Poisson(rate=nu*self.tau*self.b).sample(sample_shape=(batch_size,)).reshape(batch_size, size, 1), 0, 1),  # control region: Nb
        ))

    def simulate_for_test_statistic(
        self,
        size: int,
        estimation_method: str
    ) -> Tuple[torch.Tensor]:
        if estimation_method in ['likelihood', 'prediction', 'posterior']:
            mu = self.poi_prior.sample(sample_shape=(size, )).reshape(-1, 1)
            nu = self.nuisance_prior.sample(sample_shape=(size, )).reshape(-1, 1)
            params = torch.hstack((mu, nu))
            samples = self(param=params, batch_size=1).reshape(size, self.data_dim)
            return params, samples
        else:
            raise ValueError(f"Only one of ['likelihood', 'prediction', 'posterior'] is supported, got {estimation_method}")

    def simulate_for_critical_values(
        self,
        size: int
    ) -> Tuple[torch.Tensor]:
        # parameters
        mu = self.poi_prior.sample(sample_shape=(size, )).reshape(-1, 1)
        nu = self.nuisance_prior.sample(sample_shape=(size, )).reshape(-1, 1)
        params = torch.hstack((mu, nu))
        # samples
        samples = self(param=params, batch_size=self.batch_size)
        assert samples.shape == (size, self.batch_size, self.data_dim)
        return params, samples

    def simulate_for_diagnostics(
        self,
        size: int
    ) -> Tuple[torch.Tensor]:
        return self.simulate_for_critical_values(size)
