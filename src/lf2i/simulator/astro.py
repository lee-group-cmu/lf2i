from typing import Union, Dict, Sequence, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sbi.utils.torchutils import BoxUniform

from lf2i.simulator._base import Simulator


class GalaxySpectra(Simulator):

    DEFAULT_PRIOR_BOUNDS = {
        'tot_stellar_mass': [8, 13],  # total stellar mass formed
        'metallicity': [-2.0, 0.2],  # stellar metallicity in units of log(z/z_)
        'dust': [0.1, 1],  # Diffuse dust optical depth # NOTE: I think this is actually tau, not dust
        'age': [0.01, 4.0],  # Age of Galaxy (Gyr)
        'tau': [-1.0, 1.0]  # e-folding time of SFH (Gyr)  # NOTE: I think this is actually dust, not tau
    }

    def __init__(
        self,
        data_dir: str,
        poi_dim: int = 5,
        data_dim: int = 2,
        data_sample_size: int = 1,
        nuisance_dim: int = 0,
        custom_poi_prior_bounds: Dict[str, Sequence] = None,
        gaussian_noise: float = 0.05
    ) -> None:
        super().__init__(poi_dim=poi_dim, data_dim=data_dim, data_sample_size=data_sample_size, nuisance_dim=nuisance_dim)

        self.data_dir = Path(data_dir).resolve()
        
        if custom_poi_prior_bounds is None:
            self.poi_prior_bounds = self.DEFAULT_PRIOR_BOUNDS
        self._poi_ordered = ['tot_stellar_mass', 'age', 'tau', 'metallicity', 'dust']  # TODO: swap 'tau' and 'dust' if mistake confirmed
        self.prior = BoxUniform(
            low=torch.as_tensor([self.poi_prior_bounds[poi][0] for poi in self._poi_ordered]),
            high=torch.as_tensor([self.poi_prior_bounds[poi][1] for poi in self._poi_ordered]),
        )

        self.poi_data = self._transform_poi_data(pd.read_csv(self.data_dir / 'table.txt', header=None))
        self.sed_data = np.log10(pd.read_csv(self.data_dir / 'sed_resamp_all.txt', header=None, delimiter=' ').to_numpy())
        self.uncertainty_factor = np.log10(1 + gaussian_noise)

    def _simulate(
        self, 
        poi: torch.Tensor
    ) -> torch.Tensor:
        poi = poi.reshape(1, self.poi_dim)
        # pick the poi in data that is closer to the one sampled by the prior
        distances = np.linalg.norm(self.poi_data - poi.numpy(), axis=1)
        min_index = np.argmin(distances)
        # sample the corresponding SED (length = 138) and add Gaussian noise
        return torch.from_numpy(np.random.normal(self.sed_data[min_index], self.uncertainty_factor))

    def _transform_poi_data(
        self,
        poi_data: pd.DataFrame
    ) -> np.ndarray:
        """ Just copying Gourav's code here.
        """
        mass = np.log10(poi_data[1].values)
        met = poi_data[4].values
        age = poi_data[2].values
        tau = np.log10(poi_data[3].values)  # NOTE: I think this is actually dust, not tau
        dust = poi_data[5].values  # NOTE: I think this is actually tau, not dust
        return np.array([
            np.array([mass_value, age[idx], tau[idx], met[idx], dust[idx]]) for idx, mass_value in enumerate(mass)
        ])

    def __call__(
        self, 
        poi: torch.Tensor
    ) -> torch.Tensor:
        return self._simulate(poi=poi)

    def simulate_for_test_statistic(
        self, 
        B: int, 
        estimation_method: str
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError
    
    def simulate_for_critical_values(self, B_prime: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        raise NotImplementedError
    
    def simulate_for_diagnostics(self, B_doubleprime: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        raise NotImplementedError
