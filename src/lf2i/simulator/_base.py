from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

import numpy as np
import torch


class Simulator(ABC):
    """Base class for simulators. This is a template from which every simulator should inherit.

    Parameters
    ----------
    poi_dim : int
        Dimensionality of the space of parameters of interest (poi).
    data_dim : int
        Dimensionality of a single datapoint X.
    data_sample_size: int
        Size of a set of datapoints from a specific parameter configuration. Must be the same for observations and simulations.
        A simulated/observed sample from a specific parameter configuration will have dimensions `(data_sample_size, data_dim)`.
    nuisance_dim : Optional[int], optional
        Dimensionality of the space of nuisance parameters (systematics), by default 0. 
    """

    def __init__(
        self,
        poi_dim: int,
        data_dim: int,
        data_sample_size: int,
        nuisance_dim: Optional[int] = None
    ) -> None:
        
        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim if nuisance_dim is not None else 0
        self.data_dim = data_dim
        self.data_sample_size = data_sample_size

    @abstractmethod
    def simulate_for_test_statistic(
        self, 
        B: int,
        estimation_method: str
    ) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        """Simulate a training set used to estimate the test statistic.

        Parameters
        ----------
        B : int
            Number of simulations. Note that each simulated data point will have dimensions `(data_sample_size, data_dim)`.
        estimation_method : str
            The method with which the test statistic is estimated. 
            If likelihood-based test statistics are used, such as ACORE and BFF, then 'likelihood'. 
            If prediction/posterior-based test statistics are used, such as WALDO, then 'prediction' or 'posterior'.

        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor]]
            Y, parameters, samples (depending on the specific needs of the test statistic).
        """
        pass
    
    @abstractmethod
    def simulate_for_critical_values(
        self,
        B_prime: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Simulate a training set used to estimate the critical values via quantile regression.

        Parameters
        ----------
        B_prime : int
            Number of simulations. Note that each simulation will have dimensions `(data_sample_size, data_dim)`.

        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            Parameters, samples.
        """
        pass
    
    @abstractmethod
    def simulate_for_diagnostics(
        self,
        B_doubleprime: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Simulate a training set used to estimate conditional coverage via the diagnostics branch.

        Parameters
        ----------
        B_doubleprime : int
            Number of simulations. Note that each simulation will have dimensions `(data_sample_size, data_dim)`.

        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            Parameters, samples.
        """
        pass
