from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Simulator(ABC):
    """Base class for simulators. This is a template from which every simulator should inherit.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the parameters of interest.
    data_dim : int
        Dimensionality of a single input datapoint.
    data_sample_size: int
        Size of a set of datapoints from a specific parameter configuration. Must be the same for observations and simulations.
        A simulated/observed sample originated from a specific parameter configuration will have dimensions `(data_sample_size, data_dim)`.
    """

    def __init__(
        self,
        param_dim: int,
        data_dim: int,
        data_sample_size: int
    ) -> None:
        self.param_dim = param_dim
        self.data_dim = data_dim
        self.data_sample_size = data_sample_size

    @abstractmethod
    def simulate_for_test_statistic(
        self, 
        b: int
    ) -> Tuple[np.ndarray, ...]:
        """Simulate a training set used to estimate the test statistic.

        Parameters
        ----------
        b : int
            Number of simulations. Note that each simulated data point will have dimensions `(data_sample_size, data_dim)`.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Y, parameters, samples (depending on the specific needs of the test statistic).
        """
        pass
    
    @abstractmethod
    def simulate_for_critical_values(
        self,
        b_prime: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a training set used to estimate the critical values via quantile regression.

        Parameters
        ----------
        b_prime : int
            Number of simulations. Note that each simulation will have dimensions `(data_sample_size, data_dim)`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Parameters, samples.
        """
        pass
    
    @abstractmethod
    def simulate_for_diagnostics(
        self,
        b_doubleprime: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a training set used to estimate conditional coverage via the diagnostics branch.

        Parameters
        ----------
        b_doubleprime : int
            Number of simulations. Note that each simulation will have dimensions `(data_sample_size, data_dim)`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Parameters, samples.
        """
        pass
