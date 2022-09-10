from abc import ABC, abstractmethod
from typing import Tuple
from pydantic import BaseModel

import numpy as np


class Simulator(BaseModel, ABC):
    """Base class for simulators.

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
    param_dim: int
    data_dim: int
    data_sample_size: int

    @abstractmethod()
    def simulate_for_test_statistic(
        self, 
        b: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def simulate_for_quantile_regression(
        self,
        b_prime: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def simulate_for_diagnostics(
        self,
        b_double_prime: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
