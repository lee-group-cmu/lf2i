from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

import numpy as np
import torch


class Simulator(ABC):
    """Base class for simulators. This is a template from which every simulator should inherit.

    Parameters
    ----------
    poi_dim : int
        Dimensionality of the space of parameters of interest.
    data_dim : int
        Dimensionality of a single datapoint X.
    batch_size: int
        Size of data batches from a specific parameter configuration. Must be the same for observations and simulations.
        A simulated/observed sample batch from a specific parameter configuration will have dimensions `(batch_size, data_dim)`.
    nuisance_dim : Optional[int], optional
        Dimensionality of the space of nuisance parameters (systematics), by default 0. 
    """

    def __init__(
        self,
        poi_dim: int,
        data_dim: int,
        batch_size: int,
        nuisance_dim: Optional[int] = None
    ) -> None:
        
        self.poi_dim = poi_dim
        self.nuisance_dim = nuisance_dim or 0
        self.param_dim = poi_dim + self.nuisance_dim
        self.data_dim = data_dim
        self.batch_size = batch_size

    @abstractmethod
    def __call__(
        self,
        param: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Forward model / likelihood: draw samples given parameter values.

        Parameters
        ----------
        param : Union[np.ndarray, torch.Tensor]
            Parameter values, shape `(size, param_dim)`.
        batch_size : Optional[int], optional
            Number of observations to draw for each row of `param`. Defaults to `self.batch_size`.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Samples, shape `(size, batch_size, data_dim)`.
        """
        pass

    @abstractmethod
    def simulate_for_test_statistic(
        self, 
        size: int,
        estimation_method: str
    ) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        """Simulate a training set used to estimate the test statistic.

        Parameters
        ----------
        size : int
            Number of simulations.
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
        size: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Simulate a training set used to estimate the critical values via quantile regression.

        Parameters
        ----------
        size : int
            Number of simulations. Note that each simulation will be a batch with dimensions `(batch_size, data_dim)`.

        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            Parameters, samples.
        """
        pass
    
    @abstractmethod
    def simulate_for_diagnostics(
        self,
        size: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Simulate a training set used to estimate conditional coverage via the diagnostics branch.

        Parameters
        ----------
        size : int
            Number of simulations. Note that each simulation will be a batch with dimensions `(batch_size, data_dim)`.

        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            Parameters, samples.
        """
        pass
