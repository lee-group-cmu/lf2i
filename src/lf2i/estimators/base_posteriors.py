# utils/base_posteriors.py
from typing import Protocol, runtime_checkable, Union, Optional
import torch
import numpy as np


@runtime_checkable
class AbstractPosterior(Protocol):
    """
    Protocol for basic posterior distributions.
    Compatible with sbi.utils.kde.KDEWrapper and similar simple posteriors.
    """
    
    def log_prob(
        self, 
        theta: Union[torch.Tensor, np.ndarray], 
        x: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Evaluate log posterior probability log p(θ|x).
        
        Parameters
        ----------
        theta : Union[torch.Tensor, np.ndarray]
            Parameter values at which to evaluate
        x : Optional[Union[torch.Tensor, np.ndarray]]
            Observed data (optional, some posteriors may have this pre-set)
            
        Returns
        -------
        torch.Tensor
            Log probabilities
        """
        ...


@runtime_checkable
class AbstractNeuralPosterior(Protocol):
    """
    Protocol for neural posterior estimators that require training.
    Compatible with sbi.inference.posteriors.base_posterior.NeuralPosterior.
    
    Note: This extends AbstractPosterior's interface by including training methods.
    """
    
    def sample(
        self, 
        sample_shape: tuple, 
        x: Union[torch.Tensor, np.ndarray],
        show_progress_bars: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the posterior p(θ|x).
        
        Parameters
        ----------
        sample_shape : tuple
            Shape of samples to draw, e.g., (1000,) for 1000 samples
        x : Union[torch.Tensor, np.ndarray]
            Observed data
        show_progress_bars : bool
            Whether to show progress bars during sampling
            
        Returns
        -------
        torch.Tensor
            Samples from posterior with shape (sample_shape, theta_dim)
        """
        ...
    
    def log_prob(
        self, 
        theta: Union[torch.Tensor, np.ndarray], 
        x: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Evaluate log posterior probability log p(θ|x).
        
        Parameters
        ----------
        theta : Union[torch.Tensor, np.ndarray]
            Parameter values at which to evaluate
        x : Optional[Union[torch.Tensor, np.ndarray]]
            Observed data
            
        Returns
        -------
        torch.Tensor
            Log probabilities
        """
        ...


@runtime_checkable
class AbstractKDE(Protocol):
    """
    Protocol for KDE wrappers.
    Compatible with sbi.utils.kde.KDEWrapper.
    
    Note: KDE typically doesn't need the 'x' parameter since it's fit to samples.
    """
    
    def log_prob(
        self, 
        theta: Union[torch.Tensor, np.ndarray],
        **kwargs
    ) -> torch.Tensor:
        """
        Evaluate log probability under the KDE.
        
        Parameters
        ----------
        theta : Union[torch.Tensor, np.ndarray]
            Points at which to evaluate density
            
        Returns
        -------
        torch.Tensor
            Log probabilities
        """
        ...


@runtime_checkable
class AbstractNeuralPosteriorTrainer(Protocol):
    """
    Protocol for neural posterior estimators that require training.
    Compatible with sbi.inference.posteriors.base_posterior.NeuralPosterior.
    
    Note: This extends AbstractPosterior's interface by including training methods.
    """

    
    def append_simulations(
        self,
        theta: Union[torch.Tensor, np.ndarray],
        x: Union[torch.Tensor, np.ndarray],
        **kwargs
    ) -> 'AbstractNeuralPosterior':
        """
        Append training data (simulations) to the estimator.
        
        Parameters
        ----------
        theta : Union[torch.Tensor, np.ndarray]
            Simulated parameters
        x : Union[torch.Tensor, np.ndarray]
            Simulated data corresponding to theta
            
        Returns
        -------
        AbstractNeuralPosterior
            Returns self for method chaining
        """
        ...
    
    def train(self, **kwargs) -> 'AbstractNeuralPosterior':
        """
        Train the neural posterior estimator on appended simulations.
        
        Returns
        -------
        AbstractNeuralPosterior
            Returns self for method chaining
        """
        ...
    
    def build_posterior(self, **kwargs) -> 'AbstractNeuralPosterior':
        """
        Build the posterior distribution after training.
        
        Returns
        -------
        AbstractNeuralPosterior
            The trained posterior object ready for inference
        """
        ...