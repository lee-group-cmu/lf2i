# utils/base_posteriors.py
from typing import Protocol, runtime_checkable, Union, Optional
import torch
import numpy as np


@runtime_checkable
class AbstractClassifier(Protocol):
    """
    Protocol for basic likelihood ratio estimators via classification.
    """

    def predict_proba(
        self, 
        X: Union[torch.Tensor, np.ndarray],
        **kwargs
    ) -> torch.Tensor:
        """
        For X containing x and θ, predict p(y=1 | x, θ) where y is the class 
        label indicating whether the data was generated from the model with 
        parameter θ or from a reference distribution. Evaluates likelihood ratio
        p(x; θ) / [int p(x; θ') r(θ') dθ'].

        Parameters
        ----------
        theta : Union[torch.Tensor, np.ndarray]
            Parameter values at which to evaluate
        x : Optional[Union[torch.Tensor, np.ndarray]]
            Observed data (optional, some posteriors may have this pre-set)

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            Probabilities with shape (n_samples, 2) where:
            - [:, 0] contains p(y=0 | x, θ) 
            - [:, 1] contains p(y=1 | x, θ)
            Both columns must sum to 1.0 for each sample.
        """
        ...


@runtime_checkable
class AbstractClassifierTrainer(AbstractClassifier, Protocol):
    def fit(
        self, 
        X: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray],
        **kwargs
    ) -> None:
        """
        Fit the classifier to data.

        Parameters
        ----------
        X : Union[torch.Tensor, np.ndarray]
            Training data containing x and θ
        y : Union[torch.Tensor, np.ndarray]
            Class labels indicating whether each row of X was generated from the model with parameter θ or from a reference distribution.
        """
        ...
