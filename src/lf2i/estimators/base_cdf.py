from typing import Protocol, runtime_checkable, Union
import numpy as np
import torch


@runtime_checkable
class AbstractCDFEstimator(Protocol):
    """
    Protocol for parametric CDF estimators that model F(λ | θ).

    Unlike probabilistic classifiers, these receive test statistics and
    parameters of interest as separate arrays at fit time rather than a
    single combined matrix. At predict time they follow the same
    predict_proba(X) convention so that downstream p-value computation
    is uniform across both algorithm categories.

    This interface is satisfied by ParametricCDFEstimator and any custom
    estimator that models the conditional CDF of the test statistic directly.

    Implement this Protocol to provide a custom CDF estimator to
    `estimate_rejection_proba` as the `algorithm` argument.
    """

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> 'AbstractCDFEstimator':
        """
        Fit the CDF estimator to calibration data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Shape (n, 1 + poi_dim). Column 0 is the test statistic λ(x_i; θ_i),
            columns 1: are the corresponding parameters of interest θ_i.

        Returns
        -------
        self
        """
        ...

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """
        Compute p-values from the fitted CDF.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Shape (n, 1 + poi_dim). Column 0 is the test statistic λ,
            columns 1: are the parameters of interest θ.

        Returns
        -------
        np.ndarray
            Shape (n, 2).
            - [:, 0] = P(reject=0 | cutoff, θ)
            - [:, 1] = P(reject=1 | cutoff, θ)
            Rows sum to 1.
        """
        ...
