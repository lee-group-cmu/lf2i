from typing import Protocol, runtime_checkable, Union
import numpy as np
import torch


@runtime_checkable
class AbstractProbabilisticClassifier(Protocol):
    """
    Protocol for probabilistic classifiers that estimate P(reject | cutoff, θ).

    X has shape (n, 1 + poi_dim): column 0 is the resampled cutoff (test
    statistic threshold), remaining columns are parameters of interest θ.
    This interface is satisfied by sklearn-style classifiers such as
    CatBoostClassifier, LogisticRegression, and TabICLClassifier, as well
    as any custom estimator that follows the same convention.

    Implement this Protocol to provide a custom probabilistic classifier to
    `estimate_rejection_proba` as the `algorithm` argument.
    """

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> None:
        """
        Fit the classifier to augmented calibration data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Shape (n, 1 + poi_dim). Column 0 is the resampled cutoff,
            columns 1: are the parameters of interest θ. Produced by
            `lf2i.calibration.p_values.augment_calibration_set`.
        y : Union[np.ndarray, torch.Tensor]
            Shape (n,). Binary rejection indicators (0 or 1). Produced by
            `lf2i.calibration.p_values.augment_calibration_set`.
        """
        ...

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """
        Predict class probabilities for each row of X.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Shape (n, 1 + poi_dim). Column 0 is the resampled cutoff,
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
