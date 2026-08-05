from typing import Protocol, runtime_checkable, Union
import numpy as np
import torch


@runtime_checkable
class AbstractQuantileRegressor(Protocol):
    """
    Protocol for quantile regressors that estimate critical values.

    X has shape (n, param_dim): the parameters of interest θ.
    y has shape (n,): the test statistics λ(x; θ).

    This interface is satisfied by CatBoostRegressor, the FeedForwardNN
    LearnerRegression wrapper, and any custom estimator that follows
    the same fit/predict convention expected by
    `lf2i.calibration.critical_values.train_qr_algorithm`.

    Implement this Protocol to provide a custom quantile regressor to
    `train_qr_algorithm` as the `algorithm` argument.
    """

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> None:
        """
        Fit the quantile regressor to calibration data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Shape (n, param_dim). Parameters of interest θ.
        y : Union[np.ndarray, torch.Tensor]
            Shape (n,). Test statistics λ(x_i; θ_i).
        """
        ...

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """
        Predict quantile(s) of the test statistic at each parameter value.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Shape (n, param_dim). Parameters of interest θ.

        Returns
        -------
        np.ndarray
            Shape (n,) for a single quantile, or (n, k) for k quantiles.
        """
        ...
