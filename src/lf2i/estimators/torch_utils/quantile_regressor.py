from typing import Union, Dict, Any, Sequence, Optional, List
import warnings

import numpy as np
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor

from lf2i.utils.calibration_diagnostics_inputs import preprocess_train_quantile_regression
from lf2i.utils.miscellanea import select_n_jobs, to_np_if_torch
from lf2i.estimators import AbstractQuantileRegressor


class QuantileLoss(torch.nn.Module):
    """Quantile loss as a PyTorch module.
    Note that, although it supports multiple quantiles, there is currently no explicit constraint on their monotonicity to avoid quantile crossings.

    Parameters
    ----------
    quantiles : Sequence[float]
        Target quantiles. Values must be in the range `(0, 1)`.
    """
    def __init__(
        self,
        quantiles: Sequence[float]
    ) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        assert not target.requires_grad
        assert input.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target.squeeze(-1) - input[:, i]
            # "check" function
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class FeedForwardNN(torch.nn.Module):
    """Fully connected neural network.

    Parameters
    ----------
    input_d : int
        Dimensionality of the input.
    output_d : int
        Dimensionality of the output.
    hidden_layer_shapes : Sequence[int]
        The i-th element represents the number of neurons in the i-th hidden layer.
    dropout_p : float, optional
        Probability for the dropout layers, by default 0.0 (i.e., no dropout.)
    batch_norm : bool, optional
        Whether to apply batch normalization between each hidden layer or not.
    """
    def __init__(
        self,
        input_d: int,
        output_d: int,
        hidden_layer_shapes: Sequence[int],
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        dropout_p: Optional[float] = None,
        batch_norm: bool = False
    ) -> None:
        super().__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.hidden_layer_shapes = hidden_layer_shapes
        self.hidden_activation = hidden_activation

        self.build_model(batch_norm, dropout_p)

    def build_model(self, batch_norm: bool, dropout_p: Optional[float]) -> None:
        # input
        self.model = [torch.nn.Linear(self.input_d, self.hidden_layer_shapes[0]), self.hidden_activation]
        if batch_norm:
            self.model += [torch.nn.BatchNorm1d(self.hidden_layer_shapes[0])]
        if dropout_p:
            self.model += [torch.nn.Dropout(p=dropout_p)]

        # hidden
        for i in range(0, len(self.hidden_layer_shapes)-1):
            self.model += [torch.nn.Linear(self.hidden_layer_shapes[i], self.hidden_layer_shapes[i+1]), self.hidden_activation]
            if batch_norm:
                self.model += [torch.nn.BatchNorm1d(self.hidden_layer_shapes[i+1])]
            if dropout_p:
                self.model += [torch.nn.Dropout(p=dropout_p)]
        # output: no sigmoid cause we use BCEWithLogitsLoss (more numerically stable thanks to log-sum-exp trick)
        self.model += [torch.nn.Linear(self.hidden_layer_shapes[-1], self.output_d)]
        self.model = torch.nn.Sequential(*self.model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class Learner:
    """Utility class to train a neural network.

    Parameters
    ----------
    model : torch.nn.Module
        Neural Network architecture.
    optimizer : torch.optim.Optimizer
        Chosen optimizer.
    loss : torch.nn.Module
        Loss function to minimize via SGD.
    device : str, optional
        Device on which to perform computations, by default "cpu"
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        device: str = "cpu",
        verbose: bool = True
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss = loss.to(self.device)
        self.loss_trajectory: List[np.ndarray] = []
        self.verbose = verbose

    def __getstate__(self):
        # The optimizer is only needed during training; drop it to avoid pickling
        # torch.backends.* ConfigModuleInstance objects that live in its closure chain.
        state = self.__dict__.copy()
        state['optimizer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int
    ) -> None:
        self.model.train()
        if self.verbose:
            pbar = tqdm(total=epochs, desc="Training Neural Network")
        for _ in range(epochs):
            shuffled_idx = torch.randperm(X.shape[0])
            X = X[shuffled_idx, :]
            y = y[shuffled_idx]
            epoch_losses = []
            for idx in range(0, X.shape[0], batch_size):
                self.optimizer.zero_grad()

                batch_X = X[idx: min(idx + batch_size, X.shape[0]), :].float().to(self.device)
                batch_y = y[idx: min(idx + batch_size, y.shape[0])].reshape(-1, 1).float().to(self.device)

                batch_predictions = self.model(batch_X)
                batch_loss = self.loss(input=batch_predictions, target=batch_y)
                batch_loss.backward()
                self.optimizer.step()
                epoch_losses.append(batch_loss.cpu().detach().numpy())
            if self.verbose:
                pbar.update(1)
            self.loss_trajectory.append(np.mean(epoch_losses))
        if self.verbose:
            pbar.close()

    def predict(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError('Use the children classes that implement the correct methods for inference in regression or classification settings.')


class LearnerRegression(Learner):

    def predict(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        return self.model(X.float().to(self.device)).cpu().detach()


class ScaledQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Feed-forward-NN quantile regressor with input standardization and flat
    extrapolation beyond the training range -- a thin, sklearn-compatible
    (`BaseEstimator`/`RegressorMixin`) wrapper around `FeedForwardNN` +
    `LearnerRegression` + `QuantileLoss`, usable directly with
    `sklearn.model_selection.RandomizedSearchCV` (matching the `'cat-gb'` calibration
    path's own hyperparameter-search mechanism -- see
    `lf2i.calibration.critical_values.train_qr_algorithm`).

    Standardization is data-driven: `X` is min-max scaled to `[0, 1]` using its own
    training-time min/max (fit fresh in `.fit()`, reused at `.predict()` time) -- no
    external bounds needed. This matters in practice: an unscaled scalar input
    spanning thousands of raw units (e.g. an effective temperature in Kelvin) starves
    a default-initialized `Linear`+`ReLU` stack of any useful gradient signal.

    clamp_lo/clamp_hi: where clamp_extrapolation's flat region actually kicks
    in, if set -- e.g. the caller's own grid_lo/grid_hi (a chosen percentile
    of train+calib theta), not necessarily the same as _lo/_hi below (the raw
    min/max of whatever theta sample X.fit() was actually called with, used
    for scaling). Decoupled on purpose: the QR's own training theta sample can
    span nearly the full data range even when the caller wants the flat region
    to start well inside that -- leaving clamp_lo/clamp_hi at their None
    default falls back to _lo/_hi, i.e. the original (pre-decoupling)
    behavior, unchanged for any existing caller that doesn't set these.

    Parameters
    ----------
    quantiles : Sequence[float], optional
        Target quantile(s), by default (0.5,).
    poi_dim : int, optional
        Dimensionality of the parameter(s) of interest (the regressor's input), by
        default 1.
    hidden_layer_shapes : Sequence[int], optional
        Hidden layer widths, by default (64, 64).
    hidden_activation : Optional[torch.nn.Module], optional
        Passed through to `FeedForwardNN`; `None` resolves to `torch.nn.ReLU()`, by
        default None.
    dropout_p : Optional[float], optional
        Passed through to `FeedForwardNN`, by default None.
    batch_norm : bool, optional
        Passed through to `FeedForwardNN`, by default False.
    epochs : int, optional
        By default 100.
    batch_size : int, optional
        By default 64.
    clamp_extrapolation : bool
        By default True.
    clamp_lo : float, optional
    clamp_hi : float, optional
    device : str, optional
        By default "cpu".
    verbose : bool, optional
        By default False -- quieter than `Learner`'s own default, since this is
        meant to also work inside a hyperparameter search where per-candidate
        progress bars would be noisy.
    """

    def __init__(
        self,
        quantiles: Sequence[float] = (0.5,),
        poi_dim: int = 1,
        hidden_layer_shapes: Sequence[int] = (64, 64),
        hidden_activation: Optional[torch.nn.Module] = None,
        dropout_p: Optional[float] = None,
        batch_norm: bool = False,
        epochs: int = 100,
        batch_size: int = 64,
        clamp_extrapolation: bool = True,
        clamp_lo: Optional[float] = None,
        clamp_hi: Optional[float] = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        # sklearn convention: __init__ only stores constructor arguments verbatim
        # (no resolving/validating here) so get_params()/set_params()/clone() work.
        self.quantiles = quantiles
        self.poi_dim = poi_dim
        self.hidden_layer_shapes = hidden_layer_shapes
        self.hidden_activation = hidden_activation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.clamp_extrapolation = clamp_extrapolation
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.device = device
        self.verbose = verbose

        feedforward_nn = FeedForwardNN(
            input_d=self.poi_dim,
            output_d=len(self.quantiles),
            hidden_layer_shapes=list(self.hidden_layer_shapes),
            hidden_activation=self.hidden_activation or torch.nn.ReLU(),
            dropout_p=self.dropout_p,
            batch_norm=self.batch_norm,
        )
        self.model = feedforward_nn
        self._learner = LearnerRegression(
            model=feedforward_nn,
            optimizer=torch.optim.Adam,
            loss=QuantileLoss(quantiles=list(self.quantiles)),
            device=self.device,
            verbose=self.verbose,
        )
        self._lo: Optional[torch.Tensor] = None
        self._hi: Optional[torch.Tensor] = None

    def _scale(self, X: torch.Tensor) -> torch.Tensor:
        span = torch.clamp(self._hi - self._lo, min=1e-12)
        return (X - self._lo) / span

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "ScaledQuantileRegressor":
        X = X if isinstance(X, torch.Tensor) else torch.as_tensor(to_np_if_torch(X))
        y = y if isinstance(y, torch.Tensor) else torch.as_tensor(to_np_if_torch(y))
        self._lo = X.amin(dim=0, keepdim=True).float()
        self._hi = X.amax(dim=0, keepdim=True).float()
        self._learner.fit(
            X=self._scale(X.float()),
            y=y,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """By default (clamp_extrapolation=True, the CatBoost-matching behavior --
        see __init__), X is clamped before being fed to the network, so predictions
        outside the clamp range are a CONSTANT extrapolant (whatever the boundary
        predicts) rather than the network's raw, unconstrained -- and potentially
        non-monotonic -- output. The clamp range is [self.clamp_lo, self.clamp_hi]
        when those are set (e.g. to the caller's own chosen grid bounds); otherwise
        it falls back to [self._lo, self._hi] (the training data's own raw min/max,
        the original pre-decoupling behavior). Existing pickled instances predate
        both attributes; getattr defaults them to that same fallback rather than
        raising."""
        X = X if isinstance(X, torch.Tensor) else torch.as_tensor(to_np_if_torch(X))
        X = X.float()
        if getattr(self, "clamp_extrapolation", True):
            clamp_lo = getattr(self, "clamp_lo", None)
            clamp_hi = getattr(self, "clamp_hi", None)
            lo = self._lo if clamp_lo is None else torch.as_tensor(clamp_lo, dtype=self._lo.dtype, device=self._lo.device).expand_as(self._lo)
            hi = self._hi if clamp_hi is None else torch.as_tensor(clamp_hi, dtype=self._hi.dtype, device=self._hi.device).expand_as(self._hi)
            X = torch.clamp(X, min=lo, max=hi)
        return self._learner.predict(self._scale(X))


class LearnerClassification(Learner):

    def predict_proba(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        # output predicted probability for positive class
        return sigmoid(self.model(X.float().to(self.device)).cpu().detach().reshape(X.shape[0], -1))
