# Part of this code was adapted from https://colab.research.google.com/drive/1nXOlrmVHqCHiixqiMF6H8LSciz583_W2

from typing import Sequence, Optional, List
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.functional import sigmoid


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
            errors = target - input[:, i]
            # “check” function
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
        return self.model(X.to(self.device)).cpu().detach()
    

class LearnerClassification(Learner):

    def predict(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        # output predicted probability for positive class
        return sigmoid(self.model(X.float().to(self.device)).cpu().detach().reshape(X.shape[0], -1))
