# Part of this code was adapted from https://colab.research.google.com/drive/1nXOlrmVHqCHiixqiMF6H8LSciz583_W2

from typing import Union, List, Tuple
from tqdm import tqdm

from itertools import chain
import numpy as np
import torch


class QuantileLoss(torch.nn.Module):
    """Quantile loss as a PyTorch module.

    Parameters
    ----------
    quantiles : Union[Tuple, List, np.ndarray, torch.Tensor]
        Target quantiles. Values must be in the range `(0, 1)`.
    """
    def __init__(
        self, 
        quantiles: Union[Tuple, List, np.ndarray, torch.Tensor]
    ) -> None:
        super.__init__()
        self.quantiles = quantiles

    def forward(
        self, 
        targets: torch.Tensor, 
        predictions: torch.Tensor
    ) -> torch.Tensor:
        assert not targets.requires_grad
        assert predictions.size(0) == targets.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i]
            # “check” function
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class QuantileNN(torch.nn.Module):
    """Fully connected neural network for quantile regression.

    Parameters
    ----------
    n_quantiles : int
        Number of target quantiles.
    input_d : int
        Dimensionality of the input. 
    hidden_layer_shapes : Union[Tuple, List]
        The i-th element represents the number of neurons in the i-th hidden layer.
    dropout_p : float, optional
        Probability for the dropout layers, by default 0.0
    """
    def __init__(
        self,
        n_quantiles: int,
        input_d: int,
        hidden_layer_shapes: Union[Tuple, List],
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.input_d = input_d
        self.hidden_layer_shapes = hidden_layer_shapes
        self.hidden_activation = hidden_activation
        self.dropout_p = dropout_p
        
        self.build_model()
        self.init_weights()

    def build_model(self) -> None:
        # input
        self.model = [
            torch.nn.Linear(self.input_d, self.hidden_layer_shapes[0]), 
            self.hidden_activation, 
            torch.nn.Dropout(p=self.dropout_p)
        ]
        # hidden 
        for i in range(0, len(self.hidden_layer_shapes)-1):
            self.model += [
                torch.nn.Linear(self.hidden_layer_shapes[i], self.hidden_layer_shapes[i+1]), 
                self.hidden_activation, 
                torch.nn.Dropout(p=self.dropout_p)
            ]
        # output
        self.model = torch.nn.Sequential(*self.model)
        self.final_layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_layer_shapes[-1], 1) for _ in range(len(self.n_quantiles))]
        )
    
    def init_weights(self) -> None:
        torch.manual_seed(self.seed)
        for m in chain(self.model, self.final_layers):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp_ = self.model(x)
        return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)

    
class Learner:
    """Learner for the quantile neural network. Implements methods for training and inference.

    Parameters
    ----------
    model : torch.nn.Module
        Quantile Neural Network architecture.
    optimizer : torch.optim.Optimizer
        Chosen optimizer.
    loss : torch.nn.Module
        Quantile Loss.
    device : str, optional
        Device on which to perform computations, by default "cpu". Use "cuda" for GPU.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module = QuantileLoss,
        device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss = loss.to(self.device)
        self.loss_trajectory = []
    
    def fit(
        self,
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int, 
        batch_size: int
    ) -> None:
        self.model.train()
        for _ in tqdm(range(epochs), desc="Training Quantile NN"):
            shuffle_idx = np.arange(X.shape[0])
            np.random.shuffle(shuffle_idx)
            X = X[shuffle_idx, :]
            y = y[shuffle_idx]
            epoch_losses = []
            for idx in range(0, X.shape[0], batch_size):
                self.optimizer.zero_grad()
                
                batch_X = torch.from_numpy(
                    X[idx: min(idx + batch_size, X.shape[0]), :]
                ).float().to(self.device).requires_grad_(False)
                batch_y = torch.from_numpy(
                    y[idx: min(idx + batch_size, y.shape[0])].reshape(-1,1)
                ).float().to(self.device).requires_grad_(False)
                
                batch_predictions = self.model(batch_X)
                batch_loss = self.loss(batch_predictions, batch_y)
                batch_loss.backward()
                self.optimizer.step()
                epoch_losses.append(batch_loss.cpu().detach().numpy())
            self.loss_trajectory.append(np.mean(epoch_losses))

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        self.model.eval()
        return self.model(torch.from_numpy(X).to(self.device).requires_grad_(False)).cpu().detach().numpy()
