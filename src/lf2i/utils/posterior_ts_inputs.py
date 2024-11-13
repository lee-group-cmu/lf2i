from typing import Tuple
import warnings

import torch


def preprocess_estimation_evaluation(
    parameters: torch.Tensor, 
    samples: torch.Tensor,
    param_dim: int
) -> Tuple[torch.Tensor]:
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return parameters.reshape(-1, param_dim), samples
