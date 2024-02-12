from typing import Tuple
import warnings

import torch


def preprocess_estimation_evaluation(
    parameters: torch.Tensor, 
    samples: torch.Tensor,
    param_dim: int
) -> Tuple[torch.Tensor]:
    if (len(samples.shape) == 3) and (samples.shape[1] > 1):
        warnings.warn(f"You provided a simulated set with batch size = {samples.shape[1]}. This dimension will be flattened for estimation and evaluation. Is this the desired behaviour?")
    return parameters.reshape(-1, param_dim), samples.reshape(-1, samples.shape[-1])
