from typing import Tuple
import warnings

import torch


def preprocess_estimation_evaluation(
    parameters: torch.Tensor, 
    samples: torch.Tensor,
    param_dim: int
) -> Tuple[torch.Tensor]:
    
    parameters = parameters.reshape(-1, param_dim)
    
    # NOTE: SBI NPEs (e.g. SNPE) expect (n_simulations, x_dim), flatten batch dimension if present
    if samples.ndim == 3: # (size, batch_size, data_dim)

        # For batch_size=1, just squeeze; for batch_size>1, need to handle differently
        if samples.shape[1] == 1: 
            samples = samples.squeeze(1) # -> (size, data_dim)
        else:
            # Flatten: each batch element becomes a separate simulation
            n_total = samples.shape[0] * samples.shape[1]
            samples = samples.reshape(n_total, -1)
            # Need to repeat parameters accordingly
            parameters = parameters.repeat_interleave(samples.shape[0] // parameters.shape[0], dim = 0)

    elif samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return parameters, samples
