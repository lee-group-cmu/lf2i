from typing import Union, Tuple

import numpy as np
import torch

from sbi.inference.posteriors.base_posterior import NeuralPosterior

def hpd_region(
    posterior: NeuralPosterior, 
    param_grid: Union[torch.Tensor, np.ndarray], 
    x: Union[torch.Tensor, np.ndarray],
    confidence_level: float, 
    num_p_levels: int = 100_000, 
    tol: float = 0.01
) -> Tuple[float, Union[np.ndarray, torch.Tensor]]:
    """Compute a high-posterior-density region at the desired confidence (technically, credible) level.

    Parameters
    ----------
    posterior : NeuralPosterior
        Estimated posterior distribution. Must implement `log_prob(theta=..., x=...)` method.
    param_grid : Union[torch.Tensor, np.ndarray]
        Fine grid of evaluation points in the support of the posterior.
    x : Union[torch.Tensor, np.ndarray]
        Observed sample given which the posterior p(theta|x) is evaluated.
    confidence_level : float
        Desired confidence level for the credible region.
    num_p_levels : int, optional
        Number of level sets to examine, by default 100_000. More level sets imply higher precision in the confidence level.
    tol : float, optional
        Coverage levels within tol of `confidence_level` will be accepted, by default 0.01.

    Returns
    -------
    Tuple[float, Union[np.ndarray, torch.Tensor]]
        Final confidence level, hpd credible region
    """
    # evaluate posterior over fine grid of values in support
    posterior_probs = torch.exp(posterior.log_prob(theta=param_grid, x=x))
    posterior_probs /= torch.sum(posterior_probs)  # make sure they sum to 1

    # descend the level sets of the posterior (p_levels) and stop when the area above the level equals confidence level 
    p_levels = np.linspace(0.99, 0, num_p_levels)  # thresholds to include or not parameters
    current_confidence_level = 1
    new_confidence_levels = []
    idx = 0
    while np.abs(current_confidence_level - confidence_level) > tol:
        if idx == num_p_levels:  # no more to examine
            break
        new_confidence_level = torch.sum(posterior_probs[posterior_probs >= p_levels[idx]])
        new_confidence_levels.append(new_confidence_level)
        if np.abs(new_confidence_level - confidence_level) < np.abs(current_confidence_level - confidence_level):
            current_confidence_level = new_confidence_level
        idx += 1
    # all params such that p(params|x) > p_level, where p_levels is the last chosen one
    return current_confidence_level, param_grid[posterior_probs >= p_levels[idx-1], :]
