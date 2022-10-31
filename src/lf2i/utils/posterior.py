import numpy as np
import torch

def hpd_region(posterior, prior, param_grid, x, confidence_level, n_p_stars=100_000, tol=0.01):
    if posterior is None:
        # actually using prior here; naming just for consistency (should be changed)
        posterior_probs = torch.exp(prior.log_prob(torch.from_numpy(param_grid)))
    else:
        posterior_probs = torch.exp(posterior.log_prob(theta=param_grid, x=x))
    posterior_probs /= torch.sum(posterior_probs)  # normalize
    p_stars = np.linspace(0.99, 0, n_p_stars)  # thresholds to include or not parameters
    current_confidence_level = 1
    new_confidence_levels = []
    idx = 0
    while np.abs(current_confidence_level - confidence_level) > tol:
        if idx == n_p_stars:  # no more to examine
            break
        new_confidence_level = torch.sum(posterior_probs[posterior_probs >= p_stars[idx]])
        new_confidence_levels.append(new_confidence_level)
        if np.abs(new_confidence_level - confidence_level) < np.abs(current_confidence_level - confidence_level):
            current_confidence_level = new_confidence_level
        idx += 1
    # all params such that p(params|x) > p_star, where p_star is the last chosen one
    return current_confidence_level, param_grid[posterior_probs >= p_stars[idx-1], :], new_confidence_levels  