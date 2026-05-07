import numpy as np
import matplotlib.pyplot as plt
import torch

from lf2i.utils.other_methods import monte_carlo_critical_values
from lf2i.simulator import Simulator
from lf2i.test_statistics import TestStatistic
from lf2i.utils.miscellanea import to_torch_if_np, to_np_if_torch


def plot_parameter_relevance(
    test_statistic: TestStatistic,
    simulator: Simulator,
    param_bounds: dict,
    confidence_level: float,
    monte_carlo_size: int = 2_000,
    grid_size: int = 25,
    n_curves: int = 10,
    seed: int = 0,
):
    """For each parameter, plot MC critical values as a function of that parameter.

    Other parameters are drawn uniformly from their bounds (n_curves draws), and one
    curve is plotted per draw so the dependence on the swept parameter is visible across
    the range of the nuisances.
    """
    rng = np.random.default_rng(seed)
    param_names = list(param_bounds.keys())
    bounds_arr = np.array(list(param_bounds.values()))  # (n_params, 2)
    n_params = len(param_names)

    # Draw n_curves values for every parameter at once; we'll override the swept one below.
    other_draws = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1], size=(n_curves, n_params))

    fig, axs = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axs = [axs]

    for i, (name, (lo, hi)) in enumerate(param_bounds.items()):
        grid_1d = np.linspace(lo, hi, grid_size)

        for k in range(n_curves):
            param_grid = np.tile(other_draws[k], (grid_size, 1))
            param_grid[:, i] = grid_1d

            mc_cvs = monte_carlo_critical_values(
                test_statistic=test_statistic,
                simulator=simulator,
                param_grid=torch.tensor(param_grid, dtype=torch.float32),
                confidence_level=confidence_level,
                monte_carlo_size=monte_carlo_size,
            )

            label = ', '.join(
                f'{param_names[j]}={other_draws[k, j]:.2g}'
                for j in range(n_params) if j != i
            )
            axs[i].plot(grid_1d, mc_cvs, label=label if n_params > 1 else None)

        axs[i].set_xlabel(name)
        axs[i].set_ylabel('Critical Value')
        if n_params > 1:
            axs[i].legend(fontsize='x-small')

    fig.suptitle(f'MC Critical Values  (α={1 - confidence_level:.3g})', y=1.02)
    plt.tight_layout()
    return fig
