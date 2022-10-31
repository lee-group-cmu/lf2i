from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


def plot_parameter_region(
    confidence_region: np.ndarray, 
    param_dim: int,
    true_parameter: Optional[np.ndarray] = None,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    figsize=(15, 15),
    save_fig_path: Optional[str] = None
) -> None:
    if param_dim == 1:
        raise NotImplementedError
    elif param_dim == 2:
        _, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(x=confidence_region[:, 0], y=confidence_region[:, 1], s=3.5, c="green", zorder=1, label="Parameter region")
        if true_parameter is not None:
            ax.scatter(x=true_parameter.reshape(-1,)[0], y=true_parameter.reshape(-1,)[1], alpha=1, c="red", marker="*", s=500, zorder=10)

        ax.set_xlabel(r"$\theta_{{(1)}}$", fontsize=45)
        ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=45, rotation=0)
        ax.tick_params(labelsize=30)
        if parameter_space_bounds is not None:
            ax.set_xlim(parameter_space_bounds['low'], parameter_space_bounds['high'])
            ax.set_ylim(parameter_space_bounds['low'], parameter_space_bounds['high'])
        legend = ax.legend(prop={'size': 25})
        legend.legendHandles[0]._sizes = [40]
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
            ax.spines[axis].set_color('black')
                
    elif param_dim == 3:
        raise NotImplementedError
    else:
        raise ValueError("Impossible to plot a confidence region for parameters with more than 3 dimensions")

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
