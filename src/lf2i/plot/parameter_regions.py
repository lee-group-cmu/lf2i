from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


def plot_parameter_region(
    parameter_region: np.ndarray, 
    param_dim: int,
    true_parameter: Optional[np.ndarray] = None,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_fig_path: Optional[str] = None,
    **kwargs
) -> None:
    """Dispatcher to plot parameter regions of different dimensionality.
    """
    if param_dim == 1:
        plot_parameter_region_1D(parameter_region, true_parameter, parameter_space_bounds, figsize, **kwargs)
    elif param_dim == 2:
        plot_parameter_region_2D(parameter_region, true_parameter, parameter_space_bounds, figsize, **kwargs)
    elif param_dim == 3:
        raise NotImplementedError
    else:
        raise ValueError("Impossible to plot a confidence region for parameters with more than 3 dimensions")

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()


def plot_parameter_region_1D(
    parameter_region: np.ndarray,
    true_parameter: np.ndarray,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: Optional[str] = 'green'
) -> None:
    """Plot 1-dimensional parameter regions using the lower and upper bounds.
    """
    _, ax = plt.subplots(1, 1, figsize=figsize if figsize is not None else (3, 9))

    ax.scatter(x=true_parameter, y=true_parameter, alpha=1, c="red", marker="*", s=250, zorder=10)
    ax.axhline(y=np.min(parameter_region.reshape(1, -1), axis=1), xmin=0.45, xmax=0.55, label="Confidence Region", color=color)
    ax.axhline(y=np.max(parameter_region.reshape(1, -1), axis=1), xmin=0.45, xmax=0.55, color=color)
    ax.vlines(x=true_parameter, ymin=np.min(parameter_region), ymax=np.max(parameter_region), color=color)

    if parameter_space_bounds is not None:
        ax.set_ylim(parameter_space_bounds['low'], parameter_space_bounds['high'])
    ax.set_ylabel(r'$\theta$', fontsize=45, rotation=0)
    ax.get_xaxis().set_visible(False)
    ax.tick_params(labelsize=20)
    ax.legend(prop={'size': 12})
    plt.show()


def plot_parameter_region_2D(
    parameter_region: np.ndarray,
    true_parameter: Optional[np.ndarray] = None,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    plot_hull: bool = False,
    color: Optional[str] = 'green'
) -> None:
    """Plot 2-dimensional parameter regions as point clouds.
    """
    _, ax = plt.subplots(1, 1, figsize=figsize if figsize is not None else (10, 10))

    ax.scatter(x=parameter_region[:, 0], y=parameter_region[:, 1], s=3.5, c=color, zorder=1, label="Parameter region")
    if true_parameter is not None:
        ax.scatter(x=true_parameter.reshape(-1,)[0], y=true_parameter.reshape(-1,)[1], alpha=1, c="red", marker="*", s=250, zorder=10)
    if plot_hull:
        pass

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
