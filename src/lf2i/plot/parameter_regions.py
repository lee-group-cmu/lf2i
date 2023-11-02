from typing import Optional, Tuple, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgba
from matplotlib.axes._axes import Axes
import alphashape

from lf2i.plot.miscellanea import PolygonPatchFixed


def plot_parameter_region(
    parameter_region: np.ndarray, 
    param_dim: int,
    true_parameter: Optional[np.ndarray] = None,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    save_fig_path: Optional[str] = None,
    **kwargs
) -> None:
    """Dispatcher to plot parameter regions of different dimensionality.
    """
    if param_dim == 1:
        plot_parameter_region_1D(parameter_region, true_parameter, parameter_space_bounds, **kwargs)
    elif param_dim == 2:
        plot_parameter_region_2D(parameter_region=parameter_region, true_parameter=true_parameter, parameter_space_bounds=parameter_space_bounds, **kwargs)
    elif param_dim == 3:
        raise NotImplementedError
    else:
        raise ValueError("Impossible to plot a confidence region for parameters with more than 3 dimensions. Use 'parameter_regions_pairplot'.")

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
    parameter_space_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    labels: Optional[Sequence[str]] = None,
    param_names: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    alpha_shape: bool = False,
    alpha: Optional[float] = None,
    scatter: bool = True,
    color: Optional[str] = 'green',
    region_name: Optional[str] = "Parameter region",
    custom_ax: Optional[Axes] = None
) -> None:
    """Plot 2-dimensional parameter regions as point clouds.
    """
    if custom_ax is None:
        plt.figure(figsize=figsize if figsize is not None else (10, 10))
        ax = plt.gca()
    else:
        ax = custom_ax

    if scatter:
        ax.scatter(x=parameter_region[:, 0], y=parameter_region[:, 1], s=3.5, c=color, zorder=1, label=region_name)
    if alpha_shape:
        alpha_shape = alphashape.alphashape(parameter_region, alpha=alpha)
        patch = PolygonPatchFixed(alpha_shape, fc=to_rgba(color, 0.2), ec=to_rgba(color, 1), lw=2, label=region_name)
        ax.add_patch(patch)
    if true_parameter is not None:
        ax.scatter(x=true_parameter.reshape(-1,)[0], y=true_parameter.reshape(-1,)[1], alpha=1, c="red", marker="*", s=250, zorder=10)
    
    if parameter_space_bounds is not None:
        param_names = labels if param_names is None else param_names  # TODO: if none of them is supplied this throws an error
        ax.set_xlim(parameter_space_bounds[param_names[0]]['low'], parameter_space_bounds[param_names[0]]['high'])
        ax.set_ylim(parameter_space_bounds[param_names[1]]['low'], parameter_space_bounds[param_names[1]]['high'])
    if custom_ax is None:
        labels = [r"$\theta_{{(1)}}$", r"$\theta_{{(2)}}$"] if labels is None else labels
        ax.set_xlabel(labels[0], fontsize=45)
        ax.set_ylabel(labels[1], fontsize=45)
        ax.tick_params(labelsize=30)

        legend = ax.legend(prop={'size': 25})
        legend.legendHandles[0]._sizes = [40]
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
            ax.spines[axis].set_color('black')
        plt.show()
    else:
        # to plot legend in main plot
        return ax.get_legend_handles_labels()
        

def parameter_regions_pairplot(
    *parameter_regions: np.ndarray,
    true_parameter: np.ndarray,  # can plot multiple regions for the same true parameter, not different
    parameter_space_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    labels: Optional[np.ndarray] = None,
    param_names: Optional[np.ndarray] = None,
    colors: Optional[Sequence[str]] = None,
    region_names: Optional[Sequence[str]] = None,
    alpha_shape: bool = False,
    alpha: Optional[float] = None,
    scatter: bool = True,
    figsize: Optional[Sequence[int]] = (15, 15),
    save_fig_path: Optional[str] = None
) -> None:

    rows = cols = parameter_regions[0].shape[1]  # param dim
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    colors = colors or cm.rainbow(np.linspace(0, 1, len(region_names)))
    assert len(region_names) == len(colors) == len(parameter_regions)
    
    for row in range(rows):
        for col in range(cols):
            # plots
            if col <= row:
                ax[row, col].axis('off')
            else:
                for i, parameter_region in enumerate(parameter_regions):
                    leg_handles, leg_labels = plot_parameter_region_2D(
                        parameter_region=parameter_region[:, [col, row]],  # swap order to have 'row' parameter on y axis
                        true_parameter=true_parameter[[col, row]],
                        parameter_space_bounds={
                            param: dict(zip(['low', 'high'], parameter_space_bounds[param])) 
                            for param in param_names[[col, row]]
                        },
                        labels=None,
                        param_names=param_names[[col, row]],
                        color=colors[i],
                        region_name=region_names[i],
                        alpha_shape=alpha_shape,
                        alpha=alpha,
                        scatter=scatter,
                        custom_ax=ax[row, col]
                    )
                # labels
                if col == row+1:
                    ax[row, col].set_xlabel(r'$\theta_{}$'.format(col) if labels is None else labels[col], fontsize=20)
                    ax[row, col].tick_params(axis='x', labelsize=12)
                    ax[row, col].set_ylabel(r'$\theta_{}$'.format(row) if labels is None else labels[row], fontsize=20, labelpad=3)
                    ax[row, col].tick_params(axis='y', labelsize=12)
                else:
                    ax[row, col].tick_params(labelleft=False, labelbottom=False)
    
    legend = fig.legend(leg_handles, leg_labels, bbox_to_anchor=(0.5, 0.5))
    legend.legendHandles[0]._sizes = [40]
    
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
