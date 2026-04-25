from typing import Optional, Tuple, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgba
from matplotlib.axes._axes import Axes
from matplotlib.patches import Rectangle
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


def _oat_interval(
    confidence_region: np.ndarray,
    point_estimate: np.ndarray,
    dim: int
) -> Tuple[Optional[float], Optional[float]]:
    """Compute one-at-a-time (OAT) interval for a single dimension.

    Fix all parameter coordinates at their `point_estimate` values except for `dim`, then
    find the range of `dim`-coordinate values that appear in `confidence_region`.

    Parameters
    ----------
    confidence_region : np.ndarray
        Array of shape (n_points, param_dim) containing the grid points in the confidence set.
    point_estimate : np.ndarray
        1-D array of shape (param_dim,) with the point estimate (e.g. MPE).
    dim : int
        Index of the parameter dimension for which to compute the interval.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (lower, upper) bounds of the OAT interval, or (None, None) if no matching points found.
    """
    param_dim = confidence_region.shape[1]
    if param_dim == 1:
        vals = confidence_region[:, 0]
        return float(np.min(vals)), float(np.max(vals))

    other_dims = [k for k in range(param_dim) if k != dim]
    # Chebyshev distance to MPE in the other dimensions
    distances = np.max(np.abs(confidence_region[:, other_dims] - point_estimate[other_dims]), axis=1)
    min_dist = np.min(distances)
    mask = distances <= min_dist + 1e-10 * (1 + min_dist)  # relative tolerance for floating-point comparison
    filtered = confidence_region[mask, dim]
    if len(filtered) == 0:
        return None, None
    return float(np.min(filtered)), float(np.max(filtered))


def plot_parameter_intervals(
    confidence_region: np.ndarray,
    point_estimate: np.ndarray,
    param_dim: int,
    param_names: Optional[Sequence[str]] = None,
    parameter_space_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    color: str = 'green',
    figsize: Optional[Tuple[float, float]] = None,
    save_fig_path: Optional[str] = None
) -> None:
    """Plot one-at-a-time (OAT) 1D confidence intervals for each parameter dimension.

    For each dimension ``j`` the interval is constructed by fixing all other coordinates at their
    point-estimate values and reporting the range of ``j``-coordinate values that lie in
    ``confidence_region``.  The point estimate itself is marked with a vertical tick.

    Parameters
    ----------
    confidence_region : np.ndarray
        Array of shape ``(n_points, param_dim)`` containing the grid points in the confidence set
        (e.g. one element of the list returned by :meth:`LF2I.inference`).
    point_estimate : np.ndarray
        1-D array of shape ``(param_dim,)`` with the point estimate for the same observation
        (e.g. one row of the ``point_estimates`` array returned by :meth:`LF2I.inference` when
        ``return_point_estimate=True``).
    param_dim : int
        Number of parameter dimensions.
    param_names : Sequence[str], optional
        Labels for each parameter dimension, by default ``[r"$\\theta_{(1)}$", ...]``.
    parameter_space_bounds : Dict[str, Tuple[float, float]], optional
        Mapping from parameter name to ``(low, high)`` bounds used to set x-axis limits.
        Keys must match entries in ``param_names``.  If *None* the limits are inferred from the
        confidence region.
    color : str, optional
        Color used to draw the intervals, by default ``'green'``.
    figsize : Tuple[float, float], optional
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
    save_fig_path : str, optional
        If provided, the figure is saved to this path before being shown.
    """
    if param_names is None:
        param_names = [r"$\theta_{{({})}}$".format(j + 1) for j in range(param_dim)]

    confidence_region = np.asarray(confidence_region).reshape(-1, param_dim)
    point_estimate = np.asarray(point_estimate).reshape(param_dim)

    dist_between_axes_in = 0.6
    default_figsize = (8, dist_between_axes_in * param_dim + 0.4)
    fig, axs = plt.subplots(
        param_dim, 1,
        figsize=figsize if figsize is not None else default_figsize,
        sharex=False
    )
    if param_dim == 1:
        axs = [axs]

    interval_thickness = 0.25

    for j, ax in enumerate(axs):
        lo, hi = _oat_interval(confidence_region, point_estimate, j)

        # Determine x-axis limits
        if parameter_space_bounds is not None and param_names[j] in parameter_space_bounds:
            bounds = parameter_space_bounds[param_names[j]]
            xlim = (bounds[0], bounds[1])
        elif lo is not None and hi is not None:
            margin = max((hi - lo) * 0.15, 1e-6)
            xlim = (lo - margin, hi + margin)
        else:
            # Fallback: use full range of the confidence region in this dimension
            vals = confidence_region[:, j]
            margin = max((vals.max() - vals.min()) * 0.15, 1e-6)
            xlim = (float(vals.min()) - margin, float(vals.max()) + margin)

        ax.set_xlim(xlim)
        ax.set_ylim(-0.6, 0.6)

        # Hide y-axis and decorative spines
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))

        # Draw the OAT interval as a filled rectangle
        if lo is not None and hi is not None:
            ax.add_patch(Rectangle(
                (lo, -interval_thickness / 2),
                hi - lo,
                interval_thickness,
                linewidth=0,
                color=color,
                zorder=2
            ))
            # Closed-interval bracket markers: '[' at lower bound, ']' at upper bound
            bracket_kw = dict(color=color, linewidth=2, zorder=3)
            bracket_h = interval_thickness * 1.5
            ax.vlines(lo, -bracket_h / 2, bracket_h / 2, **bracket_kw)
            ax.vlines(hi, -bracket_h / 2, bracket_h / 2, **bracket_kw)

        # Point estimate marker (vertical line in a contrasting style)
        mpe_val = float(point_estimate[j])
        ax.vlines(mpe_val, -interval_thickness * 1.8, interval_thickness * 1.8,
                  color='black', linewidth=1.5, linestyle='--', zorder=4)

        ax.set_xlabel(param_names[j], fontsize=12)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
