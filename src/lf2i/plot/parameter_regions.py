from typing import Optional, Tuple, Dict, Sequence
import warnings

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgba
from matplotlib.axes._axes import Axes
import alphashape
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.patches as  mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerPatch


def _hpd_1d(samples: np.ndarray, credibility: float) -> Tuple[float, float]:
    """Shortest interval containing `credibility` fraction of samples."""
    s = np.sort(samples)
    n = len(s)
    window = max(1, int(np.ceil(credibility * n)))
    if window >= n:
        return float(s[0]), float(s[-1])
    widths = s[window - 1 : n] - s[: n - window + 1]
    idx = int(np.argmin(widths))
    return float(s[idx]), float(s[idx + window - 1])

from lf2i.plot.miscellanea import PolygonPatchFixed
from lf2i.utils.miscellanea import to_np_if_torch


def plot_parameter_regions(
    *parameter_regions: np.ndarray,
    param_dim: int,
    true_parameter: Optional[np.ndarray] = None,
    prior_samples: Optional[np.ndarray] = None,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    colors: Optional[Sequence[str]] = None,
    region_names: Optional[Sequence[str]] = None,
    labels: Optional[np.ndarray] = None,
    linestyles: Optional[Sequence[str]] = None,
    param_names: Optional[np.ndarray] = None,
    alpha_shape: bool = False,
    alpha: Optional[float] = None,
    scatter: bool = True,
    log_scale: bool = False,
    title: Optional[str] = None,
    figsize: Optional[Sequence[int]] = (15, 15),
    save_fig_path: Optional[str] = None,
    remove_legend: bool = False,
    custom_ax: Optional[Axes] = None,
    show_diagonal: Optional[bool] = False,
    diagonal_type: Optional[str] = 'hist',  # 'hist', 'kde', 'confidence', or 'posterior'
    diagonal_pvalues: Optional[np.ndarray] = None,
    diagonal_grid: Optional[np.ndarray] = None,
    diagonal_levels: Optional[Sequence[float]] = None,
    posterior_estimator=None,
    posterior_observations: Optional[Sequence] = None,
    filter_subset: Optional[bool] = False,
    subset_threshold: Optional[float] = 1.0
) -> None:
    """Dispatcher to plot parameter regions of different dimensionality.
    
    For param_dim > 2, creates a pairplot showing all 2D projections.
    """
    if param_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize if figsize is not None else (3, 9))

        linestyles = cycle(linestyles) if linestyles else cycle(['-', '--', '-.', ':'])
        colors = colors or cm.rainbow(np.linspace(0, 1, len(region_names)))
        assert len(region_names) == len(colors) == len(parameter_regions)
        for i, param_reg in enumerate(parameter_regions):
            try:
                leg_handles, leg_labels = plot_parameter_region_1D(
                    param_reg, true_parameter, parameter_space_bounds, color=colors[i], region_name=region_names[i], linestyle=next(linestyles), custom_ax=ax
                )
            except Exception as e:
                warnings.warn(f"Failed to plot 1D region {i}: {e}")
        
        if parameter_space_bounds is not None:
            ax.set_ylim(parameter_space_bounds['low'], parameter_space_bounds['high'])
        ax.set_ylabel(r'$\theta$', fontsize=45, rotation=0, labelpad=15)
        ax.get_xaxis().set_visible(False)
        if log_scale:
            ax.set_yscale('log')
        ax.tick_params(labelsize=20)
        ax.legend(prop={'size': 12})
        ax.set_title(title, fontsize=15)
        
    elif param_dim == 2:
        colors = colors or cm.rainbow(np.linspace(0, 1, len(region_names)))
        linestyles = cycle(linestyles) if linestyles else cycle(['-', '--', '-.', ':'])
        assert len(region_names) == len(colors) == len(parameter_regions)
        if custom_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = custom_ax

        if prior_samples is not None:
            kde = gaussian_kde(prior_samples.T)
            x = np.linspace(*parameter_space_bounds[param_names[0]].values(), 100)
            y = np.linspace(*parameter_space_bounds[param_names[1]].values(), 100)
            xx, yy = np.meshgrid(x, y)
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(grid_coords).reshape(xx.shape)
            contour_levels = [lvl for lvl in np.linspace(density.min(), density.max(), 9) if lvl > 1e-10]
            ax.contourf(xx, yy, density, levels=contour_levels, cmap=sns.color_palette('Greys', as_cmap=True), alpha=0.8, zorder=1, locator=ticker.MaxNLocator(prune = 'lower'))
            ax.contour(xx, yy, density, levels=contour_levels, colors='darkgrey', linewidths=0.5, linestyles='-', zorder=1)

        for i, param_reg in enumerate(parameter_regions):
            try:
                leg_handles, _ = plot_parameter_region_2D(
                    parameter_region=param_reg,
                    true_parameter=true_parameter,
                    parameter_space_bounds=parameter_space_bounds,
                    labels=labels,
                    param_names=param_names,
                    color=colors[i],
                    linestyle=next(linestyles),
                    region_name=region_names[i],
                    alpha_shape=alpha_shape,
                    alpha=alpha,
                    scatter=scatter,
                    custom_ax=ax
                )
                merged_handle = mpatches.Patch()
                merged_handle.patches = leg_handles
                leg_handles = [merged_handle]
                leg_labels = [region_names[0].split(' ')[0] + ' ' + '-'.join([rn.split(' ')[1][:-2] for rn in region_names]) + '\%']
            except Exception as e:
                warnings.warn(f"Failed to plot 2D region {i}: {e}")

        if prior_samples is not None and 'FreB' not in region_names[0]:
            leg_handles += [mpatches.Patch(edgecolor='darkgrey', facecolor=plt.cm.Greys(80), linewidth=2, label='Prior')]
            leg_labels += ['Prior']
        
        if not remove_legend:
            legend = ax.legend(
                leg_handles, leg_labels, handler_map={leg_handles[0]: MergedPatchHandler(num_patches=len(parameter_regions), gap_ratio=0.1)}, 
                prop={'size': 18}, loc='lower left', handlelength=3
            )
            if alpha_shape:
                legend.legend_handles[0]._sizes = [40]

        ax.set_xlabel(r'$\theta_0$' if labels is None else labels[0], fontsize=25, labelpad=3)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_ylabel(r'$\theta_1$' if labels is None else labels[1], fontsize=25, labelpad=10, rotation=0)
        ax.tick_params(axis='y', labelsize=18)
        
        if parameter_space_bounds is not None and param_names is not None:
            # Set limits based on parameter_space_bounds
            x_low, x_high = parameter_space_bounds[param_names[0]].values()
            y_low, y_high = parameter_space_bounds[param_names[1]].values()
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)
            
            # Set ticks based on the actual bounds
            ax.set_xticks(np.linspace(x_low, x_high, 5))
            ax.set_xticklabels(np.linspace(x_low, x_high, 5))
            ax.set_yticks(np.linspace(y_low, y_high, 5))
            ax.set_yticklabels(np.linspace(y_low, y_high, 5))
        else:
            # Fallback to hardcoded values
            ax.set_xticks(np.linspace(-10, 10, 5))
            ax.set_xticklabels(np.linspace(-10, 10, 5))
            ax.set_yticks(np.linspace(-10, 10, 5))
            ax.set_yticklabels(np.linspace(-10, 10, 5))

        if title is not None:
            ax.set_title(title, size=25, pad=20)
            
    elif param_dim >= 3:
        # Create pairplot for higher dimensions
        colors = colors or cm.rainbow(np.linspace(0, 1, len(region_names)))
        linestyles_list = linestyles or ['-', '--', '-.', ':']
        assert len(region_names) == len(colors) == len(parameter_regions)
        
        # Determine figure size
        if figsize is None:
            figsize = (4 * param_dim, 4 * param_dim)
        
        fig, axes = plt.subplots(param_dim, param_dim, figsize=figsize)
        
        # Generate parameter labels if not provided
        if labels is None:
            labels = [rf'$\theta_{{{i}}}$' for i in range(param_dim)]
        
        # Helper function to filter points based on distance from true_parameter
        def filter_by_proximity(param_reg, dims_to_plot, threshold=1.0, true_parameter=true_parameter):
            """
            Filter parameter region to only include points within threshold distance
            from true_parameter in all dimensions except those being plotted.
            
            Parameters:
            -----------
            param_reg : array
                Parameter region to filter
            dims_to_plot : list
                Dimensions being plotted (to exclude from filtering)
            threshold : float
                Maximum distance from true_parameter in non-plotted dimensions
            
            Returns:
            --------
            filtered_reg : array
                Filtered parameter region
            """
            if not filter_subset or true_parameter is None:
                return param_reg
            
            # Get dimensions to filter on (all except those being plotted)
            dims_to_filter = [d for d in range(param_dim) if d not in dims_to_plot]
            
            if len(dims_to_filter) == 0:
                return param_reg
            
            # Calculate distance in non-plotted dimensions
            true_parameter = to_np_if_torch(true_parameter)
            param_reg = to_np_if_torch(param_reg)
            mask = np.ones(len(param_reg), dtype=bool)
            for d in dims_to_filter:
                mask &= np.abs(param_reg[:, d] - true_parameter[d]) <= threshold
            
            return param_reg[mask]
        
        # Iterate over all pairs of dimensions
        for i in range(param_dim):
            for j in range(param_dim):
                ax = axes[i, j] if param_dim > 1 else axes
                try:
                    if i == j and show_diagonal:
                        if diagonal_type == 'confidence' and diagonal_pvalues is not None and diagonal_grid is not None:
                            plot_confidence_distributions_1D(
                                all_pvalues=diagonal_pvalues[:, i:i+1, :],
                                grid_values=diagonal_grid[i],
                                confidence_levels=diagonal_levels or [],
                                true_theta=true_parameter[i:i+1].reshape(1, 1) if true_parameter is not None else None,
                                custom_ax=ax,
                            )
                        elif diagonal_type == 'posterior' and posterior_estimator is not None and posterior_observations is not None:
                            plot_posterior_distributions_1D(
                                posterior_estimator=posterior_estimator,
                                observations=posterior_observations,
                                credibility_levels=diagonal_levels or [],
                                param_dim=i,
                                true_theta=np.asarray(true_parameter).reshape(1, -1) if true_parameter is not None else None,
                                custom_ax=ax,
                            )
                        elif diagonal_type == 'hist':
                            for k, param_reg in enumerate(parameter_regions):
                                filtered_reg = filter_by_proximity(param_reg, [i])
                                if len(filtered_reg) > 0:
                                    ax.hist(filtered_reg[:, i], bins=30, alpha=0.5, color=colors[k],
                                            label=region_names[k] if i == 0 else None, density=True)
                            if true_parameter is not None:
                                ax.axvline(true_parameter[i], color='red', linestyle='--', linewidth=2, label='True' if i == 0 else None)
                            ax.set_ylabel('Density', fontsize=12)
                            if i == 0:
                                ax.legend(prop={'size': 10}, loc='upper right')
                        elif diagonal_type == 'kde':
                            for k, param_reg in enumerate(parameter_regions):
                                filtered_reg = filter_by_proximity(param_reg, [i])
                                if len(filtered_reg) > 0:
                                    kde = gaussian_kde(filtered_reg[:, i])
                                    x_range = np.linspace(filtered_reg[:, i].min(), filtered_reg[:, i].max(), 100)
                                    ax.plot(x_range, kde(x_range), color=colors[k],
                                            linestyle=linestyles_list[k % len(linestyles_list)],
                                            label=region_names[k] if i == 0 else None)
                            if true_parameter is not None:
                                ax.axvline(true_parameter[i], color='red', linestyle='--', linewidth=2, label='True' if i == 0 else None)
                            ax.set_ylabel('Density', fontsize=12)
                            if i == 0:
                                ax.legend(prop={'size': 10}, loc='upper right')

                    elif i > j:
                        # Lower triangle: scatter plots with 2D regions
                        # Plot prior samples if provided
                        if prior_samples is not None:
                            kde = gaussian_kde(prior_samples[:, [j, i]].T)
                            if param_names is not None and parameter_space_bounds is not None:
                                x = np.linspace(*parameter_space_bounds[param_names[j]].values(), 100)
                                y = np.linspace(*parameter_space_bounds[param_names[i]].values(), 100)
                            else:
                                x = np.linspace(prior_samples[:, j].min(), prior_samples[:, j].max(), 100)
                                y = np.linspace(prior_samples[:, i].min(), prior_samples[:, i].max(), 100)
                            xx, yy = np.meshgrid(x, y)
                            grid_coords = np.vstack([xx.ravel(), yy.ravel()])
                            density = kde(grid_coords).reshape(xx.shape)
                            contour_levels = [lvl for lvl in np.linspace(density.min(), density.max(), 9) if lvl > 1e-10]
                            ax.contourf(xx, yy, density, levels=contour_levels, cmap=sns.color_palette('Greys', as_cmap=True),
                                       alpha=0.5, zorder=1, locator=ticker.MaxNLocator(prune='lower'))

                        # Plot parameter regions
                        linestyles_cycle = cycle(linestyles_list)
                        for k, param_reg in enumerate(parameter_regions):
                            # Filter to subset if enabled
                            filtered_reg = filter_by_proximity(param_reg, [j, i])

                            if len(filtered_reg) > 0:
                                # Extract 2D projection
                                param_reg_2d = filtered_reg[:, [j, i]]
                                true_param_2d = true_parameter[[j, i]] if true_parameter is not None else None
                                param_names_2d = [param_names[j], param_names[i]] if param_names is not None else None
                                labels_2d = [labels[j], labels[i]]

                                plot_parameter_region_2D(
                                    parameter_region=param_reg_2d,
                                    true_parameter=true_param_2d,
                                    parameter_space_bounds=parameter_space_bounds,
                                    labels=labels_2d,
                                    param_names=param_names_2d,
                                    color=colors[k],
                                    linestyle=next(linestyles_cycle),
                                    region_name=region_names[k],
                                    alpha_shape=alpha_shape,
                                    alpha=alpha,
                                    scatter=scatter,
                                    custom_ax=ax
                                )
                    else:
                        # Upper triangle: hide or show correlation/info
                        ax.axis('off')

                    # Set labels only on edges
                    if i == param_dim - 1:
                        ax.set_xlabel(labels[j], fontsize=14)
                    else:
                        ax.set_xticklabels([])

                    if j == 0 and i != j:
                        ax.set_ylabel(labels[i], fontsize=14, rotation=0, labelpad=20)
                    elif i != j:
                        ax.set_yticklabels([])
                except Exception as e:
                    warnings.warn(f"Failed to plot panel (i={i}, j={j}): {e}")
        
        plt.tight_layout()
        if title is not None:
            fig.suptitle(title, fontsize=20, y=1.02)
    else:
        raise ValueError("param_dim must be a positive integer")

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
        plt.close()
    if custom_ax is None:
        plt.show()


def plot_parameter_region_1D(
    parameter_region: np.ndarray,
    true_parameter: np.ndarray,
    parameter_space_bounds: Optional[Dict[str, float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: Optional[str] = 'green',
    log_scale: bool = False,
    region_name: str = 'Parameter Region',
    linestyle: str = '-',
    custom_ax: Optional[Axes] = None
) -> None:
    """Plot 1-dimensional parameter regions using the lower and upper bounds.
    """

    if custom_ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize if figsize is not None else (3, 9))
    else:
        ax = custom_ax

    if parameter_region is None or len(parameter_region) == 0:
        if custom_ax is not None:
            return ax.get_legend_handles_labels()
        return

    parameter_region = to_np_if_torch(parameter_region).reshape(-1)
    ax.scatter(x=true_parameter, y=true_parameter, alpha=1, c="red", marker="*", s=250, zorder=10)
    ax.axhline(y=np.min(parameter_region), xmin=0.45, xmax=0.55, label=region_name, color=color, linestyle=linestyle)
    ax.axhline(y=np.max(parameter_region), xmin=0.45, xmax=0.55, color=color, linestyle=linestyle)
    ax.vlines(x=true_parameter, ymin=np.min(parameter_region), ymax=np.max(parameter_region), color=color, linestyle=linestyle)

    if custom_ax is None:
        if parameter_space_bounds is not None:
            ax.set_ylim(parameter_space_bounds['low'], parameter_space_bounds['high'])
        ax.set_ylabel(r'$\theta$', fontsize=45, rotation=0)
        ax.get_xaxis().set_visible(False)
        if log_scale:
            ax.set_yscale('log')
        ax.tick_params(labelsize=20)
        ax.legend(prop={'size': 12})
        plt.show()
    else:
        # to plot legend in main plot
        return ax.get_legend_handles_labels()


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
    linestyle: Optional[str] = '-',
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

    if parameter_region is None or len(parameter_region) == 0:
        if custom_ax is not None:
            return ax.get_legend_handles_labels()
        return

    if not scatter:
        warnings.warn("Contour might be unreliable if alpha is not chosen properly. Please plot scatter as well, and choose alpha appropriately (try from 1 to 20).")

    if scatter:
        ax.scatter(x=parameter_region[:, 0], y=parameter_region[:, 1], s=3.5, color=to_rgba(color, 1), zorder=1, label=region_name)
    if alpha_shape:
        alpha_shape = alphashape.alphashape(parameter_region, alpha=alpha)
        patch = PolygonPatchFixed(alpha_shape, fc=to_rgba(color, 0.2), ec=to_rgba(color, 1), lw=5, label=region_name, linestyle=linestyle)
        ax.add_patch(patch)
    if true_parameter is not None:
        ax.scatter(x=true_parameter.reshape(-1,)[0], y=true_parameter.reshape(-1,)[1], alpha=1, marker='*', facecolor='white', edgecolor='white', s=300, linewidth=2, zorder=10)
        ax.scatter(x=true_parameter.reshape(-1,)[0], y=true_parameter.reshape(-1,)[1], alpha=1, marker='*', facecolor='none', edgecolor='red', s=300, linewidth=2, zorder=10)
    
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
        for handle in legend.legend_handles:
            handle.set_sizes([30])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
            ax.spines[axis].set_color('black')
        plt.show()
    else:
        # to plot legend in main plot
        return ax.get_legend_handles_labels()
        

def parameter_regions_pairplot(
    *parameter_regions: np.ndarray,
    true_parameter: np.ndarray,
    parameter_space_bounds: Optional[Dict[str, Tuple[float]]] = None,
    labels: Optional[np.ndarray] = None,
    param_names: Optional[np.ndarray] = None,
    colors: Optional[Sequence[str]] = None,
    region_names: Optional[Sequence[str]] = None,
    alpha_shape: bool = False,
    alpha: Optional[float] = None,
    scatter: bool = True,
    show_diagonal: bool = False,
    diagonal_type: str = 'confidence',
    diagonal_pvalues: Optional[np.ndarray] = None,
    diagonal_grid: Optional[np.ndarray] = None,
    diagonal_levels: Optional[Sequence[float]] = None,
    posterior_estimator=None,
    posterior_observations: Optional[Sequence] = None,
    figsize: Optional[Sequence[int]] = (15, 15),
    show_legend: bool = True,
    save_fig_path: Optional[str] = None
) -> None:
    """Plot a pairplot of 2D parameter region projections.

    Each upper-triangle cell (row, col) shows the 2D projection onto
    dimensions [row, col].  The diagonal shows either a confidence curve or
    posterior density.  Lower triangle is hidden.

    Parameters
    ----------
    *parameter_regions :
        One or more confidence/credible sets, each of shape ``(n_pts, param_dim)``.
    true_parameter :
        1-D array of shape ``(param_dim,)`` for the single observation being plotted.
    diagonal_pvalues :
        Shape ``(1, param_dim, grid_size)`` — p-values for one observation.
    diagonal_grid :
        Shape ``(param_dim, grid_size)`` — grid coordinates per dimension.
    diagonal_levels :
        Confidence or credibility levels for the diagonal curves.
    """
    rows = cols = parameter_regions[0].shape[1]
    colors = colors or cm.rainbow(np.linspace(0, 1, len(region_names)))
    assert len(region_names) == len(colors) == len(parameter_regions)

    param_names = np.asarray(param_names) if param_names is not None else np.array([rf'$\theta_{{{i}}}$' for i in range(rows)])
    true_parameter = np.asarray(true_parameter).reshape(-1)

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    leg_handles, leg_labels = [], []
    for row in range(rows):
        for col in range(cols):
            if row == col:
                if show_diagonal:
                    if diagonal_type == 'confidence' and diagonal_pvalues is not None and diagonal_grid is not None:
                        plot_confidence_distributions_1D(
                            all_pvalues=diagonal_pvalues[0:1, row:row+1, :],
                            grid_values=diagonal_grid[row],
                            confidence_levels=diagonal_levels or [],
                            true_theta=true_parameter[row:row+1].reshape(1, 1),
                            colors=colors,
                            # xlim=parameter_space_bounds[param_names[row]] if parameter_space_bounds is not None and param_names is not None else None,
                            custom_ax=ax[row, col],
                        )
                    elif diagonal_type == 'posterior' and posterior_estimator is not None and posterior_observations is not None:
                        plot_posterior_distributions_1D(
                            posterior_estimator=posterior_estimator,
                            observations=posterior_observations,
                            credibility_levels=diagonal_levels or [],
                            param_dim=row,
                            true_theta=true_parameter.reshape(1, -1),
                            colors=colors,
                            # xlim=parameter_space_bounds[param_names[row]] if parameter_space_bounds is not None and param_names is not None else None,
                            custom_ax=ax[row, col],
                        )
                    ax[row, col].legend().set_visible(False)  # Hide legend for diagonal plots
                else:
                    ax[row, col].axis('off')
            elif col < row:
                ax[row, col].axis('off')
                ax[row, col].legend().set_visible(False)
            else:
                for i, parameter_region in enumerate(parameter_regions):
                    leg_handles, leg_labels = plot_parameter_region_2D(
                        parameter_region=parameter_region[:, [row, col]],
                        true_parameter=true_parameter[[row, col]],
                        parameter_space_bounds=parameter_space_bounds,
                        param_names=param_names[[row, col]],
                        color=colors[i],
                        region_name=region_names[i],
                        alpha_shape=alpha_shape,
                        alpha=alpha,
                        scatter=scatter,
                        custom_ax=ax[row, col],
                    )
                ax[row, col].set_xlabel(param_names[row] if labels is None else labels[row])
                ax[row, col].set_ylabel(param_names[col] if labels is None else labels[col])

    if leg_handles and show_legend:
        legend = fig.legend(leg_handles, leg_labels, bbox_to_anchor=(0.5, 0.5))
        if alpha_shape:
            legend.legend_handles[0]._sizes = [40]

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_parameter_intervals(
    *parameter_regions: np.ndarray,
    param_dim: int,
    point_estimates: Optional[Sequence[np.ndarray]] = None,
    true_parameters: Optional[Sequence[np.ndarray]] = None,
    interval_type: str = 'projection',
    oat_intervals: Optional[Sequence[np.ndarray]] = None,
    param_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence] = None,
    region_names: Optional[Sequence[str]] = None,
    parameter_space_bounds: Optional[Dict] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_fig_path: Optional[str] = None,
) -> None:
    """Plot 1D interval summaries of ND confidence sets.

    Each parameter dimension gets its own horizontal number line showing the interval
    as a closed segment with bracket-style endpoint markers and an optional point estimate
    indicator.

    Parameters
    ----------
    *parameter_regions : np.ndarray
        One or more confidence sets for a **single** observation, each of shape
        ``(n_grid_pts, param_dim)``.  For 1D parameters, shape ``(n_grid_pts,)`` is
        also accepted.
    param_dim : int
        Number of parameter dimensions.
    point_estimates : sequence of np.ndarray, optional
        One array of shape ``(param_dim,)`` per region giving the maximum-p-value
        estimate θ^Focal for that region.  Used both to label the indicator and (when
        ``interval_type='slice'``) to construct the slice intervals.
    true_parameters : sequence of np.ndarray, optional
        One array of shape ``(param_dim,)`` per region giving the true parameter
        values for that region.  Plotted as a red star.
    interval_type : str, optional
        How to derive 1D intervals from the ND confidence set:

        * ``'projection'`` (default) — take ``[min, max]`` of each column.
        * ``'slice'`` — fix all dimensions except *d* at the nearest grid value to
          θ^Focal, then take ``[min, max]`` of column *d*.  Requires ``point_estimates``.
        * ``'oat'`` — use pre-computed OAT intervals passed via ``oat_intervals``.
    oat_intervals : sequence of np.ndarray, optional
        Pre-computed one-at-a-time intervals from ``LF2I.oat_intervals``.  Each element
        has shape ``(param_dim, 2)`` and corresponds to one region.  When provided,
        ``interval_type`` is ignored for interval derivation and ``*parameter_regions``
        may be omitted.
    param_names : sequence of str, optional
        Axis labels; falls back to ``θ_0, θ_1, …`` if not supplied.
    colors : sequence, optional
        One colour per region; defaults to a rainbow palette.
    region_names : sequence of str, optional
        Legend labels for the regions.
    parameter_space_bounds : dict, optional
        ``{param_name: {'low': float, 'high': float}}`` used to set per-axis xlim.
    title : str, optional
        Figure suptitle.
    figsize : tuple of int, optional
        ``(width, height)`` in inches.  Defaults to ``(8, 0.9 * param_dim)``.
    save_fig_path : str, optional
        If given, save the figure to this path.
    """
    if interval_type == 'slice' and point_estimates is None:
        raise ValueError("interval_type='slice' requires point_estimates")
    if oat_intervals is not None and len(parameter_regions) == 0:
        n_regions = len(oat_intervals)
    else:
        n_regions = len(parameter_regions)

    colors = list(colors) if colors is not None else list(cm.rainbow(np.linspace(0, 1, n_regions)))
    region_names = list(region_names) if region_names is not None else [f'Region {i}' for i in range(n_regions)]
    param_names = list(param_names) if param_names is not None else [rf'$\theta_{{{d}}}$' for d in range(param_dim)]

    # Normalise regions to 2D arrays (only needed when deriving intervals from ND sets)
    regions_2d = []
    for cs in parameter_regions:
        cs = to_np_if_torch(cs)
        if cs.ndim == 1:
            cs = cs.reshape(-1, 1)
        regions_2d.append(cs)

    # --- compute intervals for each region and dimension ---
    def _projection_interval(cs, d):
        col = cs[:, d]
        return col.min(), col.max()

    def _slice_interval(cs, d, pe):
        other_dims = [j for j in range(param_dim) if j != d]
        mask = np.ones(len(cs), dtype=bool)
        for j in other_dims:
            nearest_val = cs[np.argmin(np.abs(cs[:, j] - pe[j])), j]
            mask &= (cs[:, j] == nearest_val)
        slice_pts = cs[mask, d]
        if len(slice_pts) == 0:
            return np.nan, np.nan
        return slice_pts.min(), slice_pts.max()

    # intervals[k][d] = (lo, hi)
    if oat_intervals is not None:
        intervals = [
            [(float(oat_intervals[k][d, 0]), float(oat_intervals[k][d, 1])) for d in range(param_dim)]
            for k in range(n_regions)
        ]
    else:
        intervals = []
        for k, cs in enumerate(regions_2d):
            pe = to_np_if_torch(point_estimates[k]) if point_estimates is not None else None
            row = []
            for d in range(param_dim):
                if interval_type == 'slice':
                    lo, hi = _slice_interval(cs, d, pe)
                else:
                    lo, hi = _projection_interval(cs, d)
                row.append((lo, hi))
            intervals.append(row)

    # --- layout ---
    dist_in = 0.9
    if figsize is None:
        figsize = (8, max(dist_in * param_dim, 1.5))

    fig, axs = plt.subplots(param_dim, 1, figsize=figsize)
    if param_dim == 1:
        axs = [axs]

    thickness = 0.18
    # vertical offsets so multiple regions don't overlap
    y_offsets = np.linspace(-thickness * (n_regions - 1) / 2,
                             thickness * (n_regions - 1) / 2, n_regions)

    legend_handles = []
    for k in range(n_regions):
        patch = mpatches.Patch(color=colors[k], label=region_names[k])
        legend_handles.append(patch)

    for d, ax in enumerate(axs):
        ax.yaxis.set_visible(False)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))

        # determine xlim for this dimension
        if parameter_space_bounds is not None and param_names[d] in parameter_space_bounds:
            bounds = parameter_space_bounds[param_names[d]]
            xlim = (bounds['low'], bounds['high'])
        else:
            all_vals = []
            for k in range(n_regions):
                lo, hi = intervals[k][d]
                if not (np.isnan(lo) or np.isnan(hi)):
                    all_vals.extend([lo, hi])
                if point_estimates is not None:
                    pe = to_np_if_torch(point_estimates[k])
                    all_vals.append(float(pe[d]) if param_dim > 1 else float(pe))
            if all_vals:
                span = max(all_vals) - min(all_vals)
                pad = span * 0.15 if span > 0 else 1.0
                xlim = (min(all_vals) - pad, max(all_vals) + pad)
            else:
                xlim = (-1, 1)
        ax.set_xlim(*xlim)
        ax.set_ylim(-thickness * 2, thickness * 2)

        for k in range(n_regions):
            lo, hi = intervals[k][d]
            y = y_offsets[k]
            color = colors[k]

            if not (np.isnan(lo) or np.isnan(hi)):
                # filled rectangle (semitransparent)
                ax.add_patch(mpatches.Rectangle(
                    (lo, y - thickness / 2), hi - lo, thickness,
                    linewidth=0, color=color, alpha=0.4, zorder=2
                ))

                # bracket symbols at endpoints
                bracket_fs = 14
                ax.text(lo, y, '[', ha='right', va='center', color=color,
                        fontsize=bracket_fs, fontweight='bold', zorder=3,
                        transform=ax.transData)
                ax.text(hi, y, ']', ha='left', va='center', color=color,
                        fontsize=bracket_fs, fontweight='bold', zorder=3,
                        transform=ax.transData)

            # point estimate: star in region color
            if point_estimates is not None:
                pe = to_np_if_torch(point_estimates[k])
                pe_val = float(pe[d]) if param_dim > 1 else float(pe)
                ax.plot(pe_val, y, marker='*', color=color,
                        markersize=10, zorder=4, linestyle='none',
                        label='Focal point')

            # true parameter: red star
            if true_parameters is not None:
                tp = to_np_if_torch(true_parameters[k])
                tp_val = float(tp[d]) if param_dim > 1 else float(tp)
                ax.plot(tp_val, y, marker='*', color='red',
                        markersize=10, zorder=5, linestyle='none',
                        label='Truth')

        ax.set_xlabel(param_names[d], fontsize=12, labelpad=4)
        ax.tick_params(axis='x', labelsize=10)

    star_patch = mlines.Line2D([], [], color=color, marker='*', linestyle='None',
                               markersize=10, label='Focal point')
    if star_patch not in legend_handles:
        legend_handles.append(star_patch)

    red_star_patch = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                                   markersize=10, label='Truth')
    if red_star_patch not in legend_handles:
        legend_handles.append(red_star_patch)

    if title is not None:
        fig.suptitle(title, fontsize=13, y=1.01)

    if n_regions > 0:
        fig.legend(handles=legend_handles, loc='lower center', fontsize=10,
                   ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout()

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confidence_distributions_1D(
    all_pvalues: np.ndarray,
    grid_values: np.ndarray,
    confidence_levels: Optional[Sequence[float]] = None,
    point_estimates: Optional[Sequence[np.ndarray]] = None,
    true_theta: Optional[np.ndarray] = None,
    param_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    xlim: Optional[Tuple[float, float]] = (-10, 10),
    save_fig_path: Optional[str] = None,
    custom_ax: Optional[Axes] = None,
) -> None:
    """Plot the confidence distribution (p-value curve) for 1D parameter sweeps.

    Creates one figure per observation showing the normalised p-value curve,
    interval bars for each confidence level, and optional point-estimate /
    true-parameter markers.

    Parameters
    ----------
    all_pvalues : np.ndarray
        Raw p-values with shape ``(n_obs, 1, n_grid)``.
    grid_values : np.ndarray
        Grid of parameter values with shape ``(1, n_grid)`` or ``(n_grid,)``.
    confidence_levels : sequence of float
        Confidence levels (e.g. ``[0.9, 0.95]``).  Each produces one
        horizontal threshold line and one row of interval bars.
    point_estimates : sequence of np.ndarray, optional
        One array of shape ``(1,)`` per observation for the point estimate.
    true_theta : np.ndarray, optional
        True parameters with shape ``(n_obs, 1)`` or ``(n_obs,)``.
    param_names : sequence of str, optional
        Axis label for the parameter; defaults to ``['$\\theta_1$']``.
    colors : sequence, optional
        Colors for each confidence level; defaults to a rainbow palette.
    title : str, optional
        Suptitle applied to every figure.
    figsize : tuple of int, optional
        ``(width, height)`` in inches.  Defaults to ``(7, 5)``.
    xlim : tuple of float, optional
        ``(low, high)`` x-axis limits.  Inferred from ``grid_values`` if omitted.
    save_fig_path : str, optional
        If given, figures are saved as ``<save_fig_path>_<obs_idx>.png`` and
        not displayed interactively.
    custom_ax : Axes, optional
        If provided, draw into this existing axes instead of creating a new
        figure.  Figure creation, ``plt.show()``, and ``save_fig_path`` are
        all skipped; the caller is responsible for display/saving.  Intended
        for embedding into a larger layout (e.g. a pairplot diagonal).
    """
    all_pvalues = to_np_if_torch(all_pvalues)
    grid = to_np_if_torch(grid_values).reshape(-1)

    n_obs = all_pvalues.shape[0]
    n_levels = len(confidence_levels) if confidence_levels is not None else 0

    param_names = list(param_names) if param_names is not None else [r'$\theta_1$']
    colors = list(colors) if colors is not None else list(cm.rainbow(np.linspace(0, 1, max(1, n_levels))))
    figsize = figsize if figsize is not None else (7, 5)
    x_low, x_high = xlim if xlim is not None else (float(grid.min()), float(grid.max()))

    for xdx in range(n_obs):
        raw_pv = all_pvalues[xdx, 0, :]
        pv_max = raw_pv.max()
        pvalues = raw_pv / pv_max if pv_max > 0 else raw_pv

        if custom_ax is not None:
            ax = custom_ax
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(grid, pvalues, color=colors[0], label='p-value curve', lw=5, clip_on=False)

        if confidence_levels is not None:
            for tdx, cl in enumerate(confidence_levels):
                threshold = 1.0 - cl
                color = colors[tdx % len(colors)]

                in_interval = pvalues >= threshold
                crossings = np.where(np.diff(in_interval.astype(int)))[0]
                interval_endpoints = []
                for idx in crossings:
                    x0, x1 = grid[idx], grid[idx + 1]
                    p0, p1 = pvalues[idx], pvalues[idx + 1]
                    if p1 != p0:
                        x_cross = x0 + (threshold - p0) / (p1 - p0) * (x1 - x0)
                    else:
                        x_cross = (x0 + x1) / 2.0
                    interval_endpoints.append(x_cross)
                if in_interval[0]:
                    interval_endpoints.insert(0, grid[0])
                if in_interval[-1]:
                    interval_endpoints.append(grid[-1])

                bar_y = -0.1 - 0.08 * (n_levels - tdx)
                for i in range(0, len(interval_endpoints) - 1, 2):
                    lo, hi = interval_endpoints[i], interval_endpoints[i + 1]
                    ax.fill_between([lo, hi], bar_y - 0.02, bar_y + 0.02,
                                    color=color, alpha=0.3, clip_on=False, zorder=4)
                    ax.fill_between([lo + 0.01, hi - 0.01], bar_y - 0.01, bar_y + 0.01,
                                    color=color, alpha=0.7, clip_on=False, zorder=5)
                    for x_end in (lo, hi):
                        ax.plot([x_end, x_end], [bar_y, threshold], color=color,
                                linestyle='--', alpha=0.6, clip_on=False)

                ax.axhline(y=threshold, color='grey', linestyle='--',
                        label=f'{int(cl * 100):.0f}% Confidence Level')

        if point_estimates is not None:
            pe = to_np_if_torch(point_estimates[xdx]).reshape(-1)
            ax.scatter(pe[0], 0, edgecolor='blue', facecolor='white', marker='*',
                       label='Point Estimate', clip_on=False, zorder=5)
            ax.axvline(x=pe[0], color='blue', alpha=0.5, linestyle='--',
                       label='_nolegend_', clip_on=False)

        if true_theta is not None:
            tt = to_np_if_torch(true_theta).reshape(n_obs, -1)
            ax.scatter(tt[xdx, 0], 0, edgecolor='red', facecolor='white', marker='*',
                       label='True Parameter', clip_on=False, s=300, linewidth=2, zorder=5)
            ax.axvline(x=tt[xdx, 0], color='red', alpha=0.5, linestyle='--',
                       label='_nolegend_', clip_on=False, lw=5)

        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_xlim(x_low, x_high)
        ax.set_ylabel('Confidence distribution', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
        if title is not None:
            ax.set_title(title, fontsize=13)

        if custom_ax is None:
            if save_fig_path is not None:
                fig.savefig(f'{save_fig_path}_{xdx}.png', bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


def plot_posterior_distributions_1D(
    posterior_estimator,
    observations: Sequence,
    credibility_levels: Sequence[float],
    n_samples: int = 10_000,
    param_dim: int = 0,
    point_estimates: Optional[Sequence[np.ndarray]] = None,
    true_theta: Optional[np.ndarray] = None,
    param_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    xlim: Optional[Tuple[float, float]] = (-10, 10),
    n_grid: int = 500,
    save_fig_path: Optional[str] = None,
    custom_ax: Optional[Axes] = None,
) -> None:
    """Plot the marginal posterior KDE with HPD intervals for 1D sweeps.

    Mirrors the layout of ``plot_confidence_distributions_1D``: a normalized
    density curve, one horizontal threshold line per credibility level (at the
    normalized density of the HPD boundary), and interval bars below the axis.

    Parameters
    ----------
    posterior_estimator :
        Object with a ``.sample((n_samples,), x=obs)`` method returning an
        array of shape ``(n_samples, d_theta)``.
    observations :
        Sequence of individual observations, one per figure.
    credibility_levels :
        Credibility levels, e.g. ``[0.9, 0.95]``.
    n_samples :
        Number of posterior samples to draw per observation.
    param_dim :
        Which parameter dimension to marginalize to for plotting.
    point_estimates :
        Optional sequence of arrays of shape ``(d_theta,)`` or ``(1,)``, one
        per observation.
    true_theta :
        Optional array of shape ``(n_obs, d_theta)`` or ``(n_obs,)``.
    param_names :
        x-axis label; defaults to ``['$\\theta_1$']``.
    colors :
        Colors for each credibility level; defaults to a rainbow palette.
    title :
        Suptitle applied to every figure.
    figsize :
        ``(width, height)`` in inches.  Defaults to ``(7, 5)``.
    xlim :
        ``(low, high)`` x-axis limits.  Inferred from samples if omitted.
    n_grid :
        Number of points used to evaluate the KDE curve.
    save_fig_path :
        If given, figures are saved as ``<save_fig_path>_<obs_idx>.png``.
    custom_ax : Axes, optional
        If provided, draw into this existing axes instead of creating a new
        figure.  Figure creation, ``plt.show()``, and ``save_fig_path`` are
        all skipped; the caller is responsible for display/saving.  Intended
        for embedding into a larger layout (e.g. a pairplot diagonal).
    """
    n_obs = len(observations)
    n_levels = len(credibility_levels)

    param_names = list(param_names) if param_names is not None else [r'$\theta_1$']
    colors = list(colors) if colors is not None else list(cm.rainbow(np.linspace(0, 1, n_levels)))
    figsize = figsize if figsize is not None else (7, 5)

    if true_theta is not None:
        true_theta = np.asarray(true_theta).reshape(n_obs, -1)

    for xdx, obs in enumerate(observations):
        raw_samples = posterior_estimator.sample((n_samples,), x=obs)
        try:
            marginal = np.asarray(raw_samples)[:, param_dim]
        except (IndexError, TypeError):
            marginal = np.asarray(raw_samples).reshape(-1)

        kde = gaussian_kde(marginal)

        x_low = xlim[0] if xlim is not None else float(marginal.min())
        x_high = xlim[1] if xlim is not None else float(marginal.max())
        grid = np.linspace(x_low, x_high, n_grid)

        density = kde(grid)
        density_max = density.max()
        norm_density = density / density_max if density_max > 0 else density

        map_x = float(grid[np.argmax(density)])

        if custom_ax is not None:
            ax = custom_ax
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(grid, norm_density, color=colors[0], label='Posterior density', lw=5, clip_on=False)

        if point_estimates:
            ax.scatter(map_x, 0.0, marker='*', edgecolor=colors[0], facecolor='white', zorder=5,
                    label='MAP', clip_on=False)
            ax.axvline(x=map_x, color='grey', alpha=0.4, linestyle='--', clip_on=False,
                    label='_nolegend_')

        for tdx, cl in enumerate(credibility_levels):
            color = colors[tdx % len(colors)]

            lo, hi = _hpd_1d(marginal, cl)

            boundary_density = float(np.mean([kde(lo), kde(hi)])) / density_max
            boundary_density = float(np.clip(boundary_density, 0.0, 1.0))

            in_hpd = (grid >= lo) & (grid <= hi)
            bar_y = -0.1 - 0.08 * (n_levels - tdx)

            ax.fill_between(grid, bar_y - 0.02, bar_y + 0.02,
                            where=in_hpd, color=color, alpha=0.3,
                            clip_on=False, zorder=4)
            ax.fill_between(grid, bar_y - 0.01, bar_y + 0.01,
                            where=in_hpd, color=color, alpha=0.7,
                            clip_on=False, zorder=5)
            for x_end in (lo, hi):
                ax.plot([x_end, x_end], [bar_y, boundary_density],
                        color=color, linestyle='--', alpha=0.6, clip_on=False)

            ax.axhline(y=boundary_density, color=color, linestyle='--', alpha=0.5,
                       label=f'{int(cl * 100):.0f}% HPD')

        if true_theta is not None:
            tt_val = float(true_theta[xdx, param_dim])
            ax.scatter(tt_val, 0, edgecolor='red', facecolor='white', marker='*',
                       label='True parameter', clip_on=False, s=300, linewidth=2, zorder=5)
            ax.axvline(x=tt_val, color='red', alpha=0.5, linestyle='--',
                       clip_on=False, label='_nolegend_', lw=5)

        ax.set_xlabel(param_names[0], fontsize=12)
        ax.set_xlim(x_low, x_high)
        ax.set_ylabel('Normalized posterior density', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
        if title is not None:
            ax.set_title(title, fontsize=13)

        if custom_ax is None:
            if save_fig_path is not None:
                fig.savefig(f'{save_fig_path}_{xdx}.png', bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


class MergedPatchHandler(HandlerPatch):
    def __init__(self, num_patches, gap_ratio=0.05, **kwargs):
        self.num_patches = num_patches
        self.gap_ratio = gap_ratio
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        gap = width * self.gap_ratio  
        rect_width = (width - (self.num_patches - 1) * gap) / self.num_patches

        patches = []
        for i, patch in enumerate(orig_handle.patches):
            # Get the linestyle and handle dash patterns
            linestyle = patch.get_linestyle()

            # Map dash patterns to standard linestyle strings
            if isinstance(linestyle, (list, tuple)):
                # Common dash pattern mappings
                dash_map = {
                    (0.0, None): '-',      # solid
                    (None, None): '-',     # solid
                }
                linestyle = dash_map.get(tuple(linestyle) if isinstance(linestyle, list) else linestyle, '-')

            new_patch = mpatches.Rectangle(
                [xdescent + i * (rect_width + gap), ydescent],
                rect_width, height, transform=trans,
                edgecolor=patch.get_edgecolor(), 
                facecolor=patch.get_facecolor(),
                linewidth=patch.get_linewidth(),
                linestyle=linestyle,
            )
            patches.append(new_patch)

        return patches
