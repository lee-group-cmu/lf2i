from typing import Optional, Tuple, Union, Dict, List, Sequence, Any
from warnings import simplefilter

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes._axes import Axes
import matplotlib.colors as mcolors
import seaborn as sns


def coverage_probability_plot(
    parameters: np.ndarray,
    coverage_probability: np.ndarray,
    confidence_level: float,
    param_dim: int,
    upper_proba: Optional[np.ndarray] = None,
    lower_proba: Optional[np.ndarray] = None,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (10, 8),
    xlims: Optional[Tuple[float]] = None,
    ylims: Optional[Tuple[float]] = None,
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    vmin_vmax: Optional[Union[Tuple, List]] = None,
    custom_ax: Optional[Axes] = None,  # if passing custom ax for pairplot
    show_text: bool = False,
    show_undercoverage: bool = False,
    title: Optional[str] = None,
    n_bins: int = 30,
    n_levels: int = 15
) -> None:
    # TODO: Plot variance in coverage
    if param_dim == 1:
        df_plot = pd.DataFrame({
            "parameters": parameters.reshape(-1,),
            "mean_proba": coverage_probability.reshape(-1,),
            # "lower_proba": lower_proba.reshape(-1,),
            # "upper_proba": upper_proba.reshape(-1,)
        }).sort_values(by="parameters")

        _, ax = plt.subplots(1, 1)
        ax.plot(df_plot.parameters, df_plot.mean_proba, color='crimson', label='Estimated Coverage')
        # ax.plot(df_plot.parameters, df_plot.lower_proba, color='crimson')
        # ax.plot(df_plot.parameters, df_plot.upper_proba, color='crimson')
        # ax.fill_between(x=df_plot.parameters, y1=df_plot.lower_proba, y2=df_plot.upper_proba, alpha=0.2, color='crimson')
        ax.axhline(y=confidence_level, color='black', linestyle="--", linewidth=3, 
                    label=f"Nominal coverage = {round(100 * confidence_level, 1)} %", zorder=10)
        
        if params_labels is None:
            ax.set_xlabel(r"$\theta$", fontsize=45)
        else:
            ax.set_xlabel(params_labels[0], fontsize=45)

        ax.set_ylabel("Coverage", fontsize=45)
        ax.set_ylim(*ylims if ylims is not None else (0, 1))
        ax.legend()
    else:
        vmin_vmax = vmin_vmax or (0, 100)
        if param_dim == 2:
            if custom_ax is None:
                fig = plt.figure()
            ax = custom_ax or plt.gca()

            x_bins = np.histogram_bin_edges(parameters[:, 0], bins=n_bins)
            y_bins = np.histogram_bin_edges(parameters[:, 1], bins=n_bins)
            binned_sum_proba, xedges, yedges = np.histogram2d(
                parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins], weights=np.round(coverage_probability*100, 2)
            )
            bin_counts, _, _ = np.histogram2d(parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins]) 
            heatmap_values = binned_sum_proba / bin_counts

            levels = np.linspace(vmin_vmax[0], vmin_vmax[1], num=n_levels)
            # Create mesh grid for plotting - use bin CENTERS not edges
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            y_centers = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers)

            cmap, norm = create_jetr_cmap(confidence_level=confidence_level*100)
            contour_filled = ax.contourf(
                X, Y, heatmap_values.T,
                levels=levels, cmap=cmap, norm=norm, extend='both'
            )
            contour_lines = ax.contour(
                X, Y, heatmap_values.T,
                levels=levels, colors="white", linewidths=0.8
            )
            # clabels = ax.clabel(contour_lines, levels[::2], inline=True, fontsize=25, fmt="%1.1f%%")
            # for txt in clabels:
            #     txt.set_color("black")
            #     txt.set_bbox(dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.1"))

            # if show_text:
            #     # Add numerical labels at contour centroids
            #     for y_index in range(len(yedges)-1):
            #         for x_index in range(len(xedges)-1):
            #             label = heatmap_values.T[::-1, :][y_index, x_index]
            #             if not np.isnan(label):
            #                 ax.text(xedges[x_index] + 0.5*(xedges[1]-xedges[0]), 
            #                         yedges[y_index] + 0.5*(yedges[1]-yedges[0]), 
            #                         f'{label:.1f}', color='black', ha='center', va='center', fontsize=7)

            if show_undercoverage:
                binned_sum_upper_proba, _, _ = np.histogram2d(
                    parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins], weights=np.round(upper_proba*100, 2)
                )
                bin_counts_upper, _, _ = np.histogram2d(parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins]) 
                heatmap_upper_values = binned_sum_upper_proba / bin_counts_upper

                for y_index in range(len(yedges)-1):
                    for x_index in range(len(xedges)-1):
                        if heatmap_upper_values.T[::-1, :][y_index, x_index] < confidence_level * 100:
                            ax.scatter(xedges[x_index] + 0.5*(xedges[1]-xedges[0]), 
                                       yedges[y_index] + 0.5*(yedges[1]-yedges[0]), 
                                       marker='X', color='red', s=200)

            if custom_ax is None:
                # Use a common colorbar
                cbar = fig.colorbar(contour_filled, format='%1.2f')
                standard_ticks = np.round(np.linspace(vmin_vmax[0], vmin_vmax[1], num=6), 1)
                all_ticks = np.unique(np.sort(np.append(standard_ticks[:-1], confidence_level * 100)))
                tick_labels = [f"{label:.0f}\%" for label in all_ticks]
                for i, label in enumerate(all_ticks):
                    if abs(label - confidence_level*100) <= 1e-6:
                        tick_labels[i] = r"$\mathbf{{{label}}}$\textbf{{\%}}".format(label=int(label))
                cbar.ax.yaxis.set_ticks(all_ticks)
                cbar.ax.set_yticklabels(tick_labels, fontsize=30)
                cbar.ax.axhline(y=confidence_level*100, xmin=0, xmax=1, color="black", linestyle="--", linewidth=2.5)

            ax.tick_params(axis='both', labelsize=20)
            if params_labels is None:
                ax.set_xlabel(r"$\theta^{{(1)}}$", fontsize=25, labelpad=3)
                ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=25, labelpad=10, rotation=0)
            else:
                ax.set_xlabel(params_labels[0], fontsize=25, labelpad=3)
                ax.set_ylabel(params_labels[1], fontsize=25, labelpad=10, rotation=0)

            # Set limits properly to show all data
            if xlims is not None:
                ax.set_xlim(*xlims)
            else:
                ax.set_xlim(parameters[:, 0].min(), parameters[:, 0].max())
                
            if ylims is not None:
                ax.set_ylim(*ylims)
            else:
                ax.set_ylim(parameters[:, 1].min(), parameters[:, 1].max())

            if xlims is not None and ylims is not None:
                # Set limits based on parameter_space_bounds
                x_low, x_high = xlims
                y_low, y_high = ylims
                ax.set_xlim(x_low, x_high)
                ax.set_ylim(y_low, y_high)
                
                # Set ticks based on the actual bounds
                ax.set_xticks(np.linspace(x_low, x_high, 5))
                ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(x_low, x_high, 5)])
                ax.set_yticks(np.linspace(y_low, y_high, 5))
                ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(y_low, y_high, 5)])
            else:
                # Use data-driven ticks
                x_range = parameters[:, 0]
                y_range = parameters[:, 1]
                ax.set_xticks(np.linspace(x_range.min(), x_range.max(), 5))
                ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(x_range.min(), x_range.max(), 5)])
                ax.set_yticks(np.linspace(y_range.min(), y_range.max(), 5))
                ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(y_range.min(), y_range.max(), 5)])

        elif param_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_box_aspect(aspect=None, zoom=0.85)
            scatter = ax.scatter(parameters[:, 0], parameters[:, 1], parameters[:, 2], c=np.round(coverage_probability*100, 2), 
                                cmap=cm.get_cmap(name='inferno'), vmin=vmin_vmax[0], vmax=vmin_vmax[1], alpha=1)
            cbar = fig.colorbar(scatter, format='%1.2f', location='top', orientation='horizontal', pad=-0.05)
            cbar.ax.xaxis.set_ticks(np.linspace(0, 100, num=6, dtype=int))
            cbar.ax.xaxis.set_ticklabels([str(label)+r"$\%$" for label in np.linspace(0, 100, num=6, dtype=int)], fontsize=30)
        else:
            raise ValueError("Impossible to plot coverage for a parameter with more than three dimensions")

        if custom_ax is None:
            # handle formatting within function passing custom ax. Used only for `param_dim == 2`
            cbar.set_label('Estimated Coverage', fontsize=40, labelpad=10)
            cbar.ax.tick_params(labelsize=30)
            simplefilter(action="ignore", category=UserWarning)
        
            ax.tick_params(axis='both', labelsize=20)
            if params_labels is None:
                ax.set_xlabel(r"$\theta^{{(1)}}$", fontsize=25, labelpad=3)
                ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=25, rotation=0, labelpad=3)
                if param_dim == 3:
                    ax.set_zlabel(r"$\theta^{{(3)}}$", fontsize=25, rotation=180, labelpad=3)
            else:
                ax.set_xlabel(params_labels[0], fontsize=40, labelpad=3)
                ax.set_ylabel(params_labels[1], fontsize=40, rotation=0, labelpad=10)
                if param_dim == 3:
                    ax.set_zlabel(params_labels[2], fontsize=25, rotation=180, labelpad=3)
            if xlims is not None:
                ax.set_xticks(np.linspace(xlims[0], xlims[1], 5))
                ax.set_xticklabels(np.linspace(xlims[0], xlims[1], 5), fontsize=30)
            if ylims is not None:
                ax.set_yticks(np.linspace(ylims[0], ylims[1], 5))
                ax.set_yticklabels(np.linspace(ylims[0], ylims[1], 5), fontsize=30)
            ax.tick_params(labelsize=30)

    if title is not None:
        ax.set_title(title, size=25, pad=20)
    
    if custom_ax is None:
        # save and show only if this is the primary figure. Used only for `param_dim == 2`
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()
    else:
        # avoid showing subfigure separately from main plot when calling plt.show()
        # plt.close()
        # return to attach global colorbar
        return contour_filled
    

def coverage_regions_plot(
    parameters: np.ndarray,
    confidence_level: float,
    coverage_probability: np.ndarray, 
    upper_proba: np.ndarray,
    lower_proba: np.ndarray,
    param_dim: int,
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    figsize: Tuple = (5, 5),
    save_fig_path: Optional[str] = None,
    custom_ax: Optional[Axes] = None
) -> None:
    undercoverage = upper_proba < confidence_level
    overcoverage = lower_proba > confidence_level
    coverage_regions = np.zeros(shape=(coverage_probability.shape[0], ))
    coverage_regions[undercoverage] = -1
    coverage_regions[overcoverage] = 1
    regions_colors = []
    for color_code in coverage_regions:
        if color_code == 0:
            regions_colors.append('green')
        elif color_code == -1:
            regions_colors.append('red')
        elif color_code == 1:
            regions_colors.append('yellow')
    
    if param_dim == 1:
        raise NotImplementedError
    elif param_dim == 2:
        fig = plt.figure(figsize=figsize)
        ax = custom_ax or plt.gca()
        ax.scatter(parameters[:, 0], parameters[:, 1], c=regions_colors, alpha=1)
    elif param_dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect(aspect=None, zoom=0.7)
        ax.scatter(parameters[:, 0], parameters[:, 1], parameters[:, 2], c=regions_colors, alpha=1)
    else:
        raise ValueError("Impossible to plot coverage for a parameter with more than three dimensions")

    if custom_ax is None:
        # handle formatting within function passing custom ax. save and show only if this is the primary figure
        # used only for `param_dim == 2`
        ax.tick_params(axis='both', labelsize=20)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='undercoverage', markerfacecolor='red', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='correct coverage', markerfacecolor='green', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='overcoverage', markerfacecolor='yellow', markersize=15)
        ]
        ax.legend(handles=legend_elements)
        if params_labels is None:
            ax.set_xlabel(r"$\theta^{{(1)}}$", fontsize=45, labelpad=30)
            ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=45, rotation=0, labelpad=30)
            if param_dim == 3:
                ax.set_zlabel(r"$\theta^{{(3)}}$", fontsize=45, rotation=180, labelpad=30)
        else:
            ax.set_xlabel(params_labels[0], fontsize=45, labelpad=30)
            ax.set_ylabel(params_labels[1], fontsize=45, rotation=0, labelpad=30)
            if param_dim == 3:
                ax.set_zlabel(params_labels[2], fontsize=45, rotation=180, labelpad=30)

        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()
    else:
        # avoid showing subfigure separately from main plot when calling plt.show()
        plt.close()


def coverage_pairplot(
    plot_type: str,
    parameters: np.ndarray,
    probabilities: Union[Dict[str, Dict[str, np.ndarray]], np.ndarray],
    confidence_level: float,
    diagnostics_estimator: Optional[Any] = None,  # used for partial dependence pairplots
    aggregate_fun: Optional[str] = None,  # used for partial dependence pairplots (mean/min/max over other params)
    vmin_vmax: Optional[Sequence[float]] = None,
    params_labels: Optional[Sequence] = None,
    plot_title: Optional[str] = None,
    figsize: Tuple = (15, 15),
    save_fig_path: Optional[str] = None,
    **kwargs
) -> None:
    assert plot_type in ['proba_partial_dependence_mean', 'proba_partial_dependence_min', 'coverage_regions_marginal']
    
    rows = cols = parameters.shape[1]  # param dim
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    
    for row in range(rows):
        for col in range(cols):
            # plots
            if col <= row:
                ax[row, col].axis('off')
            else:
                if plot_type == 'proba_partial_dependence_mean':
                    heatmap = coverage_probability_plot(
                        parameters=parameters[:, [col, row]],  # swap order to have 'row' parameter on y axis
                        coverage_probability=probabilities[f'{row}{col}']['mean_proba'],  # dictionary
                        confidence_level=confidence_level,
                        param_dim=2,  # pairplot
                        vmin_vmax=vmin_vmax,
                        custom_ax=ax[row, col],
                        **kwargs
                    )
                elif plot_type == 'proba_partial_dependence_min':
                    raise NotImplementedError
                elif plot_type == 'coverage_regions_marginal':
                    raise NotImplementedError  # TODO: need to double check
                    coverage_regions_plot(
                        parameters=parameters[:, [col, row]],  # swap order to have 'row' parameter on y axis
                        confidence_level=confidence_level,
                        coverage_probability=probabilities[f'{row}{col}']['mean_proba'],
                        upper_proba=probabilities[f'{row}{col}']['upper_proba'],
                        lower_proba=probabilities[f'{row}{col}']['lower_proba'],
                        param_dim=2,  # pairplot
                        custom_ax=ax[row, col],
                        **kwargs
                    )
                else:
                    raise NotImplementedError

                # labels
                if col == row+1:
                    ax[row, col].set_xlabel(r'$\theta_{}$'.format(col) if params_labels is None else params_labels[col], fontsize=20)
                    ax[row, col].tick_params(axis='x', labelsize=12)
                    ax[row, col].set_ylabel(r'$\theta_{}$'.format(row) if params_labels is None else params_labels[row], fontsize=20, labelpad=3)
                    ax[row, col].tick_params(axis='y', labelsize=12)
                else:
                    ax[row, col].tick_params(labelleft=False, labelbottom=False)
    
    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.3, 0.02, 0.5])
    cbar = fig.colorbar(heatmap, format='%1.2f', cax=cbar_ax)
    cbar.ax.plot(0.5, confidence_level*100, 'w*', markersize=7)
    # colorbar formatting
    cbar.ax.yaxis.set_ticks(np.round(np.linspace(vmin_vmax[0], vmin_vmax[1], num=5), 1))
    cbar.ax.yaxis.set_ticklabels([str(label)+"%" for label in np.round(np.linspace(vmin_vmax[0], vmin_vmax[1], num=5), 1)])
    cbar.set_label('Estimated Coverage', fontsize=25, labelpad=10)
    try:
        cbar.ax.tick_params(labelsize=15)
    except TypeError:
        print('type error cbar.ax.tick_params(labelsize=15)', flush=True)
    simplefilter(action="ignore", category=UserWarning)
    
    if plot_title is not None:
        fig.suptitle(t=plot_title, fontsize=30)
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()


def coverage_barplot(
    confidence_level: float,
    mean_proba: np.ndarray, 
    upper_proba: Optional[np.ndarray] = None,
    lower_proba: Optional[np.ndarray] = None,
    save_fig_path: Optional[str] = None,
    tol: Optional[float] = None,
    figsize: Tuple = (5, 5)
) -> None:
    sns.set_style("whitegrid")
    if (upper_proba is None) and (lower_proba is None):
        assert tol is not None, "Must specify a coverage tolerance if no upper/lower bounds are provided"
        proportion_undercoverage = np.sum(mean_proba < confidence_level - tol) / len(mean_proba)
        proportion_overcoverage = np.sum(mean_proba > confidence_level + tol) / len(mean_proba)
    else:
        proportion_undercoverage = np.sum(upper_proba < confidence_level) / len(upper_proba)
        proportion_overcoverage = np.sum(lower_proba > confidence_level) / len(lower_proba)
    
    df_barplot = pd.DataFrame({
        "x": [" "]*3,  # hack to have three adjacent coloured bars
        "coverage":  ["Undercoverage", "Correct Coverage", "Overcoverage"],
        "proportion": [proportion_undercoverage, 1-(proportion_overcoverage+proportion_undercoverage), proportion_overcoverage]   
    })
    _, ax = plt.subplots(1, 1, figsize=figsize)
    plot = sns.barplot(data=df_barplot, x="x", y="proportion", hue="coverage", errorbar=None, ax=ax, 
                       palette={'Undercoverage': 'red', "Correct Coverage": 'green', "Overcoverage": 'yellow'})

    plot.set(xlabel=None, xticks=[])
    ax.set_ylabel("Proportion", fontsize=15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([r'0%', r'20%', r'40%', r'60%', r'80%', r'100%'])
    ax.tick_params(labelsize=25)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(0.74, 0))

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()


def coverage_boxplot(
    probabilities: Sequence[np.ndarray],
    labels: Sequence[str],
    confidence_level: float,
    whiskers_loc: Union[Tuple[float, float], float] = 1.5,
    plot_fliers: bool = True,
    ylim: Optional[Sequence[float]] = None,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (5, 5)
) -> None:
    plt.figure(figsize=figsize)

    plt.boxplot(x=probabilities, notch=False, labels=labels, whis=whiskers_loc, sym=None if plot_fliers else '')
    plt.axhline(y=confidence_level, label=f'Nominal Coverage Level: {round(confidence_level*100, 1)}%', linestyle='--', linewidth=1, color='red')
    whiskers_vals = (f'0.25-{round(whiskers_loc, 1)}IQR', f'0.75+{round(whiskers_loc, 1)}IQR') if isinstance(whiskers_loc, float) else (whiskers_loc[0]/100, whiskers_loc[1]/100)
    plt.plot([], [], ' ', label=f"\nBox: (0.25, 0.5, 0.75) \nWhiskers: {whiskers_vals}")  # just to add explanation on boxplots

    sns.set_style('whitegrid')
    plt.title('Joint Coverage Probability', fontsize=20)
    plt.legend(loc='lower right')

    if ylim:
        plt.ylim(*ylim)
        yticks = np.arange(start=ylim[0], stop=ylim[1]+0.1, step=0.1)
        plt.yticks(ticks=yticks, labels=[f'{round(i*100, 0)}%' for i in yticks])
    else:
        yticks, _ = plt.yticks()
        plt.yticks(ticks=yticks, labels=[f'{round(i*100, 0)}%' for i in yticks])
    plt.tick_params(axis='both', labelsize=15)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()

def coverage_nominal_actual_boxplot(
        probabilities: Sequence[np.ndarray],
        confidence_levels: np.ndarray,
        whiskers_loc: Union[Tuple[float, float], float] = 1.5,
        plot_fliers: bool = True,
        ylim: Optional[Sequence[float]] = None,
        save_fig_path: Optional[str] = None,
        figsize: Tuple = (5, 5)
) -> None:
    plt.figure(figsize=figsize)

    positions = np.arange(1, len(confidence_levels) + 1)

    plt.boxplot(
        x=probabilities,
        positions=positions,
        notch=False,
        whis=whiskers_loc,
        sym=None if plot_fliers else '',
        widths=0.4,
        patch_artist=True
    )

    ref = np.linspace(confidence_levels.min(), confidence_levels.max(), 200)
    plt.plot(
        np.interp(ref, confidence_levels, positions),
        ref,
        linestyle='--', linewidth=1, color='red', label='Nominal = Actual'
    )
    plt.scatter(positions, confidence_levels, color='red', s=20, zorder=5)
    
    
    whiskers_vals = (f'0.25-{round(whiskers_loc, 1)}IQR', f'0.75+{round(whiskers_loc, 1)}IQR') if isinstance(whiskers_loc, float) else (whiskers_loc[0]/100, whiskers_loc[1]/100)
    plt.plot([], [], ' ', label=f"\nBox: (0.25, 0.5, 0.75) \nWhiskers: {whiskers_vals}")

    sns.set_style('whitegrid')
    plt.title('Nominal vs. Actual Coverage', fontsize=20)

    plt.xticks(
        ticks=positions,
        labels=[f'{round(cl * 100, 1)}%' for cl in confidence_levels],
        fontsize=11,
        rotation=45 if len(confidence_levels) > 6 else 0,
    )
    plt.xlabel('Nominal Coverage', fontsize=14)
    plt.ylabel('Actual Coverage', fontsize=14)

    if ylim:
        plt.ylim(*ylim)
        yticks = np.arange(start=ylim[0], stop=ylim[1] + 0.01, step=0.05)
    else:
        plt.ylim(0, 1.05)
        yticks = np.arange(0, 1.05, 0.1)

    plt.yticks(
        ticks=yticks,
        labels=[f'{round(v * 100, 0):.0f}%' for v in yticks],
        fontsize=11,
    )

    plt.tick_params(axis='both', labelsize=11)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()

class PinGreenNormalize(mcolors.Normalize):
    def __init__(self, vmin=0, vmax=100, vcenter=40, green_index=0.35, clip=False):
        super().__init__(vmin, vmax, clip)
        self.vcenter = vcenter
        self.green_index = green_index

    def __call__(self, value, clip=None):
        x = np.ma.masked_array(value, np.isnan(value))
        result = np.ma.empty(x.shape, dtype=float)

        # below the confidence level
        idx_below = x <= self.vcenter
        if self.vmin < self.vcenter:
            result[idx_below] = (x[idx_below] - self.vmin) / (self.vcenter - self.vmin) * self.green_index
        else:
            # vmin == vcenter
            result[idx_below] = 0.0
        
        # above the confidence level
        idx_above = x > self.vcenter
        if self.vcenter < self.vmax:
            result[idx_above] = self.green_index + (
                (x[idx_above] - self.vcenter) / (self.vmax - self.vcenter)
                * (1.0 - self.green_index)
            )
        else:
            # vcenter == vmax
            result[idx_above] = 1.0
        
        return result

def create_jetr_cmap(confidence_level=40):
    """
    Returns a reversed jet colormap plus a custom normalization that pins 
    'confidence_level' to a bright green region.
    """
    green_index = 0.47  # 0.35 in jet_r is roughly bright green
    
    cmap = plt.get_cmap('jet_r')
    norm = PinGreenNormalize(vmin=0, vmax=100, vcenter=confidence_level, green_index=green_index)
    return cmap, norm