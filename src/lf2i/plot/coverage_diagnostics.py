from typing import Optional, Tuple, Union, Dict, List, Sequence, Any
from warnings import simplefilter

import numpy as np
from rpy2.robjects.vectors import ListVector
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes._axes import Axes
import seaborn as sns

from lf2i.diagnostics.diagnostics import predict_r_estimator


def coverage_probability_plot(
    parameters: np.ndarray,
    coverage_probability: np.ndarray,
    confidence_level: float,
    param_dim: int,
    upper_proba: Optional[np.ndarray] = None,
    lower_proba: Optional[np.ndarray] = None,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (10, 15),
    ylims: Tuple = (0, 1),
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    vmin_vmax: Optional[Union[Tuple, List]] = None,
    custom_ax: Optional[Axes] = None,  # if passing custom ax for pairplot
    show_text: bool = False,
    show_undercoverage: bool = False
) -> None:
    if param_dim == 1:
        df_plot = pd.DataFrame({
            "parameters": parameters.reshape(-1,),
            "mean_proba": coverage_probability.reshape(-1,),
            "lower_proba": lower_proba.reshape(-1,),
            "upper_proba": upper_proba.reshape(-1,)
        }).sort_values(by="parameters")

        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(df_plot.parameters, df_plot.mean_proba, color='crimson', label='Estimated Coverage')
        ax.plot(df_plot.parameters, df_plot.lower_proba, color='crimson')
        ax.plot(df_plot.parameters, df_plot.upper_proba, color='crimson')
        ax.fill_between(x=df_plot.parameters, y1=df_plot.lower_proba, y2=df_plot.upper_proba, alpha=0.2, color='crimson')
        ax.axhline(y=confidence_level, color='black', linestyle="--", linewidth=3, 
                    label=f"Nominal coverage = {round(100 * confidence_level, 1)} %", zorder=10)
        
        if params_labels is None:
            ax.set_xlabel(r"$\theta$", fontsize=45)
        else:
            ax.set_xlabel(params_labels[0], fontsize=45)
        ax.set_ylabel("Coverage", fontsize=45)
        ax.set_ylim(*ylims)
        ax.legend()
    else:
        vmin_vmax = vmin_vmax or (0, 100)
        if param_dim == 2:
            fig = plt.figure(figsize=figsize)
            ax = custom_ax or plt.gca()

            x_bins = np.histogram_bin_edges(parameters[:, 0], bins='auto')
            y_bins = np.histogram_bin_edges(parameters[:, 1], bins='auto')
            binned_sum_proba, xedges, yedges = np.histogram2d(parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins], weights=np.round(coverage_probability*100, 2))
            bin_counts, xedges, yedges = np.histogram2d(parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins]) 
            heatmap_values = binned_sum_proba/bin_counts
            heatmap = ax.imshow(heatmap_values.T, cmap='inferno', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=vmin_vmax[0], vmax=vmin_vmax[1])
            if show_text:
                # Add the text
                jump_x = (xedges[-1] - xedges[0]) / (2.0 * len(x_bins))
                jump_y = (yedges[-1] - yedges[0]) / (2.0 * len(y_bins))
                x_positions = np.linspace(start=xedges[0], stop=xedges[-1], num=len(x_bins)-1, endpoint=False)
                y_positions = np.linspace(start=yedges[0], stop=yedges[-1], num=len(y_bins)-1, endpoint=False)
                for y_index, y in enumerate(y_positions):
                    for x_index, x in enumerate(x_positions):
                        label = heatmap_values.T[::-1, :][y_index, x_index]
                        text_x = x + jump_x
                        text_y = y + jump_y
                        ax.text(text_x, text_y, f'{label:.1f}', color='black', ha='center', va='center', fontsize=7)
            if show_undercoverage:
                binned_sum_upper_proba, _, _ = np.histogram2d(parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins], weights=np.round(upper_proba*100, 2))
                bin_counts_upper, _, _ = np.histogram2d(parameters[:, 0], parameters[:, 1], bins=[x_bins, y_bins]) 
                heatmap_upper_values = binned_sum_upper_proba/bin_counts_upper
                jump_x = (xedges[-1] - xedges[0]) / (2.0 * len(x_bins))
                jump_y = (yedges[-1] - yedges[0]) / (2.0 * len(y_bins))
                x_positions = np.linspace(start=xedges[0], stop=xedges[-1], num=len(x_bins)-1, endpoint=False)
                y_positions = np.linspace(start=yedges[0], stop=yedges[-1], num=len(y_bins)-1, endpoint=False)
                for y_index, y in enumerate(y_positions):
                    for x_index, x in enumerate(x_positions):
                        if heatmap_upper_values.T[::-1, :][y_index, x_index] < confidence_level*100:
                            plt.scatter(x + jump_x, y + jump_y, marker='X', color='red', s=200)
            if custom_ax is None:
                # use one commmon colorbar on main figure
                cbar = fig.colorbar(heatmap, format='%1.2f')
                cbar.ax.yaxis.set_ticks(np.round(np.linspace(vmin_vmax[0], vmin_vmax[1], num=5), 1))
                cbar.ax.yaxis.set_ticklabels([str(label)+"%" for label in np.round(np.linspace(vmin_vmax[0], vmin_vmax[1], num=5), 1)])
        elif param_dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
            ax.set_box_aspect(aspect=None, zoom=0.85)
            scatter = ax.scatter(parameters[:, 0], parameters[:, 1], parameters[:, 2], c=np.round(coverage_probability*100, 2), 
                                cmap=cm.get_cmap(name='inferno'), vmin=vmin_vmax[0], vmax=vmin_vmax[1], alpha=1)
            cbar = fig.colorbar(scatter, format='%1.2f', location='top', orientation='horizontal', pad=-0.05)
            cbar.ax.xaxis.set_ticks(np.linspace(0, 100, num=11, dtype=int))
            cbar.ax.xaxis.set_ticklabels([str(label)+"%" for label in np.linspace(0, 100, num=11, dtype=int)])
        else:
            raise ValueError("Impossible to plot coverage for a parameter with more than three dimensions")

        if custom_ax is None:
            # handle formatting within function passing custom ax. Used only for `param_dim == 2`
            cbar.set_label('Estimated Coverage', fontsize=30, labelpad=10)
            cbar.ax.tick_params(labelsize=15)
            simplefilter(action="ignore", category=UserWarning)
        
            ax.tick_params(axis='both', labelsize=20)
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
    
    if custom_ax is None:
        # save and show only if this is the primary figure. Used only for `param_dim == 2`
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()
    else:
        # avoid showing subfigure separately from main plot when calling plt.show()
        plt.close()
        # return to attach global colorbar
        return heatmap
    

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
                    if (diagnostics_estimator) and isinstance(diagnostics_estimator, ListVector):  # rpy2 ListVector; support only gam for now. TODO: other prob clfs
                        # for partial dependence plots
                        assert aggregate_fun is not None
                        other_params_idx = list(set(range(rows)).difference(set([row, col])))
                        if aggregate_fun == 'mean':
                            which_other_params = parameters[np.argmin(np.abs(probabilities - np.mean(probabilities))), other_params_idx]
                        elif aggregate_fun == 'min':
                            which_other_params = parameters[np.argmin(probabilities), other_params_idx]
                        elif aggregate_fun == 'max':
                            which_other_params = parameters[np.argmax(probabilities), other_params_idx]
                        else:
                            raise NotImplementedError
                        assert which_other_params.shape == (len(other_params_idx), )
                        partial_dependence_params = np.copy(parameters)
                        partial_dependence_params[:, other_params_idx] = which_other_params
                        partial_dependence_probabilities, _, _ = predict_r_estimator(
                            fitted_estimator=diagnostics_estimator,
                            parameters=partial_dependence_params,
                            param_dim=rows,
                            n_sigma=2  # unused
                        )
                
                        heatmap = coverage_probability_plot(
                            parameters=parameters[:, [col, row]],  # swap order to have 'row' parameter on y axis
                            coverage_probability=partial_dependence_probabilities,  # array
                            confidence_level=confidence_level,
                            param_dim=2,  # pairplot
                            vmin_vmax=vmin_vmax,
                            custom_ax=ax[row, col],
                            **kwargs
                        )
                    else:
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
