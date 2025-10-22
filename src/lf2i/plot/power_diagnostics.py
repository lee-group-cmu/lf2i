from typing import Optional, Tuple, Union, List, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import seaborn as sns


def set_size_plot(
    parameters: np.ndarray,
    set_sizes: np.ndarray,
    param_dim: int,
    figsize: Tuple = (10, 8),
    xlims: Optional[Tuple[float]] = None,
    ylims: Optional[Tuple[float]] = None,
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    vmin_vmax: Optional[Union[Tuple, List]] = None,
    custom_ax: Optional[Axes] = None,
    show_text: bool = False,
    title: Optional[str] = None,
    save_fig_path: Optional[str] = None
) -> None:
    """Plot average confidence set sizes across parameter space."""
    
    if param_dim == 1:
        df_plot = pd.DataFrame({
            "parameters": parameters.reshape(-1,),
            "set_size": set_sizes.reshape(-1,)
        }).sort_values(by="parameters")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(df_plot.parameters, df_plot.set_size, color='steelblue', linewidth=2)
        
        if params_labels is None:
            ax.set_xlabel(r"$\theta$", fontsize=25)
        else:
            ax.set_xlabel(params_labels[0], fontsize=25)
        ax.set_ylabel("Average Set Size", fontsize=25)
        if ylims is not None:
            ax.set_ylim(*ylims)
        ax.tick_params(axis='both', labelsize=20)
        
    elif param_dim == 2:
        if custom_ax is None:
            fig = plt.figure(figsize=figsize)
        ax = custom_ax or plt.gca()

        # Create bins and heatmap
        x_bins = np.histogram_bin_edges(parameters[:, 0], bins='auto')
        y_bins = np.histogram_bin_edges(parameters[:, 1], bins='auto')
        
        binned_sum_sizes, xedges, yedges = np.histogram2d(
            parameters[:, 0], parameters[:, 1], 
            bins=[x_bins, y_bins], 
            weights=set_sizes
        )
        bin_counts, _, _ = np.histogram2d(
            parameters[:, 0], parameters[:, 1], 
            bins=[x_bins, y_bins]
        )
        heatmap_values = binned_sum_sizes / bin_counts

        # Determine colormap range
        if vmin_vmax is None:
            vmin_vmax = (np.nanmin(heatmap_values), np.nanmax(heatmap_values))
        
        levels = np.linspace(vmin_vmax[0], vmin_vmax[1], num=15)
        
        # Plot
        contour_filled = ax.contourf(
            xedges[:-1], yedges[:-1], heatmap_values.T[::-1], 
            levels=levels, cmap='viridis', extend='both'
        )
        contour_lines = ax.contour(
            xedges[:-1], yedges[:-1], heatmap_values.T[::-1], 
            levels=levels, colors="white", linewidths=0.8
        )
        clabels = ax.clabel(
            contour_lines, levels[::2], 
            inline=True, fontsize=15, fmt="%1.2f"
        )
        for txt in clabels:
            txt.set_color("black")
            txt.set_bbox(dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.1"))

        if show_text:
            for y_index in range(len(yedges)-1):
                for x_index in range(len(xedges)-1):
                    label = heatmap_values.T[::-1, :][y_index, x_index]
                    if not np.isnan(label):
                        ax.text(
                            xedges[x_index] + 0.5*(xedges[1]-xedges[0]), 
                            yedges[y_index] + 0.5*(yedges[1]-yedges[0]), 
                            f'{label:.2f}', 
                            color='black', ha='center', va='center', fontsize=8
                        )

        if custom_ax is None:
            cbar = fig.colorbar(contour_filled, format='%1.2f')
            cbar.set_label('Average Set Size', fontsize=25, labelpad=10)
            cbar.ax.tick_params(labelsize=20)
        
        ax.tick_params(axis='both', labelsize=20)
        if params_labels is None:
            ax.set_xlabel(r"$\theta^{{(1)}}$", fontsize=25, labelpad=3)
            ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=25, labelpad=10, rotation=0)
        else:
            ax.set_xlabel(params_labels[0], fontsize=25, labelpad=3)
            ax.set_ylabel(params_labels[1], fontsize=25, labelpad=10, rotation=0)
        
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)

        if xlims is not None and ylims is not None:
            # Set limits based on parameter_space_bounds
            x_low, x_high = xlims
            y_low, y_high = ylims
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
            
    elif param_dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect(aspect=None, zoom=0.85)
        
        if vmin_vmax is None:
            vmin_vmax = (np.min(set_sizes), np.max(set_sizes))
        
        scatter = ax.scatter(
            parameters[:, 0], parameters[:, 1], parameters[:, 2], 
            c=set_sizes, cmap='viridis', 
            vmin=vmin_vmax[0], vmax=vmin_vmax[1], alpha=0.8
        )
        
        cbar = fig.colorbar(scatter, format='%1.2f', location='top', 
                           orientation='horizontal', pad=-0.05)
        cbar.set_label('Average Set Size', fontsize=25, labelpad=10)
        cbar.ax.tick_params(labelsize=20)
        
        ax.tick_params(axis='both', labelsize=20)
        if params_labels is None:
            ax.set_xlabel(r"$\theta^{{(1)}}$", fontsize=25, labelpad=3)
            ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=25, labelpad=3)
            ax.set_zlabel(r"$\theta^{{(3)}}$", fontsize=25, rotation=180, labelpad=3)
        else:
            ax.set_xlabel(params_labels[0], fontsize=25, labelpad=3)
            ax.set_ylabel(params_labels[1], fontsize=25, labelpad=3)
            ax.set_zlabel(params_labels[2], fontsize=25, rotation=180, labelpad=3)
    else:
        raise ValueError("Cannot plot for parameter dimension > 3")

    if title is not None:
        ax.set_title(title, size=25, pad=20)
    
    if custom_ax is None:
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        return contour_filled if param_dim == 2 else None


def set_size_boxplot(
    set_sizes: Sequence[np.ndarray],
    labels: Sequence[str],
    whiskers_loc: Union[Tuple[float, float], float] = 1.5,
    plot_fliers: bool = True,
    ylim: Optional[Sequence[float]] = None,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (8, 6)
) -> None:
    """Boxplot comparing set sizes across different methods."""
    
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    plt.boxplot(
        x=set_sizes, notch=False, labels=labels, 
        whis=whiskers_loc, sym=None if plot_fliers else ''
    )
    
    whiskers_vals = (
        f'0.25-{round(whiskers_loc, 1)}IQR', 
        f'0.75+{round(whiskers_loc, 1)}IQR'
    ) if isinstance(whiskers_loc, float) else (whiskers_loc[0]/100, whiskers_loc[1]/100)
    
    plt.plot([], [], ' ', label=f"Box: (0.25, 0.5, 0.75)\nWhiskers: {whiskers_vals}")

    plt.title('Confidence Set Sizes', fontsize=20)
    plt.ylabel('Set Size', fontsize=18)
    plt.legend(loc='upper right')
    
    if ylim:
        plt.ylim(*ylim)
    
    plt.tick_params(axis='both', labelsize=15)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
