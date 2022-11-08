from typing import Optional, Tuple
from warnings import simplefilter

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns


def coverage_regions_plot(
    parameters: np.ndarray,
    mean_proba: np.ndarray, 
    upper_proba: Optional[np.ndarray],
    lower_proba: Optional[np.ndarray],
    confidence_level: float,
    param_dim: int,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (10, 15),
    ylims: Tuple = (0, 1)
) -> None:  
    if param_dim == 1:
        df_plot = pd.DataFrame({
            "parameters": parameters.reshape(-1,),
            "mean_proba": mean_proba.reshape(-1,),
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
        
        ax.set_xlabel(r"$\theta$", fontsize=45)
        ax.set_ylabel("Coverage", fontsize=45)
        ax.set_ylim(*ylims)
        ax.legend()
                    
    elif param_dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        scatter = plt.scatter(parameters[:, 0], parameters[:, 1], c=np.round(mean_proba*100, 2), 
                                cmap=cm.get_cmap(name='inferno'), vmin=0, vmax=100, alpha=1)
        cbar = fig.colorbar(scatter, format='%1.2f')
        
        cbar.set_label('Estimated Coverage', fontsize=30)
        cbar.ax.tick_params(labelsize=15)
        simplefilter(action="ignore", category=UserWarning)
        cbar.ax.yaxis.set_ticks(np.linspace(0, 100, num=11, dtype=int))
        cbar.ax.yaxis.set_ticklabels([str(label)+"%" for label in np.linspace(0, 100, num=11, dtype=int)])
        ax.set_xlabel(r"$\theta^{{(1)}}$", fontsize=45)
        ax.set_ylabel(r"$\theta^{{(2)}}$", fontsize=45, rotation=0, labelpad=40)
        plt.tick_params(axis='both', labelsize=20)
        
    elif param_dim == 3:
        raise NotImplementedError
    else:
        raise ValueError("Impossible to plot coverage for a parameter with more than three dimensions")
    
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
    plot = sns.barplot(data=df_barplot, x="x", y="proportion", hue="coverage", errorbar=None, ax=ax)

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
