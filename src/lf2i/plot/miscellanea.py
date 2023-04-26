from typing import Optional, Tuple, List, Union, Sequence
from warnings import simplefilter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from descartes.patch import Polygon


def hist_pairplot(
    data: np.ndarray,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (15, 15),
    labels: Optional[Union[List, Tuple]] = None
) -> None:
    rows = cols = data.shape[1]  # data dim
    _, ax = plt.subplots(rows, cols, figsize=figsize)
    
    for row in range(rows):
        for col in range(cols):
            # plots
            if col > row:
                ax[row, col].axis('off')
            elif col == row:
                ax[row, col].hist(x=data[:, row], bins=100, density=True)
            else:
                ax[row, col].hist2d(x=data[:, col], y=data[:, row], density=False, cmin=1, bins=100)

            # labels
            if row == rows-1:
                ax[row, col].set_xlabel(r'$x_{}$'.format(col) if labels is None else labels[col], fontsize=25)
                ax[row, col].tick_params(axis='x', labelsize=12)
            else:
                ax[row, col].tick_params(axis='x', labelbottom=False)
            if col == 0:
                ax[row, col].set_ylabel(r'$x_{}$'.format(row) if labels is None else labels[row], fontsize=25, rotation=0, labelpad=20)
                ax[row, col].tick_params(axis='y', labelsize=12)
            else:
                if row != col:  # axis is different for 1D hist
                    ax[row, col].tick_params(axis='y', labelleft=False)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()


def check_probs_classifier(
    y_true: np.ndarray,
    y_pred_positive_proba: np.ndarray,
    parameters: np.ndarray,
    confidence_level: float,
    param_dim: int,
    params_labels: Optional[Sequence] = None,
    figsize: Tuple = (15, 15),
    save_fig_path: Optional[str] = None
) -> None:
    
    # [i:i+1] is just to have it as array, otherwise indexing returns float and log_loss expects array
    log_loss_single_sample = np.array([log_loss(y_true=y_true[i:i+1], y_pred=y_pred_positive_proba[i:i+1]) for i in range(len(y_true))])
    vmin, vmax = np.min(log_loss_single_sample), np.max(log_loss_single_sample)

    if param_dim == 1:
        raise NotImplementedError
    elif param_dim == 2:
        raise NotImplementedError
    else:
        rows = cols = parameters.shape[1]  # param dim
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        
        for row in range(rows):
            for col in range(cols):
                # plots
                if col <= row:
                    ax[row, col].axis('off')
                else:
                    x_bins = np.histogram_bin_edges(parameters[:, col], bins='auto')  # swap order to have 'row' parameter on y axis
                    y_bins = np.histogram_bin_edges(parameters[:, row], bins='auto')
                    binned_sum_proba, xedges, yedges = np.histogram2d(parameters[:, col], parameters[:, row], bins=[x_bins, y_bins], weights=log_loss_single_sample)
                    bin_counts, xedges, yedges = np.histogram2d(parameters[:, col], parameters[:, row], bins=[x_bins, y_bins]) 
                    heatmap_values = binned_sum_proba/bin_counts
                    heatmap = ax[row, col].imshow(heatmap_values.T, cmap='inferno', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=vmin, vmax=vmax)

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
        cbar.ax.yaxis.set_ticks(np.round(np.linspace(vmin, vmax, num=5), 1))
        cbar.ax.yaxis.set_ticklabels([str(label)+"%" for label in np.round(np.linspace(vmin, vmax, num=5), 1)])
        cbar.set_label('Estimated Coverage', fontsize=25, labelpad=10)
        cbar.ax.tick_params(labelsize=15)
        simplefilter(action="ignore", category=UserWarning)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()


def PolygonPathFixed(polygon):
    """FIXED: shapely changed how it handles Polygon exteriors and descartes hasn't been updated.
    Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    if hasattr(polygon, 'geom_type'):  # Shapely
        ptype = polygon.geom_type
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    else:  # GeoJSON
        polygon = getattr(polygon, '__geo_interface__', polygon)
        ptype = polygon["type"]
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon['coordinates']]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    vertices = np.concatenate([
        np.concatenate([np.asarray(t.exterior.coords)[:, :2]] +
                    [np.asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])
    codes = np.concatenate([
        np.concatenate([coding(t.exterior.coords)] +
                    [coding(r) for r in t.interiors]) for t in polygon])

    return Path(vertices, codes)


def PolygonPatchFixed(polygon, **kwargs):
    """FIXED: shapely changed how it handles Polygon exteriors and descartes hasn't been updated.
    Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPathFixed(polygon), **kwargs)