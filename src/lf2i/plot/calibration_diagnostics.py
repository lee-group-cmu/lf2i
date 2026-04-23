from typing import Optional, Tuple, Union, Dict, List, Sequence, Any
from warnings import simplefilter

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

# ---------------------------------------------------------------------------
# Calibration-quality score plots (MSE / CRPS / pinball loss)
# ---------------------------------------------------------------------------

_DEFAULT_SCORE_LABELS: Dict[str, str] = {
    'mse': 'MSE',
    'crps': 'CRPS (T-scale)',
}


def _score_label(key: str) -> str:
    if key in _DEFAULT_SCORE_LABELS:
        return _DEFAULT_SCORE_LABELS[key]
    if key.startswith('pinball_'):
        try:
            return f'Pinball α={float(key[8:]):.2f}'
        except ValueError:
            pass
    return key


def _bin_edges_from_unique(vals: np.ndarray) -> np.ndarray:
    """Bin edges that place each unique value at the centre of its own bin."""
    vals = np.sort(np.unique(vals))
    if len(vals) == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    d = np.diff(vals)
    return np.concatenate([[vals[0] - d[0] / 2], vals[:-1] + d / 2, [vals[-1] + d[-1] / 2]])


def calibration_score_plot(
    parameters: np.ndarray,
    scores: np.ndarray,
    score_label: str,
    param_dim: int,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (6, 5),
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    vmax: Optional[float] = None,
    custom_ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Optional[Any]:
    """
    Heatmap of a single per-theta calibration score (e.g. MSE, CRPS, pinball
    loss) returned by monte_carlo_pvalue_diagnostics.

    Green = 0 (best), red = vmax (worst).  Mirrors the interface of
    coverage_probability_plot: when custom_ax is None the function owns the
    figure, adds a colorbar, and calls plt.show(); when custom_ax is supplied
    it returns the pcolormesh artist so the caller can attach a shared colorbar.
    """
    own_fig = custom_ax is None

    if param_dim == 1:
        if own_fig:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = custom_ax
        sorted_idx = np.argsort(parameters.reshape(-1))
        ax.plot(parameters.reshape(-1)[sorted_idx], scores[sorted_idx], color='crimson')
        ax.set_xlabel(
            params_labels[0] if params_labels else r'$\theta$', fontsize=20
        )
        ax.set_ylabel(score_label, fontsize=20)
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)
        mesh = None

    elif param_dim == 2:
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = custom_ax
            fig = ax.get_figure()

        x_unique = np.unique(parameters[:, 0])
        y_unique = np.unique(parameters[:, 1])
        x_edges = _bin_edges_from_unique(x_unique)
        y_edges = _bin_edges_from_unique(y_unique)

        binned_sum, _, _ = np.histogram2d(
            parameters[:, 0], parameters[:, 1],
            bins=[x_edges, y_edges], weights=scores,
        )
        bin_counts, _, _ = np.histogram2d(
            parameters[:, 0], parameters[:, 1], bins=[x_edges, y_edges]
        )
        with np.errstate(invalid='ignore'):
            Z = binned_sum / bin_counts  # NaN only if a grid point has no samples

        X, Y = np.meshgrid(x_unique, y_unique)

        _vmax = vmax if vmax is not None else float(np.nanmax(Z))
        mesh = ax.pcolormesh(X, Y, Z.T, cmap='RdYlGn_r', vmin=0, vmax=_vmax, shading='auto')

        x_low  = xlims[0] if xlims else float(parameters[:, 0].min())
        x_high = xlims[1] if xlims else float(parameters[:, 0].max())
        y_low  = ylims[0] if ylims else float(parameters[:, 1].min())
        y_high = ylims[1] if ylims else float(parameters[:, 1].max())
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.set_xticks(np.linspace(x_low, x_high, 5))
        ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(x_low, x_high, 5)])
        ax.set_yticks(np.linspace(y_low, y_high, 5))
        ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(y_low, y_high, 5)])
        ax.set_xlabel(
            params_labels[0] if params_labels else r'$\theta^{(1)}$',
            fontsize=20, labelpad=3,
        )
        ax.set_ylabel(
            params_labels[1] if params_labels else r'$\theta^{(2)}$',
            fontsize=20, labelpad=5, rotation=0,
        )
        ax.tick_params(axis='both', labelsize=12)

        if own_fig:
            cbar = fig.colorbar(mesh)
            cbar.set_label(score_label, fontsize=16, labelpad=8)
            cbar.ax.tick_params(labelsize=12)

    else:
        raise ValueError(
            f'calibration_score_plot: param_dim={param_dim} not supported (max 2)'
        )

    if title is not None:
        ax.set_title(title, fontsize=16, pad=10)

    if own_fig:
        simplefilter(action='ignore', category=UserWarning)
        plt.tight_layout()
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()
    else:
        return mesh


def calibration_score_panel(
    evaluation_grid: np.ndarray,
    estimation_errors: Dict[str, np.ndarray],
    param_dim: int,
    score_labels: Optional[Dict[str, str]] = None,
    save_fig_path: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Side-by-side heatmap panels for every score key in estimation_errors.
    Direct counterpart of coverage_pairplot for calibration-quality diagnostics.

    Intended to be called with the output of monte_carlo_pvalue_diagnostics::

        grid, errors = lf2i_obj.monte_carlo_pvalue_diagnostics(...)
        calibration_score_panel(grid, errors, param_dim=2)

    Each panel gets an independent colorbar anchored at zero (green, best) up
    to the per-panel maximum (red, worst).  Use score_labels to override the
    default display names for any key.
    """
    keys = list(estimation_errors.keys())
    n_panels = len(keys)
    _label_overrides = score_labels or {}

    if figsize is None:
        figsize = (5 * n_panels, 4)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        label = _label_overrides.get(key, _score_label(key))
        mesh = calibration_score_plot(
            parameters=evaluation_grid,
            scores=estimation_errors[key],
            score_label=label,
            param_dim=param_dim,
            xlims=xlims,
            ylims=ylims,
            params_labels=params_labels,
            custom_ax=ax,
        )
        if mesh is not None:
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(label, fontsize=13, labelpad=6)
            cbar.ax.tick_params(labelsize=11)

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    simplefilter(action='ignore', category=UserWarning)
    fig.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
