from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.axes._axes import Axes

from lf2i.utils.miscellanea import to_np_if_torch


def plot_point_estimate_population(
    true_parameters: np.ndarray,
    point_estimates: np.ndarray,
    param_names: Optional[Sequence[str]] = None,
    true_color: str = 'grey',
    estimate_color: str = 'steelblue',
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_fig_path: Optional[str] = None,
) -> None:
    """Overlay true parameters and focal point estimates in parameter space.

    True parameters are rendered as large, semi-transparent markers; point
    estimates as small, opaque markers so that deviations from the truth are
    immediately visible.

    Parameters
    ----------
    true_parameters : np.ndarray, shape (n_obs, param_dim)
        Holdout true parameter values.
    point_estimates : np.ndarray, shape (n_obs, param_dim)
        Focal point estimates corresponding to each observation.
    param_names : sequence of str, optional
        Axis labels; defaults to θ_0, θ_1, …
    true_color : str, optional
        Colour for the true-parameter markers. Default ``'grey'``.
    estimate_color : str, optional
        Colour for the estimate markers. Default ``'steelblue'``.
    title : str, optional
        Figure title.
    figsize : tuple of int, optional
        ``(width, height)`` in inches.
    save_fig_path : str, optional
        If given, save to this path instead of displaying interactively.
    """
    true_parameters = np.asarray(to_np_if_torch(true_parameters))
    point_estimates = np.asarray(to_np_if_torch(point_estimates))

    if true_parameters.ndim == 1:
        true_parameters = true_parameters.reshape(-1, 1)
    if point_estimates.ndim == 1:
        point_estimates = point_estimates.reshape(-1, 1)

    n_obs, param_dim = true_parameters.shape
    param_names = list(param_names) if param_names is not None else [rf'$\theta_{{{d}}}$' for d in range(param_dim)]

    if param_dim == 1:
        figsize = figsize or (4, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.05, 0.05, n_obs)
        ax.scatter(jitter, true_parameters[:, 0], s=120, color=true_color,
                   alpha=0.3, label='True', zorder=2)
        ax.scatter(jitter, point_estimates[:, 0], s=20, color=estimate_color,
                   alpha=1.0, label='Estimate', zorder=3)
        ax.set_ylabel(param_names[0], fontsize=13)
        ax.set_xticks([])
        ax.legend(fontsize=11)

    elif param_dim == 2:
        figsize = figsize or (6, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(true_parameters[:, 0], true_parameters[:, 1], s=80,
                   color=true_color, alpha=0.3, label='True', zorder=2)
        ax.scatter(point_estimates[:, 0], point_estimates[:, 1], s=15,
                   color=estimate_color, alpha=1.0, label='Estimate', zorder=3)
        ax.set_xlabel(param_names[0], fontsize=13)
        ax.set_ylabel(param_names[1], fontsize=13)
        ax.legend(fontsize=11)

    else:
        n_panels = param_dim * (param_dim - 1) // 2
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        figsize = figsize or (4 * ncols, 4 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = np.array(axes).reshape(-1)
        panel = 0
        for d0 in range(param_dim):
            for d1 in range(d0 + 1, param_dim):
                ax = axes_flat[panel]
                ax.scatter(true_parameters[:, d0], true_parameters[:, d1], s=80,
                           color=true_color, alpha=0.3, label='True' if panel == 0 else None, zorder=2)
                ax.scatter(point_estimates[:, d0], point_estimates[:, d1], s=15,
                           color=estimate_color, alpha=1.0, label='Estimate' if panel == 0 else None, zorder=3)
                ax.set_xlabel(param_names[d0], fontsize=12)
                ax.set_ylabel(param_names[d1], fontsize=12)
                panel += 1
        for ax in axes_flat[panel:]:
            ax.axis('off')
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, 0.0), frameon=False)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_fig_path is not None:
        fig.savefig(save_fig_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_point_estimate_residuals(
    true_parameters: np.ndarray,
    point_estimates: np.ndarray,
    n_bins: int = 4,
    param_names: Optional[Sequence[str]] = None,
    color: str = 'steelblue',
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_fig_path: Optional[str] = None,
) -> None:
    """Tomographically binned histograms of component-wise signed residuals.

    For each parameter dimension the true-parameter range is divided into
    ``n_bins`` equal-count bins. Within each bin the distribution of signed
    residuals (estimate − truth) is shown as a histogram, so that bias and
    variance can be assessed across the parameter space.

    Parameters
    ----------
    true_parameters : np.ndarray, shape (n_obs, param_dim)
        Holdout true parameter values.
    point_estimates : np.ndarray, shape (n_obs, param_dim)
        Focal point estimates corresponding to each observation.
    n_bins : int, optional
        Number of equal-count tomographic bins per dimension. Default 4.
    param_names : sequence of str, optional
        Row labels; defaults to θ_0, θ_1, …
    color : str, optional
        Histogram bar colour. Default ``'steelblue'``.
    title : str, optional
        Figure suptitle.
    figsize : tuple of int, optional
        ``(width, height)`` in inches.
    save_fig_path : str, optional
        If given, save to this path instead of displaying interactively.
    """
    true_parameters = np.asarray(to_np_if_torch(true_parameters))
    point_estimates = np.asarray(to_np_if_torch(point_estimates))

    if true_parameters.ndim == 1:
        true_parameters = true_parameters.reshape(-1, 1)
    if point_estimates.ndim == 1:
        point_estimates = point_estimates.reshape(-1, 1)

    param_dim = true_parameters.shape[1]
    residuals = point_estimates - true_parameters
    param_names = list(param_names) if param_names is not None else [rf'$\theta_{{{d}}}$' for d in range(param_dim)]

    figsize = figsize or (3 * n_bins, 3 * param_dim)
    fig, axes = plt.subplots(param_dim, n_bins, figsize=figsize, squeeze=False)

    for d in range(param_dim):
        quantile_edges = np.quantile(true_parameters[:, d],
                                     np.linspace(0, 1, n_bins + 1))
        # Ensure strictly increasing edges for searchsorted
        quantile_edges = np.unique(quantile_edges)
        actual_bins = len(quantile_edges) - 1

        for b in range(actual_bins):
            ax = axes[d, b]
            lo, hi = quantile_edges[b], quantile_edges[b + 1]
            if b == actual_bins - 1:
                mask = (true_parameters[:, d] >= lo) & (true_parameters[:, d] <= hi)
            else:
                mask = (true_parameters[:, d] >= lo) & (true_parameters[:, d] < hi)
            resid_b = residuals[mask, d]
            if len(resid_b) > 0:
                ax.hist(resid_b, bins='auto', color=color, alpha=0.75, density=True)
            ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
            ax.set_title(f'[{lo:.2g}, {hi:.2g}]', fontsize=9)
            if b == 0:
                ax.set_ylabel(f'{param_names[d]}\nresidual density', fontsize=10)
            if d == param_dim - 1:
                ax.set_xlabel('estimate − truth', fontsize=9)

        for b in range(actual_bins, n_bins):
            axes[d, b].axis('off')

    if title is not None:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    if save_fig_path is not None:
        fig.savefig(save_fig_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_point_estimator_comparison(
    true_parameters: np.ndarray,
    *point_estimates: np.ndarray,
    estimator_names: Sequence[str],
    param_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_fig_path: Optional[str] = None,
) -> None:
    """Per-component scatter of signed residuals across estimators.

    For each parameter dimension, plots the residuals (estimate − truth) of the
    first estimator on the x-axis against the residuals of every other estimator
    on the y-axis, with zero lines and a y=x reference diagonal. Points above
    the diagonal indicate where the y-axis estimator has a larger (more positive)
    residual than the x-axis estimator.

    Parameters
    ----------
    true_parameters : np.ndarray, shape (n_obs, param_dim)
        Holdout true parameter values.
    *point_estimates : np.ndarray, each shape (n_obs, param_dim)
        Point estimates from two or more estimators. Must pass at least two.
    estimator_names : sequence of str
        Names corresponding to each array in ``point_estimates``.
    param_names : sequence of str, optional
        Per-dimension labels; defaults to θ_0, θ_1, …
    colors : sequence of str, optional
        One colour per estimator pair (x-axis estimator vs each y-axis estimator).
        Defaults to a rainbow palette over the number of pairs.
    title : str, optional
        Figure suptitle.
    figsize : tuple of int, optional
        ``(width, height)`` in inches.
    save_fig_path : str, optional
        If given, save to this path instead of displaying interactively.
    """
    if len(point_estimates) < 2:
        raise ValueError("At least two estimators are required for comparison.")
    if len(estimator_names) != len(point_estimates):
        raise ValueError("`estimator_names` must have one entry per estimator array.")

    true_parameters = np.asarray(to_np_if_torch(true_parameters))
    estimates = [np.asarray(to_np_if_torch(pe)) for pe in point_estimates]

    if true_parameters.ndim == 1:
        true_parameters = true_parameters.reshape(-1, 1)
    estimates = [e.reshape(-1, 1) if e.ndim == 1 else e for e in estimates]

    param_dim = true_parameters.shape[1]
    param_names = list(param_names) if param_names is not None else [rf'$\theta_{{{d}}}$' for d in range(param_dim)]

    # x-axis: first estimator; y-axis: each subsequent estimator
    n_pairs = len(estimates) - 1
    default_colors = list(cm.rainbow(np.linspace(0, 1, n_pairs)))
    colors = list(colors) if colors is not None else default_colors

    figsize = figsize or (4 * param_dim, 4 * n_pairs)
    fig, axes = plt.subplots(n_pairs, param_dim, figsize=figsize, squeeze=False)

    resid_ref = estimates[0] - true_parameters  # (n_obs, param_dim)

    for pair_idx, (est, name) in enumerate(zip(estimates[1:], estimator_names[1:])):
        resid_other = est - true_parameters
        color = colors[pair_idx % len(colors)]

        for d in range(param_dim):
            ax = axes[pair_idx, d]
            rx = resid_ref[:, d]
            ry = resid_other[:, d]

            ax.scatter(rx, ry, s=12, color=color, alpha=0.5, zorder=3)

            # Reference lines
            lim = np.max(np.abs(np.concatenate([rx, ry]))) * 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color='black', linewidth=0.7, linestyle='--', zorder=1)
            ax.axvline(0, color='black', linewidth=0.7, linestyle='--', zorder=1)
            ax.plot([-lim, lim], [-lim, lim], color='grey', linewidth=0.9,
                    linestyle=':', zorder=2, label='y = x')

            if pair_idx == 0:
                ax.set_title(param_names[d], fontsize=13)
            if d == 0:
                ax.set_ylabel(f'{name}\nresidual', fontsize=11)
            if pair_idx == n_pairs - 1:
                ax.set_xlabel(f'{estimator_names[0]}\nresidual', fontsize=11)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.01)

    plt.tight_layout()
    if save_fig_path is not None:
        fig.savefig(save_fig_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
