from typing import Optional, Tuple, Union, Dict, List, Sequence, Any
from warnings import simplefilter

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Calibration-quality score plots (MSE / CRPS / pinball loss)
# ---------------------------------------------------------------------------

_DEFAULT_SCORE_LABELS: Dict[str, str] = {
    'mse': 'MSE',
    'crps': 'CRPS',
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
    n_bins: Optional[int] = None,
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

        if n_bins is not None:
            x_edges = np.linspace(parameters[:, 0].min(), parameters[:, 0].max(), n_bins + 1)
            y_edges = np.linspace(parameters[:, 1].min(), parameters[:, 1].max(), n_bins + 1)
        else:
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

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

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


def plot_cdf_comparison(
    test_statistic: Any,
    calib_model: Any,
    acceptance_region: str,
    theta_eval: np.ndarray,
    simulator: Any,
    monte_carlo_size: int = 2000,
    n_grid: int = 500,
    title: Optional[str] = None,
    figsize: Tuple = (7, 5),
    save_fig_path: Optional[str] = None,
    custom_ax: Optional[Axes] = None,
) -> None:
    """Plot the MC empirical CDF vs. the calibration model's predicted CDF at a fixed theta.

    Parameters
    ----------
    test_statistic : TestStatistic
        Fitted test statistic with an ``evaluate`` method.
    calib_model : Any
        A single calibration model (a value from ``lf2i.calibration_model``), must have ``predict_proba``.
    acceptance_region : str
        ``'left'`` or ``'right'``.
    theta_eval : np.ndarray, shape (param_dim,) or (1, param_dim)
        The parameter value at which to evaluate.
    simulator : Simulator
        lf2i Simulator used to draw MC samples.
    monte_carlo_size : int
        Number of MC draws. Default 2000.
    n_grid : int
        Resolution of the lambda grid for the parametric CDF curve. Default 500.
    title : str, optional
    figsize : Tuple
    save_fig_path : str, optional
    custom_ax : Axes, optional
        If provided, draw onto this axes without calling ``plt.show()``.
    """
    import torch
    from lf2i.utils.calibration_diagnostics_inputs import preprocess_predict_p_values
    from lf2i.utils.miscellanea import to_np_if_torch

    theta_eval = np.asarray(theta_eval, dtype=np.float32)
    if theta_eval.ndim == 1:
        theta_eval = theta_eval.reshape(1, -1)

    theta_mc = torch.tensor(theta_eval).repeat(monte_carlo_size, 1)
    samples_mc = simulator(theta_mc)
    ts_mc = to_np_if_torch(
        test_statistic.evaluate(parameters=theta_mc, samples=samples_mc, mode='diagnostics')
    ).reshape(-1)

    ts_sorted = np.sort(ts_mc)
    ecdf = np.arange(1, monte_carlo_size + 1) / monte_carlo_size

    lambda_grid = np.linspace(ts_sorted.min(), ts_sorted.max(), n_grid)
    theta_rep = np.tile(theta_eval, (n_grid, 1))
    X_eval = preprocess_predict_p_values('diagnostics', lambda_grid, theta_rep, calib_model)
    proba = calib_model.predict_proba(X=X_eval)
    # For acceptance_region='right': col 1 = P(T <= lambda | theta) = CDF
    # For acceptance_region='left':  col 0 = 1 - P(T >= lambda | theta) = CDF
    cdf_hat = proba[:, 1] if acceptance_region == 'right' else proba[:, 0]

    own_fig = custom_ax is None
    if own_fig:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = custom_ax

    ax.step(ts_sorted, ecdf, label='Truth', color='steelblue', lw=2, where='post')
    ax.plot(lambda_grid, cdf_hat, label='Estimate', color='tomato', lw=2, ls='--')
    ax.set_xlabel(r'Test statistic ($\lambda$)')
    ax.set_ylabel(r'CDF ($F(\lambda \mid \theta)$)')
    ax.legend(fontsize=9)

    if title is not None:
        ax.set_title(title, fontsize=12)

    if own_fig:
        simplefilter(action='ignore', category=UserWarning)
        plt.tight_layout()
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()


def calibration_cdf_panel(
    evaluation_grid: np.ndarray,
    estimation_errors: Dict[str, np.ndarray],
    test_statistic: Any,
    calibration_model: Dict,
    simulator: Any,
    param_dim: int,
    score_key: str = 'crps',
    monte_carlo_size: int = 2000,
    n_grid: int = 500,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    n_bins: Optional[int] = None,
    params_labels: Optional[Union[Tuple[str], List[str]]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    save_fig_path: Optional[str] = None,
    query_points: Optional[np.ndarray] = None,
) -> None:
    """Three-panel diagnostic figure combining a calibration score heatmap with CDF comparisons.

    Panels
    ------
    Left          : calibration score heatmap (from :func:`calibration_score_plot`).
    Middle/Right  : CDF comparisons. When ``query_points`` is None (default), these
                    show the worst-match theta (highest ``score_key``) and best-match
                    theta (lowest ``score_key``). When ``query_points`` is provided,
                    one panel is produced per query point and the points are indexed
                    on the score heatmap.

    Parameters
    ----------
    evaluation_grid : np.ndarray
        Grid of parameter values, as returned by :func:`monte_carlo_pvalue_diagnostics`.
    estimation_errors : Dict[str, np.ndarray]
        Per-theta error arrays, as returned by :func:`monte_carlo_pvalue_diagnostics`.
    test_statistic : TestStatistic
        Fitted test statistic.
    calibration_model : Dict
        The ``lf2i.calibration_model`` dict (key -> model).
    simulator : Simulator
        lf2i Simulator used to draw MC samples.
    param_dim : int
        Dimensionality of the parameter.
    score_key : str
        Key in ``estimation_errors`` used to rank thetas. Default ``'crps'``.
    monte_carlo_size : int
        MC draws per theta for the CDF panels. Default 2000.
    n_grid : int
        Lambda-grid resolution for parametric CDF curves. Default 500.
    xlims, ylims : Tuple[float, float], optional
        Axis limits for the score heatmap.
    params_labels : list of str, optional
    title : str, optional
        Overall figure suptitle.
    figsize : Tuple, optional
    save_fig_path : str, optional
    query_points : np.ndarray, optional
        Array of shape ``(n_query, param_dim)`` (or ``(param_dim,)`` for a single
        point). When provided, CDF comparisons are shown for each of these points
        instead of the automatic worst/best selection, and the points are marked
        with index labels on the score heatmap.
    """
    from lf2i.utils.miscellanea import to_np_if_torch

    scores = estimation_errors[score_key]

    calib_key = (
        'multiple_levels' if 'multiple_levels' in calibration_model
        else next(iter(calibration_model))
    )
    calib_model_single = calibration_model[calib_key]
    acceptance_region = test_statistic.acceptance_region

    grid_np = to_np_if_torch(evaluation_grid)
    if grid_np.ndim == 1:
        grid_np = grid_np.reshape(-1, 1)

    score_label = _score_label(score_key)

    if query_points is not None:
        qp = np.atleast_2d(np.asarray(query_points, dtype=np.float32))
        n_query = len(qp)
        n_panels = 1 + n_query

        if figsize is None:
            figsize = (6 * n_panels, 5)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize)

        # Left: calibration score heatmap
        mesh = calibration_score_plot(
            parameters=grid_np,
            scores=scores,
            score_label=score_label,
            param_dim=param_dim,
            xlims=xlims,
            ylims=ylims,
            params_labels=params_labels,
            custom_ax=axes[0],
            n_bins=n_bins,
        )
        if mesh is not None:
            cbar = fig.colorbar(mesh, ax=axes[0])
            cbar.set_label(score_label, fontsize=12, labelpad=6)
            cbar.ax.tick_params(labelsize=10)

        # Index query points on the score panel
        if param_dim == 2:
            for i, pt in enumerate(qp):
                axes[0].scatter(pt[0], pt[1], color='black', s=120, zorder=5)
                axes[0].annotate(
                    str(i + 1), (pt[0], pt[1]),
                    ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold', zorder=6,
                )
            axes[0].set_title('Local calibration risk', fontsize=14, pad=10)
        else:
            ymax = axes[0].get_ylim()[1]
            for i, pt in enumerate(qp):
                axes[0].axvline(pt[0], color='black', linestyle='--', linewidth=1, zorder=5)
                axes[0].text(
                    pt[0], ymax, str(i + 1),
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                )

        # CDF comparison panels — one per query point
        for i, (ax, pt) in enumerate(zip(axes[1:], qp)):
            theta = pt.reshape(1, -1)
            theta_str = ', '.join(f'{v:.2f}' for v in pt)
            plot_cdf_comparison(
                test_statistic=test_statistic,
                calib_model=calib_model_single,
                acceptance_region=acceptance_region,
                theta_eval=theta,
                simulator=simulator,
                monte_carlo_size=monte_carlo_size,
                n_grid=n_grid,
                title=f'Query point {i + 1}',
                custom_ax=ax,
            )
            ax.annotate(
                rf'$\theta_{{{i + 1}}} = ({theta_str})$',
                xy=(0.5, -0.25), xycoords='axes fraction',
                ha='center', va='top', fontsize=14,
            )

    else:
        worst_idx = int(np.argmax(scores))
        best_idx = int(np.argmin(scores))

        worst_theta = grid_np[worst_idx:worst_idx + 1]
        best_theta = grid_np[best_idx:best_idx + 1]

        if figsize is None:
            figsize = (18, 5)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Left: calibration score heatmap
        mesh = calibration_score_plot(
            parameters=grid_np,
            scores=scores,
            score_label=score_label,
            param_dim=param_dim,
            xlims=xlims,
            ylims=ylims,
            params_labels=params_labels,
            custom_ax=axes[0],
            n_bins=n_bins
        )
        if mesh is not None:
            cbar = fig.colorbar(mesh, ax=axes[0])
            cbar.set_label(score_label, fontsize=12, labelpad=6)
            cbar.ax.tick_params(labelsize=10)

        # Annotate worst/best on the heatmap for 2-D parameter spaces
        if param_dim == 2:
            axes[0].scatter(
                worst_theta[0, 0], worst_theta[0, 1],
                color='black', marker='v', s=120, zorder=5,
            )
            axes[0].scatter(
                best_theta[0, 0], best_theta[0, 1],
                color='white', marker='^', s=120, zorder=5,
                edgecolors='black', linewidths=1,
            )
            legend_elements = [
                Line2D([0], [0], marker='v', color='w', markerfacecolor='black', markersize=14, label='Worst'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
                       markeredgecolor='black', markersize=14, label='Best'),
            ]
            axes[0].legend(handles=legend_elements, fontsize=14, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.25))
            axes[0].set_title('Local calibration risk', fontsize=14, pad=10)

        # Middle/right: CDF comparisons
        for ax, theta, label in [
            (axes[1], worst_theta, '▼ Worst'),
            (axes[2], best_theta, '△ Best'),
        ]:
            theta_str = ', '.join(f'{v:.2f}' for v in theta[0])
            plot_cdf_comparison(
                test_statistic=test_statistic,
                calib_model=calib_model_single,
                acceptance_region=acceptance_region,
                theta_eval=theta,
                simulator=simulator,
                monte_carlo_size=monte_carlo_size,
                n_grid=n_grid,
                title=label,
                custom_ax=ax,
            )
            ax.annotate(
                rf'$\theta = ({theta_str})$',
                xy=(0.5, -0.25), xycoords='axes fraction',
                ha='center', va='top', fontsize=14,
            )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    simplefilter(action='ignore', category=UserWarning)
    fig.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
