from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

from lf2i.utils.miscellanea import to_np_if_torch


def parameter_regions_pairplot_animation(
    *parameter_regions: np.ndarray,
    posterior_regions: Sequence[np.ndarray],
    true_parameter: np.ndarray,
    n_frames: int = 60,
    diagonal_pvalues: Optional[np.ndarray] = None,
    diagonal_grid: Optional[np.ndarray] = None,
    diagonal_levels: Optional[Sequence[float]] = None,
    posterior_estimator=None,
    posterior_observations: Optional[Sequence] = None,
    n_posterior_samples: int = 10_000,
    parameter_space_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    param_names: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    colors: Optional[Sequence[str]] = None,
    region_names: Optional[Sequence[str]] = None,
    figsize: Sequence[int] = (15, 15),
    show_legend: bool = True,
    fps: int = 30,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Animate the morphing of a posterior pairplot into a confidence-region pairplot.

    Diagonal panels interpolate between a normalized posterior KDE curve (frame 0)
    and a normalized confidence p-value curve (last frame).  Off-diagonal panels
    alpha-crossfade from posterior sample clouds to confidence region point clouds.

    Parameters
    ----------
    *parameter_regions :
        Confidence region point clouds, each of shape ``(n_pts, n_dims)``.
    posterior_regions :
        Posterior sample clouds corresponding 1-to-1 with ``parameter_regions``,
        each of shape ``(n_pts, n_dims)``.
    true_parameter :
        1-D array of shape ``(n_dims,)``.
    n_frames :
        Total number of animation frames.
    diagonal_pvalues :
        Shape ``(1, n_dims, grid_size)`` — normalized p-values for the confidence
        diagonal curves.
    diagonal_grid :
        Shape ``(n_dims, grid_size)`` — x-axis grid coordinates per dimension.
    diagonal_levels :
        Unused in the animation (kept for API parity); confidence/credibility level
        threshold lines are omitted to keep frames clean.
    posterior_estimator :
        Object with a ``.sample((n,), x=obs)`` method used to draw posterior samples
        for the diagonal KDE curves.
    posterior_observations :
        Observations passed to ``posterior_estimator``; the first element is used.
    n_posterior_samples :
        Number of posterior samples to draw for each diagonal KDE.
    parameter_space_bounds :
        Dict mapping each param name to ``{'low': float, 'high': float}``.
        When provided, sets axis limits on all panels.
    param_names :
        Dimension labels; defaults to ``θ_0, θ_1, ...``.
    labels :
        Axis labels overriding ``param_names`` when provided.
    colors :
        One color per region pair; defaults to a rainbow palette.
    region_names :
        Legend labels for each region pair.
    figsize :
        Figure size passed to ``plt.subplots``.
    show_legend :
        Whether to add a figure-level legend.
    fps :
        Frames per second when saving.
    save_path :
        If given, save the animation to this path via ffmpeg.

    Returns
    -------
    FuncAnimation
    """
    n_regions = len(parameter_regions)
    n_dims = parameter_regions[0].shape[1]

    true_parameter = np.asarray(true_parameter).reshape(-1)
    colors = list(colors) if colors is not None else list(cm.rainbow(np.linspace(0, 1, n_regions)))
    region_names = list(region_names) if region_names is not None else [f"Region {i}" for i in range(n_regions)]
    param_names = (
        np.asarray(param_names)
        if param_names is not None
        else np.array([rf"$\theta_{{{i}}}$" for i in range(n_dims)])
    )

    # --- Pre-compute diagonal curves on the shared confidence grid ---
    has_diagonal = (
        diagonal_pvalues is not None
        and diagonal_grid is not None
        and posterior_estimator is not None
        and posterior_observations is not None
    )

    post_y: dict[int, np.ndarray] = {}
    conf_y: dict[int, np.ndarray] = {}

    if has_diagonal:
        diagonal_pvalues = to_np_if_torch(diagonal_pvalues)
        diagonal_grid = to_np_if_torch(diagonal_grid)

        raw_samples = posterior_estimator.sample(
            (n_posterior_samples,), x=posterior_observations[0]
        )
        raw_samples = np.asarray(raw_samples)

        for d in range(n_dims):
            grid_d = diagonal_grid[d].reshape(-1)

            # Confidence curve
            raw_pv = diagonal_pvalues[0, d, :]
            pv_max = raw_pv.max()
            conf_y[d] = raw_pv / pv_max if pv_max > 0 else raw_pv.copy()

            # Posterior KDE on same grid
            try:
                marginal = raw_samples[:, d]
            except (IndexError, TypeError):
                marginal = raw_samples.reshape(-1)
            kde = gaussian_kde(marginal)
            density = kde(grid_d)
            d_max = density.max()
            post_y[d] = density / d_max if d_max > 0 else density.copy()

    # --- Build figure ---
    fig, ax = plt.subplots(n_dims, n_dims, figsize=figsize)
    if n_dims == 1:
        ax = np.array([[ax]])

    diag_lines: dict[int, plt.Line2D] = {}
    conf_scats: dict[tuple, list] = {}
    post_scats: dict[tuple, list] = {}

    leg_handles, leg_labels_list = [], []

    for row in range(n_dims):
        for col in range(n_dims):
            if row == col:
                if has_diagonal:
                    grid_d = diagonal_grid[col].reshape(-1)
                    (line,) = ax[row, col].plot(grid_d, post_y[col], color=colors[0])
                    diag_lines[col] = line
                    ax[row, col].axvline(true_parameter[col], color="black", linewidth=1, linestyle="--")
                    ax[row, col].set_ylim(0, 1)
                    ax[row, col].set_ylabel("Normalized density")
                    x_label = param_names[col] if labels is None else labels[col]
                    ax[row, col].set_xlabel(x_label)
                    if parameter_space_bounds is not None and param_names[col] in parameter_space_bounds:
                        b = parameter_space_bounds[param_names[col]]
                        ax[row, col].set_xlim(b['low'], b['high'])
                else:
                    ax[row, col].axis("off")

            elif col < row:
                ax[row, col].axis("off")

            else:
                # Upper triangle: col > row
                cell_conf_scats = []
                cell_post_scats = []
                for i in range(n_regions):
                    conf_pts = np.asarray(parameter_regions[i])
                    post_pts = np.asarray(posterior_regions[i])

                    start_rgba = np.append(to_rgba(colors[0])[:3], 1.0)
                    ps = ax[row, col].scatter(
                        post_pts[:, col], post_pts[:, row],
                        s=4, marker=".", label=None,
                    )
                    ps.set_facecolor(start_rgba)

                    cs = ax[row, col].scatter(
                        conf_pts[:, col], conf_pts[:, row],
                        s=4, marker=".", label=region_names[i],
                    )
                    cs.set_facecolor(np.append(to_rgba(colors[0])[:3], 0.0))
                    cell_conf_scats.append(cs)
                    cell_post_scats.append(ps)

                    if row == 0 and col == 1:
                        import matplotlib.lines as mlines
                        handle = mlines.Line2D(
                            [], [], color=colors[i], marker="o", linestyle="None",
                            markersize=6, label=region_names[i],
                        )
                        leg_handles.append(handle)
                        leg_labels_list.append(region_names[i])

                ax[row, col].scatter(
                    true_parameter[col], true_parameter[row],
                    marker="*", s=80, color="white", edgecolors="red", linewidths=0.8, zorder=5,
                )

                x_label = param_names[col] if labels is None else labels[col]
                y_label = param_names[row] if labels is None else labels[row]
                ax[row, col].set_xlabel(x_label)
                ax[row, col].set_ylabel(y_label)
                if parameter_space_bounds is not None:
                    if param_names[col] in parameter_space_bounds:
                        b = parameter_space_bounds[param_names[col]]
                        ax[row, col].set_xlim(b['low'], b['high'])
                    if param_names[row] in parameter_space_bounds:
                        b = parameter_space_bounds[param_names[row]]
                        ax[row, col].set_ylim(b['low'], b['high'])

                conf_scats[(row, col)] = cell_conf_scats
                post_scats[(row, col)] = cell_post_scats

    if show_legend and leg_handles:
        fig.legend(leg_handles, leg_labels_list, bbox_to_anchor=(0.5, 0.5))

    plt.tight_layout()

    diag_color_start = np.array(to_rgba(colors[0]))
    diag_color_end = np.array(to_rgba(colors[-1]))

    # --- Animation update ---
    def _update(frame: int):
        t = frame / max(n_frames - 1, 1)

        for d, line in diag_lines.items():
            y = (1.0 - t) * post_y[d] + t * conf_y[d]
            line.set_ydata(y)
            line.set_color((1.0 - t) * diag_color_start + t * diag_color_end)

        blended_rgb = (1.0 - t) * diag_color_start[:3] + t * diag_color_end[:3]

        for key, scats in conf_scats.items():
            for sc in scats:
                sc.set_facecolor(np.append(blended_rgb, t))
        for key, scats in post_scats.items():
            for sc in scats:
                sc.set_facecolor(np.append(blended_rgb, 1.0 - t))

        return []

    _update(0)

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        blit=False,
        interval=1000 // fps,
    )

    if save_path is not None:
        anim.save(save_path, writer="ffmpeg", fps=fps)

    return anim
