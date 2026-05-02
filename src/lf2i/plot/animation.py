from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgba
from matplotlib.animation import FuncAnimation
from matplotlib.patches import PathPatch
import alphashape
from scipy.stats import gaussian_kde

from lf2i.plot.miscellanea import PolygonPatchFixed
from lf2i.utils.miscellanea import to_np_if_torch


def _alpha_patch(pts: np.ndarray, color, alpha_param: Optional[float]) -> Optional[PathPatch]:
    """Return a PolygonPatchFixed for pts, or None if the shape is degenerate."""
    if len(pts) < 3:
        return None
    try:
        shape = alphashape.alphashape(pts, alpha=alpha_param if alpha_param is not None else 0)
        if shape is None or shape.is_empty or shape.geom_type not in ("Polygon", "MultiPolygon"):
            return None
        return PolygonPatchFixed(shape, fc=to_rgba(color, 0.2), ec=to_rgba(color, 1.0), lw=2)
    except Exception:
        return None


def parameter_regions_pairplot_animation(
    *parameter_regions: np.ndarray,
    posterior_regions: Sequence[np.ndarray],
    true_parameter: np.ndarray,
    n_frames: int = 60,
    alpha: Optional[float] = None,
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
    morph between point clouds via alpha shapes: points are added and removed in
    order of proximity to the combined centroid of both clouds (posterior outer
    points disappear first; confidence inner points appear first), and an alpha
    shape contour is redrawn each frame over the current visible point set.

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
    alpha :
        Alpha parameter passed to ``alphashape.alphashape`` controlling contour
        tightness.  ``None`` or ``0`` produces the convex hull.
    diagonal_pvalues :
        Shape ``(1, n_dims, grid_size)`` — normalized p-values for the confidence
        diagonal curves.
    diagonal_grid :
        Shape ``(n_dims, grid_size)`` — x-axis grid coordinates per dimension.
    diagonal_levels :
        Unused in the animation (kept for API parity).
    posterior_estimator :
        Object with a ``.sample((n,), x=obs)`` method used to draw posterior samples
        for the diagonal KDE curves.
    posterior_observations :
        Observations passed to ``posterior_estimator``; the first element is used.
    n_posterior_samples :
        Number of posterior samples to draw for each diagonal KDE.
    parameter_space_bounds :
        Dict mapping each param name to ``{'low': float, 'high': float}``.
        When provided, sets axis limits on diagonal panels.
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

            raw_pv = diagonal_pvalues[0, d, :]
            pv_max = raw_pv.max()
            conf_y[d] = raw_pv / pv_max if pv_max > 0 else raw_pv.copy()

            try:
                marginal = raw_samples[:, d]
            except (IndexError, TypeError):
                marginal = raw_samples.reshape(-1)
            kde = gaussian_kde(marginal)
            density = kde(grid_d)
            d_max = density.max()
            post_y[d] = density / d_max if d_max > 0 else density.copy()

    # --- Pre-sort off-diagonal point clouds by proximity to combined centroid ---
    # Points nearest the combined centroid persist longest (posterior) or appear
    # first (confidence), so the shape morphs from the inside out.
    post_sorted: dict[tuple, list] = {}
    conf_sorted: dict[tuple, list] = {}

    for row in range(n_dims):
        for col in range(n_dims):
            if col <= row:
                continue
            cell_post, cell_conf = [], []
            for i in range(n_regions):
                post_2d = np.asarray(posterior_regions[i])[:, [col, row]]
                conf_2d = np.asarray(parameter_regions[i])[:, [col, row]]
                center = np.mean(np.vstack([post_2d, conf_2d]), axis=0)
                post_idx = np.argsort(np.linalg.norm(post_2d - center, axis=1))
                conf_idx = np.argsort(np.linalg.norm(conf_2d - center, axis=1))
                cell_post.append(post_2d[post_idx])
                cell_conf.append(conf_2d[conf_idx])
            post_sorted[(row, col)] = cell_post
            conf_sorted[(row, col)] = cell_conf

    # --- Build figure ---
    fig, ax = plt.subplots(n_dims, n_dims, figsize=figsize)
    if n_dims == 1:
        ax = np.array([[ax]])

    diag_lines: dict[int, plt.Line2D] = {}
    # patches[(row, col)] = list of current PathPatch (one per region), may be None
    patches: dict[tuple, list] = {}

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
                # Upper triangle: col > row — draw initial alpha shapes (full posterior, t=0)
                cell_patches = []
                for i in range(n_regions):
                    pts = post_sorted[(row, col)][i]
                    patch = _alpha_patch(pts, colors[i], alpha)
                    if patch is not None:
                        ax[row, col].add_patch(patch)
                    cell_patches.append(patch)

                    if row == 0 and col == 1:
                        import matplotlib.patches as mpatches
                        handle = mpatches.Patch(
                            facecolor=to_rgba(colors[i], 0.2),
                            edgecolor=to_rgba(colors[i], 1.0),
                            label=region_names[i],
                        )
                        leg_handles.append(handle)
                        leg_labels_list.append(region_names[i])

                patches[(row, col)] = cell_patches

                ax[row, col].scatter(
                    true_parameter[col], true_parameter[row],
                    marker="*", s=80, color="white", edgecolors="red", linewidths=0.8, zorder=5,
                )

                # Fix axis limits from the full union of both clouds.
                all_x = np.concatenate(
                    [post_sorted[(row, col)][i][:, 0] for i in range(n_regions)]
                    + [conf_sorted[(row, col)][i][:, 0] for i in range(n_regions)]
                )
                all_y = np.concatenate(
                    [post_sorted[(row, col)][i][:, 1] for i in range(n_regions)]
                    + [conf_sorted[(row, col)][i][:, 1] for i in range(n_regions)]
                )
                margin_x = (all_x.max() - all_x.min()) * 0.05 or 0.5
                margin_y = (all_y.max() - all_y.min()) * 0.05 or 0.5
                ax[row, col].set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
                ax[row, col].set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

                x_label = param_names[col] if labels is None else labels[col]
                y_label = param_names[row] if labels is None else labels[row]
                ax[row, col].set_xlabel(x_label)
                ax[row, col].set_ylabel(y_label)

    if show_legend and leg_handles:
        fig.legend(leg_handles, leg_labels_list, bbox_to_anchor=(0.5, 0.5))

    plt.tight_layout()

    # --- Animation update ---
    def _update(frame: int):
        t = frame / max(n_frames - 1, 1)

        for d, line in diag_lines.items():
            line.set_ydata((1.0 - t) * post_y[d] + t * conf_y[d])

        for (row, col), cell_patches in patches.items():
            for i, old_patch in enumerate(cell_patches):
                if old_patch is not None:
                    old_patch.remove()

                n_post = round((1.0 - t) * len(post_sorted[(row, col)][i]))
                n_conf = round(t * len(conf_sorted[(row, col)][i]))
                visible = np.vstack([
                    post_sorted[(row, col)][i][:n_post],
                    conf_sorted[(row, col)][i][:n_conf],
                ]) if n_post + n_conf > 0 else np.empty((0, 2))

                new_patch = _alpha_patch(visible, colors[i], alpha)
                if new_patch is not None:
                    ax[row, col].add_patch(new_patch)
                patches[(row, col)][i] = new_patch

        return []

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
