from typing import Callable, Sequence
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
from matplotlib.colors import to_rgba

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.0

from monaqa2.data.plot_main import (
    plot_spectral_gap_vs_n,
    plot_spectral_gap_vs_beta,
    plot_last_step_classical_queries_and_spectral_gap_vs_n,
    plot_annealing_classical_and_quantum_queries_vs_n,
)


_SPECTRAL_GAP_BETA_YTICKS = [1e-1, 1e-4, 1e-7, 1e-10, 1e-13, 1e-16]


def _make_table_axes(n_items: int, ncols: int, figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, np.ndarray]:
    if n_items <= 0:
        raise ValueError("n_items must be positive.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")

    nrows = int(np.ceil(n_items / ncols))
    if figsize is None:
        figsize = (7.2 * ncols, 5.8 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax in axes[n_items:]:
        ax.axis("off")

    return fig, axes


def _finish_table(fig: plt.Figure, hspace: float, wspace: float) -> None:
    fig.subplots_adjust(hspace=hspace, wspace=wspace)


def _format_large_fit_constant(label: str) -> str:
    def repl(match: re.Match) -> str:
        prefix, value_text, suffix = match.groups()
        value = float(value_text)
        if abs(value) <= 99.0:
            return match.group(0)

        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10.0**exponent
        mantissa = round(mantissa, 1)

        if abs(mantissa) >= 10.0:
            mantissa /= 10.0
            exponent += 1

        return rf"{prefix}{mantissa:.1f}\,10^{{{exponent}}}{suffix}"

    return re.sub(r"(\$)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(\s*\\times)", repl, label)


def _fit_only_label(label: str) -> str:
    if ": fit " in label:
        return _format_large_fit_constant(label.split(": fit ", 1)[1])
    if ": log-linear interpolation" in label:
        return "log-linear"
    if " queries: " in label:
        return _format_large_fit_constant(label.split(" queries: ", 1)[1])
    if " inverse gap: " in label:
        return _format_large_fit_constant(label.split(" inverse gap: ", 1)[1])
    if label.endswith(" classical annealing queries"):
        return "classical"
    if label.endswith(" quantum-walk queries"):
        return "quantum"
    return label


def _line_color(handle):
    return handle.get_color() if hasattr(handle, "get_color") else "black"


def _line_style(handle):
    return handle.get_linestyle() if hasattr(handle, "get_linestyle") else "-"


def _line_width(handle):
    return handle.get_linewidth() if hasattr(handle, "get_linewidth") else 2.0


def _line_alpha(handle):
    alpha = handle.get_alpha() if hasattr(handle, "get_alpha") else None
    return 1.0 if alpha is None else alpha


def _lighten_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="0.92", alpha=0.45, linewidth=0.6)
    ax.grid(False, which="minor")


def _enlarge_axis_labels(ax: plt.Axes, label_fontsize: int, tick_labelsize: int, title_fontsize: int) -> None:
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    ax.title.set_size(title_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.tick_params(axis="both", which="minor", labelsize=max(1, tick_labelsize - 2))


def _keep_axis_labels_only_on_outer_edges(axes: np.ndarray, n_items: int, ncols: int, gap_axes: Sequence[plt.Axes | None] | None = None) -> None:
    nrows = int(np.ceil(n_items / ncols))
    for idx, ax in enumerate(axes[:n_items]):
        row = idx // ncols
        col = idx % ncols
        if row != nrows - 1:
            ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")
        if gap_axes is not None and idx < len(gap_axes) and gap_axes[idx] is not None:
            gap_axes[idx].set_ylabel("")


def _keep_single_axis_y_ticks_only_on_outer_edges(
    axes: np.ndarray,
    n_items: int,
    ncols: int,
) -> None:
    nrows = int(np.ceil(n_items / ncols))
    for idx, ax in enumerate(axes[:n_items]):
        row = idx // ncols
        col = idx % ncols
        is_left_edge = col == 0
        is_right_edge = ncols > 1 and col == ncols - 1

        if row != nrows - 1:
            ax.set_xlabel("")

        if is_left_edge:
            ax.tick_params(axis="y", which="both", left=True, labelleft=True, right=False, labelright=False)
        elif is_right_edge:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False, right=True, labelright=True)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False, right=False, labelright=False)


def _replace_legend_with_line_fit_rows(
    ax: plt.Axes,
    ncol: int = 3,
    y: float = -0.17,
    row_gap: float = 0.030,
    group_gap: float = 0.135,
    line_half_width: float = 0.055,
    fontsize: int = 12,
    label_formatter: Callable[[str], str] | None = None,
) -> None:
    legend = ax.get_legend()
    if legend is None:
        return

    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", None)
    if handles is None:
        return

    formatter = _fit_only_label if label_formatter is None else label_formatter
    labels = [formatter(text.get_text()) for text in legend.get_texts()]
    legend.remove()

    for group_start in range(0, len(handles), ncol):
        group_handles = handles[group_start:group_start + ncol]
        group_labels = labels[group_start:group_start + ncol]
        count = len(group_handles)
        xs = np.linspace(0.14, 0.86, count) if count > 1 else np.array([0.5])
        y_line = y - (group_start // ncol) * group_gap
        y_text = y_line - row_gap

        for x, handle, label in zip(xs, group_handles, group_labels):
            ax.plot(
                [x - line_half_width, x + line_half_width],
                [y_line, y_line],
                transform=ax.transAxes,
                color=_line_color(handle),
                linestyle=_line_style(handle),
                linewidth=_line_width(handle),
                alpha=_line_alpha(handle),
                solid_capstyle="butt",
                clip_on=False,
            )
            ax.text(
                x,
                y_text,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )


def _apply_spectral_gap_beta_y_scale(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1.0)
    ax.set_yticks(_SPECTRAL_GAP_BETA_YTICKS)
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))


def _scale_inset_bounds(
    inset_bounds: tuple[float, float, float, float],
    miniature_scale: float,
) -> tuple[float, float, float, float]:
    if miniature_scale <= 0.0:
        raise ValueError("miniature_scale must be positive.")

    x0, y0, width, height = inset_bounds
    width = min(width * miniature_scale, max(0.01, 0.98 - x0))
    height = min(height * miniature_scale, max(0.01, 0.98 - y0))
    return x0, y0, width, height


def _add_spectral_gap_beta_inset(
    fig: plt.Figure,
    ax: plt.Axes,
    fixed_n: int,
    inset_bounds: tuple[float, float, float, float],
    inset_tick_labelsize: int,
    **kwargs,
) -> plt.Axes:
    inset_kwargs = dict(kwargs)
    inset_kwargs.pop("beta_plot_min", None)
    inset_kwargs.pop("beta_plot_max", None)
    inset_kwargs.pop("show_legend", None)
    inset_kwargs.pop("title", None)
    inset_kwargs.pop("fig", None)
    inset_kwargs.pop("ax", None)

    inset_ax = ax.inset_axes(inset_bounds)
    plot_spectral_gap_vs_beta(
        fixed_n=fixed_n,
        fig=fig,
        ax=inset_ax,
        title="",
        show_legend=False,
        beta_plot_min=0.25,
        beta_plot_max=4.0,
        **inset_kwargs,
    )
    inset_ax.set_xlim(0.25, 4.0)
    inset_ax.set_ylim(1e-4, 1.0)
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.set_title("")
    inset_ax.set_xticks([0.25, 1.0, 4.0])
    inset_ax.set_xticklabels([r"$0.25$", r"$1$", r"$4$"])
    inset_ax.set_yticks([1e-1, 1e-4])
    inset_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    inset_ax.tick_params(axis="both", which="major", labelsize=inset_tick_labelsize, pad=1)
    inset_ax.tick_params(axis="both", which="minor", labelsize=max(1, inset_tick_labelsize - 2), pad=1)
    inset_ax.grid(True, which="major", color="0.90", alpha=0.55, linewidth=0.55)
    inset_ax.grid(False, which="minor")
    return inset_ax



def _spectral_gap_n_legend_label(label: str) -> str:
    if ": fit " not in label:
        return _fit_only_label(label)

    fit_text = label.split(": fit ", 1)[1]
    match = re.search(r"2\^\{([^}]*)\}", fit_text)
    if match is None:
        return _fit_only_label(label)

    exponent_text = match.group(1)
    alpha_match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", exponent_text)
    if alpha_match is None:
        return _fit_only_label(label)

    alpha = abs(float(alpha_match.group(0)))
    return rf"$\lambda={alpha:.3f}$"


def _split_fit_lines_at_n(
    ax: plt.Axes,
    extrapolation_start_n: float,
    extrapolation_alpha_factor: float = 0.55,
    extrapolation_linewidth_factor: float = 0.95,
) -> None:
    for line in list(ax.lines):
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size < 4 or y.size != x.size:
            continue
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            continue
        if np.nanmax(x) <= extrapolation_start_n:
            continue
        if np.nanmin(x) >= extrapolation_start_n:
            continue

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        y_start = np.interp(extrapolation_start_n, x, y)

        solid_mask = x <= extrapolation_start_n
        dashed_mask = x >= extrapolation_start_n

        x_solid = np.concatenate([x[solid_mask], [extrapolation_start_n]])
        y_solid = np.concatenate([y[solid_mask], [y_start]])
        x_dashed = np.concatenate([[extrapolation_start_n], x[dashed_mask]])
        y_dashed = np.concatenate([[y_start], y[dashed_mask]])

        color = line.get_color()
        alpha = line.get_alpha()
        alpha = 1.0 if alpha is None else alpha
        linewidth = line.get_linewidth()

        line.set_data(x_solid, y_solid)
        ax.plot(
            x_dashed,
            y_dashed,
            color=color,
            linestyle="--",
            linewidth=linewidth * extrapolation_linewidth_factor,
            alpha=alpha * extrapolation_alpha_factor,
            zorder=line.get_zorder(),
        )


def _soften_plot_elements(
    ax: plt.Axes,
    scatter_size: float = 22.0,
    band_alpha: float = 0.16,
) -> None:
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            offsets = collection.get_offsets()
            if offsets is not None and len(offsets) > 0:
                collection.set_sizes(np.full(len(offsets), scatter_size))
        elif isinstance(collection, PolyCollection):
            collection.set_alpha(band_alpha)


def _positive_limits_from_axes(axes: Sequence[plt.Axes]) -> tuple[float, float]:
    lows = []
    highs = []
    for ax in axes:
        y0, y1 = ax.get_ylim()
        if np.isfinite(y0) and y0 > 0:
            lows.append(y0)
        if np.isfinite(y1) and y1 > 0:
            highs.append(y1)
    if not lows or not highs:
        return 1e-16, 1.0

    lower = min(lows)
    upper = max(highs)
    lower = 10.0 ** np.floor(np.log10(lower))
    upper = 10.0 ** np.ceil(np.log10(upper))
    if lower >= upper:
        lower = upper * 1e-3
    return lower, upper


def _apply_rowwise_y_limits(axes: np.ndarray, n_items: int, ncols: int) -> None:
    nrows = int(np.ceil(n_items / ncols))
    for row in range(nrows):
        row_start = row * ncols
        row_end = min((row + 1) * ncols, n_items)
        row_axes = list(axes[row_start:row_end])
        y_limits = _positive_limits_from_axes(row_axes)
        for ax in row_axes:
            ax.set_ylim(y_limits)


def _ceil_to_multiple(value: float, step: float) -> float:
    return step * np.ceil(value / step)


def _plain_one_log_formatter(value: float, position: int | None = None) -> str:
    if np.isclose(value, 1.0):
        return "1"
    if value <= 0.0 or not np.isfinite(value):
        return ""
    exponent = int(np.round(np.log10(value)))
    return rf"$10^{{{exponent}}}$"


def _measurement_y_values_from_axes(axes: Sequence[plt.Axes]) -> np.ndarray:
    values = []
    for ax in axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            offsets = collection.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue
            y = np.asarray(offsets[:, 1], dtype=float)
            y = y[np.isfinite(y) & (y > 0.0)]
            if y.size:
                values.append(y)

    if not values:
        return np.array([], dtype=float)

    return np.concatenate(values)


def _spectral_gap_n_row_y_limit_from_measurements(
    row_axes: Sequence[plt.Axes],
    measurement_scale: float,
    max_extra_orders: float,
    tick_order_step: int,
) -> tuple[float, float, np.ndarray]:
    if measurement_scale < 1.0:
        raise ValueError("measurement_scale must be at least 1.0.")
    if max_extra_orders < 0.0:
        raise ValueError("max_extra_orders must be non-negative.")
    if tick_order_step <= 0:
        raise ValueError("tick_order_step must be positive.")

    y = _measurement_y_values_from_axes(row_axes)
    if y.size == 0:
        lower, _ = _positive_limits_from_axes(row_axes)
        data_orders = max(0.0, np.log10(1.0 / lower))
    else:
        data_min = min(float(np.min(y)), 1.0)
        data_orders = max(0.0, np.log10(1.0 / data_min))

    scaled_orders = measurement_scale * data_orders
    capped_orders = min(scaled_orders, data_orders + max_extra_orders)
    limit_orders = int(max(tick_order_step, _ceil_to_multiple(capped_orders, tick_order_step)))

    lower = 10.0 ** (-limit_orders)
    upper = 1.0
    ticks = 10.0 ** (-np.arange(0, limit_orders + 1, tick_order_step, dtype=float))
    return lower, upper, ticks


def _apply_rowwise_spectral_gap_n_y_limits_from_measurements(
    axes: np.ndarray,
    n_items: int,
    ncols: int,
    measurement_scale: float = 1.5,
    max_extra_orders: float = 6.0,
    tick_order_step: int = 3,
) -> None:
    nrows = int(np.ceil(n_items / ncols))
    formatter = FuncFormatter(_plain_one_log_formatter)
    for row in range(nrows):
        row_start = row * ncols
        row_end = min((row + 1) * ncols, n_items)
        row_axes = list(axes[row_start:row_end])
        lower, upper, ticks = _spectral_gap_n_row_y_limit_from_measurements(
            row_axes,
            measurement_scale=measurement_scale,
            max_extra_orders=max_extra_orders,
            tick_order_step=tick_order_step,
        )
        for ax in row_axes:
            ax.set_yscale("log")
            ax.set_ylim(lower, upper)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(formatter)


def _apply_spectral_gap_n_x_scale(ax: plt.Axes, n_plot_min: float, n_plot_max: float) -> None:
    left = max(1.0, float(n_plot_min))
    right = float(n_plot_max)
    ax.set_xlim(left, right)

    ticks = [float(k) for k in range(1, 11) if left <= k <= right]
    if left <= 20.0 <= right:
        ticks.append(20.0)

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])


_LOCAL_MOVE_COLOR = "#56B4E9"


def _rgba_close(color, target_color: str, tol: float = 1e-2) -> bool:
    try:
        rgba = np.asarray(to_rgba(color), dtype=float)
        target = np.asarray(to_rgba(target_color), dtype=float)
        return bool(np.allclose(rgba[:3], target[:3], atol=tol, rtol=0.0))
    except ValueError:
        return False


def _collection_has_color(collection, target_color: str) -> bool:
    for getter_name in ("get_facecolors", "get_edgecolors"):
        if not hasattr(collection, getter_name):
            continue
        colors = getattr(collection, getter_name)()
        if colors is None or len(colors) == 0:
            continue
        if any(_rgba_close(color, target_color) for color in colors):
            return True
    return False


def _remove_artists_with_color(ax: plt.Axes, target_color: str) -> None:
    for line in list(ax.lines):
        if _rgba_close(line.get_color(), target_color):
            line.remove()

    for collection in list(ax.collections):
        if _collection_has_color(collection, target_color):
            collection.remove()


def _remove_local_move_artists(ax: plt.Axes, ax_gap: plt.Axes | None = None) -> None:
    _remove_artists_with_color(ax, _LOCAL_MOVE_COLOR)
    if ax_gap is not None:
        _remove_artists_with_color(ax_gap, _LOCAL_MOVE_COLOR)


def _last_step_scaling_label(label: str) -> str:
    if " queries: " in label:
        prefix = r"$Q(n): \lambda="
    elif " inverse gap: " in label:
        prefix = r"$\delta(n): \lambda="
    else:
        return _fit_only_label(label)

    match = re.search(r"2\^\{([^}]*)\}", label)
    if match is None:
        return _fit_only_label(label)

    exponent_text = match.group(1)
    lambda_match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", exponent_text)
    if lambda_match is None:
        return _fit_only_label(label)

    lam = abs(float(lambda_match.group(0)))
    return rf"{prefix}{lam:.3f}$"


def _style_last_step_observable_lines(
    ax: plt.Axes,
    ax_gap: plt.Axes | None,
    query_linewidth: float,
    gap_linewidth: float,
    query_alpha: float,
    gap_alpha: float,
) -> None:
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size < 4 or y.size != x.size:
            continue
        line.set_linestyle("-")
        line.set_linewidth(query_linewidth)
        line.set_alpha(query_alpha)

    if ax_gap is None:
        return

    for line in ax_gap.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size < 4 or y.size != x.size:
            continue
        line.set_linestyle("-")
        line.set_linewidth(gap_linewidth)
        line.set_alpha(gap_alpha)


def _replace_last_step_legend_with_scaling_rows(
    ax: plt.Axes,
    ncol: int = 3,
    y: float = -0.17,
    row_gap: float = 0.030,
    group_gap: float = 0.145,
    line_half_width: float = 0.050,
    fontsize: int = 12,
    remove_local: bool = False,
    query_linewidth: float = 2.0,
    gap_linewidth: float = 1.2,
    query_alpha: float = 0.95,
    gap_alpha: float = 0.60,
    x_center: float = 0.5,
    x_span: float = 0.54,
    two_column_x_span: float = 0.34,
) -> None:
    legend = ax.get_legend()
    if legend is None:
        return

    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", None)
    if handles is None:
        return

    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()

    filtered_handles = []
    filtered_labels = []
    filtered_raw_labels = []
    for handle, label in zip(handles, labels):
        if remove_local and label.startswith("Local spin-flip"):
            continue
        filtered_handles.append(handle)
        filtered_labels.append(_last_step_scaling_label(label))
        filtered_raw_labels.append(label)

    for group_start in range(0, len(filtered_handles), ncol):
        group_handles = filtered_handles[group_start:group_start + ncol]
        group_labels = filtered_labels[group_start:group_start + ncol]
        group_raw_labels = filtered_raw_labels[group_start:group_start + ncol]
        count = len(group_handles)
        if count == 0:
            continue

        span = two_column_x_span if count == 2 else x_span
        xs = np.linspace(x_center - 0.5 * span, x_center + 0.5 * span, count) if count > 1 else np.array([x_center])
        y_line = y - (group_start // ncol) * group_gap
        y_text = y_line - row_gap

        for x, handle, label, raw_label in zip(xs, group_handles, group_labels, group_raw_labels):
            is_gap = " inverse gap: " in raw_label
            ax.plot(
                [x - line_half_width, x + line_half_width],
                [y_line, y_line],
                transform=ax.transAxes,
                color=_line_color(handle),
                linestyle="-",
                linewidth=gap_linewidth if is_gap else query_linewidth,
                alpha=gap_alpha if is_gap else query_alpha,
                solid_capstyle="butt",
                clip_on=False,
            )
            ax.text(
                x,
                y_text,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )


def _positive_y_values_from_axes(axes: Sequence[plt.Axes]) -> np.ndarray:
    values = []
    for ax in axes:
        if ax is None:
            continue

        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            y = y[np.isfinite(y) & (y > 0.0)]
            if y.size:
                values.append(y)

        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                offsets = collection.get_offsets()
                if offsets is None or len(offsets) == 0:
                    continue
                y = np.asarray(offsets[:, 1], dtype=float)
                y = y[np.isfinite(y) & (y > 0.0)]
                if y.size:
                    values.append(y)

    if not values:
        return np.array([], dtype=float)

    return np.concatenate(values)


def _log_limits_and_ticks_for_last_step_row(
    row_axes: Sequence[plt.Axes],
    row_gap_axes: Sequence[plt.Axes | None],
    tick_order_step: int,
    include_unit_tick: bool = False,
) -> tuple[float, float, np.ndarray]:
    if tick_order_step <= 0:
        raise ValueError("tick_order_step must be positive.")

    y = _positive_y_values_from_axes(list(row_axes) + [ax for ax in row_gap_axes if ax is not None])
    if y.size == 0:
        return 1.0, 1e3, np.array([1.0, 1e3], dtype=float)

    lower = float(np.min(y))
    upper = float(np.max(y))

    if lower <= 0.0 or not np.isfinite(lower):
        lower = 1.0
    if upper <= lower or not np.isfinite(upper):
        upper = lower * 10.0 ** tick_order_step

    if include_unit_tick:
        lower = min(lower, 1.0)
        upper = max(upper, 1.0)

    min_order = int(np.floor(np.log10(lower)))
    max_order = int(np.ceil(np.log10(upper)))
    first_tick_order = int(tick_order_step * np.ceil(min_order / tick_order_step))
    last_tick_order = int(tick_order_step * np.floor(max_order / tick_order_step))

    if first_tick_order > last_tick_order:
        ticks = np.array([], dtype=float)
    else:
        ticks = 10.0 ** np.arange(first_tick_order, last_tick_order + 1, tick_order_step, dtype=float)
        ticks = ticks[(ticks >= lower) & (ticks <= upper)]

    if include_unit_tick:
        ticks = np.unique(np.concatenate([ticks, np.array([1.0], dtype=float)]))
        ticks = ticks[(ticks >= lower) & (ticks <= upper)]

    if ticks.size == 0:
        middle_order = tick_order_step * np.round((np.log10(lower) + np.log10(upper)) / (2.0 * tick_order_step))
        tick = 10.0 ** middle_order
        ticks = np.array([tick], dtype=float) if lower <= tick <= upper else np.array([], dtype=float)

    return lower, upper, ticks


def _apply_rowwise_last_step_y_limits(
    axes: np.ndarray,
    gap_axes: Sequence[plt.Axes | None],
    n_items: int,
    ncols: int,
    y_tick_order_step: int = 3,
    include_unit_tick: bool = False,
) -> None:
    nrows = int(np.ceil(n_items / ncols))
    formatter = FuncFormatter(_plain_one_log_formatter)

    for row in range(nrows):
        row_start = row * ncols
        row_end = min((row + 1) * ncols, n_items)
        row_axes = list(axes[row_start:row_end])
        row_gap_axes = [gap_axes[idx] if idx < len(gap_axes) else None for idx in range(row_start, row_end)]
        lower, upper, ticks = _log_limits_and_ticks_for_last_step_row(
            row_axes,
            row_gap_axes,
            y_tick_order_step,
            include_unit_tick=include_unit_tick,
        )

        for idx in range(row_start, row_end):
            ax = axes[idx]
            ax.set_yscale("log")
            ax.set_ylim(lower, upper)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(formatter)

            if idx < len(gap_axes) and gap_axes[idx] is not None:
                ax_gap = gap_axes[idx]
                ax_gap.set_yscale("log")
                ax_gap.set_ylim(lower, upper)
                ax_gap.set_yticks(ticks)
                ax_gap.yaxis.set_major_formatter(formatter)


def _keep_last_step_axis_labels_only_on_outer_edges(
    axes: np.ndarray,
    n_items: int,
    ncols: int,
    gap_axes: Sequence[plt.Axes | None],
) -> None:
    nrows = int(np.ceil(n_items / ncols))
    for idx, ax in enumerate(axes[:n_items]):
        row = idx // ncols
        col = idx % ncols
        if row != nrows - 1:
            ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", labelleft=False)

        if idx < len(gap_axes) and gap_axes[idx] is not None:
            ax_gap = gap_axes[idx]
            if col != ncols - 1 and idx != n_items - 1:
                ax_gap.set_ylabel("")
                ax_gap.tick_params(axis="y", which="both", labelright=False)

def plot_spectral_gap_vs_n_table(
    fixed_betas: Sequence[float],
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    hspace: float = 0.42,
    wspace: float = 0.28,
    legend_y: float = -0.14,
    last_row_legend_y: float | None = -0.18,
    show_legend: bool = True,
    label_fontsize: int = 14,
    tick_labelsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 12,
    extrapolation_start_n: float | None = None,
    extrapolation_alpha_factor: float = 0.55,
    scatter_size: float = 22.0,
    band_alpha: float = 0.16,
    rowwise_y_limits: bool = True,
    y_limit_measurement_scale: float = 1.5,
    y_limit_max_extra_orders: float = 6.0,
    y_tick_order_step: int = 3,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = _make_table_axes(len(fixed_betas), ncols, figsize)

    n_plot_min = kwargs.setdefault("n_plot_min", 1)
    n_plot_max = kwargs.setdefault("n_plot_max", 20)

    if extrapolation_start_n is None:
        extrapolation_start_n = kwargs.get("n_fit_max", 10)
        if extrapolation_start_n is None:
            extrapolation_start_n = 10

    n_items = len(fixed_betas)
    nrows = int(np.ceil(n_items / ncols))

    for idx, (fixed_beta, ax) in enumerate(zip(fixed_betas, axes)):
        row = idx // ncols
        is_last_row = row == nrows - 1
        current_legend_y = legend_y
        if is_last_row and last_row_legend_y is not None:
            current_legend_y = last_row_legend_y

        plot_spectral_gap_vs_n(
            fixed_beta=fixed_beta,
            fig=fig,
            ax=ax,
            title=rf"$\beta={fixed_beta}$",
            show_legend=show_legend,
            **kwargs,
        )
        _split_fit_lines_at_n(
            ax,
            extrapolation_start_n=float(extrapolation_start_n),
            extrapolation_alpha_factor=extrapolation_alpha_factor,
        )
        _soften_plot_elements(ax, scatter_size=scatter_size, band_alpha=band_alpha)
        _apply_spectral_gap_n_x_scale(ax, n_plot_min=n_plot_min, n_plot_max=n_plot_max)
        if show_legend:
            _replace_legend_with_line_fit_rows(
                ax,
                ncol=3,
                y=current_legend_y,
                fontsize=legend_fontsize,
                label_formatter=_spectral_gap_n_legend_label,
            )
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    if rowwise_y_limits:
        _apply_rowwise_spectral_gap_n_y_limits_from_measurements(
            axes,
            len(fixed_betas),
            ncols,
            measurement_scale=y_limit_measurement_scale,
            max_extra_orders=y_limit_max_extra_orders,
            tick_order_step=y_tick_order_step,
        )

    _keep_single_axis_y_ticks_only_on_outer_edges(axes, len(fixed_betas), ncols)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(fixed_betas)]

def plot_spectral_gap_vs_beta_table(
    fixed_ns: Sequence[int],
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    hspace: float = 0.38,
    wspace: float = 0.28,
    legend_y: float = -0.18,
    show_legend: bool = False,
    label_fontsize: int = 14,
    tick_labelsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 12,
    add_inset: bool = True,
    inset_bounds: tuple[float, float, float, float] = (0.075, 0.075, 0.438, 0.369),
    miniature_scale: float = 1.0,
    inset_tick_labelsize: int = 8,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = _make_table_axes(len(fixed_ns), ncols, figsize)

    for fixed_n, ax in zip(fixed_ns, axes):
        plot_spectral_gap_vs_beta(fixed_n=fixed_n, fig=fig, ax=ax, title=rf"$n={fixed_n}$", show_legend=show_legend, **kwargs)
        _apply_spectral_gap_beta_y_scale(ax)
        if add_inset:
            scaled_inset_bounds = _scale_inset_bounds(inset_bounds, miniature_scale)
            _add_spectral_gap_beta_inset(fig, ax, fixed_n, scaled_inset_bounds, inset_tick_labelsize, **kwargs)
        if show_legend:
            _replace_legend_with_line_fit_rows(ax, ncol=3, y=legend_y, fontsize=legend_fontsize)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    _keep_single_axis_y_ticks_only_on_outer_edges(axes, len(fixed_ns), ncols)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(fixed_ns)]


def plot_last_step_classical_queries_and_spectral_gap_vs_n_table(
    betas: Sequence[float],
    epsilon: float,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    hspace: float = 0.56,
    wspace: float = 0.34,
    legend_y: float = -0.14,
    last_row_legend_y: float | None = -0.22,
    show_legend: bool = True,
    label_fontsize: int = 14,
    tick_labelsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 12,
    extrapolation_start_n: float | None = None,
    extrapolation_alpha_factor: float = 0.30,
    extrapolation_linewidth_factor: float = 0.85,
    query_linewidth: float = 2.0,
    gap_linewidth: float = 1.2,
    query_alpha: float = 0.95,
    gap_alpha: float = 0.60,
    legend_x_span: float = 0.54,
    legend_two_column_x_span: float = 0.34,
    legend_line_half_width: float = 0.050,
    scatter_size: float = 22.0,
    band_alpha: float = 0.16,
    rowwise_y_limits: bool = True,
    y_tick_order_step: int = 3,
    include_unit_y_tick: bool = True,
    remove_local_beta_threshold: float | None = 4.0,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray, list[plt.Axes | None]]:
    fig, axes = _make_table_axes(len(betas), ncols, figsize)
    gap_axes = []

    n_plot_min = kwargs.setdefault("n_plot_min", 1)
    n_plot_max = kwargs.setdefault("n_plot_max", 20)

    if extrapolation_start_n is None:
        extrapolation_start_n = kwargs.get("n_fit_max", 10)
        if extrapolation_start_n is None:
            extrapolation_start_n = 10

    n_items = len(betas)
    nrows = int(np.ceil(n_items / ncols))

    for idx, (beta, ax) in enumerate(zip(betas, axes)):
        row = idx // ncols
        is_last_row = row == nrows - 1
        current_legend_y = legend_y
        if is_last_row and last_row_legend_y is not None:
            current_legend_y = last_row_legend_y

        _, _, ax_gap = plot_last_step_classical_queries_and_spectral_gap_vs_n(
            beta=beta,
            epsilon=epsilon,
            fig=fig,
            ax=ax,
            title=rf"$\beta={beta}$, $\epsilon={epsilon:g}$",
            show_legend=show_legend,
            **kwargs,
        )

        remove_local = remove_local_beta_threshold is not None and float(beta) > float(remove_local_beta_threshold)
        if remove_local:
            _remove_local_move_artists(ax, ax_gap)

        _style_last_step_observable_lines(
            ax,
            ax_gap,
            query_linewidth=query_linewidth,
            gap_linewidth=gap_linewidth,
            query_alpha=query_alpha,
            gap_alpha=gap_alpha,
        )

        _split_fit_lines_at_n(
            ax,
            extrapolation_start_n=float(extrapolation_start_n),
            extrapolation_alpha_factor=extrapolation_alpha_factor,
            extrapolation_linewidth_factor=extrapolation_linewidth_factor,
        )
        if ax_gap is not None:
            _split_fit_lines_at_n(
                ax_gap,
                extrapolation_start_n=float(extrapolation_start_n),
                extrapolation_alpha_factor=extrapolation_alpha_factor,
                extrapolation_linewidth_factor=extrapolation_linewidth_factor,
            )

        _soften_plot_elements(ax, scatter_size=scatter_size, band_alpha=band_alpha)
        _apply_spectral_gap_n_x_scale(ax, n_plot_min=n_plot_min, n_plot_max=n_plot_max)

        if show_legend:
            _replace_last_step_legend_with_scaling_rows(
                ax,
                ncol=2 if remove_local else 3,
                y=current_legend_y,
                fontsize=legend_fontsize,
                remove_local=remove_local,
                query_linewidth=query_linewidth,
                gap_linewidth=gap_linewidth,
                query_alpha=query_alpha,
                gap_alpha=gap_alpha,
                line_half_width=legend_line_half_width,
                x_span=legend_x_span,
                two_column_x_span=legend_two_column_x_span,
            )

        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)
        if ax_gap is not None:
            ax_gap.yaxis.label.set_size(label_fontsize)
            ax_gap.tick_params(axis="y", which="major", labelsize=tick_labelsize)
            ax_gap.grid(False)
        gap_axes.append(ax_gap)

    if rowwise_y_limits:
        _apply_rowwise_last_step_y_limits(
            axes,
            gap_axes,
            len(betas),
            ncols,
            y_tick_order_step=y_tick_order_step,
            include_unit_tick=include_unit_y_tick,
        )

    _keep_last_step_axis_labels_only_on_outer_edges(axes, len(betas), ncols, gap_axes)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(betas)], gap_axes

def plot_annealing_classical_and_quantum_queries_vs_n_table(
    betas: Sequence[float],
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    hspace: float = 0.38,
    wspace: float = 0.28,
    legend_y: float = -0.18,
    show_legend: bool = False,
    label_fontsize: int = 14,
    tick_labelsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 12,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = _make_table_axes(len(betas), ncols, figsize)

    for beta, ax in zip(betas, axes):
        plot_annealing_classical_and_quantum_queries_vs_n(beta=beta, epsilon=epsilon, annealing_schedule_generator=annealing_schedule_generator, fig=fig, ax=ax, title=rf"$\beta_F={beta}$, $\epsilon={epsilon:g}$", show_legend=show_legend, **kwargs)
        if show_legend:
            _replace_legend_with_line_fit_rows(ax, ncol=3, y=legend_y, fontsize=legend_fontsize)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    _keep_axis_labels_only_on_outer_edges(axes, len(betas), ncols)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(betas)]
