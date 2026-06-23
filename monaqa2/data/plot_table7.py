from typing import Callable, Sequence
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext

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
    ax.grid(True, which="major", color="0.88", alpha=0.55, linewidth=0.7)
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


def _replace_legend_with_line_fit_rows(
    ax: plt.Axes,
    ncol: int = 3,
    y: float = -0.17,
    row_gap: float = 0.030,
    group_gap: float = 0.135,
    line_half_width: float = 0.055,
    fontsize: int = 12,
) -> None:
    legend = ax.get_legend()
    if legend is None:
        return

    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", None)
    if handles is None:
        return

    labels = [_fit_only_label(text.get_text()) for text in legend.get_texts()]
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


def plot_spectral_gap_vs_n_table(
    fixed_betas: Sequence[float],
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
    fig, axes = _make_table_axes(len(fixed_betas), ncols, figsize)

    for fixed_beta, ax in zip(fixed_betas, axes):
        plot_spectral_gap_vs_n(fixed_beta=fixed_beta, fig=fig, ax=ax, title=rf"$\beta={fixed_beta}$", show_legend=show_legend, **kwargs)
        if show_legend:
            _replace_legend_with_line_fit_rows(ax, ncol=3, y=legend_y, fontsize=legend_fontsize)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    _keep_axis_labels_only_on_outer_edges(axes, len(fixed_betas), ncols)
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
    inset_tick_labelsize: int = 8,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = _make_table_axes(len(fixed_ns), ncols, figsize)

    for fixed_n, ax in zip(fixed_ns, axes):
        plot_spectral_gap_vs_beta(fixed_n=fixed_n, fig=fig, ax=ax, title=rf"$n={fixed_n}$", show_legend=show_legend, **kwargs)
        _apply_spectral_gap_beta_y_scale(ax)
        if add_inset:
            _add_spectral_gap_beta_inset(fig, ax, fixed_n, inset_bounds, inset_tick_labelsize, **kwargs)
        if show_legend:
            _replace_legend_with_line_fit_rows(ax, ncol=3, y=legend_y, fontsize=legend_fontsize)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    _keep_axis_labels_only_on_outer_edges(axes, len(fixed_ns), ncols)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(fixed_ns)]


def plot_last_step_classical_queries_and_spectral_gap_vs_n_table(
    betas: Sequence[float],
    epsilon: float,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    hspace: float = 0.44,
    wspace: float = 0.34,
    legend_y: float = -0.18,
    show_legend: bool = False,
    label_fontsize: int = 14,
    tick_labelsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 12,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray, list[plt.Axes | None]]:
    fig, axes = _make_table_axes(len(betas), ncols, figsize)
    gap_axes = []

    for beta, ax in zip(betas, axes):
        _, _, ax_gap = plot_last_step_classical_queries_and_spectral_gap_vs_n(beta=beta, epsilon=epsilon, fig=fig, ax=ax, title=rf"$\beta={beta}$, $\epsilon={epsilon:g}$", show_legend=show_legend, **kwargs)
        if show_legend:
            _replace_legend_with_line_fit_rows(ax, ncol=3, y=legend_y, fontsize=legend_fontsize)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)
        if ax_gap is not None:
            ax_gap.yaxis.label.set_size(label_fontsize)
            ax_gap.tick_params(axis="y", which="major", labelsize=tick_labelsize)
        gap_axes.append(ax_gap)

    _keep_axis_labels_only_on_outer_edges(axes, len(betas), ncols, gap_axes)
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
