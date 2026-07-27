from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.runtime import get_annealing_queries_quantum_walks
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_n
from monaqa2.data.classical_query import get_classical_query_fit_by_n


CM = 1.0 / 2.54
PT_TO_CM = 2.54 / 72.0
APS_FIGURESTAR_WIDTH_CM = 17.8
APS_FIGURESTAR_HEIGHT_CM = 5.8

PROPOSAL_COLORS = {
    # Neutral family: near-black and gray
    "local1_classical": "#222222",
    "local1_quantum": "#999999",

    # Cold family: navy blue and cyan
    "uniform_classical": "#004488",
    "uniform_quantum": "#66CCEE",

    # Warm family: magenta and orange
    "layden_classical": "#EE7733",
    "layden_quantum": "#AA3377",
}

QUERY_SERIES = ("uniform_classical", "uniform_quantum", "layden_quantum")

QUERY_LABELS = {
    "uniform_classical": "Best classical walk",
    "uniform_quantum": "Quantized classical walk",
    "layden_quantum": "This work",
}

QUERY_COLORS = {
    "uniform_classical": PROPOSAL_COLORS["uniform_classical"],
    "uniform_quantum": PROPOSAL_COLORS["uniform_quantum"],
    "layden_quantum": PROPOSAL_COLORS["layden_quantum"],
}


def _n_grid(n_plot_min: int | None, n_plot_max: int | None) -> np.ndarray:
    n_plot_min = 3 if n_plot_min is None else int(n_plot_min)
    n_plot_max = 120 if n_plot_max is None else int(n_plot_max)
    if n_plot_max < n_plot_min:
        raise ValueError("n_plot_max must be greater than or equal to n_plot_min.")
    return np.arange(n_plot_min, n_plot_max + 1, dtype=int)


def _compute_query_curves(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    n_vals: np.ndarray,
    classical_query_file: Path,
    spectral_gap_file: Path,
    statistic: str,
    n_fit_min: int | None,
    n_fit_max: int | None,
    debug: bool = True,
) -> dict[str, np.ndarray]:
    curves = {key: [] for key in QUERY_SERIES}
    schedules_by_n = []
    for n in n_vals:
        schedule = [float(beta_t) for beta_t in annealing_schedule_generator(int(n), float(beta))]
        schedules_by_n.append(schedule)
        uniform_classical_total = 0.0
        uniform_gaps = []
        layden_gaps = []
        if debug:
            print(f"\n[debug] n={int(n)}, beta={float(beta):.12g}, epsilon={float(epsilon):.12g}, schedule={schedule}")
        for beta_t in schedule:
            A_q, b_q = get_classical_query_fit_by_n(proposal="uniform", a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            uniform_classical_contribution = float(A_q * np.exp(b_q * int(n)))
            uniform_classical_total += uniform_classical_contribution
            local1_A_g, local1_b_g = get_spectral_gap_fit_by_n(proposal="local1", a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            uniform_A_g, uniform_b_g = get_spectral_gap_fit_by_n(proposal="uniform", a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            layden_A_g, layden_b_g = get_spectral_gap_fit_by_n(proposal="layden", a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            local1_b_g = max(float(local1_b_g), 0.0)
            uniform_b_g = max(float(uniform_b_g), 0.0)
            layden_b_g = max(float(layden_b_g), 0.0)
            local1_gap = float(local1_A_g * np.exp(-local1_b_g * int(n)))
            uniform_gap = float(uniform_A_g * np.exp(-uniform_b_g * int(n)))
            layden_gap = float(layden_A_g * np.exp(-layden_b_g * int(n)))
            uniform_gaps.append(uniform_gap)
            layden_gaps.append(layden_gap)
            if debug:
                print(f"  beta_t={beta_t:.12g}")
                print(f"    uniform classical fit: A={float(A_q):.12g}, b/log2={float(b_q) / np.log(2):+.12g}, value={uniform_classical_contribution:.12g}")
                print(f"    local1 gap fit: A={float(local1_A_g):.12g}, b/log2={local1_b_g / np.log(2):+.12g}, gap={local1_gap:.12g}")
                print(f"    uniform gap fit: A={float(uniform_A_g):.12g}, b/log2={uniform_b_g / np.log(2):+.12g}, gap={uniform_gap:.12g}")
                print(f"    layden gap fit: A={float(layden_A_g):.12g}, b/log2={layden_b_g / np.log(2):+.12g}, gap={layden_gap:.12g}")
        curves["uniform_classical"].append(uniform_classical_total)
        uniform_quantum_queries = float(get_annealing_queries_quantum_walks(int(n), epsilon, uniform_gaps))
        layden_quantum_queries = float(get_annealing_queries_quantum_walks(int(n), epsilon, layden_gaps))
        if debug:
            print(f"  uniform_gaps={uniform_gaps}")
            print(f"  layden_gaps={layden_gaps}")
            print(f"  uniform_quantum_queries={uniform_quantum_queries:.12g}")
            print(f"  layden_quantum_queries={layden_quantum_queries:.12g}")
        curves["uniform_quantum"].append(uniform_quantum_queries)
        curves["layden_quantum"].append(layden_quantum_queries)
    curves = {key: np.asarray(values, dtype=float) for key, values in curves.items()}
    if debug:
        _debug_print_query_curves(n_vals, curves, schedules_by_n)
    return curves


def _debug_print_query_curves(n_vals: np.ndarray, curves: dict[str, np.ndarray], schedules_by_n: list[list[float]]) -> None:
    for key, values in curves.items():
        decreases = np.where(values[1:] < values[:-1])[0]
        if len(decreases):
            print(f"\n[debug] decreases for {QUERY_LABELS[key]}")
        for idx in decreases:
            n0 = int(n_vals[idx])
            n1 = int(n_vals[idx + 1])
            y0 = float(values[idx])
            y1 = float(values[idx + 1])
            ratio = y1 / y0 if y0 > 0.0 else np.nan
            print(f"  n {n0}->{n1}: {y0:.6g}->{y1:.6g} ratio={ratio:.6g} schedule_len={len(schedules_by_n[idx])}->{len(schedules_by_n[idx + 1])}")
            print(f"    previous schedule={schedules_by_n[idx]}")
            print(f"    current  schedule={schedules_by_n[idx + 1]}")


def _plot_query_curves(ax: plt.Axes, n_vals: np.ndarray, curves: dict[str, np.ndarray], line_width: float, line_alpha: float) -> tuple[list[plt.Line2D], list[str]]:
    handles = []
    labels = []
    for zorder, key in enumerate(QUERY_SERIES, start=2):
        (line,) = ax.plot(n_vals, curves[key], color=QUERY_COLORS[key], linewidth=line_width, linestyle="-", alpha=line_alpha, label=QUERY_LABELS[key], zorder=zorder)
        handles.append(line)
        labels.append(QUERY_LABELS[key])
    return handles, labels


def _finish_query_axis(ax: plt.Axes, n_vals: np.ndarray, title: str | None, xlabel: str | None, ylabel: str | None, show_numerical_boundary: bool, numerical_boundary_n: float | None, grid_color: str, grid_linewidth: float, x_right_padding: float = 0.0) -> None:
    ax.set_yscale("log")
    ax.set_xlim(float(n_vals[0]), float(n_vals[-1]) + max(0.0, float(x_right_padding)))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if show_numerical_boundary and numerical_boundary_n is not None and float(n_vals[0]) <= float(numerical_boundary_n) <= float(n_vals[-1]):
        ax.axvline(float(numerical_boundary_n), color="0.62", linewidth=0.55, linestyle=(0, (2.0, 2.0)), alpha=0.80, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    ax.grid(False, which="major", axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _positive_log_ylim_from_curves(curve_groups: Sequence[dict[str, np.ndarray]]) -> tuple[float, float] | None:
    values = []
    for curves in curve_groups:
        for key in QUERY_SERIES:
            arr = np.asarray(curves[key], dtype=float)
            values.append(arr[np.isfinite(arr) & (arr > 0.0)])
    values = [arr for arr in values if arr.size]
    if not values:
        return None
    merged = np.concatenate(values)
    y_min = float(np.min(merged))
    y_max = float(np.max(merged))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min <= 0.0 or y_max <= 0.0:
        return None
    if np.isclose(y_min, y_max):
        return y_min / 2.0, y_max * 2.0
    return 10.0 ** (np.log10(y_min) - 0.06 * (np.log10(y_max) - np.log10(y_min))), 10.0 ** (np.log10(y_max) + 0.06 * (np.log10(y_max) - np.log10(y_min)))


def _validate_legend_placement(legend_placement: str) -> None:
    if legend_placement not in {"top_left", "out"}:
        raise ValueError("legend_placement must be either 'top_left' or 'out'.")


def _format_last_value(value: float) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    exponent_int = int(exponent)
    exponent_text = f"{exponent_int:02d}" if exponent_int >= 0 else f"-{abs(exponent_int):02d}"
    return rf"${mantissa}e{exponent_text}$"


def _draw_last_value_labels(ax: plt.Axes, n_vals: np.ndarray, curves: dict[str, np.ndarray], fontsize: float = 7.0) -> None:
    x_last = float(n_vals[-1])
    for key in QUERY_SERIES:
        values = np.asarray(curves[key], dtype=float)
        if values.size == 0 or not np.isfinite(values[-1]) or values[-1] <= 0.0:
            continue
        ax.annotate(_format_last_value(float(values[-1])), xy=(x_last, float(values[-1])), xytext=(-4.0, 0.0), textcoords="offset points", ha="right", va="center", fontsize=fontsize, color=QUERY_COLORS[key], clip_on=True, zorder=20)


def _add_single_axis_legend(fig: plt.Figure, ax: plt.Axes, handles: list[plt.Line2D], labels: list[str], legend_placement: str, legend_y_shift: float) -> None:
    if legend_placement == "top_left":
        ax.legend(handles, labels, frameon=False, loc="upper left", ncol=1, borderaxespad=0.35, handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    else:
        ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.0, legend_y_shift), ncol=1, borderaxespad=0.0, handlelength=2.2, columnspacing=1.0, labelspacing=0.35)


def _add_zoom_inset(
    ax: plt.Axes,
    n_vals: np.ndarray,
    curves: dict[str, np.ndarray],
    line_width: float,
    line_alpha: float,
    grid_color: str,
    grid_linewidth: float,
    zoom_xlim: tuple[float, float],
    zoom_ylim: tuple[float, float],
    zoom_bbox: tuple[float, float, float, float],
    zoom_tick_fontsize: float | None,
    show_zoom_in_border: bool,
) -> plt.Axes:
    inset_ax = ax.inset_axes(zoom_bbox)
    _plot_query_curves(inset_ax, n_vals, curves, line_width, line_alpha)
    inset_ax.set_yscale("log")
    inset_ax.set_xlim(*zoom_xlim)
    inset_ax.set_ylim(*zoom_ylim)
    inset_ax.set_xticks([int(zoom_xlim[0]), int(zoom_xlim[1])])
    inset_ax.set_yticks([1e3, 1e4])
    inset_ax.set_yticklabels([r"$10^3$", r"$10^4$"])
    tick_fontsize = max(1.0, float(plt.rcParams.get("font.size", 10.0)) - 1.0) if zoom_tick_fontsize is None else float(zoom_tick_fontsize)
    inset_ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, length=2.4, pad=1.4)
    inset_ax.set_axisbelow(True)
    inset_ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    inset_ax.grid(False, which="major", axis="x")
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    if show_zoom_in_border:
        for spine in ["top", "right"]:
            inset_ax.spines[spine].set_visible(True)
            inset_ax.spines[spine].set_color("black")
            inset_ax.spines[spine].set_linewidth(0.65)
    else:
        for spine in ["top", "right"]:
            inset_ax.spines[spine].set_visible(False)
    return inset_ax


def plot_annealing_classical_and_quantum_queries_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    debug: bool = False,
    show_numerical_boundary: bool = False,
    add_regime_separator: bool = False,
    numerical_boundary_n: float | None = None,
    legend_placement: str = "top_left",
    legend_y_shift: float = -0.22,
    show_last_value: bool = False,
    last_value_fontsize: float = 7.0,
    xlabel_labelpad: float = 4,
    line_width: float = 2.0,
    line_alpha: float = 0.94,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    x_right_padding: float = 0.0,
    show_zoom_in: bool = True,
    zoom_xlim: tuple[float, float] = (3.0, 10.0),
    zoom_ylim: tuple[float, float] = (1e2, 1e4),
    zoom_bbox: tuple[float, float, float, float] = (0.10, 0.57, 0.42, 0.36),
    zoom_tick_fontsize: float | None = None,
    show_zoom_in_border: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot compact-no-Layden annealing query estimates versus n."""
    _validate_legend_placement(legend_placement)
    n_vals = _n_grid(n_plot_min, n_plot_max)
    show_numerical_boundary = show_numerical_boundary or add_regime_separator
    numerical_boundary_n = float(n_fit_max) if numerical_boundary_n is None and n_fit_max is not None else numerical_boundary_n
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8.5 * CM, 5.6 * CM))
        if legend_placement == "out":
            fig.subplots_adjust(left=0.18, right=0.985, top=0.985, bottom=0.43)
        else:
            fig.subplots_adjust(left=0.18, right=0.985, top=0.985, bottom=0.17)
    curves = _compute_query_curves(beta, epsilon, annealing_schedule_generator, n_vals, classical_query_file, spectral_gap_file, statistic, n_fit_min, n_fit_max, debug)
    handles, labels = _plot_query_curves(ax, n_vals, curves, line_width, line_alpha)
    ylim = _positive_log_ylim_from_curves([curves])
    if ylim is not None:
        ax.set_ylim(*ylim)
    _finish_query_axis(ax, n_vals, title, r"$n$", r"Queries", show_numerical_boundary, numerical_boundary_n, grid_color, grid_linewidth, x_right_padding=x_right_padding)
    ax.xaxis.labelpad = xlabel_labelpad
    if show_last_value:
        _draw_last_value_labels(ax, n_vals, curves, fontsize=last_value_fontsize)
    if show_zoom_in:
        _add_zoom_inset(ax, n_vals, curves, line_width, line_alpha, grid_color, grid_linewidth, zoom_xlim, zoom_ylim, zoom_bbox, zoom_tick_fontsize, show_zoom_in_border)
    if show_legend:
        _add_single_axis_legend(fig, ax, handles, labels, legend_placement, legend_y_shift)
    return fig, ax
