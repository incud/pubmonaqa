import re
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from monaqa2.data.classical_query import get_classical_query_fit_by_n, get_classical_query_stats
from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.runtime import (
    get_annealing_queries_quantum_walks,
    get_one_step_quantum_walk_queries,
    get_annealing_time_classical_walk_local,
    get_annealing_time_classical_walk_qemc,
    get_annealing_time_classical_walk_uniform,
    get_annealing_time_quantum_walk_local,
    get_annealing_time_quantum_walk_qemc,
    get_annealing_time_quantum_walk_uniform,
    split_quantum_error_budget,
)
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_beta, get_spectral_gap_fit_by_n, get_spectral_gap_stats

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.0


PROPOSALS_SORTED = ["local1", "uniform", "layden"]


PROPOSAL_LABELS = {
    "layden": "Quantum enhanced",
    "local1": "Local spin-flip",
    "local2": "Local spin-flip (double)",
    "local3": "Local spin-flip (triple)",
    "uniform": "Uniform",
    "qemc": "Quantum enhanced (best hyperparameters)",
}

PROPOSAL_COLORS = {
    "local1_classical": "#777777",
    "local1_quantum": "#000000",
    "uniform_classical": "#66CCEE",
    "uniform_quantum": "#4477AA",
    "layden_classical": "#CC6677",
    "layden_quantum": "#AA3377",
}


MOVE_LEGEND_LABELS = ["local", "uniform", "dynamics"]


MOVE_LABELS = {
    "local1": "local move",
    "uniform": "uniform move",
    "layden": "quantum-enhanced move",
}


def _proposal_color(proposal: str, kind: str) -> str:
    key = f"{proposal}_{kind}"
    try:
        return PROPOSAL_COLORS[key]
    except:
        raise ValueError(f"Cannot find proposal color with {key=}. Available: {PROPOSAL_COLORS.keys()}")


def _proposal_base_color(proposal: str) -> str:
    return _proposal_color(proposal, "quantum")


def _walk_label(proposal: str, kind: str) -> str:
    walk = "Classical walk" if kind == "classical" else "Quantum walk"
    return f"{walk}, {MOVE_LABELS.get(proposal, PROPOSAL_LABELS[proposal].lower())}"


def _compact_spectral_gap_points(proposal: str, beta: float, statistic: str = "mean+std", in_file: Path = SPECTRAL_GAP_FILE):
    table = get_spectral_gap_stats(proposal=proposal, a=np.inf, in_file=in_file, statistic=statistic)
    table = table[np.isclose(table["beta"].astype(float), float(beta)) & (table["count"].astype(int) >= 1) & np.isfinite(table["n"].astype(float)) & np.isfinite(table["center"].astype(float)) & (table["center"].astype(float) > 0.0)].copy()
    table["n"] = table["n"].astype(float)
    table["center"] = table["center"].astype(float)
    table["spread"] = table["spread"].fillna(0.0).astype(float)
    table = table.sort_values("n")
    return table["n"], table["center"], table["spread"]
    

def _compact_classical_query_points(proposal: str, beta: float, epsilon: float, statistic: str = "mean+std", in_file: Path = CLASSICAL_QUERY_FILE, min_count: int = 1):
    table = get_classical_query_stats(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, in_file=in_file, statistic=statistic)
    table = table[np.isclose(table["beta"].astype(float), float(beta)) & (table["count"].astype(int) >= min_count) & np.isfinite(table["n"].astype(float)) & np.isfinite(table["center"].astype(float)) & (table["center"].astype(float) > 0.0)].copy()
    table["n"] = table["n"].astype(float)
    table["center"] = table["center"].astype(float)
    table["spread"] = table["spread"].fillna(0.0).astype(float)
    table = table.sort_values("n")
    return table["n"], table["center"], table["spread"]


def _positive_band(center: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = center - spread
    upper = center + spread
    positive = np.concatenate([center[center > 0.0], upper[upper > 0.0]])
    floor = np.min(positive) * 1e-2 if positive.size else np.finfo(float).tiny
    return np.maximum(lower, floor), np.maximum(upper, floor)




def _n_plot_grid(n_plot_min: int | float, n_plot_max: int | float) -> np.ndarray:
    """Return the integer n-grid used by the annealing plots."""
    return np.arange(int(np.ceil(n_plot_min)), int(np.floor(n_plot_max)) + 1, dtype=float)


def make_prefix_stable_schedule_generator(
    beta_max: float,
    n_ref: int | float,
    base_schedule_generator: Callable[[int | float, float], list[float]],
) -> Callable[[int | float, float], list[float]]:
    """Return a schedule generator using one fixed master beta grid.

    The returned function ignores the input ``n`` for the internal beta grid.
    For any requested final beta, it returns all master points below that beta
    and then appends the requested beta as the final capped point. Therefore,
    schedules at different final beta values share the same prefix, up to the
    last capped point.

    :param beta_max: Largest beta initially used to build the master schedule.
    :param n_ref: Reference n used to build the master schedule, usually the
        largest n in the plot.
    :param base_schedule_generator: Original adaptive schedule generator.
    :return: Prefix-stable schedule generator.
    """
    n_ref = float(n_ref)
    master_beta_max = float(beta_max)
    master_schedule = [float(b) for b in base_schedule_generator(n_ref, master_beta_max)]

    def schedule(n: int | float, beta: float) -> list[float]:
        nonlocal master_beta_max, master_schedule

        beta = float(beta)
        if beta <= 0.0:
            return []

        if beta > master_beta_max and not np.isclose(beta, master_beta_max):
            master_beta_max = beta
            master_schedule = [float(b) for b in base_schedule_generator(n_ref, master_beta_max)]

        result = []
        for b in master_schedule:
            b = float(b)
            if b < beta and not np.isclose(b, beta):
                if not result or not np.isclose(result[-1], b):
                    result.append(b)

        if not result or not np.isclose(result[-1], beta):
            result.append(beta)

        return result

    return schedule


def _schedule_change_positions(n_vals: np.ndarray, beta: float, annealing_schedule_generator: Callable[[int, float], list[float]]) -> list[float]:
    """Return n values where the number of charged temperatures changes."""
    positions = []
    prev_len = None
    for n in n_vals:
        length = len(annealing_schedule_generator(float(n), beta))
        if prev_len is not None and length != prev_len:
            positions.append(float(n))
        prev_len = length
    return positions


def _draw_schedule_change_lines(ax: plt.Axes, positions: list[float]) -> None:
    """Draw very thin grey vertical lines at schedule-length changes."""
    for n in positions:
        ax.axvline(n, color="0.72", linewidth=0.45, linestyle="-", alpha=0.50, zorder=0)


def _plot_with_projection_style(
    ax: plt.Axes,
    n_vals: np.ndarray,
    y_vals: list[float],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    label: str,
    n_fit_max: int | None,
    zorder: int,
):
    """Plot a curve with solid calibrated and projected parts."""
    y = np.asarray(y_vals, dtype=float)
    if n_fit_max is None:
        (line,) = ax.plot(n_vals, y, color=color, linewidth=linewidth, linestyle="-", alpha=alpha, label=label, zorder=zorder)
        return line

    calibrated = n_vals <= float(n_fit_max)
    projected = n_vals >= float(n_fit_max)

    line = None
    if np.any(calibrated):
        (line,) = ax.plot(n_vals[calibrated], y[calibrated], color=color, linewidth=linewidth, linestyle="-", alpha=alpha, label=label, zorder=zorder)
    if np.any(projected):
        (proj_line,) = ax.plot(n_vals[projected], y[projected], color=color, linewidth=linewidth, linestyle="-", alpha=alpha, label=None if line is not None else label, zorder=zorder)
        if line is None:
            line = proj_line
    return line


def _debug_print_series_drops(
    proposal: str,
    name: str,
    n_vals: np.ndarray,
    values: list[float],
    records: list[dict],
    *,
    rtol: float = 1e-10,
) -> None:
    """Print decreases in a plotted series and the local fit contributions causing them."""
    arr = np.asarray(values, dtype=float)
    printed = False
    for i in range(1, len(arr)):
        if np.isfinite(arr[i - 1]) and np.isfinite(arr[i]) and arr[i] < arr[i - 1] * (1.0 - rtol):
            if not printed:
                print(f"\n[debug] decreases for {proposal} {name}")
                printed = True
            prev_record = records[i - 1]
            curr_record = records[i]
            print(
                f"  n {n_vals[i - 1]:.0f}->{n_vals[i]:.0f}: "
                f"{arr[i - 1]:.6g}->{arr[i]:.6g} "
                f"ratio={arr[i] / arr[i - 1]:.6g} "
                f"schedule_len={len(prev_record['schedule'])}->{len(curr_record['schedule'])}"
            )
            if prev_record["schedule"] != curr_record["schedule"]:
                print(f"    previous schedule={prev_record['schedule']}")
                print(f"    current  schedule={curr_record['schedule']}")
            for row in curr_record["steps"]:
                print(
                    f"    beta_t={row['beta_t']:.12g} "
                    f"b_q/log2={row['b_q'] / np.log(2):+.6g} "
                    f"q={row['q_contrib']:.6g} "
                    f"b_g/log2={row['b_g'] / np.log(2):+.6g} "
                    f"delta={row['delta']:.6g}"
                )


def _debug_print_schedule_changes(n_vals: np.ndarray, beta: float, annealing_schedule_generator: Callable[[int, float], list[float]]) -> None:
    """Print all n values where the charged schedule length changes."""
    prev_schedule = None
    for n in n_vals:
        schedule = annealing_schedule_generator(float(n), beta)
        if prev_schedule is not None and len(schedule) != len(prev_schedule):
            direction = "increases" if len(schedule) > len(prev_schedule) else "decreases"
            print(
                f"[debug] schedule length {direction} at n={n:.0f}: "
                f"{len(prev_schedule)} -> {len(schedule)}"
            )
            print(f"        previous={prev_schedule}")
            print(f"        current ={schedule}")
        prev_schedule = schedule


def plot_spectral_gap_vs_n(
    fixed_beta: float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot spectral gap versus n for fixed beta.

    Scatter points show the statistic center over instances.
    Transparent bands show center +/- spread.
    Solid lines show fits delta(n) = A * exp(-b n).
    """
    if n_plot_min is None or n_plot_max is None:
        n_plot_min = 1
        n_plot_max = 100

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

    n_grid = np.linspace(float(n_plot_min), float(n_plot_max), 300)
    legend_handles = {}
    legend_labels = {}

    for proposal in PROPOSALS_SORTED:

        color = _proposal_base_color(proposal)

        # Scatter points of the spectral gap
        n_vals, center, spread = _compact_spectral_gap_points(proposal, fixed_beta, statistic, in_file)
        if show_spread:
            lower, upper = _positive_band(center, spread)
            ax.fill_between(n_vals, lower, upper, color=color, alpha=0.25, linewidth=0.0, zorder=1)
        ax.scatter(n_vals, center, s=36, color=color, edgecolors="none", alpha=0.95, zorder=3)

        # Fitting line
        A, b = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=fixed_beta, in_file=in_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        delta_fit = A * np.exp(-b * n_grid)
        (fit_line,) = ax.plot(n_grid, delta_fit, color=color, linewidth=2.0, linestyle="-", alpha=0.90, zorder=2)

        # Legend
        legend_handles[proposal] = fit_line
        legend_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]}: fit ${A:.3f} \times 2^{{-{b / np.log(2):.3f} n}}$"
    
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Spectral gap $\delta$")
    ax.set_title(title if title is not None else rf"Spectral gap over instances, $\beta={fixed_beta}$, statistic={statistic}")

    ax.set_axisbelow(True)
    ax.grid(False)

    if show_legend:
        handles = [legend_handles[p] for p in PROPOSALS_SORTED if p in legend_handles]
        labels = [legend_labels[p] for p in PROPOSALS_SORTED if p in legend_labels]
        ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax


def plot_spectral_gap_vs_beta(
    fixed_n: int,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    beta_max: float = 100.0,
    beta_step: float = 0.01,
    beta_plot_min: float | None = None,
    beta_plot_max: float | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot spectral gap versus beta for fixed n.

    Scatter points show the statistic center over instances.
    Transparent bands show center +/- spread.
    Solid lines show log-linear interpolation of delta(beta).
    """
    beta_plot_min = beta_step if beta_plot_min is None else float(beta_plot_min)
    beta_plot_max = beta_max if beta_plot_max is None else float(beta_plot_max)

    if beta_plot_min <= 0.0:
        raise ValueError("beta_plot_min must be positive because the x-axis is logarithmic.")

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

    legend_handles = {}
    legend_labels = {}

    for proposal in PROPOSALS_SORTED:

        color = _proposal_base_color(proposal)

        table = get_spectral_gap_stats(proposal=proposal, a=np.inf, in_file=in_file, statistic=statistic)
        table = table[
            (table["n"].astype(int) == int(fixed_n))
            & (table["count"].astype(int) >= 1)
            & np.isfinite(table["beta"].astype(float))
            & np.isfinite(table["center"].astype(float))
            & (table["beta"].astype(float) > 0.0)
            & (table["center"].astype(float) > 0.0)
        ].copy()

        table["beta"] = table["beta"].astype(float)
        table["center"] = table["center"].astype(float)
        table["spread"] = table["spread"].fillna(0.0).astype(float)
        table = table.sort_values("beta")

        table_plot = table[(table["beta"] >= beta_plot_min) & (table["beta"] <= beta_plot_max)]

        if not table_plot.empty:
            beta_vals = table_plot["beta"].to_numpy(dtype=float)
            center = table_plot["center"].to_numpy(dtype=float)
            spread = table_plot["spread"].to_numpy(dtype=float)

            if show_spread:
                lower, upper = _positive_band(center, spread)
                ax.fill_between(beta_vals, lower, upper, color=color, alpha=0.25, linewidth=0.0, zorder=1)

            ax.scatter(beta_vals, center, s=36, color=color, edgecolors="none", alpha=0.95, zorder=3)

        grid = get_spectral_gap_fit_by_beta(proposal=proposal, a=np.inf, fixed_n=fixed_n, beta_max=beta_max, beta_step=beta_step, spectral_gap_file=in_file, statistic=statistic)
        grid = grid[(grid[:, 0] >= beta_plot_min) & (grid[:, 0] <= beta_plot_max) & (grid[:, 1] > 0.0)]

        (line,) = ax.plot(grid[:, 0], grid[:, 1], color=color, linewidth=2.0, linestyle="-", alpha=0.90, zorder=2)

        legend_handles[proposal] = line
        legend_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]}: log-linear interpolation"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(beta_plot_min, beta_plot_max)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"Spectral gap $\delta$")
    ax.set_title(title if title is not None else rf"Spectral gap over instances, $n={fixed_n}$, statistic={statistic}")

    ax.set_axisbelow(True)
    ax.grid(False)

    if show_legend:
        handles = [legend_handles[p] for p in PROPOSALS_SORTED if p in legend_handles]
        labels = [legend_labels[p] for p in PROPOSALS_SORTED if p in legend_labels]
        ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax


def plot_last_step_classical_queries_and_spectral_gap_vs_n(
    beta: float,
    epsilon: float,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    show_inverse_gap: bool = True,
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes, plt.Axes | None]:
    """
    Plot classical queries versus n for fixed beta and epsilon. This is for the last
    step of the annealing only (guaranteed warm start).

    Scatter points show the statistic center over instances.
    Transparent bands show center +/- spread.
    Solid lines show fits T(n) = A exp(b n).
    Dashed lines show inverse spectral-gap fits 1/delta(n), if enabled.
    """
    if n_plot_min is None or n_plot_max is None:
        n_plot_min = 1
        n_plot_max = 100

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax_gap = ax.twinx() if show_inverse_gap else None
    n_grid = np.linspace(float(n_plot_min), float(n_plot_max), 300)

    query_handles = {}
    query_labels = {}
    gap_handles = {}
    gap_labels = {}

    for proposal in PROPOSALS_SORTED:

        color = _proposal_base_color(proposal)

        # Scatter points of the classical query count
        n_vals, center, spread = _compact_classical_query_points(proposal, beta, epsilon, statistic, classical_query_file, min_count)
        if show_spread:
            lower, upper = _positive_band(center, spread)
            ax.fill_between(n_vals, lower, upper, color=color, alpha=0.20, linewidth=0.0, zorder=1)
        ax.scatter(n_vals, center, s=36, color=color, edgecolors="none", alpha=0.95, zorder=3)

        # Fitting line for the classical query count
        A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        query_fit = A_q * np.exp(b_q * n_grid)
        (query_line,) = ax.plot(n_grid, query_fit, color=color, linewidth=2.0, linestyle="-", alpha=0.90, zorder=2)

        # Legend
        query_handles[proposal] = query_line
        query_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} queries: ${A_q:.3g} \times 2^{{{b_q / np.log(2):.3f} n}}$"

        # Optional inverse spectral-gap fitting line
        if show_inverse_gap and ax_gap is not None:
            A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            b_g = max(float(b_g), 0.0)
            inv_gap_fit = (1.0 / A_g) * np.exp(b_g * n_grid)
            (gap_line,) = ax_gap.plot(n_grid, inv_gap_fit, color=color, linewidth=2.0, linestyle="--", alpha=0.90, zorder=2)

            # Legend
            gap_handles[proposal] = gap_line
            gap_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} inverse gap: ${1.0 / A_g:.3g} \times 2^{{{b_g / np.log(2):.3f} n}}$"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Classical queries")
    ax.set_title(title if title is not None else rf"Classical queries, $\beta={beta}$, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(False)

    handles = [query_handles[p] for p in PROPOSALS_SORTED if p in query_handles]
    labels = [query_labels[p] for p in PROPOSALS_SORTED if p in query_labels]

    if show_inverse_gap and ax_gap is not None:
        ax_gap.set_yscale("log")
        ax_gap.set_ylabel(r"Inverse spectral gap $1/\delta$")
        ax_gap.grid(False)

        left_ylim = ax.get_ylim()
        right_ylim = ax_gap.get_ylim()
        shared_ylim = (min(left_ylim[0], right_ylim[0]), max(left_ylim[1], right_ylim[1]))
        ax.set_ylim(shared_ylim)
        ax_gap.set_ylim(shared_ylim)

        handles += [gap_handles[p] for p in PROPOSALS_SORTED if p in gap_handles]
        labels += [gap_labels[p] for p in PROPOSALS_SORTED if p in gap_labels]

    if show_legend:
        ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax, ax_gap


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
    schedule_diagnostic_generator: Callable[[int, float], list[float]] | None = None,
    show_schedule_panel: bool = True,
    show_schedule_vertical_lines: bool = True,
    legend_y_shift: float = -0.82,
    xlabel_labelpad: float = 18,
    mode: str = "full",
    legend_placement: str = "legend_out",
    line_width: float = 2.0,
    projected_alpha: float = 0.58,
    calibrated_alpha: float = 0.94,
    line_label_x_fraction: float = 0.82,
    line_label_y_multiplier: float = 1.08,
    line_label_fontsize: float = 9.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot total annealing classical-query estimates and quantum-walk query estimates versus n.

    :param beta: Final inverse temperature.
    :param epsilon: Target TV-distance error.
    :param annealing_schedule_generator: Schedule generator used for the plotted curves.
    :param classical_query_file: Classical-query data file.
    :param spectral_gap_file: Spectral-gap data file.
    :param statistic: Statistic used in the aggregate tables.
    :param n_fit_min: Minimum n used in the exponential fits.
    :param n_fit_max: Maximum n used in the exponential fits.
    :param n_plot_min: Minimum n shown in the plot.
    :param n_plot_max: Maximum n shown in the plot.
    :param fig: Optional figure.
    :param ax: Optional main axis.
    :param title: Optional title. If None, no title is drawn.
    :param show_legend: Whether to show labels, either as an external legend or line labels.
    :param debug: If True, print schedule changes and curve decreases.
    :param schedule_diagnostic_generator: Schedule generator used only for the lower schedule panel. If None, the function uses ``annealing_schedule_generator.base_schedule_generator`` when available, otherwise ``annealing_schedule_generator`` itself.
    :param show_schedule_panel: Whether to show the lower schedule-length panel.
    :param show_schedule_vertical_lines: Whether to show vertical lines at schedule-length changes.
    :param legend_y_shift: Vertical legend anchor. More negative moves the legend farther down.
    :param xlabel_labelpad: Padding for the x-axis label.
    :param mode: Either ``"full"``, ``"compact"``, or ``"compact_no_layden"``.
    :param legend_placement: Either ``"legend_out"`` for an external legend or ``"legend_line"`` for direct labels on the curves.
    :param line_width: Common linewidth for all query curves.
    :param projected_alpha: Alpha used for the projected part after ``n_fit_max``.
    :param calibrated_alpha: Alpha used for the calibrated part up to ``n_fit_max``.
    :param line_label_x_fraction: Relative x-position used for direct line labels when ``legend_placement="legend_line"``.
    :param line_label_y_multiplier: Multiplicative vertical offset for direct line labels.
    :param line_label_fontsize: Font size for direct line labels.
    :return: Tuple ``(fig, ax)``.
    """
    if mode not in {"full", "compact", "compact_no_layden"}:
        raise ValueError(f"Unknown mode={mode}. Expected 'full', 'compact', or 'compact_no_layden'.")
    if legend_placement not in {"legend_out", "legend_line"}:
        raise ValueError(f"Unknown legend_placement={legend_placement}. Expected 'legend_out' or 'legend_line'.")

    if n_plot_min is None:
        n_plot_min = 3
    if n_plot_max is None:
        n_plot_max = 120

    n_vals = np.arange(int(n_plot_min), int(n_plot_max) + 1, dtype=int)

    if mode == "full":
        proposals_to_plot = [proposal for proposal in PROPOSALS_SORTED if proposal in {"local1", "uniform", "layden"}]
    elif mode == "compact":
        proposals_to_plot = ["uniform", "layden"]
    else:
        proposals_to_plot = ["uniform", "layden"]

    hide_classical_layden = mode == "compact_no_layden"

    if schedule_diagnostic_generator is None:
        schedule_diagnostic_generator = getattr(annealing_schedule_generator, "base_schedule_generator", annealing_schedule_generator)

    if fig is None or ax is None:
        if show_schedule_panel:
            fig, (ax, ax_sched) = plt.subplots(2, 1, figsize=(8.5, 5.45), sharex=True, gridspec_kw={"height_ratios": [4.0, 0.75], "hspace": 0.06})
            fig.subplots_adjust(bottom=0.30)
        else:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            fig.subplots_adjust(bottom=0.26)
            ax_sched = None
    else:
        ax_sched = None

    def _split_projection(n_values: np.ndarray, y_values: np.ndarray):
        if n_fit_max is None:
            return n_values, y_values, np.array([], dtype=int), np.array([], dtype=float)
        fit_mask = n_values <= int(n_fit_max)
        proj_mask = n_values >= int(n_fit_max)
        return n_values[fit_mask], y_values[fit_mask], n_values[proj_mask], y_values[proj_mask]

    def _plot_curve_with_projection(y_values: list[float], color: str, label: str, zorder: int):
        y_values = np.asarray(y_values, dtype=float)
        n_fit, y_fit, n_proj, y_proj = _split_projection(n_vals, y_values)
        line = None
        if len(n_fit):
            (line,) = ax.plot(n_fit, y_fit, color=color, linewidth=line_width, linestyle="-", alpha=calibrated_alpha, zorder=zorder, label=label)
        if len(n_proj):
            (proj_line,) = ax.plot(n_proj, y_proj, color=color, linewidth=line_width, linestyle="-", alpha=projected_alpha, zorder=zorder, label=label if line is None else None)
            if line is None:
                line = proj_line
        return line

    def _schedule_diagnostics(schedule_generator):
        schedules = []
        lengths = []
        for n in n_vals:
            schedule = schedule_generator(int(n), beta)
            schedules.append(schedule)
            lengths.append(len(schedule))
        lengths = np.asarray(lengths, dtype=int)
        change_positions = []
        for i in range(1, len(n_vals)):
            if lengths[i] != lengths[i - 1]:
                change_positions.append(int(n_vals[i]))
        return schedules, lengths, change_positions

    def _legend_ncol() -> int:
        if mode == "full":
            return 3
        if mode == "compact":
            return 2
        return 3

    def _compact_no_layden_label(proposal: str, kind: str) -> str:
        if proposal == "uniform" and kind == "classical":
            return "Best classical"
        if proposal == "uniform" and kind == "quantum":
            return "Quantized best classical"
        if proposal == "layden" and kind == "quantum":
            return "Our approach"
        return _walk_label(proposal, kind)

    def _curve_label(proposal: str, kind: str) -> str:
        if mode == "compact_no_layden":
            return _compact_no_layden_label(proposal, kind)
        return _walk_label(proposal, kind)

    def _draw_line_labels(line_records: list[tuple[plt.Line2D, str, str, np.ndarray]]) -> None:
        if not line_records:
            return
        x_left, x_right = float(n_plot_min), float(n_plot_max)
        x_span = x_right - x_left
        if x_span <= 0.0:
            return
        label_x = x_left + float(line_label_x_fraction) * x_span
        dx = max(1.0, 0.035 * x_span)
        y_bottom, y_top = ax.get_ylim()
        log_y_min = np.log10(y_bottom)
        log_y_max = np.log10(y_top)

        for _, label, color, values in sorted(line_records, key=lambda item: item[3][-1]):
            values = np.asarray(values, dtype=float)
            valid = np.isfinite(values) & (values > 0.0) & np.isfinite(n_vals.astype(float))
            if np.count_nonzero(valid) < 2:
                continue

            x_data = n_vals.astype(float)[valid]
            log_y_data = np.log10(values[valid])
            x_label = float(np.clip(label_x, x_data[0], x_data[-1]))
            x0 = float(np.clip(x_label - dx, x_data[0], x_data[-1]))
            x1 = float(np.clip(x_label + dx, x_data[0], x_data[-1]))
            if np.isclose(x0, x1):
                continue

            log_y_curve = float(np.interp(x_label, x_data, log_y_data))
            log_y_label = log_y_curve + np.log10(float(line_label_y_multiplier))
            log_y_label = float(np.clip(log_y_label, log_y_min + 0.025, log_y_max - 0.025))

            y0 = 10.0 ** float(np.interp(x0, x_data, log_y_data))
            y1 = 10.0 ** float(np.interp(x1, x_data, log_y_data))
            p0 = ax.transData.transform((x0, y0))
            p1 = ax.transData.transform((x1, y1))
            angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))

            ax.text(x_label, 10.0 ** log_y_label, label, color=color, fontsize=line_label_fontsize, ha="center", va="bottom", rotation=angle, rotation_mode="anchor", clip_on=True, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.6})

    diagnostic_schedules, diagnostic_lengths, diagnostic_change_positions = _schedule_diagnostics(schedule_diagnostic_generator)

    if debug:
        if schedule_diagnostic_generator is annealing_schedule_generator:
            print("[debug] schedule panel uses the plotted schedule generator")
        else:
            print("[debug] schedule panel uses a separate diagnostic schedule generator")
        print(f"[debug] schedule-change positions: {diagnostic_change_positions}")
        for i in range(1, len(n_vals)):
            if diagnostic_lengths[i] != diagnostic_lengths[i - 1]:
                print(f"[debug] schedule length changes at n={n_vals[i]}: {diagnostic_lengths[i - 1]} -> {diagnostic_lengths[i]}")
                print(f"        previous={diagnostic_schedules[i - 1]}")
                print(f"        current ={diagnostic_schedules[i]}")

    if show_schedule_vertical_lines:
        for x in diagnostic_change_positions:
            ax.axvline(x, color="0.72", linewidth=0.45, alpha=0.95, zorder=0)

    handles = []
    labels = []
    line_records = []

    for proposal in proposals_to_plot:
        classical_queries = []
        quantum_queries = []
        plotted_schedules = []

        for n in n_vals:
            schedule = annealing_schedule_generator(int(n), beta)
            plotted_schedules.append(schedule)
            classical_total = 0.0
            spectral_gaps = []

            for beta_t in schedule:
                A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                classical_total += A_q * np.exp(b_q * n)
                A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                b_g = max(float(b_g), 0.0)
                spectral_gaps.append(A_g * np.exp(-b_g * n))

            classical_queries.append(classical_total)
            quantum_queries.append(get_annealing_queries_quantum_walks(int(n), epsilon, spectral_gaps))

        if not (proposal == "layden" and hide_classical_layden):
            classical_label = _curve_label(proposal, "classical")
            classical_color = _proposal_color(proposal, "classical")
            classical_line = _plot_curve_with_projection(classical_queries, color=classical_color, label=classical_label, zorder=2)
            handles.append(classical_line)
            labels.append(classical_label)
            line_records.append((classical_line, classical_label, classical_color, np.asarray(classical_queries, dtype=float)))

        quantum_label = _curve_label(proposal, "quantum")
        quantum_color = _proposal_color(proposal, "quantum")
        quantum_line = _plot_curve_with_projection(quantum_queries, color=quantum_color, label=quantum_label, zorder=3)
        handles.append(quantum_line)
        labels.append(quantum_label)
        line_records.append((quantum_line, quantum_label, quantum_color, np.asarray(quantum_queries, dtype=float)))

        if debug:
            classical_arr = np.asarray(classical_queries, dtype=float)
            quantum_arr = np.asarray(quantum_queries, dtype=float)
            for name, arr in [("classical queries", classical_arr), ("quantum-walk queries", quantum_arr)]:
                decreases = np.where(arr[1:] < arr[:-1])[0]
                if len(decreases):
                    print(f"\n[debug] decreases for {proposal} {name}")
                for idx in decreases:
                    n0 = int(n_vals[idx])
                    n1 = int(n_vals[idx + 1])
                    y0 = arr[idx]
                    y1 = arr[idx + 1]
                    ratio = y1 / y0 if y0 > 0.0 else np.nan
                    print(f"  n {n0}->{n1}: {y0:.6g}->{y1:.6g} ratio={ratio:.6g} schedule_len={len(plotted_schedules[idx])}->{len(plotted_schedules[idx + 1])}")
                    print(f"    previous schedule={plotted_schedules[idx]}")
                    print(f"    current  schedule={plotted_schedules[idx + 1]}")
                    for beta_t in plotted_schedules[idx + 1]:
                        A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                        A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                        b_g = max(float(b_g), 0.0)
                        q_contribution = A_q * np.exp(b_q * n1)
                        gap_value = A_g * np.exp(-b_g * n1)
                        print(f"    beta_t={beta_t:.12g} b_q/log2={b_q / np.log(2):+.6g} q={q_contribution:.6g} b_g/log2={b_g / np.log(2):+.6g} delta={gap_value:.6g}")

    ax.set_yscale("log")
    ax.set_xlim(float(n_plot_min), float(n_plot_max))
    ax.set_ylabel(r"Queries")
    if title is not None:
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color="0.92", linewidth=0.55, zorder=0)
    ax.grid(False, which="major", axis="x")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_schedule_panel and ax_sched is not None:
        if show_schedule_vertical_lines:
            for x in diagnostic_change_positions:
                ax_sched.axvline(x, color="0.72", linewidth=0.45, alpha=0.95, zorder=0)
        ax_sched.step(n_vals, diagnostic_lengths, where="post", color="0.30", linewidth=0.95, zorder=2)
        ax_sched.set_ylabel("Annealing\nsteps")
        ax_sched.set_xlabel(r"$n$", labelpad=xlabel_labelpad)
        ax_sched.set_xlim(float(n_plot_min), float(n_plot_max))
        ax_sched.set_axisbelow(True)
        ax_sched.grid(True, which="major", axis="y", color="0.92", linewidth=0.55, zorder=0)
        ax_sched.grid(False, which="major", axis="x")
        y_min = int(np.min(diagnostic_lengths))
        y_max = int(np.max(diagnostic_lengths))
        if y_min == y_max:
            ax_sched.set_yticks([y_min])
            ax_sched.set_ylim(y_min - 0.5, y_max + 0.5)
        else:
            ax_sched.set_yticks([y_min, y_max])
            ax_sched.set_ylim(y_min - 0.5, y_max + 0.5)
        for spine in ["top", "right"]:
            ax_sched.spines[spine].set_visible(False)
        legend_anchor = (0.5, legend_y_shift)
        legend_axis = ax_sched
    else:
        ax.set_xlabel(r"$n$", labelpad=xlabel_labelpad)
        legend_anchor = (0.5, legend_y_shift)
        legend_axis = ax

    if show_legend and legend_placement == "legend_out":
        legend_axis.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=legend_anchor, ncol=_legend_ncol(), borderaxespad=0.0, handlelength=2.4, columnspacing=1.6)
    elif show_legend and legend_placement == "legend_line":
        _draw_line_labels(line_records)

    return fig, ax

def _darken_color(color: str, factor: float = 0.58):
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(np.clip(factor * rgb, 0.0, 1.0))


def _first_crossing_x(n_vals: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> float | None:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    mask = np.isfinite(baseline) & np.isfinite(candidate) & (baseline > 0.0) & (candidate > 0.0)
    if not np.any(mask):
        return None
    x = np.asarray(n_vals, dtype=float)[mask]
    r = candidate[mask] / baseline[mask]
    if r[0] <= 1.0:
        return float(x[0])
    for i in range(1, len(x)):
        if r[i - 1] > 1.0 and r[i] <= 1.0:
            y0 = np.log(r[i - 1])
            y1 = np.log(r[i])
            if y0 == y1:
                return float(x[i])
            return float(x[i - 1] + (0.0 - y0) * (x[i] - x[i - 1]) / (y1 - y0))
    return None


def _draw_runtime_threshold(ax: plt.Axes, x: float | None, label: str, color: str = "black", linestyle: str = "-", linewidth: float = 0.9, label_y: float = 0.97):
    if x is None:
        return
    ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.85, zorder=8)
    if label:
        ax.text(x, label_y, label, rotation=90, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8, color=color)
















def plot_annealing_classical_and_quantum_runtime_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_device: str = "fpga",
    physical_error_rate_min: float = 1e-4,
    physical_error_rate_max: float = 1e-4,
    physical_operation_time_min: float = 200e-9,
    physical_operation_time_max: float = 20_000e-9,
    physical_measurement_time_min: float = 20e-9,
    physical_measurement_time_max: float = 2_000e-9,
    num_trotter_steps: int = 50,
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
    schedule_diagnostic_generator: Callable[[int, float], list[float]] | None = None,
    show_schedule_panel: bool = True,
    show_schedule_vertical_lines: bool = True,
    legend_y_shift: float = -0.82,
    xlabel_labelpad: float = 18,
    mode: str = "full",
    runtime_ymin_seconds: float | None = None,
    runtime_ymax_years: float | None = 1000.0,
    show_time_reference_lines: bool = False,
    deterministic_classical_band_fraction: float = 0.08,
    area_edge_linewidth: float = 0.8,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    hide_classical_qemc_area: bool = False,
    show_runtime_thresholds: bool = False,
    show_runtime_threshold_labels: bool = False,
    classical_color_darken: float = 0.58,
    differentiate_projection: bool = True,
    legend_placement: str = "legend_out",
    line_width: float = 2.0,
    projected_alpha: float = 0.58,
    calibrated_alpha: float = 0.94,
    line_label_x_fraction: float = 0.82,
    line_label_y_multiplier: float = 1.08,
    line_label_fontsize: float = 9.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot annealing runtime estimates versus n.

    Modes:
    - ``"full"``: local, uniform, and quantum-enhanced proposals.
    - ``"compact"``: uniform and quantum-enhanced proposals.
    - ``"compact_no_layden"``: uniform proposals and quantum-enhanced quantum-walk proposal only.
    """
    if mode not in {"full", "compact", "compact_no_layden"}:
        raise ValueError(f"Unknown mode={mode}. Expected 'full', 'compact', or 'compact_no_layden'.")
    if legend_placement not in {"legend_out", "legend_line"}:
        raise ValueError(f"Unknown legend_placement={legend_placement}. Expected 'legend_out' or 'legend_line'.")

    if n_plot_min is None:
        n_plot_min = 3
    if n_plot_max is None:
        n_plot_max = 120

    n_vals = np.arange(int(n_plot_min), int(n_plot_max) + 1, dtype=int)

    if mode == "full":
        proposals_to_plot = [proposal for proposal in PROPOSALS_SORTED if proposal in {"local1", "uniform", "layden"}]
    elif mode == "compact":
        proposals_to_plot = ["uniform", "layden"]
    else:
        proposals_to_plot = ["uniform", "layden"]

    effective_hide_classical_qemc_area = hide_classical_qemc_area or mode == "compact_no_layden"

    if schedule_diagnostic_generator is None:
        schedule_diagnostic_generator = getattr(annealing_schedule_generator, "base_schedule_generator", annealing_schedule_generator)

    if fig is None or ax is None:
        if show_schedule_panel:
            fig, (ax, ax_sched) = plt.subplots(2, 1, figsize=(8.5, 5.45), sharex=True, gridspec_kw={"height_ratios": [4.0, 0.75], "hspace": 0.06})
            fig.subplots_adjust(bottom=0.30)
        else:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            fig.subplots_adjust(bottom=0.26)
            ax_sched = None
    else:
        ax_sched = None

    one_second = 1.0
    one_minute = 60.0
    one_hour = 60.0 * one_minute
    one_day = 24.0 * one_hour
    one_year = 365.25 * one_day
    one_month = one_year / 12.0
    ten_years = 10.0 * one_year
    hundred_years = 100.0 * one_year
    thousand_years = 1000.0 * one_year

    def _projection_masks():
        if n_fit_max is None:
            return np.ones_like(n_vals, dtype=bool), np.zeros_like(n_vals, dtype=bool)
        fit_mask = n_vals <= int(n_fit_max)
        proj_mask = n_vals >= int(n_fit_max)
        return fit_mask, proj_mask

    def _fill_area_with_projection(lower_values: list[float] | np.ndarray, upper_values: list[float] | np.ndarray, color: str, label: str, alpha: float, zorder: int, edge_linewidth: float = area_edge_linewidth):
        lower_values = np.asarray(lower_values, dtype=float)
        upper_values = np.asarray(upper_values, dtype=float)
        lower = np.minimum(lower_values, upper_values)
        upper = np.maximum(lower_values, upper_values)
        positive = np.concatenate([lower[lower > 0.0], upper[upper > 0.0]])
        floor = np.min(positive) * 1e-4 if positive.size else np.finfo(float).tiny
        lower = np.maximum(lower, floor)
        upper = np.maximum(upper, floor)
        if not differentiate_projection:
            return ax.fill_between(n_vals, lower, upper, facecolor=color, edgecolor=color, alpha=alpha, linewidth=edge_linewidth, zorder=zorder, label=label)
        fit_mask, proj_mask = _projection_masks()
        handle = None
        if np.any(fit_mask):
            handle = ax.fill_between(n_vals[fit_mask], lower[fit_mask], upper[fit_mask], facecolor=color, edgecolor=color, alpha=alpha, linewidth=edge_linewidth, zorder=zorder, label=label)
        if np.any(proj_mask):
            proj_handle = ax.fill_between(n_vals[proj_mask], lower[proj_mask], upper[proj_mask], facecolor=color, edgecolor=color, alpha=0.65 * alpha, linewidth=edge_linewidth, zorder=zorder, label=label if handle is None else None)
            if handle is None:
                handle = proj_handle
        return handle

    def _plot_runtime_line(values: list[float] | np.ndarray, color: str, label: str, zorder: int = 6):
        values = np.asarray(values, dtype=float)
        positive = values[values > 0.0]
        floor = np.min(positive) * 1e-4 if positive.size else np.finfo(float).tiny
        values = np.maximum(values, floor)
        fit_mask, proj_mask = _projection_masks()
        handle = None
        if np.any(fit_mask):
            (handle,) = ax.plot(n_vals[fit_mask], values[fit_mask], color=color, linewidth=line_width, linestyle="-", alpha=calibrated_alpha, zorder=zorder, label=label)
        if np.any(proj_mask):
            (proj_handle,) = ax.plot(n_vals[proj_mask], values[proj_mask], color=color, linewidth=line_width, linestyle="-", alpha=projected_alpha, zorder=zorder, label=label if handle is None else None)
            if handle is None:
                handle = proj_handle
        return handle

    def _visual_band(values: list[float] | np.ndarray, fraction: float):
        values = np.asarray(values, dtype=float)
        if fraction <= 0.0:
            return values, values
        lower = values / (1.0 + fraction)
        upper = values * (1.0 + fraction)
        return lower, upper

    def _schedule_diagnostics(schedule_generator):
        schedules = []
        lengths = []
        for n in n_vals:
            schedule = schedule_generator(int(n), beta)
            schedules.append(schedule)
            lengths.append(len(schedule))
        lengths = np.asarray(lengths, dtype=int)
        change_positions = []
        for i in range(1, len(n_vals)):
            if lengths[i] != lengths[i - 1]:
                change_positions.append(int(n_vals[i]))
        return schedules, lengths, change_positions

    def _legend_ncol() -> int:
        if mode == "full":
            return 3
        if mode == "compact":
            return 2
        return 3

    def _compact_no_layden_label(proposal: str, kind: str) -> str:
        if proposal == "uniform" and kind == "classical":
            return "Best classical"
        if proposal == "uniform" and kind == "quantum":
            return "Quantized best classical"
        if proposal == "layden" and kind == "quantum":
            return "Our approach"
        return _walk_label(proposal, kind)

    def _curve_label(proposal: str, kind: str) -> str:
        if mode == "compact_no_layden":
            return _compact_no_layden_label(proposal, kind)
        return _walk_label(proposal, kind)

    def _draw_line_labels(line_records: list[tuple[str, str, np.ndarray]]) -> None:
        if not line_records:
            return
        x_left, x_right = float(n_plot_min), float(n_plot_max)
        x_span = x_right - x_left
        if x_span <= 0.0:
            return
        label_x = x_left + float(line_label_x_fraction) * x_span
        dx = max(1.0, 0.035 * x_span)
        y_bottom, y_top = ax.get_ylim()
        log_y_min = np.log10(y_bottom)
        log_y_max = np.log10(y_top)
        for label, color, values in sorted(line_records, key=lambda item: item[2][-1]):
            values = np.asarray(values, dtype=float)
            valid = np.isfinite(values) & (values > 0.0) & np.isfinite(n_vals.astype(float))
            if np.count_nonzero(valid) < 2:
                continue
            x_data = n_vals.astype(float)[valid]
            log_y_data = np.log10(values[valid])
            x_label = float(np.clip(label_x, x_data[0], x_data[-1]))
            x0 = float(np.clip(x_label - dx, x_data[0], x_data[-1]))
            x1 = float(np.clip(x_label + dx, x_data[0], x_data[-1]))
            if np.isclose(x0, x1):
                continue
            log_y_curve = float(np.interp(x_label, x_data, log_y_data))
            log_y_label = log_y_curve + np.log10(float(line_label_y_multiplier))
            log_y_label = float(np.clip(log_y_label, log_y_min + 0.025, log_y_max - 0.025))
            y0 = 10.0 ** float(np.interp(x0, x_data, log_y_data))
            y1 = 10.0 ** float(np.interp(x1, x_data, log_y_data))
            p0 = ax.transData.transform((x0, y0))
            p1 = ax.transData.transform((x1, y1))
            angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
            ax.text(x_label, 10.0 ** log_y_label, label, color=color, fontsize=line_label_fontsize, ha="center", va="bottom", rotation=angle, rotation_mode="anchor", clip_on=True, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.6})

    diagnostic_schedules, diagnostic_lengths, diagnostic_change_positions = _schedule_diagnostics(schedule_diagnostic_generator)

    if debug:
        if schedule_diagnostic_generator is annealing_schedule_generator:
            print("[debug] schedule panel uses the plotted schedule generator")
        else:
            print("[debug] schedule panel uses a separate diagnostic schedule generator")
        print(f"[debug] schedule-change positions: {diagnostic_change_positions}")
        for i in range(1, len(n_vals)):
            if diagnostic_lengths[i] != diagnostic_lengths[i - 1]:
                print(f"[debug] schedule length changes at n={n_vals[i]}: {diagnostic_lengths[i - 1]} -> {diagnostic_lengths[i]}")
                print(f"        previous={diagnostic_schedules[i - 1]}")
                print(f"        current ={diagnostic_schedules[i]}")

    if show_schedule_vertical_lines:
        for x in diagnostic_change_positions:
            ax.axvline(x, color="0.72", linewidth=0.45, alpha=0.95, zorder=0)

    handles = []
    labels = []
    line_records = []
    runtime_curves = {}
    threshold_xs = []
    eps_SF = epsilon / 4.0

    for proposal in proposals_to_plot:
        color_quantum = _proposal_color(proposal, "quantum")
        color_classical = _proposal_color(proposal, "classical")

        classical_min = []
        classical_max = []
        quantum_min = []
        quantum_max = []
        plotted_schedules = []

        for n in n_vals:
            schedule = annealing_schedule_generator(int(n), beta)
            plotted_schedules.append(schedule)

            vec_queries = []
            spectral_gaps = []

            for beta_t in schedule:
                A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                vec_queries.append(A_q * np.exp(b_q * n))
                A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                b_g = max(float(b_g), 0.0)
                spectral_gaps.append(A_g * np.exp(-b_g * n))

            if proposal == "local1":
                classical_value = get_annealing_time_classical_walk_local(int(n), vec_queries, device=classical_device)
                c_lower, c_upper = _visual_band([classical_value], deterministic_classical_band_fraction)
                classical_min.append(float(c_lower[0]))
                classical_max.append(float(c_upper[0]))
                quantum_min.append(get_annealing_time_quantum_walk_local(int(n), epsilon, schedule, spectral_gaps, physical_operation_time_min, physical_measurement_time_min, physical_error_rate_min))
                quantum_max.append(get_annealing_time_quantum_walk_local(int(n), epsilon, schedule, spectral_gaps, physical_operation_time_max, physical_measurement_time_max, physical_error_rate_max))

            elif proposal == "uniform":
                classical_value = get_annealing_time_classical_walk_uniform(int(n), vec_queries, device=classical_device)
                c_lower, c_upper = _visual_band([classical_value], deterministic_classical_band_fraction)
                classical_min.append(float(c_lower[0]))
                classical_max.append(float(c_upper[0]))
                quantum_min.append(get_annealing_time_quantum_walk_uniform(int(n), epsilon, schedule, spectral_gaps, physical_operation_time_min, physical_measurement_time_min, physical_error_rate_min))
                quantum_max.append(get_annealing_time_quantum_walk_uniform(int(n), epsilon, schedule, spectral_gaps, physical_operation_time_max, physical_measurement_time_max, physical_error_rate_max))

            elif proposal == "layden":
                classical_min.append(get_annealing_time_classical_walk_qemc(int(n), vec_queries, physical_operation_time_min, physical_measurement_time_min, physical_error_rate_min, eps_SF, num_trotter_steps=num_trotter_steps))
                classical_max.append(get_annealing_time_classical_walk_qemc(int(n), vec_queries, physical_operation_time_max, physical_measurement_time_max, physical_error_rate_max, eps_SF, num_trotter_steps=num_trotter_steps))
                quantum_min.append(get_annealing_time_quantum_walk_qemc(int(n), epsilon, schedule, spectral_gaps, physical_operation_time_min, physical_measurement_time_min, physical_error_rate_min, num_trotter_steps=num_trotter_steps))
                quantum_max.append(get_annealing_time_quantum_walk_qemc(int(n), epsilon, schedule, spectral_gaps, physical_operation_time_max, physical_measurement_time_max, physical_error_rate_max, num_trotter_steps=num_trotter_steps))

        classical_min = np.asarray(classical_min, dtype=float)
        classical_max = np.asarray(classical_max, dtype=float)
        quantum_min = np.asarray(quantum_min, dtype=float)
        quantum_max = np.asarray(quantum_max, dtype=float)

        runtime_curves[proposal] = {"classical_min": classical_min, "classical_max": classical_max, "classical_nominal": np.sqrt(classical_min * classical_max), "quantum_min": quantum_min, "quantum_max": quantum_max, "quantum_nominal": np.sqrt(quantum_min * quantum_max)}

        classical_edge = 1.1 if proposal in {"local1", "uniform"} else area_edge_linewidth

        classical_label = _curve_label(proposal, "classical")
        quantum_label = _curve_label(proposal, "quantum")

        if proposal == "uniform":
            classical_handle = _plot_runtime_line(runtime_curves[proposal]["classical_nominal"], color=color_classical, label=classical_label)
            handles.append(classical_handle)
            labels.append(classical_label)
            line_records.append((classical_label, color_classical, runtime_curves[proposal]["classical_nominal"]))
        elif not (proposal == "layden" and effective_hide_classical_qemc_area):
            classical_handle = _fill_area_with_projection(classical_min, classical_max, color=color_classical, label=classical_label, alpha=0.30, zorder=2, edge_linewidth=line_width)
            handles.append(classical_handle)
            labels.append(classical_label)
            line_records.append((classical_label, color_classical, runtime_curves[proposal]["classical_nominal"]))

        quantum_handle = _fill_area_with_projection(quantum_min, quantum_max, color=color_quantum, label=quantum_label, alpha=0.34, zorder=3, edge_linewidth=line_width)
        handles.append(quantum_handle)
        labels.append(quantum_label)
        line_records.append((quantum_label, color_quantum, runtime_curves[proposal]["quantum_nominal"]))

        if debug:
            diagnostic_curves = [("classical runtime optimistic", classical_min), ("classical runtime pessimistic", classical_max), ("quantum-walk runtime optimistic", quantum_min), ("quantum-walk runtime pessimistic", quantum_max)]
            for name, arr in diagnostic_curves:
                decreases = np.where(arr[1:] < arr[:-1])[0]
                if len(decreases):
                    print(f"\n[debug] decreases for {proposal} {name}")
                for idx in decreases:
                    n0 = int(n_vals[idx])
                    n1 = int(n_vals[idx + 1])
                    y0 = arr[idx]
                    y1 = arr[idx + 1]
                    ratio = y1 / y0 if y0 > 0.0 else np.nan
                    print(f"  n {n0}->{n1}: {y0:.6g}->{y1:.6g} ratio={ratio:.6g} schedule_len={len(plotted_schedules[idx])}->{len(plotted_schedules[idx + 1])}")
                    print(f"    previous schedule={plotted_schedules[idx]}")
                    print(f"    current  schedule={plotted_schedules[idx + 1]}")

    if show_runtime_thresholds and "uniform" in runtime_curves and "layden" in runtime_curves:
        best_classical = runtime_curves["uniform"]["classical_nominal"]
        layden_classical_min = runtime_curves["layden"]["classical_min"]
        layden_classical_max = runtime_curves["layden"]["classical_max"]
        layden_quantum_min = runtime_curves["layden"]["quantum_min"]
        layden_quantum_max = runtime_curves["layden"]["quantum_max"]

        if effective_hide_classical_qemc_area:
            thresholds = [(_first_crossing_x(n_vals, best_classical, layden_quantum_min), "uniform classical vs QEMC quantum opt.", "-"), (_first_crossing_x(n_vals, best_classical, layden_quantum_max), "uniform classical vs QEMC quantum pess.", "--")]
        else:
            qemc_best_optimistic = np.minimum(layden_classical_min, layden_quantum_min)
            qemc_best_pessimistic = np.minimum(layden_classical_max, layden_quantum_max)
            thresholds = [(_first_crossing_x(n_vals, best_classical, qemc_best_optimistic), "uniform classical vs best QEMC opt.", "-"), (_first_crossing_x(n_vals, best_classical, qemc_best_pessimistic), "uniform classical vs best QEMC pess.", "--"), (_first_crossing_x(n_vals, layden_classical_min, layden_quantum_min), "QEMC classical vs quantum opt.", ":"), (_first_crossing_x(n_vals, layden_classical_max, layden_quantum_max), "QEMC classical vs quantum pess.", "-.")]

        if debug:
            for x_threshold, threshold_label, _ in thresholds:
                print(f"[debug] threshold {threshold_label}: {x_threshold}")

        for x_threshold, threshold_label, threshold_linestyle in thresholds:
            if x_threshold is not None:
                threshold_xs.append(float(x_threshold))
            _draw_runtime_threshold(ax=ax, x=x_threshold, label="" if not show_runtime_threshold_labels else threshold_label, color="0.38", linestyle=threshold_linestyle, linewidth=1.0, label_y=0.97)

    ax.set_yscale("log")
    ax.set_xlim(float(n_plot_min), float(n_plot_max))

    time_ticks = [one_second, one_minute, one_hour, one_day, one_month, one_year, ten_years, hundred_years, thousand_years]
    time_tick_labels = ["one second", "one minute", "one hour", "one day", "one month", "one year", "10 years", "100 years", "1000 years"]
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_tick_labels)

    if runtime_ymin_seconds is not None or runtime_ymax_years is not None:
        current_bottom, current_top = ax.get_ylim()
        y_bottom = current_bottom if runtime_ymin_seconds is None else float(runtime_ymin_seconds)
        y_top = current_top if runtime_ymax_years is None else float(runtime_ymax_years) * one_year
        ax.set_ylim(y_bottom, y_top)

    if threshold_xs:
        x_left, x_right = ax.get_xlim()
        base_xticks = [float(x) for x in ax.get_xticks() if x_left <= float(x) <= x_right]
        threshold_xticks = [float(int(round(x))) for x in threshold_xs if x_left <= float(int(round(x))) <= x_right]
        merged_xticks = sorted(set([round(x, 10) for x in base_xticks + threshold_xticks]))

        def _format_threshold_tick(x: float) -> str:
            if abs(x - 40.0) < 1e-6:
                return ""
            if any(abs(x - t) < 1e-6 for t in threshold_xticks):
                return f"{int(round(x))}"
            return f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:g}"

        ax.set_xticks(merged_xticks)
        ax.set_xticklabels([_format_threshold_tick(x) for x in merged_xticks])

    if show_time_reference_lines:
        for y in time_ticks:
            ax.axhline(y, color="black", linestyle=":", linewidth=1.1, alpha=0.75, zorder=0)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    ax.grid(False, which="major", axis="x")
    ax.set_ylabel(r"Runtime [s]")
    if title is not None:
        ax.set_title(title)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_schedule_panel and ax_sched is not None:
        if show_schedule_vertical_lines:
            for x in diagnostic_change_positions:
                ax_sched.axvline(x, color="0.72", linewidth=0.45, alpha=0.95, zorder=0)

        ax_sched.step(n_vals, diagnostic_lengths, where="post", color="0.30", linewidth=0.95, zorder=2)
        ax_sched.set_ylabel("Annealing\nsteps")
        ax_sched.set_xlabel(r"$n$", labelpad=xlabel_labelpad)
        ax_sched.set_xlim(float(n_plot_min), float(n_plot_max))

        y_min = int(np.min(diagnostic_lengths))
        y_max = int(np.max(diagnostic_lengths))
        if y_min == y_max:
            ax_sched.set_yticks([y_min])
            ax_sched.set_ylim(y_min - 0.5, y_max + 0.5)
        else:
            ax_sched.set_yticks([y_min, y_max])
            ax_sched.set_ylim(y_min - 0.5, y_max + 0.5)

        if threshold_xs:
            x_left, x_right = ax.get_xlim()
            base_xticks = [float(x) for x in ax.get_xticks() if x_left <= float(x) <= x_right]
            threshold_xticks = [float(int(round(x))) for x in threshold_xs if x_left <= float(int(round(x))) <= x_right]
            merged_xticks = sorted(set([round(x, 10) for x in base_xticks + threshold_xticks]))
            ax_sched.set_xticks(merged_xticks)
            ax_sched.set_xticklabels([_format_threshold_tick(x) for x in merged_xticks])
            
        ax_sched.set_axisbelow(True)
        ax_sched.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
        ax_sched.grid(False, which="major", axis="x")

        for spine in ["top", "right"]:
            ax_sched.spines[spine].set_visible(False)

        legend_anchor = (0.5, legend_y_shift)
        legend_axis = ax_sched
    else:
        ax.set_xlabel(r"$n$", labelpad=xlabel_labelpad)
        legend_anchor = (0.5, legend_y_shift)
        legend_axis = ax

    if show_legend and legend_placement == "legend_out":
        legend_axis.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=legend_anchor, ncol=_legend_ncol(), borderaxespad=0.0, handlelength=2.4, columnspacing=1.6)
    elif show_legend and legend_placement == "legend_line":
        _draw_line_labels(line_records)

    ax.set_ylim(one_second, thousand_years)

    return fig, ax












def plot_annealing_classical_and_quantum_runtime_fancy_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_device: str = "fpga",
    physical_error_rate_min: float = 1e-4,
    physical_error_rate_max: float = 1e-4,
    physical_operation_time_min: float = 200e-9,
    physical_operation_time_max: float = 20_000e-9,
    physical_measurement_time_min: float = 20e-9,
    physical_measurement_time_max: float = 2_000e-9,
    num_trotter_steps: int = 50,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    title: str | None = None,
    show_legend: bool = True,
    debug: bool = False,
    schedule_diagnostic_generator: Callable[[int, float], list[float]] | None = None,
    show_schedule_panel: bool = True,
    show_schedule_vertical_lines: bool = True,
    legend_y_shift: float = -0.82,
    xlabel_labelpad: float = 18,
    mode: str = "full",
    show_time_reference_lines: bool = False,
    deterministic_classical_band_fraction: float = 0.08,
    area_edge_linewidth: float = 0.8,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    inset_width: str = "31%",
    inset_height: str = "38%",
    inset_loc: str = "lower right",
    inset_borderpad: float = 2.0,
    hide_classical_qemc_area: bool = False,
    show_runtime_thresholds: bool = False,
    show_runtime_threshold_labels: bool = True,
    classical_color_darken: float = 0.58,
    legend_placement: str = "legend_out",
    line_width: float = 2.0,
    projected_alpha: float = 0.58,
    calibrated_alpha: float = 0.94,
    line_label_x_fraction: float = 0.82,
    line_label_y_multiplier: float = 1.08,
    line_label_fontsize: float = 9.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Fancy runtime plot with main y-range from one day to 100 years and a lower-right
    zoom-out inset from 1 microsecond to 1 billion years.

    :return: Tuple ``(fig, ax)`` for the main axes.
    """
    one_microsecond = 1e-6
    one_hour = 60.0 * 60.0
    one_day = 24.0 * one_hour
    one_year = 365.25 * one_day
    one_month = one_year / 12.0
    ten_years = 10.0 * one_year
    one_billion_years = 1e9 * one_year

    fig, ax = plot_annealing_classical_and_quantum_runtime_vs_n(
        beta=beta,
        epsilon=epsilon,
        annealing_schedule_generator=annealing_schedule_generator,
        classical_device=classical_device,
        physical_error_rate_min=physical_error_rate_min,
        physical_error_rate_max=physical_error_rate_max,
        physical_operation_time_min=physical_operation_time_min,
        physical_operation_time_max=physical_operation_time_max,
        physical_measurement_time_min=physical_measurement_time_min,
        physical_measurement_time_max=physical_measurement_time_max,
        num_trotter_steps=num_trotter_steps,
        classical_query_file=classical_query_file,
        spectral_gap_file=spectral_gap_file,
        statistic=statistic,
        n_fit_min=n_fit_min,
        n_fit_max=n_fit_max,
        n_plot_min=n_plot_min,
        n_plot_max=n_plot_max,
        fig=None,
        ax=None,
        title=title,
        show_legend=show_legend,
        debug=debug,
        schedule_diagnostic_generator=schedule_diagnostic_generator,
        show_schedule_panel=show_schedule_panel,
        show_schedule_vertical_lines=show_schedule_vertical_lines,
        legend_y_shift=legend_y_shift,
        xlabel_labelpad=xlabel_labelpad,
        mode=mode,
        runtime_ymin_seconds=one_day,
        runtime_ymax_years=100.0,
        show_time_reference_lines=show_time_reference_lines,
        deterministic_classical_band_fraction=deterministic_classical_band_fraction,
        area_edge_linewidth=area_edge_linewidth,
        grid_color=grid_color,
        grid_linewidth=grid_linewidth,
        hide_classical_qemc_area=hide_classical_qemc_area,
        show_runtime_thresholds=show_runtime_thresholds,
        show_runtime_threshold_labels=show_runtime_threshold_labels,
        classical_color_darken=classical_color_darken,
        legend_placement=legend_placement,
        line_width=line_width,
        projected_alpha=projected_alpha,
        calibrated_alpha=calibrated_alpha,
        line_label_x_fraction=line_label_x_fraction,
        line_label_y_multiplier=line_label_y_multiplier,
        line_label_fontsize=line_label_fontsize,
    )

    ax.set_ylim(one_day, 100.0 * one_year)
    ax.set_yticks([one_day, one_month, one_year, ten_years])
    ax.set_yticklabels(["one day", "one month", "one year", "10 years"])

    inset_ax = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc=inset_loc,
        borderpad=inset_borderpad,
    )

    plot_annealing_classical_and_quantum_runtime_vs_n(
        beta=beta,
        epsilon=epsilon,
        annealing_schedule_generator=annealing_schedule_generator,
        classical_device=classical_device,
        physical_error_rate_min=physical_error_rate_min,
        physical_error_rate_max=physical_error_rate_max,
        physical_operation_time_min=physical_operation_time_min,
        physical_operation_time_max=physical_operation_time_max,
        physical_measurement_time_min=physical_measurement_time_min,
        physical_measurement_time_max=physical_measurement_time_max,
        num_trotter_steps=num_trotter_steps,
        classical_query_file=classical_query_file,
        spectral_gap_file=spectral_gap_file,
        statistic=statistic,
        n_fit_min=n_fit_min,
        n_fit_max=n_fit_max,
        n_plot_min=n_plot_min,
        n_plot_max=n_plot_max,
        fig=fig,
        ax=inset_ax,
        title=None,
        show_legend=False,
        debug=False,
        schedule_diagnostic_generator=schedule_diagnostic_generator,
        show_schedule_panel=False,
        show_schedule_vertical_lines=show_schedule_vertical_lines,
        legend_y_shift=legend_y_shift,
        xlabel_labelpad=xlabel_labelpad,
        mode=mode,
        runtime_ymin_seconds=one_microsecond,
        runtime_ymax_years=1e9,
        show_time_reference_lines=show_time_reference_lines,
        deterministic_classical_band_fraction=deterministic_classical_band_fraction,
        area_edge_linewidth=area_edge_linewidth,
        grid_color=grid_color,
        grid_linewidth=grid_linewidth,
        hide_classical_qemc_area=hide_classical_qemc_area,
        show_runtime_thresholds=False,
        show_runtime_threshold_labels=False,
        classical_color_darken=classical_color_darken,
        legend_placement=legend_placement,
        line_width=line_width,
        projected_alpha=projected_alpha,
        calibrated_alpha=calibrated_alpha,
        line_label_x_fraction=line_label_x_fraction,
        line_label_y_multiplier=line_label_y_multiplier,
        line_label_fontsize=line_label_fontsize,
    )

    inset_ax.set_title("")
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.set_ylim(one_microsecond, one_billion_years)
    inset_ax.set_yticks([one_microsecond, one_day, ten_years, one_billion_years])
    inset_ax.set_yticklabels(["1 µs", "one day", "10 years", "1B years"])
    inset_ax.tick_params(axis="both", which="major", labelsize=7)
    inset_ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    inset_ax.grid(False, which="major", axis="x")

    return fig, ax


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
        projected_mask = x >= extrapolation_start_n

        x_solid = np.concatenate([x[solid_mask], [extrapolation_start_n]])
        y_solid = np.concatenate([y[solid_mask], [y_start]])
        x_projected = np.concatenate([[extrapolation_start_n], x[projected_mask]])
        y_projected = np.concatenate([[y_start], y[projected_mask]])

        color = line.get_color()
        alpha = line.get_alpha()
        alpha = 1.0 if alpha is None else alpha
        linewidth = line.get_linewidth()

        line.set_data(x_solid, y_solid)
        ax.plot(
            x_projected,
            y_projected,
            color=color,
            linestyle="-",
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


_LOCAL_MOVE_COLOR_KEYS = [key for key in PROPOSAL_COLORS if key.startswith("local1_")]
_LOCAL_MOVE_COLORS = [PROPOSAL_COLORS[key] for key in _LOCAL_MOVE_COLOR_KEYS]


def _rgba_close(color, target_color: str, tol: float = 1e-2) -> bool:
    try:
        rgba = np.asarray(to_rgba(color), dtype=float)
        target = np.asarray(to_rgba(target_color), dtype=float)
        return bool(np.allclose(rgba[:3], target[:3], atol=tol, rtol=0.0))
    except (TypeError, ValueError):
        return False


def _rgba_close_to_any(color, target_colors: Sequence[str], tol: float = 1e-2) -> bool:
    return any(_rgba_close(color, target_color, tol=tol) for target_color in target_colors)


def _collection_has_color(collection, target_colors: Sequence[str]) -> bool:
    for getter_name in ("get_facecolors", "get_edgecolors"):
        if not hasattr(collection, getter_name):
            continue
        colors = getattr(collection, getter_name)()
        if colors is None or len(colors) == 0:
            continue
        if any(_rgba_close_to_any(color, target_colors) for color in colors):
            return True
    return False


def _remove_artists_with_colors(ax: plt.Axes, target_colors: Sequence[str]) -> None:
    for line in list(ax.lines):
        if _rgba_close_to_any(line.get_color(), target_colors):
            line.remove()

    for collection in list(ax.collections):
        if _collection_has_color(collection, target_colors):
            collection.remove()


def _remove_local_move_artists(ax: plt.Axes, ax_gap: plt.Axes | None = None) -> None:
    _remove_artists_with_colors(ax, _LOCAL_MOVE_COLORS)
    if ax_gap is not None:
        _remove_artists_with_colors(ax_gap, _LOCAL_MOVE_COLORS)


def _last_step_scaling_label(label: str) -> str:
    if " queries: " not in label and " inverse gap: " not in label:
        return _fit_only_label(label)

    match = re.search(r"2\^\{([^}]*)\}", label)
    if match is None:
        return _fit_only_label(label)

    exponent_text = match.group(1)
    lambda_match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", exponent_text)
    if lambda_match is None:
        return _fit_only_label(label)

    lam = abs(float(lambda_match.group(0)))
    return rf"$\lambda={lam:.3f}$"


def _last_step_row_header(label: str) -> str:
    if " queries: " in label:
        return r"$Q(n)$"
    if " inverse gap: " in label:
        return r"$\delta(n)$"
    return ""

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
    row_header_gap: float = 0.075,
    show_row_headers: bool = True,
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
        row_header = _last_step_row_header(group_raw_labels[0])
        if show_row_headers and row_header:
            ax.text(
                xs[0] - line_half_width - row_header_gap,
                y_text,
                row_header,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )

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
    show_legend: bool = True,
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
        plot_spectral_gap_vs_beta(fixed_n=fixed_n, fig=fig, ax=ax, title=rf"$n={fixed_n}$", show_legend=False, **kwargs)
        _apply_spectral_gap_beta_y_scale(ax)
        if add_inset:
            scaled_inset_bounds = _scale_inset_bounds(inset_bounds, miniature_scale)
            _add_spectral_gap_beta_inset(fig, ax, fixed_n, scaled_inset_bounds, inset_tick_labelsize, **kwargs)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    if show_legend:
        legend_handles = [Line2D([0], [0], color=_proposal_base_color(proposal), linewidth=2.0, linestyle="-", alpha=0.90) for proposal in PROPOSALS_SORTED]
        fig.legend(legend_handles, MOVE_LEGEND_LABELS, frameon=False, loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=len(MOVE_LEGEND_LABELS), fontsize=legend_fontsize, borderaxespad=0.0, handlelength=2.4, columnspacing=1.6)

    _keep_single_axis_y_ticks_only_on_outer_edges(axes, len(fixed_ns), ncols)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(fixed_ns)]




def _exponential_fit_label(n_vals: np.ndarray, y_vals: np.ndarray) -> str:
    n_vals = np.asarray(n_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(n_vals) & np.isfinite(y_vals) & (y_vals > 0.0)
    if np.count_nonzero(mask) < 2:
        return r"$\lambda=\mathrm{nan}$"

    b, log_a = np.polyfit(n_vals[mask], np.log(y_vals[mask]), deg=1)
    a = float(np.exp(log_a))
    return rf"${a:.3g} \times 2^{{{b / np.log(2):.3f} n}}$"


def plot_last_step_classical_queries_and_quantum_queries_vs_n(
    beta: float,
    epsilon: float,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    line_width: float = 2.0,
    line_alpha: float = 0.95,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and quantum-walk query estimates versus n for the last annealing step.

    Classical curves use query-count fits. Quantum curves use the spectral-gap fits
    converted to quantum-walk queries for a one-step schedule.
    """
    if n_plot_min is None or n_plot_max is None:
        n_plot_min = 1
        n_plot_max = 100

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

    n_grid = _n_plot_grid(n_plot_min, n_plot_max)
    handles = []
    labels = []

    for proposal in PROPOSALS_SORTED:
        classical_color = _proposal_color(proposal, "classical")
        quantum_color = _proposal_color(proposal, "quantum")

        n_vals, center, spread = _compact_classical_query_points(proposal, beta, epsilon, statistic, classical_query_file, min_count)
        if show_spread:
            lower, upper = _positive_band(center, spread)
            ax.fill_between(n_vals, lower, upper, color=classical_color, alpha=0.20, linewidth=0.0, zorder=1)
        ax.scatter(n_vals, center, s=36, color=classical_color, edgecolors="none", alpha=0.95, zorder=3)

        A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        classical_fit = A_q * np.exp(b_q * n_grid)
        (classical_line,) = ax.plot(n_grid, classical_fit, color=classical_color, linewidth=line_width, linestyle="-", alpha=line_alpha, zorder=2)
        handles.append(classical_line)
        labels.append(rf"{PROPOSAL_LABELS[proposal]} classical queries: ${A_q:.3g} \times 2^{{{b_q / np.log(2):.3f} n}}$")

        A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        b_g = max(float(b_g), 0.0)
        spectral_gaps = A_g * np.exp(-b_g * n_grid)
        quantum_fit = np.asarray([get_one_step_quantum_walk_queries(int(n), epsilon, float(delta)) for n, delta in zip(n_grid, spectral_gaps)], dtype=float)
        (quantum_line,) = ax.plot(n_grid, quantum_fit, color=quantum_color, linewidth=line_width, linestyle="-", alpha=line_alpha, zorder=2)
        handles.append(quantum_line)
        labels.append(rf"{PROPOSAL_LABELS[proposal]} quantum queries: {_exponential_fit_label(n_grid, quantum_fit)}")

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Queries")
    ax.set_title(title if title is not None else rf"Last-step queries, $\beta={beta}$, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(False)

    if show_legend:
        ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax


def _last_step_query_scaling_label(label: str) -> str:
    if " classical queries: " not in label and " quantum queries: " not in label:
        return _fit_only_label(label)

    match = re.search(r"2\^\{([^}]*)\}", label)
    if match is None:
        return _fit_only_label(label)

    exponent_text = match.group(1)
    lambda_match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", exponent_text)
    if lambda_match is None:
        return _fit_only_label(label)

    lam = abs(float(lambda_match.group(0)))
    return rf"$\lambda={lam:.3f}$"


def _last_step_query_row_header(label: str) -> str:
    if " classical queries: " in label:
        return r"$Q_{\mathrm{cl}}(n)$"
    if " quantum queries: " in label:
        return r"$Q_{\mathrm{q}}(n)$"
    return ""


def _replace_last_step_query_legend_with_scaling_rows(
    ax: plt.Axes,
    ncol: int = 3,
    y: float = -0.17,
    row_gap: float = 0.030,
    group_gap: float = 0.145,
    line_half_width: float = 0.050,
    fontsize: int = 12,
    remove_local: bool = False,
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    x_center: float = 0.5,
    x_span: float = 0.54,
    two_column_x_span: float = 0.34,
    row_header_gap: float = 0.075,
    show_row_headers: bool = True,
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

    grouped: dict[str, list[tuple[object, str, str]]] = {"classical": [], "quantum": []}
    for handle, label in zip(handles, labels):
        if remove_local and label.startswith("Local spin-flip"):
            continue
        if " classical queries: " in label:
            grouped["classical"].append((handle, _last_step_query_scaling_label(label), label))
        elif " quantum queries: " in label:
            grouped["quantum"].append((handle, _last_step_query_scaling_label(label), label))

    for row_idx, key in enumerate(["classical", "quantum"]):
        entries = grouped[key]
        count = len(entries)
        if count == 0:
            continue

        span = two_column_x_span if count == 2 else x_span
        xs = np.linspace(x_center - 0.5 * span, x_center + 0.5 * span, count) if count > 1 else np.array([x_center])
        y_line = y - row_idx * group_gap
        y_text = y_line - row_gap
        row_header = _last_step_query_row_header(entries[0][2])
        if show_row_headers and row_header:
            ax.text(
                xs[0] - line_half_width - row_header_gap,
                y_text,
                row_header,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )

        for x, (handle, label, _) in zip(xs, entries):
            ax.plot(
                [x - line_half_width, x + line_half_width],
                [y_line, y_line],
                transform=ax.transAxes,
                color=_line_color(handle),
                linestyle="-",
                linewidth=line_width,
                alpha=line_alpha,
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


def plot_last_step_classical_queries_and_quantum_queries_vs_n_table(
    betas: Sequence[float],
    epsilon: float | Sequence[float],
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
    line_width: float = 2.0,
    line_alpha: float = 0.95,
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
) -> tuple[plt.Figure, np.ndarray]:
    betas = list(betas)
    if np.isscalar(epsilon):
        epsilons = [float(epsilon)] * len(betas)
    else:
        epsilons = [float(eps) for eps in epsilon]
        if len(epsilons) != len(betas):
            raise ValueError(f"epsilon must be a scalar or a sequence with len(epsilon) == len(betas). Got len(epsilon)={len(epsilons)} and len(betas)={len(betas)}.")

    fig, axes = _make_table_axes(len(betas), ncols, figsize)
    dummy_gap_axes = [None] * len(betas)

    n_plot_min = kwargs.setdefault("n_plot_min", 1)
    n_plot_max = kwargs.setdefault("n_plot_max", 20)

    if extrapolation_start_n is None:
        extrapolation_start_n = kwargs.get("n_fit_max", 10)
        if extrapolation_start_n is None:
            extrapolation_start_n = 10

    n_items = len(betas)
    nrows = int(np.ceil(n_items / ncols))

    for idx, (beta, eps, ax) in enumerate(zip(betas, epsilons, axes)):
        row = idx // ncols
        is_last_row = row == nrows - 1
        current_legend_y = legend_y
        if is_last_row and last_row_legend_y is not None:
            current_legend_y = last_row_legend_y

        plot_last_step_classical_queries_and_quantum_queries_vs_n(
            beta=beta,
            epsilon=eps,
            fig=fig,
            ax=ax,
            title=rf"$\beta={beta}$, $\epsilon={eps:g}$",
            show_legend=show_legend,
            line_width=line_width,
            line_alpha=line_alpha,
            **kwargs,
        )

        remove_local = remove_local_beta_threshold is not None and float(beta) > float(remove_local_beta_threshold)
        if remove_local:
            _remove_local_move_artists(ax, None)

        _split_fit_lines_at_n(
            ax,
            extrapolation_start_n=float(extrapolation_start_n),
            extrapolation_alpha_factor=extrapolation_alpha_factor,
            extrapolation_linewidth_factor=extrapolation_linewidth_factor,
        )

        _soften_plot_elements(ax, scatter_size=scatter_size, band_alpha=band_alpha)
        _apply_spectral_gap_n_x_scale(ax, n_plot_min=n_plot_min, n_plot_max=n_plot_max)

        if show_legend:
            _replace_last_step_query_legend_with_scaling_rows(
                ax,
                ncol=2 if remove_local else 3,
                y=current_legend_y,
                fontsize=legend_fontsize,
                remove_local=remove_local,
                line_width=line_width,
                line_alpha=line_alpha,
                line_half_width=legend_line_half_width,
                x_span=legend_x_span,
                two_column_x_span=legend_two_column_x_span,
                show_row_headers=(idx % ncols == 0),
            )

        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    if rowwise_y_limits:
        _apply_rowwise_last_step_y_limits(
            axes,
            dummy_gap_axes,
            len(betas),
            ncols,
            y_tick_order_step=y_tick_order_step,
            include_unit_tick=include_unit_y_tick,
        )

    _keep_last_step_axis_labels_only_on_outer_edges(axes, len(betas), ncols, dummy_gap_axes)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(betas)]

def plot_last_step_classical_queries_and_spectral_gap_vs_n_table(
    betas: Sequence[float],
    epsilon: float | Sequence[float],
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
    betas = list(betas)
    if np.isscalar(epsilon):
        epsilons = [float(epsilon)] * len(betas)
    else:
        epsilons = [float(eps) for eps in epsilon]
        if len(epsilons) != len(betas):
            raise ValueError(f"epsilon must be a scalar or a sequence with len(epsilon) == len(betas). Got len(epsilon)={len(epsilons)} and len(betas)={len(betas)}.")

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

    for idx, (beta, eps, ax) in enumerate(zip(betas, epsilons, axes)):
        row = idx // ncols
        is_last_row = row == nrows - 1
        current_legend_y = legend_y
        if is_last_row and last_row_legend_y is not None:
            current_legend_y = last_row_legend_y

        _, _, ax_gap = plot_last_step_classical_queries_and_spectral_gap_vs_n(
            beta=beta,
            epsilon=eps,
            fig=fig,
            ax=ax,
            title=rf"$\beta={beta}$, $\epsilon={eps:g}$",
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
                show_row_headers=(idx % ncols == 0),
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
    epsilon: float | Sequence[float],
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
    betas = list(betas)
    if np.isscalar(epsilon):
        epsilons = [float(epsilon)] * len(betas)
    else:
        epsilons = [float(eps) for eps in epsilon]
        if len(epsilons) != len(betas):
            raise ValueError(f"epsilon must be a scalar or a sequence with len(epsilon) == len(betas). Got len(epsilon)={len(epsilons)} and len(betas)={len(betas)}.")

    fig, axes = _make_table_axes(len(betas), ncols, figsize)

    for beta, eps, ax in zip(betas, epsilons, axes):
        plot_annealing_classical_and_quantum_queries_vs_n(beta=beta, epsilon=eps, annealing_schedule_generator=annealing_schedule_generator, fig=fig, ax=ax, title=rf"$\beta_F={beta}$, $\epsilon={eps:g}$", show_legend=show_legend, **kwargs)
        if show_legend:
            _replace_legend_with_line_fit_rows(ax, ncol=3, y=legend_y, fontsize=legend_fontsize)
        _lighten_grid(ax)
        _enlarge_axis_labels(ax, label_fontsize, tick_labelsize, title_fontsize)

    _keep_axis_labels_only_on_outer_edges(axes, len(betas), ncols)
    _finish_table(fig, hspace, wspace)
    return fig, axes[:len(betas)]
