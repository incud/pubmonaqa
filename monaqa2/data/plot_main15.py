from typing import Callable

import matplotlib.pyplot as plt
from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.runtime import get_annealing_queries_quantum_walks
from monaqa2.data.spectral_gap import get_spectral_gap_stats, get_spectral_gap_fit_by_n, get_spectral_gap_fit_by_beta
from monaqa2.data.classical_query import get_classical_query_stats, get_classical_query_fit_by_n
from monaqa2.data.runtime import (
    split_quantum_error_budget,
    get_annealing_time_classical_walk_local,
    get_annealing_time_classical_walk_uniform,
    get_annealing_time_classical_walk_qemc,
    get_annealing_time_quantum_walk_local,
    get_annealing_time_quantum_walk_uniform,
    get_annealing_time_quantum_walk_qemc,
)
import numpy as np
from pathlib import Path


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
    "uniform": "#7A7A7A",
    "local1": "#56B4E9",
    "local2": "#0072B2",
    "local3": "#009E73",
    "qemc": "#E69F00",
    "layden": "#D55E00",
    "layden_aux": "#E69F00",
}


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
    """Plot a curve with a solid calibrated part and a dashed projected part."""
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
        (proj_line,) = ax.plot(n_vals[projected], y[projected], color=color, linewidth=linewidth, linestyle=(0, (4, 2)), alpha=alpha, label=None if line is not None else label, zorder=zorder)
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

        color = PROPOSAL_COLORS[proposal]

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

        color = PROPOSAL_COLORS[proposal]

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

        color = PROPOSAL_COLORS[proposal]

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
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot total annealing classical-query estimates and quantum-walk query estimates versus n.

    Thin lines show classical-walk query estimates.
    Thick lines show quantum-walk query estimates.
    Dashed portions indicate projection beyond the n-fit window.
    The optional lower panel shows the number of charged annealing steps.

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
    :param title: Optional title.
    :param show_legend: Whether to show the legend.
    :param debug: If True, print schedule changes and curve decreases.
    :param schedule_diagnostic_generator: Schedule generator used only for the lower schedule panel. If None, the function uses ``annealing_schedule_generator.base_schedule_generator`` when available, otherwise ``annealing_schedule_generator`` itself.
    :param show_schedule_panel: Whether to show the lower schedule-length panel.
    :param show_schedule_vertical_lines: Whether to show vertical lines at schedule-length changes.
    :param legend_y_shift: Vertical legend anchor. More negative moves the legend farther down.
    :return: Tuple ``(fig, ax)``.
    """
    if n_plot_min is None or n_plot_max is None:
        n_plot_min = 1
        n_plot_max = 100

    n_vals = np.arange(int(n_plot_min), int(n_plot_max) + 1, dtype=int)

    if schedule_diagnostic_generator is None:
        schedule_diagnostic_generator = getattr(
            annealing_schedule_generator,
            "base_schedule_generator",
            annealing_schedule_generator,
        )

    if fig is None or ax is None:
        if show_schedule_panel:
            fig, (ax, ax_sched) = plt.subplots(
                2,
                1,
                figsize=(8.5, 5.45),
                sharex=True,
                gridspec_kw={"height_ratios": [4.0, 0.75], "hspace": 0.06},
            )
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

        return (
            n_values[fit_mask],
            y_values[fit_mask],
            n_values[proj_mask],
            y_values[proj_mask],
        )

    def _plot_curve_with_projection(
        y_values: list[float],
        color: str,
        linewidth: float,
        label: str,
        zorder: int,
    ):
        y_values = np.asarray(y_values, dtype=float)

        n_fit, y_fit, n_proj, y_proj = _split_projection(n_vals, y_values)

        line = None
        if len(n_fit):
            (line,) = ax.plot(
                n_fit,
                y_fit,
                color=color,
                linewidth=linewidth,
                linestyle="-",
                alpha=0.92,
                zorder=zorder,
                label=label,
            )

        if len(n_proj):
            (proj_line,) = ax.plot(
                n_proj,
                y_proj,
                color=color,
                linewidth=linewidth,
                linestyle=(0, (3.0, 1.7)),
                alpha=0.92,
                zorder=zorder,
                label=label if line is None else None,
            )
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

    diagnostic_schedules, diagnostic_lengths, diagnostic_change_positions = _schedule_diagnostics(
        schedule_diagnostic_generator
    )

    if debug:
        if schedule_diagnostic_generator is annealing_schedule_generator:
            print("[debug] schedule panel uses the plotted schedule generator")
        else:
            print("[debug] schedule panel uses a separate diagnostic schedule generator")
        print(f"[debug] schedule-change positions: {diagnostic_change_positions}")
        for i in range(1, len(n_vals)):
            if diagnostic_lengths[i] != diagnostic_lengths[i - 1]:
                print(
                    f"[debug] schedule length changes at n={n_vals[i]}: "
                    f"{diagnostic_lengths[i - 1]} -> {diagnostic_lengths[i]}"
                )
                print(f"        previous={diagnostic_schedules[i - 1]}")
                print(f"        current ={diagnostic_schedules[i]}")

    if show_schedule_vertical_lines:
        for x in diagnostic_change_positions:
            ax.axvline(
                x,
                color="0.72",
                linewidth=0.45,
                alpha=0.95,
                zorder=0,
            )

    classical_handles = {}
    classical_labels = {}
    quantum_handles = {}
    quantum_labels = {}

    for proposal in PROPOSALS_SORTED:
        color = PROPOSAL_COLORS[proposal]
        classical_queries = []
        quantum_queries = []
        plotted_schedules = []

        for n in n_vals:
            schedule = annealing_schedule_generator(int(n), beta)
            plotted_schedules.append(schedule)

            classical_total = 0.0
            spectral_gaps = []

            for beta_t in schedule:
                A_q, b_q = get_classical_query_fit_by_n(
                    proposal=proposal,
                    a=np.inf,
                    q0_mode="bhattacharyya",
                    epsilon=epsilon,
                    beta=beta_t,
                    in_file=classical_query_file,
                    statistic=statistic,
                    n_min=n_fit_min,
                    n_max=n_fit_max,
                )
                classical_total += A_q * np.exp(b_q * n)

                A_g, b_g = get_spectral_gap_fit_by_n(
                    proposal=proposal,
                    a=np.inf,
                    fixed_beta=beta_t,
                    in_file=spectral_gap_file,
                    statistic=statistic,
                    n_min=n_fit_min,
                    n_max=n_fit_max,
                )
                b_g = max(float(b_g), 0.0)
                spectral_gaps.append(A_g * np.exp(-b_g * n))

            classical_queries.append(classical_total)
            quantum_queries.append(get_annealing_queries_quantum_walks(int(n), epsilon, spectral_gaps))

        classical_line = _plot_curve_with_projection(
            classical_queries,
            color=color,
            linewidth=1.05,
            label=rf"{PROPOSAL_LABELS[proposal]} classical walk",
            zorder=2,
        )
        quantum_line = _plot_curve_with_projection(
            quantum_queries,
            color=color,
            linewidth=2.35,
            label=rf"{PROPOSAL_LABELS[proposal]} quantum walk",
            zorder=3,
        )

        classical_handles[proposal] = classical_line
        quantum_handles[proposal] = quantum_line
        classical_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} classical walk"
        quantum_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} quantum walk"

        if debug:
            classical_arr = np.asarray(classical_queries, dtype=float)
            quantum_arr = np.asarray(quantum_queries, dtype=float)

            for name, arr in [
                ("classical queries", classical_arr),
                ("quantum-walk queries", quantum_arr),
            ]:
                decreases = np.where(arr[1:] < arr[:-1])[0]
                if len(decreases):
                    print(f"\n[debug] decreases for {proposal} {name}")
                for idx in decreases:
                    n0 = int(n_vals[idx])
                    n1 = int(n_vals[idx + 1])
                    y0 = arr[idx]
                    y1 = arr[idx + 1]
                    ratio = y1 / y0 if y0 > 0.0 else np.nan
                    print(
                        f"  n {n0}->{n1}: {y0:.6g}->{y1:.6g} "
                        f"ratio={ratio:.6g} "
                        f"schedule_len={len(plotted_schedules[idx])}->{len(plotted_schedules[idx + 1])}"
                    )
                    print(f"    previous schedule={plotted_schedules[idx]}")
                    print(f"    current  schedule={plotted_schedules[idx + 1]}")

                    for beta_t in plotted_schedules[idx + 1]:
                        A_q, b_q = get_classical_query_fit_by_n(
                            proposal=proposal,
                            a=np.inf,
                            q0_mode="bhattacharyya",
                            epsilon=epsilon,
                            beta=beta_t,
                            in_file=classical_query_file,
                            statistic=statistic,
                            n_min=n_fit_min,
                            n_max=n_fit_max,
                        )
                        A_g, b_g = get_spectral_gap_fit_by_n(
                            proposal=proposal,
                            a=np.inf,
                            fixed_beta=beta_t,
                            in_file=spectral_gap_file,
                            statistic=statistic,
                            n_min=n_fit_min,
                            n_max=n_fit_max,
                        )
                        b_g = max(float(b_g), 0.0)
                        q_contribution = A_q * np.exp(b_q * n1)
                        gap_value = A_g * np.exp(-b_g * n1)
                        print(
                            f"    beta_t={beta_t:.12g} "
                            f"b_q/log2={b_q / np.log(2):+.6g} "
                            f"q={q_contribution:.6g} "
                            f"b_g/log2={b_g / np.log(2):+.6g} "
                            f"delta={gap_value:.6g}"
                        )

    ax.set_yscale("log")
    ax.set_ylabel(r"Queries")
    ax.set_title(title if title is not None else rf"Annealing queries, $\beta_F={beta}$, $\epsilon={epsilon:g}$")
    ax.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    handles = [classical_handles[p] for p in PROPOSALS_SORTED if p in classical_handles]
    labels = [classical_labels[p] for p in PROPOSALS_SORTED if p in classical_labels]
    handles += [quantum_handles[p] for p in PROPOSALS_SORTED if p in quantum_handles]
    labels += [quantum_labels[p] for p in PROPOSALS_SORTED if p in quantum_labels]

    if show_schedule_panel and ax_sched is not None:
        if show_schedule_vertical_lines:
            for x in diagnostic_change_positions:
                ax_sched.axvline(
                    x,
                    color="0.72",
                    linewidth=0.45,
                    alpha=0.95,
                    zorder=0,
                )

        ax_sched.step(
            n_vals,
            diagnostic_lengths,
            where="post",
            color="0.30",
            linewidth=0.95,
            zorder=2,
        )

        ax_sched.set_ylabel("Annealing\nsteps")
        ax_sched.set_xlabel(r"$n$", labelpad=xlabel_labelpad)
        ax_sched.grid(False)

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

    if show_legend:
        legend_axis.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=legend_anchor,
            ncol=1,
            borderaxespad=0.0,
            handlelength=2.4,
        )

    return fig, ax


from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from matplotlib.colors import to_rgb


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
    grid_color: str = "0.88",
    grid_linewidth: float = 0.55,
    hide_classical_qemc_area: bool = False,
    show_runtime_thresholds: bool = False,
    show_runtime_threshold_labels: bool = False,
    classical_color_darken: float = 0.58,
    differentiate_projection: bool = True,
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

    if n_plot_min is None:
        n_plot_min = 1
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

    def _plot_black_runtime_line(values: list[float] | np.ndarray, label: str, zorder: int = 6):
        values = np.asarray(values, dtype=float)
        positive = values[values > 0.0]
        floor = np.min(positive) * 1e-4 if positive.size else np.finfo(float).tiny
        values = np.maximum(values, floor)
        (handle,) = ax.plot(n_vals, values, color="black", linewidth=1.35, linestyle="-", alpha=0.95, zorder=zorder, label=label)
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
    runtime_curves = {}
    threshold_xs = []
    eps_SF = epsilon / 4.0

    for proposal in proposals_to_plot:
        color_quantum = PROPOSAL_COLORS[proposal]
        color_classical_raw = PROPOSAL_COLORS.get("layden_aux", color_quantum) if proposal == "layden" else color_quantum
        color_classical = _darken_color(color_classical_raw, factor=classical_color_darken)

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

        if proposal == "uniform":
            classical_handle = _plot_black_runtime_line(runtime_curves[proposal]["classical_nominal"], label=rf"{PROPOSAL_LABELS[proposal]} classical walk")
            handles.append(classical_handle)
            labels.append(rf"{PROPOSAL_LABELS[proposal]} classical walk")
        elif not (proposal == "layden" and effective_hide_classical_qemc_area):
            classical_handle = _fill_area_with_projection(classical_min, classical_max, color=color_classical, label=rf"{PROPOSAL_LABELS[proposal]} classical walk", alpha=0.30, zorder=2, edge_linewidth=classical_edge)
            handles.append(classical_handle)
            labels.append(rf"{PROPOSAL_LABELS[proposal]} classical walk")

        quantum_handle = _fill_area_with_projection(quantum_min, quantum_max, color=color_quantum, label=rf"{PROPOSAL_LABELS[proposal]} quantum walk", alpha=0.34, zorder=3, edge_linewidth=area_edge_linewidth)
        handles.append(quantum_handle)
        labels.append(rf"{PROPOSAL_LABELS[proposal]} quantum walk")

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
    ax.grid(True, which="major", axis="both", color=grid_color, linewidth=grid_linewidth, zorder=0)
    ax.set_ylabel(r"Runtime [s]")
    ax.set_title(title if title is not None else rf"Annealing runtime, $\beta_F={beta}$, $\epsilon={epsilon:g}$")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_schedule_panel and ax_sched is not None:
        if show_schedule_vertical_lines:
            for x in diagnostic_change_positions:
                ax_sched.axvline(x, color="0.72", linewidth=0.45, alpha=0.95, zorder=0)

        ax_sched.step(n_vals, diagnostic_lengths, where="post", color="0.30", linewidth=0.95, zorder=2)
        ax_sched.set_ylabel("Annealing\nsteps")
        ax_sched.set_xlabel(r"$n$", labelpad=xlabel_labelpad)

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
        ax_sched.grid(True, which="major", axis="both", color=grid_color, linewidth=grid_linewidth, zorder=0)

        for spine in ["top", "right"]:
            ax_sched.spines[spine].set_visible(False)

        legend_anchor = (0.5, legend_y_shift)
        legend_axis = ax_sched
    else:
        ax.set_xlabel(r"$n$", labelpad=xlabel_labelpad)
        legend_anchor = (0.5, legend_y_shift)
        legend_axis = ax

    if show_legend:
        legend_axis.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=legend_anchor, ncol=1, borderaxespad=0.0, handlelength=2.4)

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
    grid_color: str = "0.88",
    grid_linewidth: float = 0.55,
    inset_width: str = "31%",
    inset_height: str = "38%",
    inset_loc: str = "lower right",
    inset_borderpad: float = 2.0,
    hide_classical_qemc_area: bool = False,
    show_runtime_thresholds: bool = False,
    show_runtime_threshold_labels: bool = True,
    classical_color_darken: float = 0.58,
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
        title="",
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
    )

    inset_ax.set_title("")
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.set_ylim(one_microsecond, one_billion_years)
    inset_ax.set_yticks([one_microsecond, one_day, ten_years, one_billion_years])
    inset_ax.set_yticklabels(["1 µs", "one day", "10 years", "1B years"])
    inset_ax.tick_params(axis="both", which="major", labelsize=7)
    inset_ax.grid(
        True,
        which="major",
        axis="both",
        color=grid_color,
        linewidth=grid_linewidth,
        zorder=0,
    )

    return fig, ax