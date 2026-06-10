from typing import Callable

import matplotlib.pyplot as plt
from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.runtime import get_annealing_queries_quantum_walks
from monaqa2.data.spectral_gap import get_spectral_gap_stats, get_spectral_gap_fit_by_n
from monaqa2.data.classical_query import get_classical_query_stats, get_classical_query_fit_by_n
from monaqa2.data.runtime import (
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


def plot_spectral_gap_vs_n(
    beta: float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
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

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n_grid = np.linspace(float(n_plot_min), float(n_plot_max), 300)
    legend_handles = {}
    legend_labels = {}

    for proposal in PROPOSALS_SORTED:

        color = PROPOSAL_COLORS[proposal]

        # Scatter points of the spectral gap
        n_vals, center, spread = _compact_spectral_gap_points(proposal, beta, statistic, in_file)
        if show_spread:
            lower, upper = _positive_band(center, spread)
            ax.fill_between(n_vals, lower, upper, color=color, alpha=0.25, linewidth=0.0, zorder=1)
        ax.scatter(n_vals, center, s=36, color=color, edgecolors="none", alpha=0.95, zorder=3)

        # Fitting line
        A, b = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta, in_file=in_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        delta_fit = A * np.exp(-b * n_grid)
        (fit_line,) = ax.plot(n_grid, delta_fit, color=color, linewidth=2.0, linestyle="-", alpha=0.90, zorder=2)

        # Legend
        legend_handles[proposal] = fit_line
        legend_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]}: fit ${A:.3f} \times \exp(-{b:.3f} n)$"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Spectral gap $\delta$")
    ax.set_title(rf"Spectral gap over instances, $\beta={beta}$, statistic={statistic}")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

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
        query_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} queries: ${A_q:.3g}\exp({b_q:.3f} n)$"

        # Optional inverse spectral-gap fitting line
        if show_inverse_gap and ax_gap is not None:
            A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            b_g = max(float(b_g), 0.0)
            inv_gap_fit = (1.0 / A_g) * np.exp(b_g * n_grid)
            (gap_line,) = ax_gap.plot(n_grid, inv_gap_fit, color=color, linewidth=2.0, linestyle="--", alpha=0.90, zorder=2)

            # Legend
            gap_handles[proposal] = gap_line
            gap_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} inverse gap: ${1.0 / A_g:.3g}\exp({b_g:.3f} n)$"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Classical queries")
    ax.set_title(rf"Classical queries, $\beta={beta}$, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

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

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax, ax_gap



def plot_annealing_classical_queries_and_quantum_queries_vs_n(
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
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot total annealing classical-query estimates and quantum-walk query estimates versus n.

    Solid lines use fitted classical last-step query counts summed over the annealing schedule.
    Dashed lines use fitted spectral gaps converted into QSP/QSVT filter degrees.
    """
    if n_plot_min is None or n_plot_max is None:
        n_plot_min = 1
        n_plot_max = 100

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n_vals = np.linspace(float(n_plot_min), float(n_plot_max), 300)
    classical_handles = {}
    classical_labels = {}
    quantum_handles = {}
    quantum_labels = {}

    for proposal in PROPOSALS_SORTED:

        color = PROPOSAL_COLORS[proposal]
        classical_queries = []
        quantum_queries = []

        for n in n_vals:
            schedule = annealing_schedule_generator(n, beta)
            classical_total = 0.0
            spectral_gaps = []

            for beta_t in schedule:
                A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                classical_total += A_q * np.exp(b_q * n)

                A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                spectral_gaps.append(A_g * np.exp(-b_g * n))

            classical_queries.append(classical_total)
            quantum_queries.append(get_annealing_queries_quantum_walks(n, epsilon, spectral_gaps))
            
        (classical_line,) = ax.plot(n_vals, classical_queries, color=color, linewidth=2.0, linestyle="--", alpha=0.90)
        (quantum_line,) = ax.plot(n_vals, quantum_queries, color=color, linewidth=2.0, linestyle="-", alpha=0.90)

        classical_handles[proposal] = classical_line
        quantum_handles[proposal] = quantum_line
        classical_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} classical annealing queries"
        quantum_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} quantum-walk queries"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Queries")
    ax.set_title(rf"Annealing queries, $\beta_F={beta}$, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    handles = [classical_handles[p] for p in PROPOSALS_SORTED if p in classical_handles]
    labels = [classical_labels[p] for p in PROPOSALS_SORTED if p in classical_labels]
    handles += [quantum_handles[p] for p in PROPOSALS_SORTED if p in quantum_handles]
    labels += [quantum_labels[p] for p in PROPOSALS_SORTED if p in quantum_labels]

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax


def plot_annealing_classical_queries_and_quantum_runtime_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_device: str = "fpga",
    physical_operation_time: float = 1e-8, 
    physical_measurement_time: float = 1e-7, 
    physical_error_rate: float = 1e-5, 
    num_trotter_steps: int = 50,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot annealing runtime estimates versus n.

    Dashed lines use fitted classical query counts converted to runtime.
    Solid lines use fitted spectral gaps converted to QSP/QSVT quantum-walk runtime.
    """
    if n_plot_min is None or n_plot_max is None:
        n_plot_min = 1
        n_plot_max = 100

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n_vals = np.linspace(float(n_plot_min), float(n_plot_max), 300)

    classical_handles = {}
    classical_labels = {}
    quantum_handles = {}
    quantum_labels = {}

    for proposal in PROPOSALS_SORTED:

        print(f"\n{proposal=}: ", end="")
        color = PROPOSAL_COLORS[proposal]
        classical_runtime = []
        quantum_runtime = []

        for n in n_vals:
            print(f"{n} ", end="", flush=True)
            schedule = annealing_schedule_generator(n, beta)
            vec_queries = []
            spectral_gaps = []

            for beta_t in schedule:
                A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                vec_queries.append(A_q * np.exp(b_q * n))

                A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
                spectral_gaps.append(A_g * np.exp(-b_g * n))

            if proposal == "local1":
                classical_runtime.append(get_annealing_time_classical_walk_local(n, vec_queries, device=classical_device))
                quantum_runtime.append(get_annealing_time_quantum_walk_local(n, epsilon, schedule, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, num_trotter_steps))
            elif proposal == "uniform":
                classical_runtime.append(get_annealing_time_classical_walk_uniform(n, vec_queries, device=classical_device))
                quantum_runtime.append(get_annealing_time_quantum_walk_uniform(n, epsilon, schedule, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, num_trotter_steps))
            elif proposal == "layden":
                classical_runtime.append(get_annealing_time_classical_walk_qemc(n, vec_queries, physical_operation_time, physical_measurement_time, physical_error_rate, num_trotter_steps))
                quantum_runtime.append(get_annealing_time_quantum_walk_qemc(n, epsilon, schedule, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, num_trotter_steps))

        (classical_line,) = ax.plot(n_vals, classical_runtime, color=color, linewidth=2.0, linestyle="--", alpha=0.90)
        (quantum_line,) = ax.plot(n_vals, quantum_runtime, color=color, linewidth=2.0, linestyle="-", alpha=0.90)

        classical_handles[proposal] = classical_line
        quantum_handles[proposal] = quantum_line
        classical_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} classical runtime"
        quantum_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} quantum-walk runtime"

    # Info about the scale size of the 
    one_day = 24 * 60 * 60
    one_year = 365.25 * one_day
    ten_years = 10 * one_year
    ax.axhline(one_day, color="black", linestyle=":", linewidth=1.5, alpha=0.85, zorder=0)
    ax.axhline(ten_years, color="black", linestyle=":", linewidth=1.5, alpha=0.85, zorder=0)


    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Runtime [s]")
    ax.set_title(rf"Annealing runtime, $\beta_F={beta}$, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    handles = [classical_handles[p] for p in PROPOSALS_SORTED if p in classical_handles]
    labels = [classical_labels[p] for p in PROPOSALS_SORTED if p in classical_labels]
    handles += [quantum_handles[p] for p in PROPOSALS_SORTED if p in quantum_handles]
    labels += [quantum_labels[p] for p in PROPOSALS_SORTED if p in quantum_labels]

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax