from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_n
from monaqa2.data.classical_query import get_classical_query_stats, get_classical_query_fit_by_n


# PROPOSALS = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
PROPOSALS = ["local1", "uniform", "layden"]

PROPOSAL_LABELS = {
    "uniform": "Uniform",
    "local1": "Local spin-flip",
    "local2": "Local spin-flip (double)",
    "local3": "Local spin-flip (triple)",
    "qemc": "Quantum enhanced (best hyperparameters)",
    "layden": "Quantum enhanced",
}

PROPOSAL_COLORS = {
    "uniform": "#7A7A7A",
    "local1": "#56B4E9",
    "local2": "#0072B2",
    "local3": "#009E73",
    "qemc": "#E69F00",
    "layden": "#D55E00",
}


def _positive_band(center: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = center - spread
    upper = center + spread
    positive = np.concatenate([center[center > 0.0], upper[upper > 0.0]])
    floor = np.min(positive) * 1e-2 if positive.size else np.finfo(float).tiny
    return np.maximum(lower, floor), np.maximum(upper, floor)


def plot_classical_queries_vs_n(
    beta: float,
    a: int | float,
    q0_mode: str,
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
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes | None]:
    """
    Plot classical queries versus n for fixed beta, acceptance parameter, initialization, and epsilon.

    Left axis:
        scatter = classical-query statistic center over instances;
        transparent band = center +/- spread;
        solid line = fit T(n) = A exp(b n).

    Right axis, enabled by show_inverse_gap:
        dashed line = inverse spectral-gap fit, 1/delta(n), using delta(n) = A exp(-b n).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    query_tables = {}
    all_n = []

    for proposal in PROPOSALS:
        table = get_classical_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=classical_query_file, statistic=statistic)
        table = table[np.isclose(table["beta"].astype(float), float(beta)) & (table["count"].astype(int) >= min_count) & np.isfinite(table["n"].astype(float)) & np.isfinite(table["center"].astype(float)) & (table["center"].astype(float) > 0.0)].copy()

        if table.empty:
            continue

        table["n"] = table["n"].astype(float)
        table["center"] = table["center"].astype(float)
        table["spread"] = table["spread"].fillna(0.0).astype(float)
        table = table.sort_values("n")
        query_tables[proposal] = table
        all_n.extend(table["n"].tolist())

    if not all_n:
        raise ValueError(f"No valid classical-query data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}.")

    n_min = float(np.min(all_n)) if n_plot_min is None else float(n_plot_min)
    n_max = float(np.max(all_n)) if n_plot_max is None else float(n_plot_max)
    n_grid = np.linspace(n_min, n_max, 300)
    ax_gap = ax.twinx() if show_inverse_gap else None

    query_handles = {}
    query_labels = {}
    gap_handles = {}
    gap_labels = {}

    for proposal in PROPOSALS:
        if proposal not in query_tables:
            continue

        color = PROPOSAL_COLORS[proposal]
        table = query_tables[proposal]
        n_vals = table["n"].to_numpy(dtype=float)
        center = table["center"].to_numpy(dtype=float)
        spread = table["spread"].to_numpy(dtype=float)

        if show_spread:
            lower, upper = _positive_band(center, spread)
            ax.fill_between(n_vals, lower, upper, color=color, alpha=0.20, linewidth=0.0, zorder=1)

        ax.scatter(n_vals, center, s=36, color=color, edgecolors="none", alpha=0.95, zorder=3)

        try:
            A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, beta=beta, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        except ValueError:
            continue

        query_fit = A_q * np.exp(b_q * n_grid)
        (query_line,) = ax.plot(n_grid, query_fit, color=color, linewidth=2.0, linestyle="-", alpha=0.90, zorder=2)
        query_handles[proposal] = query_line
        query_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} queries: ${A_q:.3g}\exp({b_q:.3f}n)$"

        if show_inverse_gap and ax_gap is not None:
            try:
                A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=a, fixed_beta=beta, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
            except ValueError:
                continue

            if A_g <= 0.0 or not np.isfinite(A_g):
                continue

            b_g = max(float(b_g), 0.0)
            inv_gap_fit = (1.0 / A_g) * np.exp(b_g * n_grid)
            (gap_line,) = ax_gap.plot(n_grid, inv_gap_fit, color=color, linewidth=2.0, linestyle="--", alpha=0.90, zorder=2)
            gap_handles[proposal] = gap_line
            gap_labels[proposal] = rf"{PROPOSAL_LABELS[proposal]} inverse gap: ${1.0 / A_g:.3g}\exp({b_g:.3f}n)$"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Classical queries")
    ax.set_title(rf"Classical queries, $\beta={beta}$, $a={a}$, $q_0={q0_mode}$, $\epsilon={epsilon:g}$")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    ax_gap.set_yscale("log")
    ax_gap.set_ylabel(r"Inverse spectral gap $1/\delta$")
    ax_gap.grid(False)

    left_ylim = ax.get_ylim()
    right_ylim = ax_gap.get_ylim()
    shared_ylim = (min(left_ylim[0], right_ylim[0]), max(left_ylim[1], right_ylim[1]))
    ax.set_ylim(shared_ylim)
    ax_gap.set_ylim(shared_ylim)

    handles = [query_handles[p] for p in PROPOSALS if p in query_handles]
    labels = [query_labels[p] for p in PROPOSALS if p in query_labels]

    if show_inverse_gap:
        handles += [gap_handles[p] for p in PROPOSALS if p in gap_handles]
        labels += [gap_labels[p] for p in PROPOSALS if p in gap_labels]

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax, ax_gap
