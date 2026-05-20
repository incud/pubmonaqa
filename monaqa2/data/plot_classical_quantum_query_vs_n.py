from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE
from monaqa2.data.classical_query import get_classical_query_stats, get_classical_query_fit
from monaqa2.data.quantum_query import get_quantum_query_stats, get_quantum_query_fit


# PLOT_ORDER = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
# LEGEND_ORDER = ["local1", "local2", "local3", "uniform", "qemc", "layden"]

PLOT_ORDER = ["local1", "uniform", "layden"]
LEGEND_ORDER = ["local1", "uniform", "layden"]

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


def _filter_stats_table(table, beta: float, min_count: int):
    table = table[np.isclose(table["beta"].astype(float), float(beta)) & (table["count"].astype(int) >= min_count) & np.isfinite(table["n"].astype(float)) & np.isfinite(table["center"].astype(float)) & (table["center"].astype(float) > 0.0)].copy()

    if table.empty:
        return table

    table["n"] = table["n"].astype(float)
    table["center"] = table["center"].astype(float)
    table["spread"] = table["spread"].fillna(0.0).astype(float)

    return table.sort_values("n")


def plot_classical_and_quantum_queries(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and quantum query counts versus n for fixed beta, acceptance a, and initialization mode.

    Points show the selected statistic over instances. Lines are the fitted forms Q(n)=A exp(b n), using the common query fitting utilities. Classical traces are solid lines with circle markers; quantum traces are dashed lines with square markers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    stats = {"classical": {}, "quantum": {}}
    all_n = []

    for proposal in PLOT_ORDER:
        classical_table = get_classical_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=classical_query_file, statistic=statistic, only_ok=only_ok)
        classical_table = _filter_stats_table(classical_table, beta=beta, min_count=min_count)
        if not classical_table.empty:
            stats["classical"][proposal] = classical_table
            all_n.extend(classical_table["n"].tolist())

        quantum_table = get_quantum_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, in_file=quantum_query_file, statistic=statistic, only_ok=only_ok)
        quantum_table = _filter_stats_table(quantum_table, beta=beta, min_count=min_count)
        if not quantum_table.empty:
            stats["quantum"][proposal] = quantum_table
            all_n.extend(quantum_table["n"].tolist())

    if not all_n:
        raise ValueError(f"No valid data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}, statistic={statistic}.")

    n_min = float(np.min(all_n)) if n_plot_min is None else float(n_plot_min)
    n_max = float(np.max(all_n)) if n_plot_max is None else float(n_plot_max)
    n_grid = np.linspace(n_min, n_max, 300)

    styles = {
        "classical": {"linestyle": "-", "marker": "o", "alpha_fill": 0.18, "markersize": 34, "linewidth": 2.2, "name": "classical"},
        "quantum": {"linestyle": "--", "marker": "s", "alpha_fill": 0.10, "markersize": 30, "linewidth": 2.0, "name": "quantum"},
    }

    legend_handles = {}
    legend_labels = {}

    for proposal in PLOT_ORDER:
        color = PROPOSAL_COLORS[proposal]

        for kind in ("classical", "quantum"):
            if proposal not in stats[kind]:
                continue

            table = stats[kind][proposal]
            n_vals = table["n"].to_numpy(dtype=float)
            center = table["center"].to_numpy(dtype=float)
            spread = table["spread"].to_numpy(dtype=float)
            style = styles[kind]

            if show_spread:
                lower, upper = _positive_band(center, spread)
                ax.fill_between(n_vals, lower, upper, color=color, alpha=style["alpha_fill"], linewidth=0.0, zorder=1)

            ax.scatter(n_vals, center, s=style["markersize"], marker=style["marker"], color=color, edgecolors="none", alpha=0.95, zorder=4)

            try:
                if kind == "classical":
                    A, b = get_classical_query_fit(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, beta=beta, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max, only_ok=only_ok)
                else:
                    A, b = get_quantum_query_fit(proposal=proposal, a=a, q0_mode=q0_mode, beta=beta, in_file=quantum_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max, only_ok=only_ok)
            except ValueError:
                continue

            y_fit = A * np.exp(b * n_grid)
            (line,) = ax.plot(n_grid, y_fit, color=color, linewidth=style["linewidth"], linestyle=style["linestyle"], alpha=0.95, zorder=3)
            legend_handles[(proposal, kind)] = line
            legend_labels[(proposal, kind)] = f"{PROPOSAL_LABELS[proposal]} ({style['name']}): {A:.3g} exp({b:.3f} n)"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Number of queries")
    ax.set_title(f"Classical and quantum queries, beta={beta}, a={a}, q0={q0_mode}, epsilon={epsilon:g}")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    handles = []
    labels = []
    for proposal in LEGEND_ORDER:
        if (proposal, "classical") in legend_handles:
            handles.append(legend_handles[(proposal, "classical")])
            labels.append(legend_labels[(proposal, "classical")])
        if (proposal, "quantum") in legend_handles:
            handles.append(legend_handles[(proposal, "quantum")])
            labels.append(legend_labels[(proposal, "quantum")])

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.6)

    return fig, ax
