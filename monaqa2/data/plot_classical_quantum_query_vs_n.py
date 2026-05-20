from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE
from monaqa2.data.table_classical_quantum_query_vs_n import get_classical_quantum_query_vs_n_table


# PLOT_ORDER = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
# LEGEND_ORDER = ["local1", "local2", "local3", "uniform", "qemc", "layden"]

PLOT_ORDER = ["local1", "uniform", "qemc"]
LEGEND_ORDER = ["local1", "uniform", "qemc"]

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


def plot_classical_and_quantum_queries(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_project_values=tuple(range(10, 101, 10)),
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and quantum query counts from the table returned by get_classical_quantum_query_vs_n_table.

    Observed points are scatter markers. Projected points are connected into the fitted line A exp(b n).
    """
    table = get_classical_quantum_query_vs_n_table(beta=beta, a=a, q0_mode=q0_mode, epsilon=epsilon, classical_query_file=classical_query_file, quantum_query_file=quantum_query_file, statistic=statistic, min_count=min_count, n_fit_min=n_fit_min, n_fit_max=n_fit_max, n_project_values=n_project_values, moves=tuple(PLOT_ORDER), only_ok=only_ok)

    if table.empty:
        raise ValueError(f"No valid data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}, statistic={statistic}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    styles = {
        "classical": {"linestyle": "-", "marker": "o", "markersize": 34, "linewidth": 2.2, "name": "classical"},
        "quantum": {"linestyle": "--", "marker": "s", "markersize": 30, "linewidth": 2.0, "name": "quantum"},
    }

    legend_handles = {}
    legend_labels = {}

    for move in PLOT_ORDER:
        color = PROPOSAL_COLORS[move]

        for walk in ("classical", "quantum"):
            sub = table[(table["move"] == move) & (table["walk"] == walk)]
            observed = sub[sub["source"] == "observed"].sort_values("n")
            projected = sub[sub["source"] == "projected"].sort_values("n")

            if observed.empty and projected.empty:
                continue

            style = styles[walk]

            if not observed.empty:
                ax.scatter(observed["n"].to_numpy(dtype=float), observed["num_queries"].to_numpy(dtype=float), s=style["markersize"], marker=style["marker"], color=color, edgecolors="none", alpha=0.95, zorder=4)

            if not projected.empty:
                (line,) = ax.plot(projected["n"].to_numpy(dtype=float), projected["num_queries"].to_numpy(dtype=float), color=color, linewidth=style["linewidth"], linestyle=style["linestyle"], alpha=0.95, zorder=3)
                legend_handles[(move, walk)] = line
                A = float(projected["A"].iloc[0])
                b = float(projected["b"].iloc[0])
                legend_labels[(move, walk)] = f"{PROPOSAL_LABELS[move]} ({style['name']}): {A:.3g} exp({b:.3f} n)"

    if n_plot_min is not None or n_plot_max is not None:
        ax.set_xlim(left=n_plot_min, right=n_plot_max)

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Number of queries")
    ax.set_title(f"Classical and quantum queries, beta={beta}, a={a}, q0={q0_mode}, epsilon={epsilon:g}")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    handles = []
    labels = []
    for move in LEGEND_ORDER:
        if (move, "classical") in legend_handles:
            handles.append(legend_handles[(move, "classical")])
            labels.append(legend_labels[(move, "classical")])
        if (move, "quantum") in legend_handles:
            handles.append(legend_handles[(move, "quantum")])
            labels.append(legend_labels[(move, "quantum")])

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.6)

    return fig, ax
