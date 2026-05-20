from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE
from monaqa2.data.table_classical_quantum_query_vs_n import get_classical_quantum_query_vs_n_table
from monaqa2.data.runtime_new import (
    get_time_classical_walk_local,
    get_time_classical_walk_qemc,
    get_time_classical_walk_uniform,
    get_time_quantum_walk_local,
    get_time_quantum_walk_qemc,
    get_time_quantum_walk_uniform,
)


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


def _runtime_from_queries(n: int, walk: str, move: str, num_queries: float, num_trotter_steps: int) -> float:
    q = int(np.ceil(float(num_queries)))

    if walk == "classical" and move == "uniform":
        return float(get_time_classical_walk_uniform(n, q))
    if walk == "classical" and move.startswith("local"):
        return float(get_time_classical_walk_local(n, q))
    if walk == "classical" and move in ["qemc", "layden"]:
        return float(get_time_classical_walk_qemc(n, q, num_trotter_steps))

    if walk == "quantum" and move == "uniform":
        return float(get_time_quantum_walk_uniform(n, q))
    if walk == "quantum" and move.startswith("local"):
        return float(get_time_quantum_walk_local(n, q))
    if walk == "quantum" and move in ["qemc", "layden"]:
        return float(get_time_quantum_walk_qemc(n, q))

    raise ValueError(f"unknown walk/move pair: walk={walk}, move={move}")


def _query_table_to_runtime_table(query_table: pd.DataFrame, num_trotter_steps: int) -> pd.DataFrame:
    rows = []

    for row in query_table.itertuples(index=False):
        n = int(row.n)
        walk = str(row.walk)
        move = str(row.move)
        runtime = _runtime_from_queries(n=n, walk=walk, move=move, num_queries=float(row.num_queries), num_trotter_steps=num_trotter_steps)

        if np.isfinite(runtime) and runtime > 0.0:
            rows.append({"n": n, "walk": walk, "move": move, "runtime": float(runtime), "source": str(row.source), "num_queries": float(row.num_queries), "query_A": float(row.A), "query_b": float(row.b)})

    return pd.DataFrame(rows, columns=["n", "walk", "move", "runtime", "source", "num_queries", "query_A", "query_b"])


def _runtime_power(walk: str, move: str) -> int:
    if walk == "classical" and move.startswith("local"):
        return 1
    if walk == "classical" and move == "uniform":
        return 2
    if walk == "classical" and move in ["qemc", "layden"]:
        return 1
    return 0


def _fit_runtime_fixed_query_exponent(table: pd.DataFrame, walk: str, move: str, n_min: int | None, n_max: int | None) -> tuple[float, float, int]:
    table = table.copy()

    if n_min is not None:
        table = table[table["n"].astype(int) >= int(n_min)]
    if n_max is not None:
        table = table[table["n"].astype(int) <= int(n_max)]

    n = table["n"].to_numpy(dtype=float)
    y = table["runtime"].to_numpy(dtype=float)
    query_b_values = pd.to_numeric(table["query_b"], errors="coerce").to_numpy(dtype=float)
    query_b_values = query_b_values[np.isfinite(query_b_values)]

    if query_b_values.size == 0:
        raise ValueError("No query exponent available. Use projected rows from get_classical_quantum_query_vs_n_table.")

    b = float(query_b_values[0])
    power = _runtime_power(walk, move)
    mask = np.isfinite(n) & np.isfinite(y) & (n > 0.0) & (y > 0.0)
    n, y = n[mask], y[mask]

    if y.size < 1:
        raise ValueError("Need at least one positive runtime point to fit.")

    log_C = float(np.mean(np.log(y) - power * np.log(n) - b * n))
    return float(np.exp(log_C)), b, int(power)


def _runtime_fit_values(A: float, b: float, power: int, n: np.ndarray) -> np.ndarray:
    return A * (n ** power) * np.exp(b * n)


def _fit_label(move: str, walk: str, A: float, b: float, power: int) -> str:
    if power == 0:
        return f"{PROPOSAL_LABELS[move]} ({walk}): {A:.3g} exp({b:.3f} n)"
    if power == 1:
        return f"{PROPOSAL_LABELS[move]} ({walk}): {A:.3g} n exp({b:.3f} n)"
    if power == 2:
        return f"{PROPOSAL_LABELS[move]} ({walk}): {A:.3g} n^2 exp({b:.3f} n)"
    return f"{PROPOSAL_LABELS[move]} ({walk}): {A:.3g} n^{power} exp({b:.3f} n)"


def plot_classical_logicalquantum_runtime_vs_n(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    num_trotter_steps: int,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 100,
    n_project_values=tuple(range(10, 101, 10)),
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and logical-quantum runtime versus n using the shared query table.

    The query table already contains observed and projected points. This function only converts num_queries to runtime and fits the runtime table. The query exponent B is inherited from the query fit num_queries(n)=A_q exp(B n). Runtime fits only the prefactor C in C P_move(n) exp(B n). Classical local/qemc use P_move(n)=n, classical uniform uses P_move(n)=n^2, and quantum uses P_move(n)=1.
    """
    query_table = get_classical_quantum_query_vs_n_table(beta=beta, a=a, q0_mode=q0_mode, epsilon=epsilon, classical_query_file=classical_query_file, quantum_query_file=quantum_query_file, statistic=statistic, min_count=min_count, n_fit_min=n_fit_min, n_fit_max=10, n_project_values=n_project_values, moves=tuple(PLOT_ORDER), only_ok=only_ok)
    runtime_table = _query_table_to_runtime_table(query_table, num_trotter_steps=num_trotter_steps)

    if runtime_table.empty:
        raise ValueError(f"No valid runtime data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}, statistic={statistic}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    n_min = float(runtime_table["n"].min()) if n_plot_min is None else float(n_plot_min)
    n_max = float(runtime_table["n"].max()) if n_plot_max is None else float(n_plot_max)
    n_grid = np.linspace(n_min, n_max, 300)

    styles = {
        "classical": {"linestyle": "-", "marker": "o", "markersize_observed": 34, "markersize_projected": 18, "linewidth": 2.2, "alpha_projected": 0.35},
        "quantum": {"linestyle": "--", "marker": "s", "markersize_observed": 30, "markersize_projected": 14, "linewidth": 2.0, "alpha_projected": 0.35},
    }

    legend_handles = {}
    legend_labels = {}

    for move in PLOT_ORDER:
        color = PROPOSAL_COLORS[move]

        for walk in ("classical", "quantum"):
            sub = runtime_table[(runtime_table["move"] == move) & (runtime_table["walk"] == walk)].copy()

            if sub.empty:
                continue

            observed = sub[sub["source"] == "observed"].sort_values("n")
            projected = sub[sub["source"] == "projected"].sort_values("n")
            fit_table = projected if not projected.empty else sub
            style = styles[walk]

            if not observed.empty:
                ax.scatter(observed["n"].to_numpy(dtype=float), observed["runtime"].to_numpy(dtype=float), s=style["markersize_observed"], marker=style["marker"], color=color, edgecolors="none", alpha=0.95, zorder=4)

            if not projected.empty:
                ax.scatter(projected["n"].to_numpy(dtype=float), projected["runtime"].to_numpy(dtype=float), s=style["markersize_projected"], marker=style["marker"], color=color, edgecolors="none", alpha=style["alpha_projected"], zorder=3)

            try:
                A, b, power = _fit_runtime_fixed_query_exponent(fit_table, walk=walk, move=move, n_min=n_fit_min, n_max=n_fit_max)
            except ValueError:
                continue

            y_fit = _runtime_fit_values(A, b, power, n_grid)
            (line,) = ax.plot(n_grid, y_fit, color=color, linewidth=style["linewidth"], linestyle=style["linestyle"], alpha=0.95, zorder=2)
            legend_handles[(move, walk)] = line
            legend_labels[(move, walk)] = _fit_label(move, walk, A, b, power)

    one_day = 24 * 60 * 60
    ten_years = 10 * 365.25 * 24 * 60 * 60
    ax.axhline(one_day, color="black", linestyle=":", linewidth=1.5, alpha=0.85, zorder=0)
    ax.axhline(ten_years, color="black", linestyle="-.", linewidth=1.5, alpha=0.85, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(f"Classical and logical-quantum runtime, beta={beta}, a={a}, q0={q0_mode}, epsilon={epsilon:g}, Trotter={num_trotter_steps}")

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
