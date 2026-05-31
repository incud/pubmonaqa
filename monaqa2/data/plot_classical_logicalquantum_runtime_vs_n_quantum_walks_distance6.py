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


def _runtime_from_queries(n: int, walk: str, move: str, num_queries: float, num_trotter_steps: int, logical_operation_time: float) -> float:
    q = int(np.ceil(float(num_queries)))

    if walk == "classical" and move == "uniform":
        return float(get_time_classical_walk_uniform(n, q))
    if walk == "classical" and move.startswith("local"):
        return float(get_time_classical_walk_local(n, q))
    if walk == "classical" and move in ["qemc", "layden"]:
        return float(get_time_classical_walk_qemc(n, q, num_trotter_steps, logical_operation_time=logical_operation_time))

    if walk == "quantum" and move == "uniform":
        return float(get_time_quantum_walk_uniform(n, q, logical_operation_time=logical_operation_time))
    if walk == "quantum" and move.startswith("local"):
        return float(get_time_quantum_walk_local(n, q, logical_operation_time=logical_operation_time))
    if walk == "quantum" and move in ["qemc", "layden"]:
        return float(get_time_quantum_walk_qemc(n, q, num_trotter_steps, logical_operation_time=logical_operation_time))

    raise ValueError(f"unknown walk/move pair: walk={walk}, move={move}")


def _query_table_to_runtime_table(query_table: pd.DataFrame, num_trotter_steps: int, logical_operation_time: float) -> pd.DataFrame:
    rows = []

    for row in query_table.itertuples(index=False):
        n = int(row.n)
        walk = str(row.walk)
        move = str(row.move)
        runtime = _runtime_from_queries(n=n, walk=walk, move=move, num_queries=float(row.num_queries), num_trotter_steps=num_trotter_steps, logical_operation_time=logical_operation_time)

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
    if walk == "quantum" and move.startswith("local"):
        return 1
    if walk == "quantum" and move == "uniform":
        return 1
    if walk == "quantum" and move in ["qemc", "layden"]:
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
        raise ValueError("No query exponent available in the query table.")

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


def _query_fit(table: pd.DataFrame, walk: str, move: str) -> tuple[float, float]:
    sub = table[(table["walk"] == walk) & (table["move"] == move)]
    A = pd.to_numeric(sub["A"], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(sub["b"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(A) & np.isfinite(b)

    if not np.any(mask):
        raise ValueError(f"No query fit for walk={walk}, move={move}.")

    return float(A[mask][0]), float(b[mask][0])


def _logical_steps_from_queries(n: int, walk: str, move: str, num_queries: float, num_trotter_steps: int) -> float:
    return _runtime_from_queries(n=n, walk=walk, move=move, num_queries=num_queries, num_trotter_steps=num_trotter_steps, logical_operation_time=1.0)


def _required_distance(num_logical_steps: float, n_spins: int, prob_phys_error: float, target_eps: float, d_min: int = 1, d_max: int = 501) -> float:
    
    C = 0.1 # pre-factor constant derived from counting error paths, typically ranging between 0.05 and 0.1
    p_th = 0.01 # circuit-level threshold of the surface code, here 1%
    
    def p_logical(d: int) -> float:
        return C * (prob_phys_error / p_th) ** ((d + 1) / 2.0)

    spacetime_volume = num_logical_steps * (22 * n_spins * n_spins)

    for d in range(int(d_min), int(d_max) + 1):
        if float(spacetime_volume) * p_logical(d) <= float(target_eps):
            return float(d)

    return np.nan


def plot_classical_logicalquantum_runtime_vs_n(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    num_trotter_steps: int,
    logical_operation_time: float = 1e-6,
    zoom_interesting_time: bool = False,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 100,
    n_fit_line_min: int = 3,
    n_fit_line_max: int = 100,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and logical-quantum runtime versus n using the shared query table.
    """
    query_table = get_classical_quantum_query_vs_n_table(beta=beta, a=a, q0_mode=q0_mode, epsilon=epsilon, classical_query_file=classical_query_file, quantum_query_file=quantum_query_file, statistic=statistic, min_count=min_count, n_fit_min=n_fit_min, n_fit_max=n_fit_max, moves=tuple(PLOT_ORDER), only_ok=only_ok)
    runtime_table = _query_table_to_runtime_table(query_table, num_trotter_steps=num_trotter_steps, logical_operation_time=logical_operation_time)

    if runtime_table.empty:
        raise ValueError(f"No valid runtime data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}, statistic={statistic}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    n_grid = np.linspace(int(n_fit_line_min), int(n_fit_line_max), 300)

    styles = {
        "classical": {"linestyle": "-", "marker": "o", "markersize": 34, "linewidth": 2.2},
        "quantum": {"linestyle": "--", "marker": "s", "markersize": 30, "linewidth": 2.0},
    }

    legend_handles = {}
    legend_labels = {}

    for move in PLOT_ORDER:
        color = PROPOSAL_COLORS[move]

        for walk in ("classical", "quantum"):
            sub = runtime_table[(runtime_table["move"] == move) & (runtime_table["walk"] == walk)].copy()

            if sub.empty:
                continue

            observed = sub.sort_values("n")
            style = styles[walk]

            ax.scatter(observed["n"].to_numpy(dtype=float), observed["runtime"].to_numpy(dtype=float), s=style["markersize"], marker=style["marker"], color=color, edgecolors="none", alpha=0.95, zorder=4)

            try:
                A, b, power = _fit_runtime_fixed_query_exponent(observed, walk=walk, move=move, n_min=n_fit_min, n_max=n_fit_max)
            except ValueError:
                continue

            y_fit = _runtime_fit_values(A, b, power, n_grid)
            (line,) = ax.plot(n_grid, y_fit, color=color, linewidth=style["linewidth"], linestyle=style["linestyle"], alpha=0.95, zorder=2)
            legend_handles[(move, walk)] = line
            legend_labels[(move, walk)] = _fit_label(move, walk, A, b, power)

    one_day = 24 * 60 * 60
    one_year = 365.25 * one_day
    ten_years = 10 * one_year

    ax.set_yscale("log")
    
    if zoom_interesting_time:
        time_ticks = [
            (one_day, "1 day"),
            (30 * one_day, "30 days"),
            (0.5 * one_year, "6 months"),
            (one_year, "1 year"),
            (2 * one_year, "2 years"),
            (5 * one_year, "5 years"),
            (10 * one_year, "10 years"),
            (20 * one_year, "20 years"),
            (50 * one_year, "50 years"),
            (100 * one_year, "100 years"),
        ]

        ax.set_ylim(one_day, 100 * one_year)
        ax.set_yticks([value for value, _ in time_ticks])
        ax.set_yticklabels([label for _, label in time_ticks])

        for value, _ in time_ticks:
            ax.axhline(value, color="lightgray", linestyle="-", linewidth=0.45, alpha=0.85, zorder=0)
    else:
        ax.axhline(one_day, color="black", linestyle=":", linewidth=1.5, alpha=0.85, zorder=0)
        ax.axhline(ten_years, color="black", linestyle="-.", linewidth=1.5, alpha=0.85, zorder=0)

    ax.set_xlim(int(n_fit_line_min), int(n_fit_line_max))
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


def plot_required_surface_code_distance_vs_n(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    num_trotter_steps: int,
    prob_phys_error: float,
    target_eps: float,
    logical_operation_time: float = 1e-6,
    aux_info_cycle: bool = True,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 100,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the surface-code distance required by num_logical_steps * p_logical(d) <= target_eps.

    Left axis: required code distance d.
    Right axis: either target cycle time logical_operation_time / d, or physical operation time
    logical_operation_time / (d * (4 + 10)).
    """
    moves = tuple(dict.fromkeys(list(PLOT_ORDER) + ["layden"]))
    query_table = get_classical_quantum_query_vs_n_table(
        beta=beta,
        a=a,
        q0_mode=q0_mode,
        epsilon=epsilon,
        classical_query_file=classical_query_file,
        quantum_query_file=quantum_query_file,
        statistic=statistic,
        min_count=min_count,
        n_fit_min=n_fit_min,
        n_fit_max=n_fit_max,
        moves=moves,
        only_ok=only_ok,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    ax_aux = ax.twinx()

    n_values = np.arange(3, 101, dtype=int)
    curves = [
        ("quantum", "local1"),
        ("quantum", "uniform"),
        ("quantum", "layden"),
        ("classical", "layden"),
    ]

    styles = {
        "quantum": {"linestyle": "--", "marker": "s"},
        "classical": {"linestyle": "-", "marker": "o"},
    }

    left_handles = []
    left_labels = []
    right_handles = []
    right_labels = []

    for walk, move in curves:
        print(f"Processing {walk} {move}")

        try:
            A, b = _query_fit(query_table, walk=walk, move=move)
        except ValueError:
            continue

        distances = []

        for n in n_values:
            num_queries = A * np.exp(b * float(n))

            if walk == "quantum":
                steps = _logical_steps_from_queries(
                    n=int(n),
                    walk=walk,
                    move=move,
                    num_queries=num_queries,
                    num_trotter_steps=num_trotter_steps,
                )
            else:
                steps = num_trotter_steps * (1 + n + 1)

            dist = _required_distance(
                num_logical_steps=steps,
                n_spins=n,
                prob_phys_error=prob_phys_error,
                target_eps=target_eps,
            )
            distances.append(dist)

            if n == 100:
                cycle_time = logical_operation_time / dist if np.isfinite(dist) and dist > 0 else np.nan
                physical_operation_time = cycle_time / (4 + 10) if np.isfinite(cycle_time) else np.nan
                print(f"\tFitting {A} * np.exp({b} n)")
                print(f"\t{num_queries=}")
                print(f"\t{steps=}")
                print(f"\t{dist=}")
                print(f"\t{cycle_time=}")
                print(f"\t{physical_operation_time=}")

        distances = np.asarray(distances, dtype=float)
        mask = np.isfinite(distances) & (distances > 0.0)

        if not np.any(mask):
            continue

        color = "green" if walk == "classical" else PROPOSAL_COLORS[move]
        label = f"{PROPOSAL_LABELS[move]} ({walk})"

        (line_left,) = ax.plot(
            n_values[mask],
            distances[mask],
            color=color,
            linestyle=styles[walk]["linestyle"],
            marker=styles[walk]["marker"],
            markersize=3.0,
            linewidth=2.0,
            label=label,
        )

        aux_y = logical_operation_time / distances[mask]
        if not aux_info_cycle:
            aux_y = aux_y / (4 + 10)

        (line_right,) = ax_aux.plot(
            n_values[mask],
            aux_y,
            color=color,
            linestyle="-",
            linewidth=1.6,
            alpha=0.45,
            label=label,
        )

        left_handles.append(line_left)
        left_labels.append(label)
        right_handles.append(line_right)
        right_labels.append(label)

    ax.set_xlim(3, 100)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Required code distance d")
    ax.set_title(
        f"Required surface-code distance, beta={beta}, q0={q0_mode}, "
        f"target_eps={target_eps:g}, p_phys={prob_phys_error:g}"
    )

    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    ax_aux.set_yscale("log")
    ax_aux.set_ylabel("Target cycle time [s]" if aux_info_cycle else "Target physical operation time [s]")
    ax_aux.grid(False)

    ax.legend(
        left_handles,
        left_labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        borderaxespad=0.0,
        title="distance",
    )

    ax_aux.legend(
        right_handles,
        right_labels,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.15, 0.0),
        borderaxespad=0.0,
        title="time",
    )

    return fig, ax



def _logical_depth_from_queries(n: int, walk: str, move: str, num_queries: float, num_trotter_steps: int) -> float:
    return _runtime_from_queries(n=n, walk=walk, move=move, num_queries=num_queries, num_trotter_steps=num_trotter_steps, logical_operation_time=1.0)


def _physical_runtime_from_queries(
    n: int,
    walk: str,
    move: str,
    num_queries: float,
    num_trotter_steps: int,
    prob_phys_error: float,
    target_eps: float,
    physical_operation_time: float,
    annealing_overhead: bool = False,
) -> float:
    if walk == "classical" and move in ["local1", "local2", "local3"]:
        return _runtime_from_queries(n=n, walk=walk, move=move, num_queries=num_queries, num_trotter_steps=num_trotter_steps, logical_operation_time=1.0)

    if walk == "classical" and move == "uniform":
        return _runtime_from_queries(n=n, walk=walk, move=move, num_queries=num_queries, num_trotter_steps=num_trotter_steps, logical_operation_time=1.0)

    depth = _logical_depth_from_queries(n=n, walk=walk, move=move, num_queries=num_queries, num_trotter_steps=num_trotter_steps)

    if annealing_overhead:
        depth *= np.sqrt(float(n))

    d = _required_distance(num_logical_steps=depth, n_spins=n, prob_phys_error=prob_phys_error, target_eps=target_eps)

    if not np.isfinite(d):
        return np.nan

    return float(physical_operation_time * (10 + 4) * d * depth)


def plot_classical_physicalquantum_runtime_vs_n(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    num_trotter_steps: int,
    prob_phys_error: float,
    target_eps: float,
    physical_operation_time: float = 1e-8,
    annealing_overhead: bool = False,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 100,
    n_fit_line_min: int = 3,
    n_fit_line_max: int = 100,
    zoom_interesting_time: bool = False,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical CPU runtime and physical quantum runtime versus n.

    Classical local/uniform walks use CPU timing. Classical layden/qemc moves and all quantum walks use physical_operation_time * 14 * d * depth, where d is chosen from the surface-code logical error budget. If annealing_overhead=True, the logical depth is multiplied by sqrt(n) before choosing d and before computing runtime.
    """
    query_table = get_classical_quantum_query_vs_n_table(beta=beta, a=a, q0_mode=q0_mode, epsilon=epsilon, classical_query_file=classical_query_file, quantum_query_file=quantum_query_file, statistic=statistic, min_count=min_count, n_fit_min=n_fit_min, n_fit_max=n_fit_max, moves=tuple(PLOT_ORDER), only_ok=only_ok)

    rows = []
    for row in query_table.itertuples(index=False):
        n = int(row.n)
        walk = str(row.walk)
        move = str(row.move)
        runtime = _physical_runtime_from_queries(
            n=n,
            walk=walk,
            move=move,
            num_queries=float(row.num_queries),
            num_trotter_steps=num_trotter_steps,
            prob_phys_error=prob_phys_error,
            target_eps=target_eps,
            physical_operation_time=physical_operation_time,
            annealing_overhead=annealing_overhead,
        )

        if np.isfinite(runtime) and runtime > 0.0:
            rows.append({"n": n, "walk": walk, "move": move, "runtime": float(runtime), "num_queries": float(row.num_queries), "query_A": float(row.A), "query_b": float(row.b)})

    runtime_table = pd.DataFrame(rows, columns=["n", "walk", "move", "runtime", "num_queries", "query_A", "query_b"])

    if runtime_table.empty:
        raise ValueError(f"No valid physical runtime data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}, statistic={statistic}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    n_values = np.arange(int(n_fit_line_min), int(n_fit_line_max) + 1, dtype=int)

    styles = {
        "classical": {"linestyle": "-", "marker": "o", "markersize": 34, "linewidth": 2.2},
        "quantum": {"linestyle": "--", "marker": "s", "markersize": 30, "linewidth": 2.0},
    }

    legend_handles = {}
    legend_labels = {}

    for move in PLOT_ORDER:
        color = PROPOSAL_COLORS[move]

        for walk in ("classical", "quantum"):
            sub = runtime_table[(runtime_table["move"] == move) & (runtime_table["walk"] == walk)].copy()

            if sub.empty:
                continue

            observed = sub.sort_values("n")
            style = styles[walk]

            ax.scatter(observed["n"].to_numpy(dtype=float), observed["runtime"].to_numpy(dtype=float), s=style["markersize"], marker=style["marker"], color=color, edgecolors="none", alpha=0.95, zorder=4)

            try:
                A_q, b_q = _query_fit(query_table, walk=walk, move=move)
            except ValueError:
                continue

            y_fit = []
            n_fit = []
            for n in n_values:
                num_queries = A_q * np.exp(b_q * float(n))
                runtime = _physical_runtime_from_queries(
                    n=int(n),
                    walk=walk,
                    move=move,
                    num_queries=num_queries,
                    num_trotter_steps=num_trotter_steps,
                    prob_phys_error=prob_phys_error,
                    target_eps=target_eps,
                    physical_operation_time=physical_operation_time,
                    annealing_overhead=annealing_overhead,
                )

                if np.isfinite(runtime) and runtime > 0.0:
                    n_fit.append(int(n))
                    y_fit.append(float(runtime))

            if not y_fit:
                continue

            (line,) = ax.plot(np.asarray(n_fit, dtype=float), np.asarray(y_fit, dtype=float), color=color, linewidth=style["linewidth"], linestyle=style["linestyle"], alpha=0.95, zorder=2)
            legend_handles[(move, walk)] = line
            legend_labels[(move, walk)] = f"{PROPOSAL_LABELS[move]} ({walk}): queries {A_q:.3g} exp({b_q:.3f} n)"

    one_day = 24 * 60 * 60
    one_year = 365.25 * one_day
    ten_years = 10 * one_year

    ax.set_yscale("log")

    if zoom_interesting_time:
        time_ticks = [
            (one_day, "1 day"),
            (30 * one_day, "30 days"),
            (0.5 * one_year, "6 months"),
            (one_year, "1 year"),
            (2 * one_year, "2 years"),
            (5 * one_year, "5 years"),
            (10 * one_year, "10 years"),
            (20 * one_year, "20 years"),
            (50 * one_year, "50 years"),
            (100 * one_year, "100 years"),
        ]

        ax.set_ylim(one_day, 100 * one_year)
        ax.set_yticks([value for value, _ in time_ticks])
        ax.set_yticklabels([label for _, label in time_ticks])

        for value, _ in time_ticks:
            ax.axhline(value, color="lightgray", linestyle="-", linewidth=0.45, alpha=0.85, zorder=0)
    else:
        ax.axhline(one_day, color="black", linestyle=":", linewidth=1.5, alpha=0.85, zorder=0)
        ax.axhline(ten_years, color="black", linestyle="-.", linewidth=1.5, alpha=0.85, zorder=0)

    mode_label = "Full annealing walk" if annealing_overhead else "Single step of the annealing"

    ax.set_xlim(int(n_fit_line_min), int(n_fit_line_max))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(f"{mode_label}, beta={beta}, a={a}, q0={q0_mode}, epsilon={epsilon:g}, Trotter={num_trotter_steps}")

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