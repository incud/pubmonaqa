from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE
from monaqa2.data.table_classical_quantum_query_vs_n import get_classical_quantum_query_vs_n_table


PROPOSAL_COLORS = {
    "cpu_local": "#7A7A7A",
    "gpu_local": "#56B4E9",
    "fpga_local": "#0072B2",
    "cpu_uniform": "#009E73",
    "gpu_uniform": "#E69F00",
    "fpga_uniform": "#D55E00",
}


def cpu_time_per_local_move(n):
    """Return the fitted CPU seconds per local spin-flip move.

    :param n: Number of spins.
    :return: Estimated CPU seconds per local move.
    """
    n = np.asarray(n, dtype=float)
    return 5.122e-09 + 1.753e-10 * n


def cpu_time_per_uniform_move(n):
    """Return the fitted CPU seconds per uniform dense move.

    :param n: Number of spins.
    :return: Estimated CPU seconds per uniform move.
    """
    n = np.asarray(n, dtype=float)
    return 3.024e-07 + 9.643e-11 * n * n


def fpga_time_per_local_move(n):
    """Return the fitted FPGA seconds per local spin-flip move.

    :param n: Number of spins.
    :return: Estimated FPGA seconds per local move.
    """
    n = np.asarray(n, dtype=float)
    return (267.900 + 1.800 * np.log2(n)) * 1e-9


def fpga_time_per_uniform_move(n):
    """Return the fitted FPGA seconds per uniform dense move.

    :param n: Number of spins.
    :return: Estimated FPGA seconds per uniform move.
    """
    n = np.asarray(n, dtype=float)
    return (254.100 + 4.200 * np.log2(n)) * 1e-9


def gpu_time_per_local_move(n):
    """Return a temporary placeholder GPU seconds per local spin-flip move.

    :param n: Number of spins.
    :return: Placeholder GPU seconds per local move.
    """
    return fpga_time_per_local_move(n) + 1.0e-6


def gpu_time_per_uniform_move(n):
    """Return a temporary placeholder GPU seconds per uniform dense move.

    :param n: Number of spins.
    :return: Placeholder GPU seconds per uniform move.
    """
    return fpga_time_per_uniform_move(n) + 1.0e-6


def plot_time_to_solve_classically_comparison(
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
    only_ok: bool = True,
    walk: str = "classical",
):
    """Plot platform/proposal cost per query and cost per final annealing step.

    :param beta: Inverse temperature.
    :param a: Query-table parameter.
    :param q0_mode: Initial-state mode.
    :param epsilon: Accuracy/tolerance parameter.
    :param classical_query_file: Path to classical query-count file.
    :param quantum_query_file: Path to quantum query-count file.
    :param statistic: Statistic used by the query table.
    :param min_count: Minimum number of observations.
    :param n_fit_min: Minimum n used for the query-count fit.
    :param n_fit_max: Maximum n used for the query-count fit.
    :param only_ok: Whether to keep only successful rows.
    :param walk: Walk type to select.
    :return: Matplotlib figure and axes.
    """
    table = get_classical_quantum_query_vs_n_table(
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
        moves=("local1", "uniform"),
        only_ok=only_ok,
    )

    n_grid = np.linspace(8, 100, 500)

    time_models = {
        "CPU local": (cpu_time_per_local_move, "cpu_local"),
        "GPU local": (gpu_time_per_local_move, "gpu_local"),
        "FPGA local": (fpga_time_per_local_move, "fpga_local"),
        "CPU uniform": (cpu_time_per_uniform_move, "cpu_uniform"),
        "GPU uniform": (gpu_time_per_uniform_move, "gpu_uniform"),
        "FPGA uniform": (fpga_time_per_uniform_move, "fpga_uniform"),
    }

    query_models = {}

    for move in ("local1", "uniform"):
        sub = table[(table["move"] == move) & (table["walk"] == walk)].copy()

        if "source" in sub.columns:
            sub = sub[sub["source"] == "observed"]

        sub = sub.sort_values("n")

        if sub.empty:
            raise ValueError(f"No observed query data found for move={move}, walk={walk}.")

        n = sub["n"].to_numpy(dtype=float)
        q = sub["num_queries"].to_numpy(dtype=float)
        ok = np.isfinite(n) & np.isfinite(q) & (q > 0)

        if np.count_nonzero(ok) < 2:
            raise ValueError(f"Not enough valid query data to fit move={move}, walk={walk}.")

        b, log_A = np.polyfit(n[ok], np.log(q[ok]), 1)
        query_models[move] = np.exp(log_A + b * n_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)

    ax = axes[0]
    for label, (time_fn, color_key) in time_models.items():
        linestyle = "-" if "local" in label else "--"
        ax.plot(
            n_grid,
            time_fn(n_grid),
            color=PROPOSAL_COLORS[color_key],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )

    ax.set_title("Cost per query")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Seconds per query")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    for label, (time_fn, color_key) in time_models.items():
        move = "local1" if "local" in label else "uniform"
        linestyle = "-" if "local" in label else "--"
        ax.plot(
            n_grid,
            query_models[move] * time_fn(n_grid),
            color=PROPOSAL_COLORS[color_key],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )

    ax.set_title("Cost per last annealing step")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Seconds")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    return fig, axes
