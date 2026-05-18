from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.instances import load_instances
from monaqa2.data.annealing import get_annealing_betas
from monaqa2.data.quantum_query import get_qpe_cost
from monaqa2.mcmc.distribution import get_gibbs_distribution
from monaqa2.data.runtime import (
    calculate_runtime_from_uniform_distribution,
    get_quantum_walk_local_steps,
    get_quantum_walk_qemc_steps,
    get_quantum_walk_uniform_steps,
    sk_spherical_energy_difference_upper_bound,
)

PLOT_ORDER = ["local1", "uniform", "qemc", "layden"]
LEGEND_ORDER = ["local1", "uniform", "qemc", "layden"]

PROPOSAL_LABELS = {
    "uniform": "Uniform",
    "local1": "Local spin-flip (single)",
    "local2": "Local spin-flip (double)",
    "local3": "Local spin-flip (triple)",
    "qemc": "Quantum enhanced (best hyperparameters)",
    "layden": "Quantum enhanced (randomized)",
}

PROPOSAL_COLORS = {
    "uniform": "#7A7A7A",
    "local1": "#56B4E9",
    "local2": "#0072B2",
    "local3": "#009E73",
    "qemc": "#E69F00",
    "layden": "#D55E00",
}

Q0_MARKERS = {"uniform": "o", "bhattacharyya": "s"}
KIND_LINESTYLES = {"classical": "-", "quantum": "--"}
KIND_ALPHA_FILL = {"classical": 0.16, "quantum": 0.08}

RUNTIME_ROW_COLUMNS = ["n", "idx", "proposal", "q0_mode", "kind", "runtime"]
RUNTIME_STATS_COLUMNS = ["proposal", "q0_mode", "kind", "n", "center", "spread", "count"]


def _a_mask(series: pd.Series, a: int | float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.isinf(values) if np.isinf(a) else np.isclose(values, float(a))


def _epsilon_column(epsilon: float) -> str:
    epsilon = float(epsilon)
    k = int(round(-np.log10(epsilon)))
    if not np.isclose(epsilon, 10.0 ** (-k)) or k < 2 or k > 8:
        raise ValueError("epsilon must be one of 1e-2, ..., 1e-8.")
    return f"queries_eps_1e-{k}"


def _overlap(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.maximum(p * q, 0.0))))


def _summarize(x: np.ndarray, statistic: str) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        return np.nan, np.nan, 0
    if statistic == "mean+std":
        return float(np.mean(x)), float(np.std(x)), int(x.size)
    if statistic == "mean+std-tail":
        q1, q3 = np.percentile(x, [25, 75])
        x = x[(x >= q1) & (x <= q3)]
        return (float(np.mean(x)), float(np.std(x)), int(x.size)) if x.size else (np.nan, np.nan, 0)
    if statistic == "median+mad":
        center = float(np.median(x))
        return center, float(np.median(np.abs(x - center))), int(x.size)
    raise ValueError("statistic must be one of ['mean+std', 'mean+std-tail', 'median+mad'].")


def _positive_band(center: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = center - spread
    upper = center + spread
    positive = np.concatenate([center[center > 0.0], upper[upper > 0.0]])
    floor = np.min(positive) * 1e-2 if positive.size else np.finfo(float).tiny
    return np.maximum(lower, floor), np.maximum(upper, floor)


def _fit_exp(table: pd.DataFrame, n_min: int | None, n_max: int | None) -> tuple[float, float]:
    table = table.copy()
    if n_min is not None:
        table = table[table["n"].astype(int) >= int(n_min)]
    if n_max is not None:
        table = table[table["n"].astype(int) <= int(n_max)]
    n = table["n"].to_numpy(dtype=float)
    y = table["center"].to_numpy(dtype=float)
    mask = np.isfinite(n) & np.isfinite(y) & (y > 0.0)
    n, y = n[mask], y[mask]
    if y.size < 2:
        raise ValueError("Need at least two positive points to fit.")
    b, log_A = np.polyfit(n, np.log(y), deg=1)
    return float(np.exp(log_A)), float(b)


def _single_walk_nc_depth(
    num_spins: int,
    proposal: str,
    beta: float,
    U_quantum_upper_bound: float | None,
    eps_per_query: float,
) -> float:
    n = int(num_spins)
    beta = float(beta)
    eps_per_query = float(eps_per_query)
    U_quantum_upper_bound = (
        sk_spherical_energy_difference_upper_bound(n)
        if U_quantum_upper_bound is None
        else float(U_quantum_upper_bound)
    )

    if proposal == "uniform":
        return get_quantum_walk_uniform_steps(
            n=n,
            U=U_quantum_upper_bound,
            beta=beta,
            epsilon=eps_per_query,
        )

    if proposal in ["local1", "local2", "local3"]:
        return get_quantum_walk_local_steps(
            n=n,
            U=U_quantum_upper_bound,
            beta=beta,
            epsilon=eps_per_query,
        )

    if proposal in ["qemc", "layden"]:
        return get_quantum_walk_qemc_steps(
            n=n,
            U=U_quantum_upper_bound,
            beta=beta,
            epsilon=eps_per_query,
            alpha=U_quantum_upper_bound,
            t=1.0,
        )

    raise ValueError(f"unknown proposal: {proposal}")


def _qec_time_from_total_nc_depth(total_nc_depth: float, eps: float, time_quantum_gate: float, time_quantum_measurement: float, num_gate_layers_per_qec_cycle: int, prob_phys_error: float) -> float:
    eps_qec = eps / 2
    def p_logical(d: int) -> float:
        return 0.1 * (100 * prob_phys_error) ** ((d + 1) / 2)
    d = 3
    while total_nc_depth * p_logical(d) > eps_qec:
        d += 2
    qec_cycle_time = num_gate_layers_per_qec_cycle * time_quantum_gate + time_quantum_measurement
    return float(total_nc_depth * d * qec_cycle_time)


def _warm_start_quantum_runtime(num_spins: int, proposal: str, beta: float, expected_qpe_queries: float, U_quantum_upper_bound: float | None, eps: float, time_quantum_gate: float, time_quantum_measurement: float, num_gate_layers_per_qec_cycle: int, prob_phys_error: float) -> float:
    if expected_qpe_queries <= 0.0 or not np.isfinite(expected_qpe_queries):
        return np.nan
    eps_per_query = (eps / 2) / float(expected_qpe_queries)
    nc_depth = _single_walk_nc_depth(num_spins=num_spins, proposal=proposal, beta=beta, U_quantum_upper_bound=U_quantum_upper_bound, eps_per_query=eps_per_query)
    return _qec_time_from_total_nc_depth(total_nc_depth=float(expected_qpe_queries) * nc_depth, eps=eps, time_quantum_gate=time_quantum_gate, time_quantum_measurement=time_quantum_measurement, num_gate_layers_per_qec_cycle=num_gate_layers_per_qec_cycle, prob_phys_error=prob_phys_error)


def _uniform_annealing_quantum_runtime(model_cache: dict, n: int, idx: int, proposal: str, a: int | float, beta: float, U_quantum_upper_bound: float | None, spectral_gap_file: Path, statistic: str, alpha_target: float, eps: float, time_quantum_gate: float, time_quantum_measurement: float, num_gate_layers_per_qec_cycle: int, prob_phys_error: float) -> float:
    key = (int(n), int(idx))
    if key not in model_cache:
        model_cache[key] = load_instances(int(n), int(idx))
    model = model_cache[key]
    schedule = get_annealing_betas(model=model, beta_final=float(beta), alpha=alpha_target)
    qpe_costs = [get_qpe_cost(n=int(n), proposal=proposal, a=a, beta=float(beta_t), spectral_gap_file=spectral_gap_file, statistic=statistic)[0] for beta_t in schedule]
    p_succ = []
    for beta_left, beta_right in zip(schedule[:-1], schedule[1:]):
        pi_left = get_gibbs_distribution(model, float(beta_left))
        pi_right = get_gibbs_distribution(model, float(beta_right))
        p_succ.append(_overlap(pi_left, pi_right) ** 2)
    return calculate_runtime_from_uniform_distribution(num_spins=int(n), num_classical_queries=0, num_quantum_queries_per_step=qpe_costs, beta_quantum_per_step=[float(x) for x in schedule], p_succ_quantum_per_step=p_succ, U_quantum_upper_bound=U_quantum_upper_bound, walk="quantum", proposal=proposal, eps=eps, time_quantum_gate=time_quantum_gate, time_quantum_measurement=time_quantum_measurement, num_gate_layers_per_qec_cycle=num_gate_layers_per_qec_cycle, prob_phys_error=prob_phys_error)


def _runtime_rows(
    beta: float,
    a: int | float,
    epsilon: float,
    U_quantum_upper_bound: float | None,
    q0_modes: tuple[str, ...],
    classical_query_file: Path,
    quantum_query_file: Path,
    spectral_gap_file: Path,
    statistic: str,
    alpha_target: float,
    only_ok: bool,
    a_cpu: float,
    b_cpu: float,
    c_cpu: float,
    num_steps_trotter: int,
    time_quantum_gate: float,
    time_quantum_measurement: float,
    num_gate_layers_per_qec_cycle: int,
    prob_phys_error: float,
) -> pd.DataFrame:
    classical_df = pd.read_pickle(classical_query_file)
    quantum_df = pd.read_pickle(quantum_query_file)
    eps_col = _epsilon_column(epsilon)
    model_cache = {}
    rows = []

    if eps_col not in classical_df.columns:
        raise ValueError(f"Classical query file does not contain column {eps_col}. Available query columns: {[c for c in classical_df.columns if str(c).startswith('queries_eps_')]}")

    if "queries" not in quantum_df.columns:
        raise ValueError("Quantum query file does not contain column 'queries'.")

    for proposal in PLOT_ORDER:
        for q0_mode in q0_modes:
            print(f"\t{proposal} {q0_mode}")

            cmask = (classical_df["proposal"].astype(str) == proposal) & _a_mask(classical_df["a"], a) & np.isclose(classical_df["beta"].astype(float), float(beta)) & (classical_df["q0_mode"].astype(str) == q0_mode)
            if only_ok and "ok" in classical_df.columns:
                cmask = cmask & classical_df["ok"].astype(bool)

            for _, row in classical_df.loc[cmask].iterrows():
                q = float(row[eps_col])
                if not np.isfinite(q) or q <= 0.0:
                    continue
                runtime = calculate_runtime_from_uniform_distribution(num_spins=int(row["n"]), num_classical_queries=int(q), num_quantum_queries_per_step=[], beta_quantum_per_step=[], p_succ_quantum_per_step=[], U_quantum_upper_bound=U_quantum_upper_bound, walk="classical", proposal=proposal, eps=epsilon, a_cpu=a_cpu, b_cpu=b_cpu, c_cpu=c_cpu, num_steps_trotter=num_steps_trotter, time_quantum_gate=time_quantum_gate, time_quantum_measurement=time_quantum_measurement, num_gate_layers_per_qec_cycle=num_gate_layers_per_qec_cycle, prob_phys_error=prob_phys_error)
                rows.append({"n": int(row["n"]), "idx": int(row["idx"]), "proposal": proposal, "q0_mode": q0_mode, "kind": "classical", "runtime": float(runtime)})

            qmask = (quantum_df["proposal"].astype(str) == proposal) & _a_mask(quantum_df["a"], a) & np.isclose(quantum_df["beta"].astype(float), float(beta)) & (quantum_df["q0_mode"].astype(str) == q0_mode)
            if only_ok and "ok" in quantum_df.columns:
                qmask = qmask & quantum_df["ok"].astype(bool)

            for _, row in quantum_df.loc[qmask].iterrows():
                if q0_mode == "uniform":
                    runtime = _uniform_annealing_quantum_runtime(model_cache=model_cache, n=int(row["n"]), idx=int(row["idx"]), proposal=proposal, a=a, beta=beta, U_quantum_upper_bound=U_quantum_upper_bound, spectral_gap_file=spectral_gap_file, statistic=statistic, alpha_target=alpha_target, eps=epsilon, time_quantum_gate=time_quantum_gate, time_quantum_measurement=time_quantum_measurement, num_gate_layers_per_qec_cycle=num_gate_layers_per_qec_cycle, prob_phys_error=prob_phys_error)
                elif q0_mode == "bhattacharyya":
                    runtime = _warm_start_quantum_runtime(num_spins=int(row["n"]), proposal=proposal, beta=beta, expected_qpe_queries=float(row["queries"]), U_quantum_upper_bound=U_quantum_upper_bound, eps=epsilon, time_quantum_gate=time_quantum_gate, time_quantum_measurement=time_quantum_measurement, num_gate_layers_per_qec_cycle=num_gate_layers_per_qec_cycle, prob_phys_error=prob_phys_error)
                else:
                    raise ValueError(f"unknown q0_mode: {q0_mode}")

                if np.isfinite(runtime) and runtime > 0.0:
                    rows.append({"n": int(row["n"]), "idx": int(row["idx"]), "proposal": proposal, "q0_mode": q0_mode, "kind": "quantum", "runtime": float(runtime)})

    if not rows:
        classical_available = classical_df[["proposal", "q0_mode", "beta", "a"]].drop_duplicates().sort_values(["proposal", "q0_mode", "beta"]).head(30).to_string(index=False)
        quantum_available = quantum_df[["proposal", "q0_mode", "beta", "a"]].drop_duplicates().sort_values(["proposal", "q0_mode", "beta"]).head(30).to_string(index=False)
        raise ValueError(f"No runtime rows matched proposal in {PLOT_ORDER}, q0_modes={q0_modes}, beta={beta}, a={a}, epsilon={epsilon}, only_ok={only_ok}.\n\nFirst available classical rows:\n{classical_available}\n\nFirst available quantum rows:\n{quantum_available}")

    return pd.DataFrame(rows, columns=RUNTIME_ROW_COLUMNS)


def _runtime_stats(runtime_rows: pd.DataFrame, statistic: str, min_count: int) -> pd.DataFrame:
    rows = []
    if runtime_rows.empty:
        return pd.DataFrame(columns=RUNTIME_STATS_COLUMNS)
    missing = [column for column in RUNTIME_ROW_COLUMNS if column not in runtime_rows.columns]
    if missing:
        raise ValueError(f"runtime_rows is missing expected columns: {missing}")
    for (proposal, q0_mode, kind, n), group in runtime_rows.groupby(["proposal", "q0_mode", "kind", "n"], sort=True):
        center, spread, count = _summarize(group["runtime"].to_numpy(dtype=float), statistic)
        if count >= min_count and np.isfinite(center) and center > 0.0:
            rows.append({"proposal": proposal, "q0_mode": q0_mode, "kind": kind, "n": int(n), "center": center, "spread": spread, "count": count})
    return pd.DataFrame(rows, columns=RUNTIME_STATS_COLUMNS)


def plot_classical_quantum_runtime_vs_n(
    beta: float,
    a: int | float,
    epsilon: float,
    U_quantum_upper_bound: float | None = None,
    q0_modes: tuple[str, ...] = ("uniform", "bhattacharyya"),
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 8,
    max_n_spins: int | None = 1000,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    alpha_target: float = np.sqrt(1.0 / np.e),
    only_ok: bool = True,
    a_cpu: float = 1e-6,
    b_cpu: float = 5e-9,
    c_cpu: float = 5e-10,
    num_steps_trotter: int = 100,
    time_quantum_gate: float = 20e-9,
    time_quantum_measurement: float = 100e-9,
    num_gate_layers_per_qec_cycle: int = 4,
    prob_phys_error: float = 1e-5,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and quantum runtime versus n for fixed beta and acceptance parameter.

    Runtime is computed from the generated query tables. Classical rows use the classical branch of `calculate_runtime_from_uniform_distribution`. Quantum rows use the uniform-start rewind schedule for q0_mode='uniform' and the warm-start expected projection count stored in the quantum table for q0_mode='bhattacharyya'. Lines are exponential fits y(n)=A exp(b n), projected up to `max_n_spins`. Horizontal black lines mark one day and ten years.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
    else:
        fig = ax.figure

    print("Calculating runtime rows...", flush=True)
    runtime_rows = _runtime_rows(beta=beta, a=a, epsilon=epsilon, U_quantum_upper_bound=U_quantum_upper_bound, q0_modes=tuple(q0_modes), classical_query_file=classical_query_file, quantum_query_file=quantum_query_file, spectral_gap_file=spectral_gap_file, statistic=statistic, alpha_target=alpha_target, only_ok=only_ok, a_cpu=a_cpu, b_cpu=b_cpu, c_cpu=c_cpu, num_steps_trotter=num_steps_trotter, time_quantum_gate=time_quantum_gate, time_quantum_measurement=time_quantum_measurement, num_gate_layers_per_qec_cycle=num_gate_layers_per_qec_cycle, prob_phys_error=prob_phys_error)
    print("Calculating runtime stats...", flush=True)
    stats = _runtime_stats(runtime_rows=runtime_rows, statistic=statistic, min_count=min_count)

    if stats.empty:
        raise ValueError(f"No valid runtime data found for beta={beta}, a={a}, epsilon={epsilon}, q0_modes={q0_modes}.")

    print("Handling graphics", flush=True)
    n_min = float(stats["n"].min()) if n_plot_min is None else float(n_plot_min)
    n_max = float(stats["n"].max()) if n_plot_max is None else float(n_plot_max)
    n_line_max = n_max if max_n_spins is None else float(max_n_spins)
    if n_line_max < n_min:
        raise ValueError("max_n_spins must be at least n_plot_min / min observed n.")
    n_grid = np.linspace(n_min, n_line_max, 300)
    handles = []
    labels = []

    for proposal in LEGEND_ORDER:
        for q0_mode in q0_modes:
            for kind in ("classical", "quantum"):
                table = stats[(stats["proposal"] == proposal) & (stats["q0_mode"] == q0_mode) & (stats["kind"] == kind)].sort_values("n").copy()
                if table.empty:
                    continue

                color = PROPOSAL_COLORS[proposal]
                linestyle = KIND_LINESTYLES[kind]
                marker = Q0_MARKERS.get(q0_mode, "o")
                n_vals = table["n"].to_numpy(dtype=float)
                center = table["center"].to_numpy(dtype=float)
                spread = table["spread"].fillna(0.0).to_numpy(dtype=float)

                if show_spread:
                    lower, upper = _positive_band(center, spread)
                    ax.fill_between(n_vals, lower, upper, color=color, alpha=KIND_ALPHA_FILL[kind], linewidth=0.0, zorder=1)

                ax.scatter(n_vals, center, s=36, marker=marker, color=color, edgecolors="none", alpha=0.95, zorder=4)

                try:
                    A, b = _fit_exp(table, n_min=n_fit_min, n_max=n_fit_max)
                except ValueError:
                    continue

                y_fit = A * np.exp(b * n_grid)
                (line,) = ax.plot(n_grid, y_fit, color=color, linestyle=linestyle, linewidth=2.1, alpha=0.95, zorder=3)
                handles.append(line)
                labels.append(rf"{PROPOSAL_LABELS[proposal]} {kind}, {q0_mode}: ${A:.3g}\exp({b:.3f}n)$")

    one_day = 24 * 60 * 60
    ten_years = 10 * 365.25 * 24 * 60 * 60
    ax.axhline(one_day, color="black", linestyle=":", linewidth=1.5, alpha=0.85, zorder=0)
    ax.axhline(ten_years, color="black", linestyle="-.", linewidth=1.5, alpha=0.85, zorder=0)
    x_label = max(n_max, n_line_max)
    ax.text(x_label, one_day, " 1 day", va="bottom", ha="left", color="black")
    ax.text(x_label, ten_years, " 10 years", va="bottom", ha="left", color="black")

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(rf"Classical and quantum runtime, $\beta={beta}$, $a={a}$, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.8)

    return fig, ax

