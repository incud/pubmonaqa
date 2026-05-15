from pathlib import Path

from monaqa2.data.filename import QUANTUM_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.instances import load_instances
from monaqa2.data.annealing import get_annealing_betas
from monaqa2.data.spectral_gap import get_spectral_gap_stats
from monaqa2.mcmc.distribution import get_gibbs_distribution, get_gibbs_distribution_with_bhattacharyya_guarantee
import numpy as np
import pandas as pd


PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
ACCEPTANCES = [np.inf, 1, 10]
Q0_MODES = ["uniform", "bhattacharyya"]
BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def overlap(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)))


def qpe_queries_from_spectral_gap(delta: float) -> int:
    phase_gap = float(np.arccos(np.clip(1.0 - float(delta), -1.0, 1.0)))
    if not np.isfinite(phase_gap) or phase_gap <= 0.0:
        raise ValueError(f"invalid phase gap from spectral gap: {delta}")
    return int(np.ceil(2.0 * np.pi / phase_gap))


def get_fitted_spectral_gap(
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    fit_cache: dict | None = None,
) -> float:
    """
    Return delta_n(beta) by log-linear interpolation of log(delta) at fixed n.
    """
    beta = float(beta)
    if beta <= 0.0:
        return 1.0

    fit_cache = {} if fit_cache is None else fit_cache
    a_key = float(a) if not np.isinf(a) else np.inf
    key = (int(n), proposal, a_key, statistic)

    if key not in fit_cache:
        table = get_spectral_gap_stats(proposal=proposal, a=a, in_file=spectral_gap_file, statistic=statistic)
        table = table[
            (table["n"].astype(int) == int(n))
            & np.isfinite(table["beta"].astype(float))
            & np.isfinite(table["center"].astype(float))
            & (table["beta"].astype(float) > 0.0)
            & (table["center"].astype(float) > 0.0)
        ].sort_values("beta")

        if table.empty:
            raise ValueError(f"No positive spectral-gap data for n={n}, proposal={proposal}, a={a}.")

        beta_grid = table["beta"].to_numpy(dtype=float)
        log_delta_grid = np.log(table["center"].to_numpy(dtype=float))
        fit_cache[key] = beta_grid, log_delta_grid

    beta_grid, log_delta_grid = fit_cache[key]
    beta = float(np.clip(beta, beta_grid[0], beta_grid[-1]))
    log_delta = float(np.interp(beta, beta_grid, log_delta_grid))
    delta = float(np.exp(log_delta))

    return float(np.clip(delta, np.finfo(float).tiny, 1.0))


def get_qpe_cost(
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    fit_cache: dict | None = None,
) -> tuple[int, float]:
    if float(beta) <= 0.0:
        return 0, 1.0

    delta = get_fitted_spectral_gap(n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, fit_cache=fit_cache)
    return qpe_queries_from_spectral_gap(delta), delta


def get_warm_start_quantum_query_cost(
    model,
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    alpha_target: float = np.sqrt(1.0 / np.e),
    fit_cache: dict | None = None,
) -> dict:
    pi_beta = get_gibbs_distribution(model, beta)
    q0, beta_0 = get_gibbs_distribution_with_bhattacharyya_guarantee(model, beta, pi_beta, alpha_target)
    initial_overlap = overlap(q0, pi_beta)
    p_success = initial_overlap ** 2

    if p_success <= 0.0 or not np.isfinite(p_success):
        raise ValueError(f"invalid warm-start success probability: {p_success}")

    C_beta, target_delta = get_qpe_cost(n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, fit_cache=fit_cache)

    return {"queries": int(np.ceil(C_beta / p_success)), "beta_0": float(beta_0), "initial_overlap": float(initial_overlap), "schedule_length": 0, "min_step_overlap": np.nan, "target_delta": float(target_delta)}


def get_uniform_annealing_quantum_query_cost(
    model,
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    alpha_target: float = np.sqrt(1.0 / np.e),
    fit_cache: dict | None = None,
) -> dict:
    pi_beta = get_gibbs_distribution(model, beta)
    pi_0 = np.full_like(pi_beta, 1.0 / pi_beta.size)
    initial_overlap = overlap(pi_0, pi_beta)

    schedule = get_annealing_betas(model=model, beta_final=beta, alpha=alpha_target)
    costs = [get_qpe_cost(n=n, proposal=proposal, a=a, beta=float(beta_t), spectral_gap_file=spectral_gap_file, statistic=statistic, fit_cache=fit_cache)[0] for beta_t in schedule]
    _, target_delta = get_qpe_cost(n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, fit_cache=fit_cache)

    step_overlaps = []
    for beta_left, beta_right in zip(schedule[:-1], schedule[1:]):
        pi_left = np.asarray(get_gibbs_distribution(model, float(beta_left)), dtype=float)
        pi_right = np.asarray(get_gibbs_distribution(model, float(beta_right)), dtype=float)
        step_overlaps.append(overlap(pi_left, pi_right))

    total = 0.0
    for j, alpha_step in enumerate(step_overlaps):
        p_step = float(alpha_step) ** 2
        if p_step <= 0.0 or not np.isfinite(p_step):
            raise ValueError(f"invalid annealing success probability at step {j}: {p_step}")
        total += costs[j + 1] + (costs[j] + costs[j + 1]) / (2.0 * p_step)

    return {"queries": int(np.ceil(total)), "beta_0": 0.0, "initial_overlap": float(initial_overlap), "schedule_length": int(max(len(schedule) - 1, 0)), "min_step_overlap": float(np.min(step_overlaps)) if step_overlaps else 1.0, "target_delta": float(target_delta)}


def run_experiment_to_generate_quantum_queries(
    n_list,
    idx_min=0,
    idx_max=None,
    out_file: Path = None,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    skip_most_acceptance: bool = True,
    statistic: str = "mean+std",
    alpha_target: float = np.sqrt(1.0 / np.e),
) -> None:
    AS = [np.inf] if skip_most_acceptance else ACCEPTANCES
    out_file = QUANTUM_QUERY_FILE if out_file is None else out_file
    columns = ["n", "idx", "proposal", "a", "beta", "q0_mode", "beta_0", "initial_overlap", "schedule_length", "min_step_overlap", "target_delta", "ok", "error_message", "queries"]

    def row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=np.nan, initial_overlap=np.nan, schedule_length=0, min_step_overlap=np.nan, target_delta=np.nan, ok=False, error_message="", queries=-1):
        return {"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "q0_mode": q0_mode, "beta_0": beta_0, "initial_overlap": initial_overlap, "schedule_length": int(schedule_length), "min_step_overlap": min_step_overlap, "target_delta": target_delta, "ok": bool(ok), "error_message": str(error_message), "queries": int(queries)}

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file) if out_file.exists() else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)
    df["queries"] = df["queries"].astype(object)

    idx_max = 100 if idx_max is None else int(idx_max)
    fit_cache = {}

    for n in n_list:
        ising = [load_instances(n, idx) for idx in range(idx_min, idx_max)]

        for local_idx, idx in enumerate(range(idx_min, idx_max)):
            print(n, idx)
            model = ising[local_idx]
            rows = []

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        for q0_mode in Q0_MODES:
                            a_col = df["a"].astype(float)
                            a_mask = np.isinf(a_col) if np.isinf(a) else np.isclose(a_col, float(a))
                            mask = (df["n"].astype(int) == int(n)) & (df["idx"].astype(int) == int(idx)) & (df["proposal"] == proposal) & a_mask & np.isclose(df["beta"].astype(float), float(beta)) & (df["q0_mode"] == q0_mode)
                            if mask.any():
                                continue

                            try:
                                if q0_mode == "bhattacharyya":
                                    res = get_warm_start_quantum_query_cost(model=model, n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, alpha_target=alpha_target, fit_cache=fit_cache)
                                elif q0_mode == "uniform":
                                    res = get_uniform_annealing_quantum_query_cost(model=model, n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, alpha_target=alpha_target, fit_cache=fit_cache)
                                else:
                                    raise ValueError(f"unknown q0_mode: {q0_mode}")

                                rows.append(row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=res["beta_0"], initial_overlap=res["initial_overlap"], schedule_length=res["schedule_length"], min_step_overlap=res["min_step_overlap"], target_delta=res["target_delta"], ok=True, queries=res["queries"]))

                            except Exception as e:
                                rows.append(row_with_counts(n, idx, proposal, a, beta, q0_mode, ok=False, error_message=repr(e)))

                            print(".", end="", flush=True)

            print("")
            if rows:
                new_df = pd.DataFrame(rows, columns=columns)
                new_df["queries"] = pd.Series([int(x) for x in new_df["queries"].tolist()], dtype=object)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_pickle(out_file)


def load_quantum_queries(
    proposal: str,
    a: int | float,
    q0_mode: str,
    in_file: Path = QUANTUM_QUERY_FILE,
    only_ok: bool = True,
) -> pd.DataFrame:
    assert proposal in PROPOSALS, f"proposal must be one of {PROPOSALS}"
    assert a in ACCEPTANCES, f"a must be one of {ACCEPTANCES}"
    assert q0_mode in Q0_MODES, f"q0_mode must be one of {Q0_MODES}"

    df = pd.read_pickle(in_file)
    a_col = df["a"].astype(float)
    a_mask = np.isinf(a_col) if np.isinf(a) else np.isclose(a_col, float(a))
    mask = (df["proposal"] == proposal) & a_mask & (df["q0_mode"] == q0_mode)

    if only_ok and "ok" in df.columns:
        mask = mask & df["ok"].astype(bool)

    out = df.loc[mask].copy()
    out["queries"] = pd.to_numeric(out["queries"], errors="coerce")

    return out.reset_index(drop=True)


def get_quantum_query_stats(
    proposal: str,
    a: int | float,
    q0_mode: str,
    in_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    only_ok: bool = True,
) -> pd.DataFrame:
    statistics = ["mean+std", "mean+std-tail", "median+mad"]
    assert statistic in statistics, f"statistic must be one of {statistics}"

    df = load_quantum_queries(proposal=proposal, a=a, q0_mode=q0_mode, in_file=in_file, only_ok=only_ok)
    rows = []

    for (n, beta), group in df.groupby(["n", "beta"], sort=True):
        x = group["queries"].to_numpy(dtype=float)
        x = x[np.isfinite(x) & (x > 0)]

        if x.size == 0:
            center, spread, count = np.nan, np.nan, 0
        elif statistic == "mean+std":
            center, spread, count = np.mean(x), np.std(x), x.size
        elif statistic == "mean+std-tail":
            q1, q3 = np.percentile(x, [25, 75])
            x = x[(x >= q1) & (x <= q3)]
            center, spread, count = (np.mean(x), np.std(x), x.size) if x.size else (np.nan, np.nan, 0)
        elif statistic == "median+mad":
            center = np.median(x)
            spread = np.median(np.abs(x - center))
            count = x.size

        rows.append({"n": int(n), "beta": float(beta), "center": float(center), "spread": float(spread), "count": int(count), "statistic": statistic, "q0_mode": q0_mode})

    columns = ["n", "beta", "center", "spread", "count", "statistic", "q0_mode"]
    return pd.DataFrame(rows, columns=columns).sort_values(["n", "beta"]).reset_index(drop=True)


def get_quantum_query_fit(
    proposal: str,
    a: int | float,
    q0_mode: str,
    beta: float,
    in_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    n_min: int | None = None,
    n_max: int | None = None,
    only_ok: bool = True,
) -> tuple[float, float]:
    """
    Fit Q(n) = A * exp(b n) at fixed beta, optionally using only n_min <= n <= n_max.
    """
    table = get_quantum_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, in_file=in_file, statistic=statistic, only_ok=only_ok)
    table = table[np.isclose(table["beta"].astype(float), float(beta))]

    if n_min is not None:
        table = table[table["n"].astype(int) >= n_min]

    if n_max is not None:
        table = table[table["n"].astype(int) <= n_max]

    if table.empty:
        raise ValueError(f"No data found for beta={beta}, q0_mode={q0_mode} in n range [{n_min}, {n_max}].")

    n = table["n"].to_numpy(dtype=float)
    y = table["center"].to_numpy(dtype=float)
    mask = np.isfinite(n) & np.isfinite(y) & (y > 0.0)
    n, y = n[mask], y[mask]

    if len(y) < 2:
        raise ValueError("Need at least two positive points to fit.")

    b, log_A = np.polyfit(n, np.log(y), deg=1)
    return float(np.exp(log_A)), float(b)