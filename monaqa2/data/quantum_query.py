from pathlib import Path

from monaqa2.data.filename import QUANTUM_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.instances import load_instances
from monaqa2.data.annealing import get_annealing_betas
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_beta, get_spectral_gap_fit_by_beta_single_value
from monaqa2.data.utils_stats import fit_exponential_from_stats, grouped_statistics
from monaqa2.mcmc.distribution import get_gibbs_distribution, get_gibbs_distribution_with_bhattacharyya_guarantee
import numpy as np
import pandas as pd


PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
ACCEPTANCES = [np.inf, 1, 10]
Q0_MODES = ["uniform", "bhattacharyya"]
BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def overlap(p: np.ndarray, q: np.ndarray) -> float:
    """
    Return the Bhattacharyya overlap sum_x sqrt(p_x q_x).

    :param p: First probability distribution.
    :param q: Second probability distribution.
    :return: Bhattacharyya overlap.
    """
    return float(np.sum(np.sqrt(p * q)))


def qpe_queries_from_spectral_gap(delta: float) -> int:
    """
    Convert a classical spectral gap into a conservative QPE grid size.

    The Szegedy phase gap is Delta_phi = arccos(1 - delta). The returned query count is ceil(2 pi / Delta_phi).

    :param delta: Classical spectral gap.
    :return: Number of controlled-walk powers/queries used by the QPE projection model.
    """
    phase_gap = float(np.arccos(np.clip(1.0 - float(delta), -1.0, 1.0)))

    if not np.isfinite(phase_gap) or phase_gap <= 0.0:
        raise ValueError(f"invalid phase gap from spectral gap: {delta}")

    return int(np.ceil(2.0 * np.pi / phase_gap))


def get_qpe_cost(
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    beta_max: float = 100.0,
    beta_step: float = 0.01,
    grid_cache: dict | None = None,
) -> tuple[int, float]:
    """
    Return the QPE walk-query cost and interpolated spectral gap at fixed (n, proposal, a, beta).

    The new spectral-gap API builds a fixed-n beta grid with get_spectral_gap_fit_by_beta and then
      queries it with get_spectral_gap_fit_by_beta_single_value. A grid cache is used here only to
        avoid rebuilding the same fixed-n interpolation grid many times inside annealing loops.

    :param n: System size.
    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param beta: Target inverse temperature.
    :param spectral_gap_file: Input spectral-gap pickle file.
    :param statistic: Statistic used to build the spectral-gap interpolation grid.
    :param beta_max: Maximum beta used for the interpolation grid.
    :param beta_step: Beta spacing used for the interpolation grid.
    :param grid_cache: Optional dictionary used to cache fixed-n beta grids.
    :return: Tuple (qpe_queries, interpolated_delta).
    """
    beta = float(beta)

    if beta <= 0.0:
        return 0, 1.0

    grid_cache = {} if grid_cache is None else grid_cache
    a_key = np.inf if np.isinf(a) else float(a)
    key = (str(spectral_gap_file), int(n), proposal, a_key, statistic, float(beta_max), float(beta_step))

    if key not in grid_cache:
        grid_cache[key] = get_spectral_gap_fit_by_beta(proposal=proposal, a=a, fixed_n=n, beta_max=beta_max, beta_step=beta_step, spectral_gap_file=spectral_gap_file, statistic=statistic)

    delta = get_spectral_gap_fit_by_beta_single_value(grid=grid_cache[key], beta=beta, beta_max=beta_max, beta_step=beta_step)

    return qpe_queries_from_spectral_gap(delta), delta


def get_quantum_query_cost_from_warm_start_distribution(
    model,
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    alpha_target: float = np.sqrt(1.0 / np.e),
    beta_max: float = 100.0,
    beta_step: float = 0.01,
    grid_cache: dict | None = None,
) -> dict:
    """
    Estimate QPE-query cost for a single warm-start projection onto pi_beta.

    The initial state is pi_{beta_0}, where beta_0 is chosen by get_gibbs_distribution_with_bhattacharyya_guarantee so that the overlap with pi_beta is at least alpha_target. The expected number of projection attempts is C_beta / |<pi_beta|pi_beta0>|^2.

    :param model: Ising model instance.
    :param n: System size.
    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param beta: Target inverse temperature.
    :param spectral_gap_file: Input spectral-gap pickle file.
    :param statistic: Statistic used for the spectral-gap interpolation.
    :param alpha_target: Minimum Bhattacharyya overlap used to choose beta_0.
    :param beta_max: Maximum beta used for the interpolation grid.
    :param beta_step: Beta spacing used for the interpolation grid.
    :param grid_cache: Optional dictionary used to cache fixed-n beta grids.
    :return: Row payload with query count and diagnostic metadata.
    """
    pi_beta = get_gibbs_distribution(model, beta)
    q0, beta_0 = get_gibbs_distribution_with_bhattacharyya_guarantee(model, beta, pi_beta, alpha_target)
    initial_overlap = overlap(q0, pi_beta)
    p_success = initial_overlap ** 2

    if p_success <= 0.0 or not np.isfinite(p_success):
        raise ValueError(f"invalid warm-start success probability: {p_success}")

    C_beta, target_delta = get_qpe_cost(n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, beta_max=beta_max, beta_step=beta_step, grid_cache=grid_cache)

    return {"queries": int(np.ceil(C_beta / p_success)), "beta_0": float(beta_0), "initial_overlap": float(initial_overlap), "schedule_length": 0, "min_step_overlap": np.nan, "target_delta": float(target_delta)}


def get_quantum_query_cost_from_uniform_distribution(
    model,
    n: int,
    proposal: str,
    a: int | float,
    beta: float,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    alpha_target: float = np.sqrt(1.0 / np.e),
    beta_max: float = 100.0,
    beta_step: float = 0.01,
    grid_cache: dict | None = None,
) -> dict:
    """
    Estimate QPE-query cost for the uniform-start (annealing-like) procedure.

    The initial state is the infinite-temperature Gibbs state. The method builds an annealing schedule 0=beta_0<...<beta_L=beta and applies the rewind expected-cost formula C_{t+1} + (C_t + C_{t+1})/(2 p_{t+1}) at each transition.

    :param model: Ising model instance.
    :param n: System size.
    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param beta: Target inverse temperature.
    :param spectral_gap_file: Input spectral-gap pickle file.
    :param statistic: Statistic used for the spectral-gap interpolation.
    :param alpha_target: Minimum adjacent Bhattacharyya overlap used by the schedule.
    :param beta_max: Maximum beta used for the interpolation grid.
    :param beta_step: Beta spacing used for the interpolation grid.
    :param grid_cache: Optional dictionary used to cache fixed-n beta grids.
    :return: Row payload with query count and diagnostic metadata.
    """
    pi_beta = get_gibbs_distribution(model, beta)
    pi_0 = np.full_like(pi_beta, 1.0 / pi_beta.size)
    initial_overlap = overlap(pi_0, pi_beta)

    schedule = get_annealing_betas(model=model, beta_final=beta, alpha=alpha_target)

    # QPE costs C_t for every schedule point beta_t, including beta_0=0.
    costs = [get_qpe_cost(n=n, proposal=proposal, a=a, beta=float(beta_t), spectral_gap_file=spectral_gap_file, statistic=statistic, beta_max=beta_max, beta_step=beta_step, grid_cache=grid_cache)[0] for beta_t in schedule]
    _, target_delta = get_qpe_cost(n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, beta_max=beta_max, beta_step=beta_step, grid_cache=grid_cache)

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

        # Rewind expected cost for transition beta_j -> beta_{j+1}.
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
    beta_max: float = 100.0,
    beta_step: float = 0.01,
) -> None:
    """
    Generate and persist QPE-based quantum-query estimates.

    The output table has one row for each (n, idx, proposal, a, beta, q0_mode). q0_mode="bhattacharyya" uses one warm-start projection. q0_mode="uniform" uses the full annealing schedule from beta=0 to the target beta. Existing rows are skipped if the output file already exists.

    :param n_list: Iterable of system sizes.
    :param idx_min: First instance index, inclusive.
    :param idx_max: Last instance index, exclusive. If None, use 100.
    :param out_file: Output pickle file. If None, use QUANTUM_QUERY_FILE.
    :param spectral_gap_file: Input spectral-gap pickle file.
    :param skip_most_acceptance: If True, only use a=np.inf; otherwise use all ACCEPTANCES.
    :param statistic: Statistic used for spectral-gap interpolation.
    :param alpha_target: Overlap threshold used for warm starts and annealing steps.
    :param beta_max: Maximum beta used for spectral-gap interpolation grids.
    :param beta_step: Beta spacing used for spectral-gap interpolation grids.
    :return: None.
    """
    AS = [np.inf] if skip_most_acceptance else ACCEPTANCES
    out_file = QUANTUM_QUERY_FILE if out_file is None else out_file
    columns = ["n", "idx", "proposal", "a", "beta", "q0_mode", "beta_0", "initial_overlap", "schedule_length", "min_step_overlap", "target_delta", "ok", "error_message", "queries"]

    def a_key(a):
        """
        Normalize acceptance parameters for hashable row keys.
        """
        return np.inf if np.isinf(float(a)) else float(a)

    def row_key(row):
        """
        Convert a row-like object into the unique experiment key.
        """
        return (int(row["n"]), int(row["idx"]), str(row["proposal"]), a_key(row["a"]), float(row["beta"]), str(row["q0_mode"]))

    def key(n, idx, proposal, a, beta, q0_mode):
        """
        Build the unique experiment key.
        """
        return (int(n), int(idx), str(proposal), a_key(a), float(beta), str(q0_mode))

    def row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=np.nan, initial_overlap=np.nan, schedule_length=0, min_step_overlap=np.nan, target_delta=np.nan, ok=False, error_message="", queries=-1):
        """
        Build one output row.
        """
        return {"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "q0_mode": q0_mode, "beta_0": beta_0, "initial_overlap": initial_overlap, "schedule_length": int(schedule_length), "min_step_overlap": min_step_overlap, "target_delta": target_delta, "ok": bool(ok), "error_message": str(error_message), "queries": int(queries)}

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file) if out_file.exists() else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)
    df["queries"] = df["queries"].astype(object)

    existing = set(row_key(row) for _, row in df.iterrows())
    idx_max = 100 if idx_max is None else int(idx_max)
    grid_cache = {}

    def append_and_save(rows):
        """
        Append rows to the output dataframe and immediately checkpoint to disk.
        """
        nonlocal df
        if not rows:
            return
        new_df = pd.DataFrame(rows, columns=columns)
        new_df["queries"] = pd.Series([int(x) for x in new_df["queries"].tolist()], dtype=object)
        df = pd.concat([df, new_df], ignore_index=True)
        for row in rows:
            existing.add(row_key(row))
        df.to_pickle(out_file)

    def instance_complete(n, idx):
        """
        Check whether all rows for one instance are already present.
        """
        return all(key(n, idx, proposal, a, beta, q0_mode) in existing for proposal in PROPOSALS for a in AS for beta in BETAS for q0_mode in Q0_MODES)

    for n in n_list:
        for idx in range(idx_min, idx_max):
            if instance_complete(n, idx):
                print(f"Skipping n={n}, idx={idx}: already in table.", flush=True)
                continue

            print(n, idx)
            model = load_instances(n, idx)

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        missing_q0_modes = [q0_mode for q0_mode in Q0_MODES if key(n, idx, proposal, a, beta, q0_mode) not in existing]

                        if not missing_q0_modes:
                            print(f"Skipping n={n}, idx={idx}, proposal={proposal}, a={a}, beta={beta}: already in table.", flush=True)
                            continue

                        for q0_mode in missing_q0_modes:
                            try:
                                if q0_mode == "bhattacharyya":
                                    res = get_quantum_query_cost_from_warm_start_distribution(model=model, n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, alpha_target=alpha_target, beta_max=beta_max, beta_step=beta_step, grid_cache=grid_cache)
                                elif q0_mode == "uniform":
                                    res = get_quantum_query_cost_from_uniform_distribution(model=model, n=n, proposal=proposal, a=a, beta=beta, spectral_gap_file=spectral_gap_file, statistic=statistic, alpha_target=alpha_target, beta_max=beta_max, beta_step=beta_step, grid_cache=grid_cache)
                                else:
                                    raise ValueError(f"unknown q0_mode: {q0_mode}")

                                append_and_save([row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=res["beta_0"], initial_overlap=res["initial_overlap"], schedule_length=res["schedule_length"], min_step_overlap=res["min_step_overlap"], target_delta=res["target_delta"], ok=True, queries=res["queries"])])

                            except Exception as e:
                                append_and_save([row_with_counts(n, idx, proposal, a, beta, q0_mode, ok=False, error_message=repr(e))])

                            print(".", end="", flush=True)

            print("")


def load_quantum_queries(
    proposal: str,
    a: int | float,
    q0_mode: str,
    in_file: Path = QUANTUM_QUERY_FILE,
    only_ok: bool = True,
) -> pd.DataFrame:
    """
    Load quantum-query rows for a fixed proposal, acceptance parameter, and initialization mode.

    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param q0_mode: Initialization mode, either "uniform" or "bhattacharyya".
    :param in_file: Input pickle file.
    :param only_ok: If True, keep only rows whose ok flag is True.
    :return: Filtered dataframe containing at least n, idx, beta, and queries.
    """
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
    """
    Calculate quantum-query statistics over instance index for each pair (n, beta).

    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param q0_mode: Initialization mode.
    :param in_file: Input pickle file.
    :param statistic: Statistic rule handled by grouped_statistics.
    :param only_ok: If True, keep only successful rows.
    :return: Statistics table with center/spread/count for each (n, beta).
    """
    df = load_quantum_queries(proposal=proposal, a=a, q0_mode=q0_mode, in_file=in_file, only_ok=only_ok)

    # Query counts are meaningful only when positive; failed rows use negative sentinels and are discarded here.
    return grouped_statistics(df=df, value_col="queries", group_cols=("n", "beta"), statistic=statistic, positive_only=True, extra_cols={"q0_mode": q0_mode})


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
    Fit quantum-query scaling Q(n) = A * exp(b n) at fixed beta.

    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param q0_mode: Initialization mode.
    :param beta: Fixed inverse temperature.
    :param in_file: Input pickle file.
    :param statistic: Statistic rule used before fitting.
    :param n_min: Optional minimum n included in the fit.
    :param n_max: Optional maximum n included in the fit.
    :param only_ok: If True, keep only successful rows.
    :return: Tuple (A, b) for Q(n) = A * exp(b n).
    """
    table = get_quantum_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, in_file=in_file, statistic=statistic, only_ok=only_ok)
    return fit_exponential_from_stats(table=table, beta=beta, n_min=n_min, n_max=n_max, sign=1)
