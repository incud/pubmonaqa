from pathlib import Path

from monaqa2.data.filename import CLASSICAL_QUERY_FILE
from monaqa2.data.hyperparams import load_best_qemc_gamma_t
from monaqa2.data.instances import load_instances
from monaqa2.data.utils_stats import fit_exponential_from_stats, grouped_statistics
from monaqa2.mcmc.distribution import get_gibbs_distribution, get_gibbs_distribution_with_bhattacharyya_guarantee
from monaqa2.mcmc.transition import create_transition_matrix
from monaqa2.mcmc.validation import is_irreducible
from monaqa2.mcmc.search import search_monotone
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from functools import cache


def connectivity_bottleneck(X: np.ndarray) -> float:
    """
    Return the bottleneck weight of the maximum spanning tree of the off-diagonal graph induced by X.

    This is used as a cheap connectivity proxy for the symmetric discriminant matrix. If the off-diagonal graph is disconnected, no spanning tree exists and the bottleneck is reported as 0.

    :param X: Symmetric nonnegative matrix, typically sqrt(P * P.T).
    :return: Minimum edge weight along the maximum spanning tree, or 0 if disconnected.
    """
    X_ = X.copy()
    np.fill_diagonal(X_, 0.0)

    # scipy gives a minimum spanning tree; applying it to -X_ gives a maximum spanning tree of X_.
    T = minimum_spanning_tree(-X_)

    if T.nnz < X_.shape[0] - 1:
        # The graph induced by X_ is not connected, so no spanning tree exists.
        return 0.0

    return float(-T.data.max())


def tv_convergence_times_column_stochastic(P, q0, pi, eps_list, tol=1e-11, max_iter=2**50):
    """
    Compute TV convergence times for a column-stochastic reversible Markov chain.

    The returned dictionary maps each epsilon in `eps_list` to the first time t such that TV(P^t q0, pi) <= epsilon. For well-conditioned stationary distributions, the method evolves the deviation in the symmetric-discriminant eigenbasis; if pi has very small entries, it falls back to direct probability evolution.

    :param P: Column-stochastic transition matrix.
    :param q0: Initial probability distribution.
    :param pi: Stationary probability distribution.
    :param eps_list: Decreasing list of target TV errors.
    :param tol: Numerical tolerance for stationarity and validity checks.
    :param max_iter: Maximum search horizon passed to search_monotone.
    :return: Dictionary epsilon -> convergence time, with negative values indicating search failure.
    """
    P = np.asarray(P, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    pi = np.asarray(pi, dtype=float)
    eps_list = [float(eps) for eps in eps_list]
    spectral_tol = 1e-8

    if any(eps_list[i] < eps_list[i + 1] for i in range(len(eps_list) - 1)):
        raise ValueError("eps_list must be sorted decreasingly")

    if not np.allclose(P @ pi, pi, atol=tol, rtol=0.0):
        stationarity_error = float(np.max(np.abs(P @ pi - pi)))
        raise ValueError(f"pi is not stationary for P: {stationarity_error=}, {tol=}")

    min_pi = float(np.min(pi))
    sqrt_pi = np.sqrt(pi)
    out = {}

    def get_qt(t):
        """
        Evolve q0 directly by matrix powers.
        """
        t = int(round(t))
        qt = np.linalg.matrix_power(P, t) @ q0
        qt = np.maximum(qt, 0.0)
        qt /= qt.sum()
        return qt

    def get_gt(t):
        """
        Evolve the pi-weighted deviation in the symmetric-discriminant eigenbasis.
        """
        assert min_pi >= spectral_tol
        t = int(round(t))
        return eigvecs @ ((eigvals ** t) * coeff0)

    # The symmetric discriminant is used because the chains checked below are reversible with respect to pi.
    X = np.sqrt(np.maximum(P * P.T, 0.0))
    X = 0.5 * (X + X.T)

    if min_pi >= spectral_tol:
        eigvals, eigvecs = np.linalg.eigh(X)
        eigvals = np.clip(eigvals, -1.0, 1.0)

        # Sort by absolute eigenvalue so the slow modes come first; this is not required for correctness but helps conditioning.
        order = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        g0 = (q0 - pi) / sqrt_pi
        coeff0 = eigvecs.T @ g0

    done = False
    for eps in eps_list:
        if done:
            out[eps] = -1
            continue

        if min_pi < spectral_tol:
            # Direct evolution is safer when division by sqrt(pi) would be numerically unstable.
            cost = lambda q: 0.5 * float(np.sum(np.abs(q - pi))) - eps
            try:
                out[eps] = int(search_monotone(lambda t: get_qt(t), cost, start_iter=1, max_iter=max_iter, info="Classical queries via direct probability evolution"))
            except ValueError as e:
                out[eps] = -2
                done = True
        else:
            # In the spectral representation, TV distance is 1/2 * sum_x sqrt(pi_x) |g_t(x)|.
            cost = lambda g: 0.5 * float(np.sum(sqrt_pi * np.abs(g))) - eps
            try:
                out[eps] = int(search_monotone(lambda t: get_gt(t), cost, start_iter=1, max_iter=max_iter, info="Classical queries via symmetric spectral decomposition"))
            except ValueError as e:
                out[eps] = -1
                done = True

    return out


def run_experiment_to_generate_classical_queries(n_list, idx_min=0, idx_max=None, out_file: Path = None, skip_most_acceptance: bool = True) -> None:
    """
    Generate and persist classical TV-convergence query counts.

    For each selected instance, proposal, acceptance parameter, beta, and initialization mode, this method computes the first classical Markov-chain time needed to reach each epsilon in 1e-2,...,1e-8. Existing rows are skipped using an in-memory key set, so the method can resume from a partially generated pickle file.

    :param n_list: Iterable of system sizes.
    :param idx_min: First instance index, inclusive.
    :param idx_max: Last instance index, exclusive. If None, use 100.
    :param out_file: Output pickle file. If None, use CLASSICAL_QUERY_FILE.
    :param skip_most_acceptance: If True, only use a=np.inf; otherwise use np.inf, 1, and 10.
    :return: None.
    """
    
    PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    AS = [np.inf] if skip_most_acceptance else [np.inf, 1, 10]
    BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    EPS_LIST = [10.0 ** (-k) for k in range(2, 9)]
    EPS_COLUMNS = [f"queries_eps_1e-{k}" for k in range(2, 9)]
    Q0_MODES = ("uniform", "bhattacharyya")
    TOL = 1e-11
    out_file = CLASSICAL_QUERY_FILE if out_file is None else out_file
    columns = ["n", "idx", "proposal", "a", "beta", "q0_mode", "beta_0", "initial_overlap", "connectivity_bottleneck", "ok", "error_message"] + EPS_COLUMNS

    def a_key(a):
        """
        Normalize the acceptance parameter for hashable row keys.
        """
        return np.inf if np.isinf(float(a)) else float(a)

    def row_key(row):
        """
        Convert a dataframe/dict row to the unique experiment key.
        """
        return (int(row["n"]), int(row["idx"]), str(row["proposal"]), a_key(row["a"]), float(row["beta"]), str(row["q0_mode"]))

    def key(n, idx, proposal, a, beta, q0_mode):
        """
        Build the unique experiment key.
        """
        return (int(n), int(idx), str(proposal), a_key(a), float(beta), str(q0_mode))

    def row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=np.nan, initial_overlap=np.nan, bottleneck=np.nan, ok=False, error_message="", query_counts=None):
        """
        Build one output row.
        """
        row = {"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "q0_mode": q0_mode, "beta_0": beta_0, "initial_overlap": initial_overlap, "connectivity_bottleneck": bottleneck, "ok": bool(ok), "error_message": str(error_message)}
        row.update({f"queries_eps_1e-{k}": int(query_counts[10.0 ** (-k)]) if query_counts is not None else -1 for k in range(2, 9)})
        return row

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file) if out_file.exists() else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)
    for c in EPS_COLUMNS:
        df[c] = df[c].astype(object)

    # Precompute completed rows so resume checks are O(1) rather than repeated dataframe scans.
    existing = set(row_key(row) for _, row in df.iterrows())

    def append_and_save(rows):
        """
        Append rows to the output dataframe and immediately checkpoint to disk.
        """
        nonlocal df
        if not rows:
            return
        new_df = pd.DataFrame(rows, columns=columns)
        for c in EPS_COLUMNS:
            new_df[c] = pd.Series([int(x) for x in new_df[c].tolist()], dtype=object)
        df = pd.concat([df, new_df], ignore_index=True)
        for row in rows:
            existing.add(row_key(row))
        df.to_pickle(out_file)

    def instance_complete(n, idx):
        """
        Check whether all rows for one instance are already present.
        """
        return all(key(n, idx, proposal, a, beta, q0_mode) in existing for proposal in PROPOSALS for a in AS for beta in BETAS for q0_mode in Q0_MODES)

    idx_max = 100 if idx_max is None else int(idx_max)

    for n in n_list:
        for idx in range(idx_min, idx_max):
            if instance_complete(n, idx):
                print(f"Skipping n={n}, idx={idx}: already in table.", flush=True)
                continue

            print(n, idx)
            model = load_instances(n, idx)
            gamma, t_qemc = load_best_qemc_gamma_t(n, idx)

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        missing_q0_modes = [q0_mode for q0_mode in Q0_MODES if key(n, idx, proposal, a, beta, q0_mode) not in existing]

                        if not missing_q0_modes:
                            print(f"Skipping n={n}, idx={idx}, proposal={proposal}, a={a}, beta={beta}: already in table.", flush=True)
                            continue

                        try:
                            pi = np.asarray(get_gibbs_distribution(model, beta), dtype=float)
                            P = np.asarray(create_transition_matrix(proposal, model, beta, a, gamma, t_qemc), dtype=float)

                            if not np.all(np.isfinite(pi)) or not np.all(np.isfinite(P)):
                                raise ValueError("P or pi contains NaN/inf")
                            if np.min(pi) < -TOL or np.min(P) < -TOL:
                                raise ValueError(f"P or pi has negative entries: min_P={float(np.min(P))}, min_pi={float(np.min(pi))}")

                            pi = np.maximum(pi, 0.0)
                            pi = pi / float(np.sum(pi))

                            # Validation here enforces the assumptions used by the symmetric spectral TV evolution.
                            col_error = float(np.max(np.abs(np.sum(P, axis=0) - 1.0)))
                            stationarity_error = float(np.max(np.abs(P @ pi - pi)))
                            reversibility_error = float(np.max(np.abs(P * pi[None, :] - P.T * pi[:, None])))

                            if col_error > TOL:
                                raise ValueError(f"P is not column-stochastic: {col_error=}, {TOL=}")
                            if stationarity_error > TOL:
                                raise ValueError(f"pi is not stationary for P: {stationarity_error=}, {TOL=}")
                            if reversibility_error > TOL:
                                raise ValueError(f"P is not reversible with respect to pi: {reversibility_error=}, {TOL=}")

                            X = np.sqrt(np.maximum(P * P.T, 0.0))
                            bottleneck = connectivity_bottleneck(X)

                        except Exception as e:
                            # If the transition matrix is invalid, mark both missing initialization modes as failed.
                            append_and_save([row_with_counts(n, idx, proposal, a, beta, q0_mode, ok=False, error_message=repr(e)) for q0_mode in missing_q0_modes])
                            print(".", end="", flush=True)
                            continue

                        for q0_mode in missing_q0_modes:
                            beta_0 = np.nan
                            initial_overlap = np.nan

                            try:
                                if q0_mode == "uniform":
                                    q0 = np.full_like(pi, 1.0 / pi.size)
                                    beta_0 = 0.0
                                elif q0_mode == "bhattacharyya":
                                    # Warm-start distribution at beta_0 chosen to guarantee a fixed Bhattacharyya overlap with pi_beta.
                                    res = get_gibbs_distribution_with_bhattacharyya_guarantee(model, beta, pi, np.sqrt(1.0 / np.e))
                                    q0, beta_0 = res if isinstance(res, tuple) else (res, np.nan)
                                    q0 = np.asarray(q0, dtype=float)
                                else:
                                    raise ValueError(f"unknown q0_mode: {q0_mode}")

                                if not np.all(np.isfinite(q0)) or np.min(q0) < -TOL:
                                    raise ValueError(f"invalid q0: min_q0={float(np.min(q0))}")

                                q0 = np.maximum(q0, 0.0)
                                q0 = q0 / float(np.sum(q0))
                                initial_overlap = float(np.sum(np.sqrt(np.maximum(q0 * pi, 0.0))))

                                if q0_mode == "uniform" and beta_0 != 0.0:
                                    raise ValueError(f"uniform q0 should have beta_0=0, got {beta_0=}")
                                if q0_mode == "bhattacharyya" and (beta_0 < -TOL or initial_overlap < np.sqrt(1.0 / np.e) - TOL):
                                    raise ValueError(f"bad bhattacharyya warm start: {beta_0=}, {initial_overlap=}")

                                query_counts = tv_convergence_times_column_stochastic(P, q0, pi, EPS_LIST, tol=TOL)
                                query_counts = {eps: (-1 if int(query_counts[eps]) < 0 else int(query_counts[eps])) for eps in EPS_LIST}

                                ok = True
                                error_message = ""
                                previous = None

                                # Smaller epsilon should never require fewer queries; flag violations as numerical/search failures.
                                for eps in EPS_LIST:
                                    t_eps = query_counts[eps]
                                    if t_eps < 0:
                                        ok = False
                                        error_message = f"search failed at eps={eps}"
                                        break
                                    if previous is not None and t_eps < previous:
                                        ok = False
                                        error_message = f"query counts are not monotone at eps={eps}"
                                        break
                                    previous = t_eps

                                append_and_save([row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0, initial_overlap, bottleneck, ok, error_message, query_counts)])

                            except Exception as e:
                                append_and_save([row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=beta_0, initial_overlap=initial_overlap, bottleneck=bottleneck, ok=False, error_message=repr(e))])

                            print(".", end="", flush=True)

            print("")


def _epsilon_to_query_column(epsilon: float) -> str:
    """
    Convert an epsilon value into the corresponding query-count column name.

    :param epsilon: Epsilon value, one of 1e-2,...,1e-8.
    :return: Column name such as "queries_eps_1e-2".
    """
    eps_map = {10.0 ** (-k): f"queries_eps_1e-{k}" for k in range(2, 9)}
    eps_key = min(eps_map, key=lambda x: abs(x - float(epsilon)))

    if not np.isclose(eps_key, float(epsilon)):
        raise ValueError(f"epsilon must be one of {sorted(eps_map)}")

    return eps_map[eps_key]


def load_classical_queries(
    proposal: str,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    in_file: Path = CLASSICAL_QUERY_FILE,
    only_ok: bool = True,
) -> pd.DataFrame:
    """
    Load classical query counts for a fixed proposal, acceptance parameter, initialization mode, and epsilon.

    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param q0_mode: Initialization mode, either "uniform" or "bhattacharyya".
    :param epsilon: TV error tolerance selecting one query-count column.
    :param in_file: Input pickle file.
    :param only_ok: If True, keep only rows whose ok flag is True.
    :return: DataFrame with columns ["n", "idx", "beta", "queries"].
    """
    proposals = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    acceptances = [np.inf, 1, 10]
    q0_modes = ["uniform", "bhattacharyya"]

    assert proposal in proposals, f"proposal must be one of {proposals}"
    assert a in acceptances, f"a must be one of {acceptances}"
    assert q0_mode in q0_modes, f"q0_mode must be one of {q0_modes}"

    col = _epsilon_to_query_column(epsilon)
    df = pd.read_pickle(in_file)

    if np.isinf(a):
        a_mask = np.isinf(df["a"].astype(float))
    else:
        a_mask = np.isclose(df["a"].astype(float), float(a))

    mask = (df["proposal"] == proposal) & a_mask & (df["q0_mode"] == q0_mode)

    if only_ok and "ok" in df.columns:
        mask = mask & df["ok"].astype(bool)

    out = df.loc[mask, ["n", "idx", "beta", col]].copy()
    out = out.rename(columns={col: "queries"})
    out["queries"] = pd.to_numeric(out["queries"], errors="coerce")

    return out.reset_index(drop=True)


@cache
def get_classical_query_stats(
    proposal: str,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    in_file: Path = CLASSICAL_QUERY_FILE,
    statistic: str = "mean+std",
    only_ok: bool = True,
) -> pd.DataFrame:
    """
    Calculate classical query-count statistics over instance index for each pair (n, beta).

    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param q0_mode: Initialization mode.
    :param epsilon: TV error tolerance.
    :param in_file: Input pickle file.
    :param statistic: Statistic rule handled by grouped_statistics.
    :param only_ok: If True, keep only successful rows.
    :return: Statistics table with center/spread/count for each (n, beta).
    """
    df = load_classical_queries(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=in_file, only_ok=only_ok)

    # Query counts are meaningful only when positive; failed rows use negative sentinels and are discarded here.
    return grouped_statistics(df=df, value_col="queries", group_cols=("n", "beta"), statistic=statistic, positive_only=True, extra_cols={"epsilon": float(epsilon), "q0_mode": q0_mode})


def get_classical_query_fit_by_n(
    proposal: str,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    beta: float,
    in_file: Path = CLASSICAL_QUERY_FILE,
    statistic: str = "mean+std",
    n_min: int | None = None,
    n_max: int | None = None,
    only_ok: bool = True,
) -> tuple[float, float]:
    """
    Fit classical query scaling T(n) = A * exp(b n) at fixed beta.

    :param proposal: Proposal name.
    :param a: Acceptance parameter.
    :param q0_mode: Initialization mode.
    :param epsilon: TV error tolerance.
    :param beta: Fixed inverse temperature.
    :param in_file: Input pickle file.
    :param statistic: Statistic rule used before fitting.
    :param n_min: Optional minimum n included in the fit.
    :param n_max: Optional maximum n included in the fit.
    :param only_ok: If True, keep only successful rows.
    :return: Tuple (A, b) for T(n) = A * exp(b n).
    """
    table = get_classical_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=in_file, statistic=statistic, only_ok=only_ok)
    return fit_exponential_from_stats(table=table, beta=beta, n_min=n_min, n_max=n_max, sign=1)
