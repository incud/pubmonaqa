from pathlib import Path

from monaqa2.data.filename import CLASSICAL_QUERY_FILE
from monaqa2.data.hyperparams import load_best_qemc_gamma_t
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.distribution import get_gibbs_distribution, get_gibbs_distribution_with_bhattacharyya_guarantee
from monaqa2.mcmc.transition import create_transition_matrix
from monaqa2.mcmc.validation import is_irreducible
from monaqa2.mcmc.search import search_monotone
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def connectivity_bottleneck(X: np.ndarray) -> float:
    X_ = X.copy()
    np.fill_diagonal(X_, 0.0)
    T = minimum_spanning_tree(-X_)
    if T.nnz < X_.shape[0] - 1:
        # The graph induced by X_ is not connected, so no spanning tree exists.
        return 0.0 
    return float(-T.data.max())


def tv_convergence_times_column_stochastic(P, q0, pi, eps_list, tol=1e-11, max_iter=2**50):
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
        t = int(round(t))
        qt = np.linalg.matrix_power(P, t) @ q0
        qt = np.maximum(qt, 0.0)
        qt /= qt.sum()
        return qt

    def get_gt(t):
        assert min_pi >= spectral_tol
        t = int(round(t))
        return eigvecs @ ((eigvals ** t) * coeff0)

    
    X = np.sqrt(np.maximum(P * P.T, 0.0))
    X = 0.5 * (X + X.T)

    if min_pi >= spectral_tol:
        eigvals, eigvecs = np.linalg.eigh(X)
        eigvals = np.clip(eigvals, -1.0, 1.0)
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
            cost = lambda q: 0.5 * float(np.sum(np.abs(q - pi))) - eps
            try:
                out[eps] = int(search_monotone(lambda t: get_qt(t), cost, start_iter=1, max_iter=max_iter, info="Classical queries via direct probability evolution"))
            except ValueError as e:
                out[eps] = -2
                done = True
        else:
            cost = lambda g: 0.5 * float(np.sum(sqrt_pi * np.abs(g))) - eps
            try:
                out[eps] = int(search_monotone(lambda t: get_gt(t), cost, start_iter=1, max_iter=max_iter, info="Classical queries via symmetric spectral decomposition"))
            except ValueError as e:
                out[eps] = -1
                done = True

    return out


def run_experiment_to_generate_classical_queries(n_list, idx_min=0, idx_max=None, out_file: Path = None, skip_most_acceptance: bool = True) -> None:
    
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
        return np.inf if np.isinf(float(a)) else float(a)

    def row_key(row):
        return (int(row["n"]), int(row["idx"]), str(row["proposal"]), a_key(row["a"]), float(row["beta"]), str(row["q0_mode"]))

    def key(n, idx, proposal, a, beta, q0_mode):
        return (int(n), int(idx), str(proposal), a_key(a), float(beta), str(q0_mode))

    def row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=np.nan, initial_overlap=np.nan, bottleneck=np.nan, ok=False, error_message="", query_counts=None):
        row = {"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "q0_mode": q0_mode, "beta_0": beta_0, "initial_overlap": initial_overlap, "connectivity_bottleneck": bottleneck, "ok": bool(ok), "error_message": str(error_message)}
        row.update({f"queries_eps_1e-{k}": int(query_counts[10.0 ** (-k)]) if query_counts is not None else -1 for k in range(2, 9)})
        return row

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file) if out_file.exists() else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)
    for c in EPS_COLUMNS:
        df[c] = df[c].astype(object)

    existing = set(row_key(row) for _, row in df.iterrows())

    def append_and_save(rows):
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
    Calculate query-count statistics over idx for each (n, beta).
    """
    statistics = ["mean+std", "mean+std-tail", "median+mad"]
    assert statistic in statistics, f"statistic must be one of {statistics}"

    df = load_classical_queries(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=in_file, only_ok=only_ok)
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

        rows.append({"n": int(n), "beta": float(beta), "center": float(center), "spread": float(spread), "count": int(count), "statistic": statistic, "epsilon": float(epsilon), "q0_mode": q0_mode})

    return pd.DataFrame(rows).sort_values(["n", "beta"]).reset_index(drop=True)


def get_classical_query_fit(
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
    Fit T(n) = A * exp(b n) at fixed beta, optionally using only n_min <= n <= n_max.
    """
    table = get_classical_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=in_file, statistic=statistic, only_ok=only_ok)
    table = table[np.isclose(table["beta"].astype(float), float(beta))]

    if n_min is not None:
        table = table[table["n"].astype(int) >= n_min]

    if n_max is not None:
        table = table[table["n"].astype(int) <= n_max]

    if table.empty:
        raise ValueError(f"No data found for beta={beta}, epsilon={epsilon}, q0_mode={q0_mode} in n range [{n_min}, {n_max}].")

    n = table["n"].to_numpy(dtype=float)
    y = table["center"].to_numpy(dtype=float)

    mask = np.isfinite(n) & np.isfinite(y) & (y > 0.0)
    n, y = n[mask], y[mask]

    if len(y) < 2:
        raise ValueError("Need at least two positive points to fit.")

    b, log_A = np.polyfit(n, np.log(y), deg=1)

    return float(np.exp(log_A)), float(b)