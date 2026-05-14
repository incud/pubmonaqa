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


def run_experiment_to_generate_classical_queries(n_list, idx_min=0, idx_max=None, out_file: Path = None) -> None:
    PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    AS = [np.inf, 1, 10]
    BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    EPS_LIST = [10.0 ** (-k) for k in range(2, 9)]
    EPS_COLUMNS = [f"queries_eps_1e-{k}" for k in range(2, 9)]
    Q0_MODES = ("uniform", "bhattacharyya")
    TOL = 1e-11
    out_file = CLASSICAL_QUERY_FILE if out_file is None else out_file
    columns = ["n", "idx", "proposal", "a", "beta", "q0_mode", "beta_0", "initial_overlap", "connectivity_bottleneck", "ok", "error_message"] + EPS_COLUMNS

    def row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=np.nan, initial_overlap=np.nan, bottleneck=np.nan, ok=False, error_message="", query_counts=None):
        row = {"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "q0_mode": q0_mode, "beta_0": beta_0, "initial_overlap": initial_overlap, "connectivity_bottleneck": bottleneck, "ok": bool(ok), "error_message": str(error_message)}
        row.update({f"queries_eps_1e-{k}": int(query_counts[10.0 ** (-k)]) if query_counts is not None else -1 for k in range(2, 9)})
        return row

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)
    for c in EPS_COLUMNS:
        df[c] = df[c].astype(object)

    idx_max = 100 if idx_max is None else int(idx_max)

    for n in n_list:
        ising = [load_instances(n, idx) for idx in range(idx_min, idx_max)]
        params = [load_best_qemc_gamma_t(n, idx) for idx in range(idx_min, idx_max)]

        for local_idx, idx in enumerate(range(idx_min, idx_max)):
            print(n, idx)
            gamma, t_qemc = params[local_idx]
            rows = []

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        try:
                            pi = np.asarray(get_gibbs_distribution(ising[local_idx], beta), dtype=float)
                            P = np.asarray(create_transition_matrix(proposal, ising[local_idx], beta, a, gamma, t_qemc), dtype=float)

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
                            for q0_mode in Q0_MODES:
                                mask = (df["n"] == n) & (df["idx"] == idx) & (df["proposal"] == proposal) & (df["a"] == a) & (df["beta"] == beta) & (df["q0_mode"] == q0_mode)
                                if not mask.any():
                                    rows.append(row_with_counts(n, idx, proposal, a, beta, q0_mode, ok=False, error_message=repr(e)))
                                    print(".", end="", flush=True)
                            continue

                        for q0_mode in Q0_MODES:
                            mask = (df["n"] == n) & (df["idx"] == idx) & (df["proposal"] == proposal) & (df["a"] == a) & (df["beta"] == beta) & (df["q0_mode"] == q0_mode)
                            if mask.any():
                                continue

                            try:
                                if q0_mode == "uniform":
                                    q0 = np.full_like(pi, 1.0 / pi.size)
                                    beta_0 = 0.0
                                elif q0_mode == "bhattacharyya":
                                    res = get_gibbs_distribution_with_bhattacharyya_guarantee(ising[local_idx], beta, pi, np.sqrt(1.0 / np.e))
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

                                rows.append(row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0, initial_overlap, bottleneck, ok, error_message, query_counts))

                            except Exception as e:
                                rows.append(row_with_counts(n, idx, proposal, a, beta, q0_mode, beta_0=locals().get("beta_0", np.nan), initial_overlap=locals().get("initial_overlap", np.nan), bottleneck=bottleneck, ok=False, error_message=repr(e)))

                            print(".", end="", flush=True)

            print("")
            if rows:
                new_df = pd.DataFrame(rows, columns=columns)
                for c in EPS_COLUMNS:
                    new_df[c] = pd.Series([int(x) for x in new_df[c].tolist()], dtype=object)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_pickle(out_file)