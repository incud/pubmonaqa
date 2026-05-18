from monaqa2.data.filename import SPECTRAL_GAP_FILE
from monaqa2.data.hyperparams import load_best_qemc_gamma_t
from monaqa2.data.instances import load_instances
from monaqa2.data.utils_stats import fit_exponential_from_stats, grouped_statistics
from monaqa2.mcmc.transition import create_transition_matrix
import numpy as np
import pandas as pd
from pathlib import Path


PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
AS = [np.inf, 1, 10]


def run_experiment_to_generate_spectral_gaps(n_list=None, idx_min=0, idx_max=None, out_file: Path = None) -> None:
    """
    Generate and persist spectral gaps for all requested instances, proposals, acceptances, and beta values.

    The generated table has columns ["n", "idx", "proposal", "a", "beta", "delta"]. Existing rows are skipped, so the method can be resumed from a partially generated pickle file.

    :param n_list: Iterable of system sizes. If None, use n=3,...,10.
    :param idx_min: First instance index, inclusive.
    :param idx_max: Last instance index, exclusive. If None, use 100.
    :param out_file: Output pickle file. If None, use SPECTRAL_GAP_FILE.
    :return: None.
    """
    
    def get_spectral_gap(Q_: np.ndarray) -> float:
        """
        Return the absolute spectral gap of the symmetric discriminant matrix sqrt(Q o Q^T).

        :param Q_: Column-stochastic transition matrix.
        :return: Absolute spectral gap 1 - max_{nontrivial j} |lambda_j|.
        """
        evals = np.linalg.eigvalsh(np.sqrt(np.maximum(Q_ * Q_.T, 0.0)))
        return float(1.0 - np.max(np.abs(evals[:-1])))

    BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    columns = ["n", "idx", "proposal", "a", "beta", "delta"]

    n_list = list(range(3, 11)) if n_list is None else list(n_list)
    idx_max = 100 if idx_max is None else int(idx_max)
    out_file = SPECTRAL_GAP_FILE if out_file is None else out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file) if out_file.exists() else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)

    for n in n_list:
        # Load each instance and the QEMC/Layden hyperparameters once per n.
        ising = [load_instances(n, idx) for idx in range(idx_min, idx_max)]
        params = [load_best_qemc_gamma_t(n, idx) for idx in range(idx_min, idx_max)]

        for local_idx, idx in enumerate(range(idx_min, idx_max)):
            print(n, idx)
            gamma, t = params[local_idx]
            rows = []

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        mask = (df["n"] == n) & (df["idx"] == idx) & (df["proposal"] == proposal) & (df["a"] == a) & (df["beta"] == beta)
                        if mask.any():
                            continue

                        Q = create_transition_matrix(proposal, ising[local_idx], beta, a, gamma, t)
                        delta = get_spectral_gap(Q)
                        rows.append({"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "delta": delta})
                        print(".", end="", flush=True)

            print("")
            if rows:
                df = pd.concat([df, pd.DataFrame(rows, columns=columns)], ignore_index=True)
                df.to_pickle(out_file)


def load_spectral_gap(proposal: str, a: int | float, in_file: Path = SPECTRAL_GAP_FILE) -> pd.DataFrame:
    """
    Load raw spectral-gap rows for a fixed proposal and acceptance parameter.

    :param proposal: Proposal name, one of PROPOSALS.
    :param a: Acceptance parameter, one of AS.
    :param in_file: Input pickle file.
    :return: DataFrame with columns ["n", "idx", "beta", "delta"].
    """
    assert proposal in PROPOSALS, f"proposal must be one of {PROPOSALS}"
    assert a in AS, f"a must be one of {AS}"

    df = pd.read_pickle(in_file)
    mask = (df["proposal"] == proposal) & (df["a"] == a)
    return df.loc[mask, ["n", "idx", "beta", "delta"]].reset_index(drop=True)


def get_spectral_gap_stats(
    proposal: str,
    a: int | float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
) -> pd.DataFrame:
    """
    Calculate spectral-gap statistics over instance index for each pair (n, beta).

    The returned table has columns ["n", "beta", "center", "spread", "count", "statistic"]. The exact meaning of center/spread is controlled by `statistic`.

    :param proposal: Proposal name, one of PROPOSALS.
    :param a: Acceptance parameter, one of AS.
    :param in_file: Input pickle file.
    :param statistic: Statistic rule handled by grouped_statistics.
    :return: Statistics table grouped by (n, beta).
    """
    assert proposal in PROPOSALS, f"proposal must be one of {PROPOSALS}"
    assert a in AS, f"a must be one of {AS}"

    df = load_spectral_gap(proposal=proposal, a=a, in_file=in_file)

    # Keep zero/non-positive gaps in the statistics stage to preserve the raw empirical summary; fits filter positivity later.
    return grouped_statistics(df=df, value_col="delta", group_cols=("n", "beta"), statistic=statistic, positive_only=False)


def get_spectral_gap_fit_by_n(
    proposal: str,
    a: int | float,
    fixed_beta: float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_min: int | None = None,
    n_max: int | None = None,
) -> tuple[float, float]:
    """
    Fit the fixed-beta scaling delta_beta(n) = A * exp(-b n).

    :param proposal: Proposal name, one of PROPOSALS.
    :param a: Acceptance parameter, one of AS.
    :param fixed_beta: Beta value at which the scaling in n is fitted.
    :param in_file: Input pickle file.
    :param statistic: Statistic used to build the grouped table before fitting.
    :param n_min: Optional minimum n included in the fit.
    :param n_max: Optional maximum n included in the fit.
    :return: Tuple (A, b) for delta_beta(n) = A * exp(-b n).
    """
    table = get_spectral_gap_stats(proposal=proposal, a=a, in_file=in_file, statistic=statistic)
    return fit_exponential_from_stats(table=table, beta=fixed_beta, n_min=n_min, n_max=n_max, sign=-1)


def get_spectral_gap_fit_by_beta(
    proposal: str,
    a: int | float,
    fixed_n: int,
    beta_max: float = 100.0,
    beta_step: float = 0.01,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
) -> np.ndarray:
    """
    Build a fixed-n interpolation grid for beta -> delta_n(beta).

    The interpolation is log-linear in the spectral gap: the method interpolates log(delta) on the empirical beta grid and then exponentiates. The returned array has shape (m, 2), with columns [beta, delta_n(beta)].

    :param proposal: Proposal name, one of PROPOSALS.
    :param a: Acceptance parameter, one of AS.
    :param fixed_n: System size at which beta dependence is interpolated.
    :param beta_max: Maximum beta in the returned grid.
    :param beta_step: Grid spacing for returned beta values.
    :param spectral_gap_file: Input spectral-gap pickle file.
    :param statistic: Statistic used to build the grouped table before interpolation.
    :return: Array with columns [beta, interpolated_delta].
    """
    beta_max = float(beta_max)
    beta_step = float(beta_step)

    if beta_max < 0.0:
        raise ValueError("beta_max must be non-negative.")
    if beta_step <= 0.0:
        raise ValueError("beta_step must be positive.")

    table = get_spectral_gap_stats(proposal=proposal, a=a, in_file=spectral_gap_file, statistic=statistic)
    table = table[
        (table["n"].astype(int) == int(fixed_n))
        & np.isfinite(table["beta"].astype(float))
        & np.isfinite(table["center"].astype(float))
        & (table["beta"].astype(float) > 0.0)
        & (table["center"].astype(float) > 0.0)
    ].sort_values("beta")

    if table.empty:
        raise ValueError(f"No positive spectral-gap data for n={fixed_n}, proposal={proposal}, a={a}.")

    beta_data = table["beta"].to_numpy(dtype=float)
    log_delta_data = np.log(table["center"].to_numpy(dtype=float))

    # Build the dense beta grid, including beta_max despite floating-point roundoff.
    beta_grid = np.arange(0.0, beta_max + 0.5 * beta_step, beta_step, dtype=float)
    beta_grid = beta_grid[beta_grid <= beta_max + 1e-12]

    if beta_grid.size == 0 or not np.isclose(beta_grid[-1], beta_max):
        beta_grid = np.append(beta_grid, beta_max)

    # At beta=0 we use delta=1 as the harmless infinite-temperature convention.
    delta_grid = np.ones_like(beta_grid, dtype=float)
    positive = beta_grid > 0.0

    # Outside the empirical beta support, clamp to the nearest empirical endpoint.
    beta_interp = np.clip(beta_grid[positive], beta_data[0], beta_data[-1])
    log_delta_interp = np.interp(beta_interp, beta_data, log_delta_data)
    delta_grid[positive] = np.exp(log_delta_interp)
    delta_grid = np.clip(delta_grid, np.finfo(float).tiny, 1.0)

    return np.column_stack([beta_grid, delta_grid])


def get_spectral_gap_fit_by_beta_single_value(grid, beta, beta_max=100.0, beta_step=0.01):
    """
    Return one interpolated spectral-gap value from a precomputed beta grid.

    The grid is assumed to have been generated by get_spectral_gap_fit_by_beta with beta values produced by np.arange(0.0, beta_max + 0.5 * beta_step, beta_step). The lookup rounds beta to the nearest grid index.

    :param grid: Array with columns [beta, delta_n(beta)].
    :param beta: Beta value to query.
    :param beta_max: Maximum beta used to generate the grid.
    :param beta_step: Beta spacing used to generate the grid.
    :return: Interpolated spectral gap delta_n(beta).
    """
    if beta <= 0.0:
        return 1.0

    # Round to the nearest grid point because the grid is uniform by construction.
    idx = int(round(min(float(beta), float(beta_max)) / float(beta_step)))
    idx = min(idx, len(grid) - 1)

    return float(grid[idx, 1])