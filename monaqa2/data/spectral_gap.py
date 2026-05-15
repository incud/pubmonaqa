from monaqa2.data.filename import SPECTRAL_GAP_FILE
from monaqa2.data.hyperparams import load_best_qemc_gamma_t
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.transition import create_transition_matrix
import numpy as np
import pandas as pd
from pathlib import Path


def run_experiment_to_generate_spectral_gaps(n_list=None, idx_min=0, idx_max=None, out_file: Path = None) -> None:
    
    def get_spectral_gap(Q_: np.ndarray) -> float:
        evals = np.linalg.eigvalsh(np.sqrt(np.maximum(Q_ * Q_.T, 0.0)))
        return float(1.0 - np.max(np.abs(evals[:-1])))

    PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    AS = [np.inf, 1, 10]
    BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    columns = ["n", "idx", "proposal", "a", "beta", "delta"]

    n_list = list(range(3, 11)) if n_list is None else list(n_list)
    idx_max = 100 if idx_max is None else int(idx_max)
    out_file = SPECTRAL_GAP_FILE if out_file is None else out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file) if out_file.exists() else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)

    for n in n_list:
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
    proposals = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    acceptances = [np.inf, 1, 10]

    assert proposal in proposals, f"proposal must be one of {proposals}"
    assert a in acceptances, f"a must be one of {acceptances}"

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
    Calculate statistics over idx for each (n, beta).
    """
    proposals = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    acceptances = [np.inf, 1, 10]
    statistics = ["mean+std", "mean+std-tail", "median+mad"]

    assert proposal in proposals, f"proposal must be one of {proposals}"
    assert a in acceptances, f"a must be one of {acceptances}"
    assert statistic in statistics, f"statistic must be one of {statistics}"

    df = load_spectral_gap(proposal=proposal, a=a, in_file=in_file)
    rows = []

    for (n, beta), group in df.groupby(["n", "beta"], sort=True):
        x = group["delta"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        if x.size == 0:
            center, spread, count = np.nan, np.nan, 0

        elif statistic == "mean+std":
            center, spread, count = np.mean(x), np.std(x), x.size

        elif statistic == "mean+std-tail":
            q1, q3 = np.percentile(x, [25, 75])
            x = x[(x >= q1) & (x <= q3)]
            center, spread, count = (
                (np.mean(x), np.std(x), x.size) if x.size else (np.nan, np.nan, 0)
            )

        elif statistic == "median+mad":
            center = np.median(x)
            spread = np.median(np.abs(x - center))
            count = x.size

        rows.append(
            {
                "n": int(n),
                "beta": float(beta),
                "center": float(center),
                "spread": float(spread),
                "count": int(count),
                "statistic": statistic,
            }
        )

    return pd.DataFrame(rows).sort_values(["n", "beta"]).reset_index(drop=True)



def get_spectral_gap_fit(
    proposal: str,
    a: int | float,
    beta: float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_min: int | None = None,
    n_max: int | None = None,
) -> tuple[float, float]:
    """
    Fit delta(n) = A * exp(-b n) at fixed beta, optionally using only n_min <= n <= n_max.
    """
    table = get_spectral_gap_stats(proposal, a, in_file, statistic)
    table = table[np.isclose(table["beta"].astype(float), float(beta))]

    if n_min is not None:
        table = table[table["n"].astype(int) >= n_min]

    if n_max is not None:
        table = table[table["n"].astype(int) <= n_max]

    if table.empty:
        raise ValueError(f"No data found for beta={beta} in n range [{n_min}, {n_max}].")

    n = table["n"].to_numpy(dtype=float)
    y = table["center"].to_numpy(dtype=float)

    mask = np.isfinite(n) & np.isfinite(y) & (y > 0.0)
    n, y = n[mask], y[mask]

    if len(y) < 2:
        raise ValueError("Need at least two positive points to fit.")

    slope, log_A = np.polyfit(n, np.log(y), deg=1)

    return float(np.exp(log_A)), float(-slope)