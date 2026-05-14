from monaqa2.data.filename import SPECTRAL_GAP_FILE
from monaqa2.data.hyperparams import load_best_qemc_gamma_t
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.transition import create_transition_matrix
import numpy as np
import pandas as pd


def run_experiment_to_generate_spectral_gaps() -> None:

    def get_spectral_gap(Q_: np.ndarray) -> float:
        evals = np.linalg.eigvalsh(np.sqrt(np.maximum(Q_ * Q_.T, 0.0)))
        return float(1.0 - np.max(np.abs(evals[:-1])))

    PROPOSALS = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    AS = [np.inf, 1, 10]
    BETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    columns = ["n", "idx", "proposal", "a", "beta", "delta"]

    SPECTRAL_GAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(SPECTRAL_GAP_FILE) if SPECTRAL_GAP_FILE.exists() else pd.DataFrame(columns=columns)
    df = df[columns] if len(df) else df

    for n in range(3, 11):
        ising = [load_instances(n, idx) for idx in range(100)]
        params = [load_best_qemc_gamma_t(n, idx) for idx in range(100)]

        for idx in range(100):
            print(n, idx)
            mask = (df["n"] == n) & (df["idx"] == idx)
            if mask.any():
                continue

            gamma, t = params[idx]
            rows = []

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        Q = create_transition_matrix(proposal, ising[idx], beta, a, gamma, t)
                        delta = get_spectral_gap(Q)
                        rows.append({"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "delta": delta})
                        print(".", end="", flush=True)
            print("")
            df = pd.concat([df, pd.DataFrame(rows, columns=columns)], ignore_index=True)
            df.to_pickle(SPECTRAL_GAP_FILE)


def load_spectral_gap(proposal: str, a: int | float) -> pd.DataFrame:
    proposals = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    acceptances = [np.inf, 1, 10]

    assert proposal in proposals, f"proposal must be one of {proposals}"
    assert a in acceptances, f"a must be one of {acceptances}"

    df = pd.read_pickle(SPECTRAL_GAP_FILE)
    mask = (df["proposal"] == proposal) & (df["a"] == a)
    return df.loc[mask, ["n", "idx", "beta", "delta"]].reset_index(drop=True)
