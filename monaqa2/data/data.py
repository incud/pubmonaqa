from collections import defaultdict
from pathlib import Path
import h5py
import json
from monaqa2.mcmc.model import IsingModel
from monaqa2.mcmc.transition import create_transition_matrix
import numpy as np
import pandas as pd


MONAQA2_PARENT = Path(__file__).resolve().parents[2]
ISING_INSTANCES_FILE = MONAQA2_PARENT / "data/ising_instances.hdf5"

BEST_HYPERPARAMS_JSON_FILE_LIST = [
    (3, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n3.json"),
    (4, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n4.json"),
    (5, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n5.json"),
    (5, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n5_fine.json"),
    (6, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n6.json"),
    (6, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n6_fine.json"),
    (7, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n7.json"),
    (7, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n7_fine.json"),
    (8, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n8.json"),
    (8, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n8_fine.json"),
    (9, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n9.json"),
    (9, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n9_fine.json"),
    (10, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n10.json"),
    (10, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n10_fine.json")
]
BEST_HYPERPARAMS_QEMC_FILE = MONAQA2_PARENT / "data/best_hyperparams_qemc.hdf5"

SPECTRAL_GAP_FILE = MONAQA2_PARENT / "data/spectral_gap.pkl"


def load_instances(n: int, idx: int) -> IsingModel:
    n = int(n)
    idx = int(idx)
    assert 3 <= n <= 10, f"The preloaded instances have between 3 and 10 spins (you asked for {n=})"
    assert 0 <= idx <= 99, f"The preloaded instances have index between 0 and 99 included (you asked for {idx=})"
    file = h5py.File(ISING_INSTANCES_FILE)
    coeffs = file['coefficients'][f'{n}'][:, idx]
    return IsingModel.from_coefficients(n=n, coefficients=coeffs)


def export_best_gamma_t_h5(n_json_list: list[tuple[int, Path]] = BEST_HYPERPARAMS_JSON_FILE_LIST, out_h5_path: Path = BEST_HYPERPARAMS_QEMC_FILE) -> Path:
    """
    Writes an HDF5 file with datasets /gamma/{n} and /t/{n},
    each of shape (100,), for n = 3, ..., 10. These contain the
    hyperparameters maximizing the spectral gap of the QEMC move
    across all JSON files provided for each n.
    """
    json_by_n = defaultdict(list)
    for n, path in n_json_list:
        json_by_n[int(n)].append(Path(path))

    with h5py.File(out_h5_path, "w") as h5:
        gamma_h5 = h5.create_group("gamma")
        t_h5 = h5.create_group("t")

        for n in range(3, 11):
            if n not in json_by_n:
                raise ValueError(f"Missing JSON files for n={n}.")

            best_delta = np.full(100, -np.inf)
            best_gamma = np.empty(100, dtype=float)
            best_t = np.empty(100, dtype=float)

            for path in json_by_n[n]:
                with open(path, "r") as f:
                    js = json.load(f)

                gamma = np.asarray(js["gamma_range"], dtype=float)
                tvec = np.asarray(js["time_range"], dtype=float)
                delta = np.asarray(js["delta"], dtype=float)

                # double check: validate the file format is ok
                assert int(js["n"]) == n
                assert int(js["num_random_models"]) == 100
                assert delta.shape == (gamma.size, tvec.size, 100)

                G, T, M = delta.shape
                flat_delta = delta.reshape(G * T, M)

                # best configuration within this file, independently for each instance
                idx = flat_delta.argmax(axis=0)
                file_best_delta = flat_delta[idx, np.arange(M)]
                file_best_gamma = gamma[idx // T]
                file_best_t = tvec[idx % T]

                # keep the best configuration across all files for this n
                improve = file_best_delta > best_delta
                best_delta[improve] = file_best_delta[improve]
                best_gamma[improve] = file_best_gamma[improve]
                best_t[improve] = file_best_t[improve]

            gamma_h5.create_dataset(str(n), data=best_gamma)
            t_h5.create_dataset(str(n), data=best_t)

    return Path(out_h5_path)


def load_best_qemc_gamma_t(n: int, idx: int) -> tuple[float, float]:
    n = int(n)
    idx = int(idx)
    assert 3 <= n <= 10, f"The preloaded instances have between 3 and 10 spins (you asked for {n=})"
    assert 0 <= idx <= 99, f"The preloaded instances have index between 0 and 99 included (you asked for {idx=})"
    file = h5py.File(BEST_HYPERPARAMS_QEMC_FILE)
    gamma = float(file['gamma'][f'{n}'][idx])
    t = float(file['t'][f'{n}'][idx])
    return (gamma, t)


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

            h, J = ising[idx].h_rescaled, ising[idx].J_rescaled
            gamma, t = params[idx]
            rows = []

            for proposal in PROPOSALS:
                for a in AS:
                    for beta in BETAS:
                        Q = create_transition_matrix(proposal, h, J, beta, a, gamma, t)
                        delta = get_spectral_gap(Q)
                        rows.append({"n": n, "idx": idx, "proposal": proposal, "a": a, "beta": beta, "delta": delta})
                        print(".", end="", flush=True)
            print("")
            df = pd.concat([df, pd.DataFrame(rows, columns=columns)], ignore_index=True)
            df.to_pickle(SPECTRAL_GAP_FILE)


def load_spectral_gap(proposal: str, a: float) -> pd.DataFrame:
    proposals = ["uniform", "local1", "local2", "local3", "qemc", "layden"]
    acceptances = [np.inf, 1, 10]

    assert proposal in proposals, f"proposal must be one of {proposals}"
    assert a in acceptances, f"a must be one of {acceptances}"

    df = pd.read_pickle(SPECTRAL_GAP_FILE)
    mask = (df["proposal"] == proposal) & (df["a"] == a)
    return df.loc[mask, ["n", "idx", "beta", "delta"]].reset_index(drop=True)