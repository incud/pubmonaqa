from collections import defaultdict
from pathlib import Path
import h5py
import json
import numpy as np
from monaqa2.data.filename import BEST_HYPERPARAMS_JSON_FILE_LIST, BEST_HYPERPARAMS_QEMC_FILE


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
