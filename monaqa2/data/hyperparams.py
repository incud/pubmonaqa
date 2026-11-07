from collections import defaultdict
from pathlib import Path
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
    import h5py

    json_by_n = defaultdict(list)
    for n, path in n_json_list:
        json_by_n[int(n)].append(Path(path))

    if any(n not in json_by_n for n in range(3, 11)):
        raise ValueError("JSON files are required for every n=3,...,10.")
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

                if (int(js["n"]) != n or int(js["num_random_models"]) != 100
                        or gamma.ndim != 1 or tvec.ndim != 1 or not gamma.size or not tvec.size
                        or delta.shape != (gamma.size, tvec.size, 100)
                        or not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(tvec))
                        or not np.all(np.isfinite(delta)) or np.any((delta < 0) | (delta > 1))):
                    raise ValueError(f"Invalid parameter grid in {path}.")

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
    """Load one best parameter pair and close its HDF5 file immediately."""
    import h5py

    n = int(n)
    idx = int(idx)
    if not 3 <= n <= 10 or not 0 <= idx < 100:
        raise ValueError("Stored parameters require 3 <= n <= 10 and 0 <= idx < 100.")
    with h5py.File(BEST_HYPERPARAMS_QEMC_FILE, "r") as file:
        gamma = float(file['gamma'][f'{n}'][idx])
        t = float(file['t'][f'{n}'][idx])
    return (gamma, t)
