
"""Compare four Ising ensembles at beta=4 using three MCMC proposals.

    python spectral_gaps_models.py generate
    python spectral_gaps_models.py analyze 0
    python spectral_gaps_models.py collide

Requires monaqa2 with create_transition_matrix(..., rescaled=False).
"""
import sys
sys.path.append("..")

from pathlib import Path

import h5py
import numpy as np

from monaqa2.data.instances import load_instances
from monaqa2.mcmc.model import IsingModel, RandomSherringtonKirkpatrickModel
from monaqa2.mcmc.transition import create_transition_matrix

BETA, SEED = 4.0, 2026
MOVES = ("local1", "uniform", "layden")
MODELS = ("normalized", "unnormalized", "unnormalized_sk", "new_sk")
NS, COUNT = range(3, 11), 100
OUT = Path(__file__).resolve().parent / "sk_variants"
INSTANCES = OUT / "ising_models.hdf5"


def coefficients(model, rescaled=False):
    h = model.h_rescaled if rescaled else model.h
    J = model.J_rescaled if rescaled else model.J
    return np.r_[h, J[np.triu_indices(model.n, 1)]]


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(INSTANCES, "a") as f:
        for n in NS:
            for idx in range(COUNT):
                if all(f"{name}/{n}/{idx}" in f for name in MODELS):
                    continue
                original = load_instances(n, idx)
                new_sk = RandomSherringtonKirkpatrickModel(
                    n, local_fields=True, seed=SEED + 1000 * n + idx)
                arrays = {
                    "normalized": coefficients(original, rescaled=True),
                    "unnormalized": coefficients(original),
                    "unnormalized_sk": coefficients(original.to_sk(local_fields=True)),
                    "new_sk": coefficients(new_sk),
                }
                for name, values in arrays.items():
                    group = f.require_group(f"{name}/{n}")
                    if str(idx) not in group:
                        group.create_dataset(str(idx), data=values)
            print(f"Generated n={n}", flush=True)
    print(f"Saved {INSTANCES}")


def get_spectral_gap(Q_):
    """Absolute spectral gap of the symmetric discriminant matrix."""
    evals = np.linalg.eigvalsh(np.sqrt(np.maximum(Q_ * Q_.T, 0.0)))
    return float(1.0 - np.max(np.abs(evals[:-1])))


def analyze(idx):
    if not 0 <= idx < COUNT:
        raise ValueError(f"idx must be between 0 and {COUNT - 1}")
    result = OUT / f"spectral_gaps_{idx}.hdf5"
    with h5py.File(INSTANCES, "r") as source, h5py.File(result, "a") as target:
        if "beta" in target.attrs and target.attrs["beta"] != BETA:
            raise ValueError(f"Cache has different beta: {result}")
        target.attrs["beta"] = BETA
        target.attrs["moves"] = ",".join(MOVES)  # Reuse caches that included local2
        for n in NS:  # Complete all models and moves before increasing n
            for name in MODELS:
                missing = [move for move in MOVES if f"{move}/{name}/{n}" not in target]
                if not missing:
                    continue
                coeffs = source[f"{name}/{n}/{idx}"][:]
                model = IsingModel.from_coefficients(n, coeffs)
                for move in missing:
                    Q = create_transition_matrix(move, model, BETA, a=np.inf, rescaled=False)
                    gap = get_spectral_gap(Q)
                    target.require_group(f"{move}/{name}").create_dataset(str(n), data=gap)
                    target.flush()
                    print(f"idx={idx:02d} n={n} {name} {move}: {gap:.6g}", flush=True)
    print(f"Saved {result}")


def collide():
    files = [OUT / f"spectral_gaps_{idx}.hdf5" for idx in range(COUNT)]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files, e.g. {missing[:5]}")
    merged = OUT / "spectral_gaps_models_merged.hdf5"
    temporary = OUT / "spectral_gaps_models_merged.tmp.hdf5"
    with h5py.File(temporary, "w") as out:
        out.attrs.update(beta=BETA, moves=",".join(MOVES))
        for idx, path in enumerate(files):
            with h5py.File(path, "r") as f:
                if f.attrs.get("beta") != BETA or not set(MOVES).issubset(
                    str(f.attrs.get("moves", "")).split(",")
                ):
                    raise ValueError(f"Inconsistent settings: {path}")
                for move in MOVES:
                    for name in MODELS:
                        for n in NS:
                            val = f[f"{move}/{name}/{n}"][()]
                            out.require_group(f"{move}/{name}/{n}").create_dataset(str(idx), data=val)
            print(f"Merged idx={idx:02d}", flush=True)
    temporary.replace(merged)  # Publish only after all 100 files merged successfully
    print(f"Saved {merged}")


if __name__ == "__main__":
    if sys.argv[1:] == ["generate"]:
        generate()
    elif len(sys.argv) == 3 and sys.argv[1] == "analyze":
        analyze(int(sys.argv[2]))
    elif sys.argv[1:] == ["collide"]:
        collide()
    else:
        sys.exit("Usage: python spectral_gaps_models.py generate | analyze <idx> | collide")
