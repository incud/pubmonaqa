
"""Spectral gaps for five Ising ensembles at beta=4.

Usage:
    python spectral_gaps_models.py generate
    python spectral_gaps_models.py analyze 0
    python spectral_gaps_models.py collide

Requires monaqa2 with create_transition_matrix(..., rescaled=False).
"""
import sys
sps.path.append("..")
from pathlib import Path

import h5py
import numpy as np

from monaqa2.data.instances import load_instances
from monaqa2.mcmc.model import IsingModel, RandomSherringtonKirkpatrickModel
from monaqa2.mcmc.transition import create_transition_matrix

BETA, SEED = 4.0, 2026
MOVES = ("local1", "uniform", "layden")
MODELS = (
    "normalized",
    "unnormalized",
    "unnormalized_sk",
    "new_sk",
    "deterministic_normalized",
)
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
            added = 0
            for idx in range(COUNT):
                missing = {
                    name for name in MODELS
                    if f"{name}/{n}/{idx}" not in f
                }
                if not missing:
                    continue

                arrays = {}

                if missing & {"normalized", "unnormalized", "unnormalized_sk"}:
                    original = load_instances(n, idx)

                    if "normalized" in missing:
                        arrays["normalized"] = coefficients(
                            original, rescaled=True
                        )
                    if "unnormalized" in missing:
                        arrays["unnormalized"] = coefficients(original)
                    if "unnormalized_sk" in missing:
                        arrays["unnormalized_sk"] = coefficients(
                            original.to_sk(local_fields=True)
                        )

                if "new_sk" in missing:
                    new_sk = RandomSherringtonKirkpatrickModel(
                        n, local_fields=True, seed=SEED + 1000 * n + idx
                    )
                    arrays["new_sk"] = coefficients(new_sk)

                if "deterministic_normalized" in missing:
                    raw = arrays.get("unnormalized")
                    if raw is None:
                        raw = f[f"unnormalized/{n}/{idx}"][:]
                    arrays["deterministic_normalized"] = (
                        raw * np.sqrt(2.0 / (n + 1))
                    )

                for name, values in arrays.items():
                    f.require_group(f"{name}/{n}").create_dataset(
                        str(idx), data=values
                    )
                    added += 1

            print(f"n={n}: added {added} instance datasets", flush=True)

    print(f"Saved {INSTANCES}")


def get_spectral_gap(Q_):
    """Absolute spectral gap of the symmetric discriminant matrix."""
    evals = np.linalg.eigvalsh(
        np.sqrt(np.maximum(Q_ * Q_.T, 0.0))
    )
    return float(1.0 - np.max(np.abs(evals[:-1])))


def analyze(idx):
    if not 0 <= idx < COUNT:
        raise ValueError(f"idx must be between 0 and {COUNT - 1}")

    result = OUT / f"spectral_gaps_{idx}.hdf5"
    added = 0

    with h5py.File(INSTANCES, "r") as source, \
         h5py.File(result, "a") as target:

        if "beta" in target.attrs and target.attrs["beta"] != BETA:
            raise ValueError(f"Cache has different beta: {result}")

        target.attrs["beta"] = BETA
        target.attrs["moves"] = ",".join(MOVES)

        # Finish all models and moves at each n before proceeding
        for n in NS:
            for name in MODELS:
                missing = [
                    move for move in MOVES
                    if f"{move}/{name}/{n}" not in target
                ]
                if not missing:
                    continue

                coeffs = source[f"{name}/{n}/{idx}"][:]
                model = IsingModel.from_coefficients(n, coeffs)

                for move in missing:
                    Q = create_transition_matrix(
                        move, model, BETA, a=np.inf, rescaled=False
                    )
                    gap = get_spectral_gap(Q)

                    target.require_group(f"{move}/{name}").create_dataset(
                        str(n), data=gap
                    )
                    target.flush()
                    added += 1

                    print(
                        f"idx={idx:02d} n={n} {name} {move}: {gap:.6g}",
                        flush=True,
                    )

    print(f"Saved {result} ({added} new gaps)")


def collide():
    files = [
        OUT / f"spectral_gaps_{idx}.hdf5"
        for idx in range(COUNT)
    ]
    missing = [f.name for f in files if not f.is_file()]

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} files, e.g. {missing[:5]}"
        )

    merged = OUT / "spectral_gaps_models_merged.hdf5"
    added = 0

    with h5py.File(merged, "a") as out:
        if "beta" in out.attrs and out.attrs["beta"] != BETA:
            raise ValueError(f"Cache has different beta: {merged}")

        out.attrs["beta"] = BETA
        out.attrs["moves"] = ",".join(MOVES)

        for idx, path in enumerate(files):
            with h5py.File(path, "r") as f:
                if f.attrs.get("beta") != BETA or not set(MOVES).issubset(
                    str(f.attrs.get("moves", "")).split(",")
                ):
                    raise ValueError(f"Inconsistent settings: {path}")

                for move in MOVES:
                    for name in MODELS:
                        for n in NS:
                            dest = f"{move}/{name}/{n}/{idx}"

                            # Preserve previously merged results
                            if dest in out:
                                continue

                            source_path = f"{move}/{name}/{n}"
                            if source_path not in f:
                                raise ValueError(
                                    f"Missing {source_path} in {path}; "
                                    f"rerun analyze {idx}"
                                )

                            out.require_group(
                                f"{move}/{name}/{n}"
                            ).create_dataset(
                                str(idx), data=f[source_path][()]
                            )
                            added += 1

            print(f"Merged idx={idx:02d}", flush=True)

    print(f"Saved {merged} ({added} new gaps)")


if __name__ == "__main__":
    if sys.argv[1:] == ["generate"]:
        generate()
    elif len(sys.argv) == 3 and sys.argv[1] == "analyze":
        analyze(int(sys.argv[2]))
    elif sys.argv[1:] == ["collide"]:
        collide()
    else:
        sys.exit(
            "Usage: python spectral_gaps_models.py "
            "generate | analyze <idx> | collide"
        )
