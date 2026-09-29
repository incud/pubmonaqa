"""Run, for example: python3 launch_randomized_grid_search.py 7 0 24.

Both instance bounds are inclusive. Place this file in the repository root.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from monaqa2.data.filename import GRIDSEARCH_RANDOMIZED_HYPERPARAMS_FILE
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.proposal import create_proposal_matrix_quantum_exact
from monaqa2.mcmc.transition import create_transition_matrix

K = 3
BETA = 4.0
TIME_LIMS = (2.0, 20.0)
GAMMA_LIMS = (0.25, 0.60)

# The same midpoint grid used to calculate the averaged proposal.
TIMES, GAMMAS = [
    lo + (np.arange(2**K) + 0.5) * (hi - lo) / 2**K
    for lo, hi in (TIME_LIMS, GAMMA_LIMS)
]


def get_spectral_gap(P):
    """Calculate the absolute spectral gap of a reversible transition matrix.

    :param P: Column-stochastic transition matrix.
    :return: One minus the largest nonstationary eigenvalue magnitude.
    """
    evals = np.linalg.eigvalsh(np.sqrt(np.maximum(P * P.T, 0.0)))
    return float(1.0 - np.max(np.abs(evals[:-1])))


def grid_search_spectral_gaps(n, idx):
    """Calculate only P gaps for one instance over the 8x8 parameter grid.

    :param n: Number of spins.
    :param idx: Instance index.
    :return: Gap table indexed by [time][gamma], ready for JSON.
    """
    ising = load_instances(n, idx)
    gaps = np.empty((len(TIMES), len(GAMMAS)))
    for a, t in enumerate(TIMES):
        for b, gamma in enumerate(GAMMAS):
            T = create_proposal_matrix_quantum_exact(
                ising.h_rescaled, ising.J_rescaled, gamma, t
            )
            P = create_transition_matrix(T, ising, BETA)
            gaps[a, b] = get_spectral_gap(P)
    return gaps.tolist()


def main():
    """Run the requested instance interval and save after each instance.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, help="Number of spins")
    parser.add_argument("min_inst", type=int, help="First instance, included")
    parser.add_argument("max_inst", type=int, help="Last instance, included")
    args = parser.parse_args()
    if args.N < 1 or not 0 <= args.min_inst <= args.max_inst:
        parser.error("Use N >= 1 and 0 <= min_inst <= max_inst.")

    # Separate files prevent different Slurm tasks from overwriting each other.
    grid_file = Path(GRIDSEARCH_RANDOMIZED_HYPERPARAMS_FILE)
    grid_file = grid_file.with_stem(
        f"{grid_file.stem}_k{K}_n{args.N}_inst{args.min_inst}_to_{args.max_inst}"
    )
    grid_file.parent.mkdir(parents=True, exist_ok=True)
    if grid_file.exists():
        with grid_file.open() as f:
            grid_spectral_gaps = json.load(f)
    else:
        grid_spectral_gaps = {}
    instances = grid_spectral_gaps.setdefault(f"k={K}", {}).setdefault(f"n={args.N}", {})
    print(f"Saving P gaps to {grid_file}", flush=True)

    for idx in range(args.min_inst, args.max_inst + 1):
        key = f"inst{idx:03d}"
        if instances.get(key) is not None:
            continue
        print(f"n={args.N}, instance={idx}", flush=True)
        instances[key] = grid_search_spectral_gaps(args.N, idx)

        # Replace the checkpoint only after the complete JSON has been written.
        temporary = grid_file.with_suffix(grid_file.suffix + ".tmp")
        with temporary.open("w") as f:
            json.dump(grid_spectral_gaps, f)
        temporary.replace(grid_file)


if __name__ == "__main__":
    main()
