"""Run, for example: python3 launch_randomized_average_behaviour.py 7 0 24.

Both instance bounds are inclusive. Place this file in the repository root.
JSON tables use [n - NS.start][instance], with null for unfinished entries.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from monaqa2.data.filename import CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.proposal import create_proposal_matrix_quantum_exact, create_proposal_matrix_quantum_layden
from monaqa2.mcmc.transition import create_transition_matrix

NS = range(3, 11)
IDXS = range(100)
KS = range(1, 6)
BETA = 4.0
TIME_LIMS = (2.0, 20.0)
GAMMA_LIMS = (0.25, 0.60)
APPROACHES = (
    ["T_exact"]
    + [f"T_grid_k{k}" for k in KS]
    + ["P_exact"]
    + [f"P_grid_k{k}" for k in KS]
)


def get_spectral_gap(Q):
    """Calculate the absolute spectral gap of a reversible stochastic matrix.

    :param Q: Proposal T or transition P, in column-stochastic convention.
    :return: One minus the largest nonstationary eigenvalue magnitude.
    """
    evals = np.linalg.eigvalsh(np.sqrt(np.maximum(Q * Q.T, 0.0)))
    return float(1.0 - np.max(np.abs(evals[:-1])))


def create_averaged_proposal(ising, k=None):
    """Average the proposal matrices over the selected parameter distribution.

    :param ising: Ising instance with rescaled h and J.
    :param k: Grid level, or None for the Layden reference.
    :return: Averaged proposal matrix T.
    """
    if k is None:
        # Continuous time average and 20 gamma midpoint values.
        return create_proposal_matrix_quantum_layden(
            ising.h_rescaled, ising.J_rescaled,
            gamma_lims=GAMMA_LIMS, gamma_steps=20, time_lims=TIME_LIMS,
        )

    # Midpoint grids: 2^k times and min(2^k, 20) gamma values.
    nt, ng = 2**k, min(2**k, 20)
    t0, t1 = TIME_LIMS
    g0, g1 = GAMMA_LIMS
    times = t0 + (np.arange(nt) + 0.5) * (t1 - t0) / nt
    gammas = g0 + (np.arange(ng) + 0.5) * (g1 - g0) / ng
    T = np.zeros((2**ising.n, 2**ising.n))
    for t in times:
        for gamma in gammas:
            T += create_proposal_matrix_quantum_exact(
                ising.h_rescaled, ising.J_rescaled, gamma, t
            )
    return T / (nt * ng)


def main():
    """Run the requested instance interval and resume completed calculations.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, choices=NS, help="Number of spins")
    parser.add_argument("min_inst", type=int, help="First instance, included")
    parser.add_argument("max_inst", type=int, help="Last instance, included")
    args = parser.parse_args()
    if not 0 <= args.min_inst <= args.max_inst < len(IDXS):
        parser.error("Use 0 <= min_inst <= max_inst < 100.")

    # Each process writes its own file, retaining the original extension.
    main_file = Path(CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE)
    average_file = main_file.with_stem(
        f"{main_file.stem}_n{args.N}_inst{args.min_inst}_to_{args.max_inst}"
    )
    average_file.parent.mkdir(parents=True, exist_ok=True)

    if main_file.exists():
        with main_file.open() as f:
            main_spectral_gaps = json.load(f)
    else:
        main_spectral_gaps = {}
    if average_file.exists():
        with average_file.open() as f:
            average_spectral_gaps = json.load(f)
    else:
        average_spectral_gaps = {}

    # Match the original [n][instance] tables, including all T and P approaches.
    for spectral_gaps in (main_spectral_gaps, average_spectral_gaps):
        for approach in APPROACHES:
            values = spectral_gaps.setdefault(
                approach, [[None] * len(IDXS) for _ in NS]
            )
            # Preserve P gaps saved by the previous nested-dictionary version.
            if isinstance(values, dict):
                table = [[None] * len(IDXS) for _ in NS]
                for n_key, instances in values.items():
                    for instance_key, gap in instances.items():
                        table[int(n_key.removeprefix("n=")) - NS.start][
                            int(instance_key.removeprefix("inst"))
                        ] = gap
                spectral_gaps[approach] = table
    print(f"Saving averaged-matrix T and P gaps to {average_file}", flush=True)

    i = args.N - NS.start
    for k in [None, *KS]:
        name = "exact" if k is None else f"grid_k{k}"
        T_key, P_key = f"T_{name}", f"P_{name}"
        for idx in range(args.min_inst, args.max_inst + 1):
            T_done = (
                main_spectral_gaps[T_key][i][idx] is not None
                or average_spectral_gaps[T_key][i][idx] is not None
            )
            P_done = (
                main_spectral_gaps[P_key][i][idx] is not None
                or average_spectral_gaps[P_key][i][idx] is not None
            )
            if T_done and P_done:
                continue
            print(f"n={args.N}, instance={idx}, approach={name}", flush=True)
            ising = load_instances(args.N, idx)
            T = create_averaged_proposal(ising, k)
            # Calculate the gap of the averaged matrix, not the average of gaps.
            if not T_done:
                average_spectral_gaps[T_key][i][idx] = get_spectral_gap(T)
            if not P_done:
                P = create_transition_matrix(T, ising, BETA)
                average_spectral_gaps[P_key][i][idx] = get_spectral_gap(P)
            with average_file.open("w") as f:
                json.dump(average_spectral_gaps, f)


if __name__ == "__main__":
    main()
