import json
import matplotlib.pyplot as plt
import numpy as np
from monaqa2.data.filename import CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.proposal import create_proposal_matrix_quantum_exact, create_proposal_matrix_quantum_layden
from monaqa2.mcmc.transition import create_transition_matrix
from monaqa2.mcmc.model import IsingModel

def get_spectral_gap(Q_: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(np.sqrt(np.maximum(Q_ * Q_.T, 0.0)))
    return float(1.0 - np.max(np.abs(evals[:-1])))


NS = range(3, 10+1)
IDXS = range(100)
KS = range(1, 5+1)

BETA = 4.0
TIME_LIMS = (2.0, 20.0)
GAMMA_LIMS = (0.25, 0.60)

APPROACHES = ["T_exact"] + [f"T_grid_k{k}" for k in KS] + ["P_exact"] + [f"P_grid_k{k}" for k in KS]

# Each approach contains a [n][instance] table of spectral gaps.
if CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE.exists():
    with open(CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE) as f:
        spectral_gaps = json.load(f)
else:
    spectral_gaps = {}

for approach in APPROACHES:
    spectral_gaps.setdefault(approach, [[None] * len(IDXS) for _ in NS])


# Exact randomized proposal: continuous time average and 20 gamma points.
for n in NS:
    print(f"\nn={n}: ", end="")
    for idx in IDXS:
        i = n - NS.start

        if spectral_gaps["T_exact"][i][idx] is not None and spectral_gaps["P_exact"][i][idx] is not None:
            continue

        print(f".", end="", flush=True)

        ising: IsingModel = load_instances(n, idx)
        T = create_proposal_matrix_quantum_layden(ising.h_rescaled, ising.J_rescaled, gamma_lims=GAMMA_LIMS, gamma_steps=20, time_lims=TIME_LIMS)

        spectral_gaps["T_exact"][i][idx] = get_spectral_gap(T)
        spectral_gaps["P_exact"][i][idx] = get_spectral_gap(create_transition_matrix(T, ising, BETA))

        # Save after every instance so interrupted calculations can resume.
        with open(CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE, "w") as f:
            json.dump(spectral_gaps, f)


# Approximated randomized proposal: discrete grids for both t and gamma.
for k in KS:
    nt = 2**k
    ng = min(2**k, 20)

    t0, t1 = TIME_LIMS
    g0, g1 = GAMMA_LIMS
    times = t0 + (np.arange(nt) + 0.5) * (t1 - t0) / nt
    gammas = g0 + (np.arange(ng) + 0.5) * (g1 - g0) / ng

    for n in NS:
        print(f"\nk={k} n={n}: ", end="")
        for idx in IDXS:
            i = n - NS.start
            T_key = f"T_grid_k{k}"
            P_key = f"P_grid_k{k}"

            if spectral_gaps[T_key][i][idx] is not None and spectral_gaps[P_key][i][idx] is not None:
                continue

            print(f".", end="", flush=True)

            ising: IsingModel = load_instances(n, idx)
            T = np.zeros((2**n, 2**n))

            # Average the proposal over the finite (t, gamma) grid.
            for t in times:
                for gamma in gammas:
                    T += create_proposal_matrix_quantum_exact(ising.h_rescaled, ising.J_rescaled, gamma, t)

            T /= nt * ng

            spectral_gaps[T_key][i][idx] = get_spectral_gap(T)
            spectral_gaps[P_key][i][idx] = get_spectral_gap(create_transition_matrix(T, ising, BETA))

            # Save after every instance so interrupted calculations can resume.
            with open(CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE, "w") as f:
                json.dump(spectral_gaps, f)
