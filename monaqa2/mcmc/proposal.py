from monaqa2.mcmc.utils_hamiltonian import hamming, energies, get_cached_mixing_hamiltonian, infer_n_from_h_J, int_to_bin
from monaqa2.mcmc.validation import check_double_stochasticity
import numpy as np
import scipy
import scipy.linalg as la


# ========================================================================
# ====== PROPOSAL MATRIX METHODS =========================================
# ========================================================================

def create_proposal_matrix_uniform(n: int) -> np.ndarray:
    """
    Uniform proposal matrix.

    Every configuration is proposed with equal probability.
    """
    if n < 1:
        raise ValueError("n must be positive.")

    d = 2**n
    P = np.ones((d, d), dtype=float) / d

    check_double_stochasticity(P)
    return P


def create_proposal_matrix_local(n: int, k: int = 1) -> np.ndarray:
    """
    Exactly-k local spin-flip proposal matrix.

    Only configurations at Hamming distance exactly k can be proposed.
    """
    if n < 1:
        raise ValueError("n must be positive.")
    if k > n or k <= 0:
        raise ValueError(f"k must satisfy 1 <= k <= n, got {k=} and {n=}.")

    d = 2**n
    prob_propose = 1.0 / scipy.special.comb(n, k, exact=False)

    P = np.zeros((d, d), dtype=float)

    for i in range(d):
        s_i = int_to_bin(i, n)

        for j in range(d):
            if i == j:
                continue

            s_j = int_to_bin(j, n)

            if hamming(s_i, s_j) != k:
                continue

            P[j, i] = prob_propose

    return P


def create_proposal_matrix_local_up_to_radius(n: int, radius: int = 2) -> np.ndarray:
    """
    Radius-r local spin-flip proposal matrix.

    Configurations at Hamming distance 1, ..., radius can be proposed.
    """
    if n < 2:
        raise ValueError("n must be positive and >= 2")
    if radius > n or radius <= 0:
        raise ValueError(f"radius must satisfy 1 <= radius <= n, got {radius=} and {n=}.")

    d = 2**n
    prob_propose = 1.0 / sum(
        scipy.special.comb(n, r, exact=False)
        for r in range(1, radius + 1)
    )

    P = np.zeros((d, d), dtype=float)

    for i in range(d):
        s_i = int_to_bin(i, n)

        for j in range(d):
            if i == j:
                continue

            s_j = int_to_bin(j, n)
            dist = hamming(s_i, s_j)

            if not (1 <= dist <= radius):
                continue

            P[j, i] = prob_propose

    return P


def create_proposal_matrix_quantum_exact(
    h: np.ndarray,
    J: np.ndarray,
    gamma: float = 0.7,
    t: float = 1.0,
) -> np.ndarray:
    """
    Quantum proposal matrix

        P_{y x} = |<y| exp(-i H t) |x>|^2,

    where

        H = (1 - gamma) diag(E) + gamma H_mixer.

    The number of spins n is inferred from h and J.
    """
    n = infer_n_from_h_J(h, J)

    H_ising = np.diag(energies(h, J))
    H_mixer = get_cached_mixing_hamiltonian(n)

    H = (1.0 - gamma) * H_ising + gamma * H_mixer
    U = la.expm(-1j * H * t)

    P = np.abs(U) ** 2

    check_double_stochasticity(P)
    return P


def create_proposal_matrix_quantum_time_avg(
    H: np.ndarray,
    time_lims: tuple[float, float] = (2.0, 20.0),
) -> np.ndarray:
    r"""
    Given a real-symmetric Hamiltonian H, return

        P = E_{t ~ Unif[t0, tf]}[ |<y| exp(-i H t) |x>|^2 ].

    If H = V diag(lambda) V.T, then with Delta_{kl} = lambda_k - lambda_l,

        P = M.T diag(w) M,

    where M_{(k,l),s} = V_{s k} V_{s l}, for k >= l, w_{kk}=1, and

        w_{kl} = 2 E_t[cos(Delta_{kl} t)]

    for k > l.
    """
    H = np.asarray(H, dtype=float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square.")
    if not np.allclose(H, H.T):
        raise ValueError("H must be real symmetric.")

    t0, tf = map(float, time_lims)

    if tf <= t0:
        raise ValueError("time_lims must satisfy tf > t0.")

    d = H.shape[0]

    lam, V = la.eigh(H)

    idx = np.arange(d)
    kk, ll = np.meshgrid(idx, idx, indexing="ij")
    tri = kk >= ll

    dlam = (lam[kk] - lam[ll])[tri]

    w = np.ones_like(dlam, dtype=float)
    mask = dlam != 0.0

    w[mask] = (
        2.0
        * (np.sin(dlam[mask] * tf) - np.sin(dlam[mask] * t0))
        / ((tf - t0) * dlam[mask])
    )

    M = la.khatri_rao(V.T, V.T)[tri.ravel()]

    P = (w[:, None] * M).T @ M
    P = np.real_if_close(P)
    P = np.asarray(P, dtype=float)

    check_double_stochasticity(P)
    return P


def create_proposal_matrix_quantum_layden(
    h: np.ndarray,
    J: np.ndarray,
    gamma_lims: tuple[float, float] = (0.25, 0.60),
    gamma_steps: int = 20,
    time_lims: tuple[float, float] = (2.0, 20.0),
) -> np.ndarray:
    r"""
    Layden et al. proposal:

        P = E_{gamma ~ Unif[g0, g1]} E_{t ~ Unif[t0, tf]}
            [ |<y| exp(-i H(gamma) t) |x>|^2 ],

    where

        H(gamma) = (1 - gamma) diag(E) + gamma H_mixer.

    The gamma average is approximated by a midpoint Riemann sum with
    gamma_steps points. The number of spins n is inferred from h and J.
    """
    if gamma_steps < 1:
        raise ValueError("gamma_steps must be positive.")

    n = infer_n_from_h_J(h, J)
    d = 2**n

    H_ising = np.diag(energies(h, J))
    H_mixer = get_cached_mixing_hamiltonian(n)

    g0, g1 = map(float, gamma_lims)

    if g1 <= g0:
        raise ValueError("gamma_lims must satisfy g1 > g0.")

    step = (g1 - g0) / gamma_steps
    gammas = g0 + (np.arange(gamma_steps) + 0.5) * step

    P = np.zeros((d, d), dtype=float)

    for gamma in gammas:
        H = (1.0 - gamma) * H_ising + gamma * H_mixer
        P += create_proposal_matrix_quantum_time_avg(H, time_lims=time_lims)

    P /= gamma_steps

    check_double_stochasticity(P)
    return P
