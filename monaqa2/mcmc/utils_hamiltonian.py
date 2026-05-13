import numpy as np
from functools import reduce


# ========================================================================
# ====== UTILITIES FUNCTION TO SUPPORT PROPOSAL MATRIX CREATION ==========
# ========================================================================

def infer_n_from_h_J(h: np.ndarray, J: np.ndarray) -> int:
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)

    if h.ndim != 1:
        raise ValueError("h must be a one-dimensional array.")

    n = h.shape[0]

    if J.shape != (n, n):
        raise ValueError(f"J must have shape {(n, n)}, got {J.shape}.")

    return n


def int_to_bin(i: int, n: int) -> str:
    """
    Convert an integer to a bitstring of fixed length using the convention
    0 -> 00...0.
    """
    return bin(i)[2:].zfill(n)


def bits(i: int, n: int) -> np.ndarray:
    """
    Big-endian bit vector matching int_to_bin(i, n).
    """
    return np.array([int(b) for b in int_to_bin(i, n)], dtype=int)


def energy(i: int, h: np.ndarray, J: np.ndarray) -> float:
    """
    Ising energy of computational-basis state i.

    Uses

        E(z) = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,

    with Z_i = 1 - 2 z_i.
    """
    n = infer_n_from_h_J(h, J)
    z = 1 - 2 * bits(i, n)

    value = float(np.dot(h, z))

    for p in range(n):
        for q in range(p + 1, n):
            value += float(J[p, q]) * int(z[p]) * int(z[q])

    return value


def energies(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    n = infer_n_from_h_J(h, J)
    return np.array([energy(i, h, J) for i in range(2**n)], dtype=float)


def hamming(s1: str, s2: str) -> int:
    """
    Hamming distance between two bit strings.
    """
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_mixing_hamiltonian(n: int) -> np.ndarray:
    """
    Dense transverse-field mixing Hamiltonian sum_i X_i.
    """
    if n < 1:
        raise ValueError("n must be positive.")

    def kron(mats: list[np.ndarray]) -> np.ndarray:
        return reduce(np.kron, mats)

    def X(i: int, n: int) -> np.ndarray:
        if i < 0 or i >= n:
            raise ValueError("Bad value of i.")

        X_mat = np.array([[0, 1], [1, 0]], dtype=float)
        I_mat = np.eye(2, dtype=float)

        return kron([X_mat if j == i else I_mat for j in range(n)])

    return sum(X(i, n) for i in range(n))


_cached_mixing_hamiltonian: dict[int, np.ndarray] = {}


def get_cached_mixing_hamiltonian(n: int) -> np.ndarray:
    if n not in _cached_mixing_hamiltonian:
        _cached_mixing_hamiltonian[n] = get_mixing_hamiltonian(n)

    return _cached_mixing_hamiltonian[n]