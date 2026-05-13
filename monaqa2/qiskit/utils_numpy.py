import numpy as np
from functools import reduce

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron(*matrices):
    """Reduce multiple matrices using tensor product."""
    return reduce(np.kron, matrices)

def sequence(*matrices):
    """Reduce multiple matrices using tensor product."""
    return reduce(np.dot, matrices)

def apply_unitary(U, targets, n):
    # 1. Reshape Identity into a tensor of shape (2, 2, ..., 2)
    # There are n 'row' axes and n 'column' axes
    T = np.eye(2**n).reshape([2] * (2 * n))
    
    # 2. Define the axis permutation: move targets to the front
    # We only need to move the 'row' axes (0 to n-1)
    others = [i for i in range(n) if i not in targets]
    perm = targets + others
    
    # 3. Transpose the tensor, apply U via matrix multiplication, then transpose back
    # We reshape to (dim_U, dim_rest, dim_U, dim_rest)
    dim_u = 2**len(targets)
    dim_rest = 2**(n - len(targets))
    
    T = T.transpose(perm + [i + n for i in perm])
    T = T.reshape(dim_u, dim_rest, dim_u, dim_rest)
    
    # Apply U: (u_in, rest_in, u_out, rest_out) -> Matrix multiply on u axes
    # This is equivalent to: T_new = U @ T_on_target_axes
    res = np.einsum('ij,jklm->iklm', U, T)
    
    # 4. Reverse the permutation and return to square matrix
    inv_perm = np.argsort(perm)
    res = res.reshape([2] * (2 * n))
    res = res.transpose(list(inv_perm) + [i + n for i in inv_perm])
    
    return res.reshape(2**n, 2**n)

def ket(i: int, n: int):
    """Return the column vector for |i> in an n-qubit space."""
    dim = 2**n
    vec = np.zeros((dim, 1), dtype=complex)
    vec[i, 0] = 1.0
    return vec

def bra(i: int, n: int):
    """Return the row vector for <i| in an n-qubit space."""
    return ket(i, n).conj().T

def ketbra(i: int, n: int):
    """Return the matrix for |i><i|"""
    return ket(i, n) @ bra(i, n)

SWAP = 0.5 * (kron(I, I) + kron(X, X) + kron(Y, Y) + kron(Z, Z))

def swap(n: int):
    """
    Return the 2n-qubit unitary that swaps two n-qubit registers:

        |a>|b> -> |b>|a>

    Basis convention matches ket(i, n), with the first register as the more
    significant n-bit block in the 2n-qubit index:
    
        index = a * 2**n + b
    """
    if n < 1:
        raise ValueError("n must be positive.")

    dim = 2 ** n
    U = np.zeros((dim * dim, dim * dim), dtype=complex)

    for a in range(dim):
        for b in range(dim):
            col = a * dim + b
            row = b * dim + a
            U[row, col] = 1.0

    return U


def _ising_hamiltonian(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    H = np.zeros((2**n, 2**n), dtype=complex)

    for i in range(n):
        ops_z = [I] * n
        ops_x = [I] * n
        ops_z[i] = Z
        ops_x[i] = X

        H += h[i] * kron(*ops_z)
        H += gamma[i] * kron(*ops_x)

    for i in range(n):
        for j in range(i + 1, n):
            ops = [I] * n
            ops[i] = Z
            ops[j] = Z

            H += J[i, j] * kron(*ops)

    return H


def _ising_alpha(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray) -> float:
    alpha = float(np.sum(np.abs(h)) + np.sum(np.abs(gamma)))

    for i in range(n):
        for j in range(i + 1, n):
            alpha += abs(float(J[i, j]))

    return alpha



def bits(index: int, n: int) -> np.ndarray:
    return np.array([(index >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)
    

def energy(x: int, n: int, h: np.ndarray, J: np.ndarray) -> float:
    z = 1 - 2 * bits(x, n)

    value = float(np.dot(h, z))

    for i in range(n):
        for j in range(i + 1, n):
            value += float(J[i, j]) * int(z[i]) * int(z[j])

    return value


def expected_glauber_amplitude(x: int, y: int, n: int, h: np.ndarray, J: np.ndarray, beta: float, a: float,) -> float:
    delta = energy(y, n, h, J) - energy(x, n, h, J)
    log_r = -beta * delta

    if np.isinf(a):
        log_acceptance = min(0.0, log_r)
    else:
        if a < 1.0:
            raise ValueError("a must satisfy a >= 1 or a = np.inf.")

        # A_a = r / (1 + r**a)**(1/a)
        # log A_a = log_r - (1/a) log(1 + exp(a log_r))
        log_acceptance = log_r - np.logaddexp(0.0, a * log_r) / a

    return float(np.exp(0.5 * log_acceptance))