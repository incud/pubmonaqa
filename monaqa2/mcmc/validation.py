import numpy as np
import networkx as nx


# ========================================================================
# ====== UTILITIES FUNCTION TO SUPPORT TRANSITION MATRIX VALIDATION ======
# ========================================================================

def is_column_stochastic(Q: np.ndarray, tol: float = 1e-10) -> bool:
    Q = np.asarray(Q)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return False
    if (Q < 0).any():
        return False
    if not np.allclose(Q.sum(axis=0), 1.0, atol=tol, rtol=0.0):
        return False

    return True


def check_column_stochasticity(Q: np.ndarray, tol: float = 1e-10) -> None:
    Q = np.asarray(Q)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Invalid MC: the matrix must be square.")

    if (Q < 0).any():
        raise ValueError(
            f"Invalid MC: the matrix is not column-stochastic "
            f"(negative entries found, {np.min(Q)=})."
        )

    col_deviation = np.max(np.abs(Q.sum(axis=0) - 1.0))
    if not np.allclose(Q.sum(axis=0), 1.0, atol=tol, rtol=0.0):
        raise ValueError(
            f"Invalid MC: the matrix is not column-stochastic "
            f"(columns do not sum to 1, max deviation={col_deviation})."
        )


def is_irreducible(Q: np.ndarray, tol: float = 1e-10) -> bool:
    Q = np.asarray(Q)
    return nx.is_strongly_connected(nx.from_numpy_array(Q > tol, create_using=nx.DiGraph))


def check_irreducibility(Q: np.ndarray, tol: float = 1e-10) -> None:
    if not is_irreducible(Q, tol=tol):
        raise ValueError("Chain is not irreducible.")


# Backward-compatible alias for the previous typo.
def check_irreduciblility(Q: np.ndarray, tol: float = 1e-10) -> None:
    check_irreducibility(Q, tol=tol)


def is_reversible(Q: np.ndarray, tol_edge: float = 1e-14, tol_log: float = 1e-8) -> bool:
    Q = np.asarray(Q)

    A = (Q > tol_edge) & (Q.T > tol_edge)
    G = nx.from_numpy_array(A, create_using=nx.Graph)

    for cycle in nx.cycle_basis(G):
        fwd = 0.0
        bwd = 0.0

        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]

            fwd += np.log(Q[a, b])
            bwd += np.log(Q[b, a])

        if abs(fwd - bwd) > tol_log:
            return False

    return True


def check_reversibility(Q: np.ndarray, tol_edge: float = 1e-14, tol_log: float = 1e-8) -> None:
    if not is_reversible(Q, tol_edge=tol_edge, tol_log=tol_log):
        raise ValueError("Detailed balance fails on a cycle.")


def is_symmetric(Q: np.ndarray, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    Q = np.asarray(Q)
    return np.allclose(Q, Q.T, atol=atol, rtol=rtol)


def check_symmetric(Q: np.ndarray, atol: float = 1e-12, rtol: float = 1e-10) -> None:
    Q = np.asarray(Q)

    if not np.allclose(Q, Q.T, atol=atol, rtol=rtol):
        diff = np.max(np.abs(Q - Q.T))
        raise ValueError(f"Not symmetric: max|Q-Q.T|={diff}")


def is_double_stochastic(Q: np.ndarray, tol: float = 1e-10) -> bool:
    Q = np.asarray(Q)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return False
    if not np.allclose(Q.sum(axis=0), 1.0, atol=tol, rtol=0.0):
        return False
    if not np.allclose(Q.sum(axis=1), 1.0, atol=tol, rtol=0.0):
        return False

    return True


def check_double_stochasticity(Q: np.ndarray, tol: float = 1e-10) -> None:
    Q = np.asarray(Q)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Matrix must be square.")

    if not np.allclose(Q.sum(axis=0), 1.0, atol=tol, rtol=0.0):
        deviation = np.max(np.abs(Q.sum(axis=0) - 1.0))
        raise ValueError(f"Columns do not sum to 1 within tolerance. Max deviation={deviation}.")

    if not np.allclose(Q.sum(axis=1), 1.0, atol=tol, rtol=0.0):
        deviation = np.max(np.abs(Q.sum(axis=1) - 1.0))
        raise ValueError(f"Rows do not sum to 1 within tolerance. Max deviation={deviation}.")

