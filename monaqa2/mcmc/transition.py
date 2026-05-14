import numpy as np
from monaqa2.mcmc.model import IsingModel
from monaqa2.mcmc.proposal import create_proposal_matrix_quantum_exact, create_proposal_matrix_local, create_proposal_matrix_quantum_layden, create_proposal_matrix_uniform


def create_acceptance_matrix(model: IsingModel, beta: float, a: float = 1.0,) -> np.ndarray:
    
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if not (np.isinf(a) or a >= 1):
        raise ValueError("a must satisfy a >= 1 or a = np.inf.")

    E = model.energies_rescaled
    delta = E[:, None] - E[None, :]
    log_r = -beta * delta

    if np.isinf(a):
        log_A = np.minimum(0.0, log_r)
    else:
        log_A = log_r - np.logaddexp(0.0, a * log_r) / a

    A = np.exp(log_A)
    np.fill_diagonal(A, 1.0)
    return A


def create_transition_matrix(P: np.ndarray | str, model: IsingModel, beta: float, a: float = 1.0, gamma: float = None, t: float = None) -> np.ndarray:
    n = model.n
    h = model.h_rescaled
    J = model.J_rescaled
    A = create_acceptance_matrix(model, beta, a)

    if isinstance(P, str):
        if P == "uniform":
            P = create_proposal_matrix_uniform(n=n)
        elif P == "local1":
            P = create_proposal_matrix_local(n=n, k=1)
        elif P == "local2":
            P = create_proposal_matrix_local(n=n, k=2)
        elif P == "local3":
            P = create_proposal_matrix_local(n=n, k=3)
        elif P == "qemc":
            if gamma is None or t is None:
                raise ValueError("Transition matrix 'qemc' must set both 'gamma' and 't' parameters")
            P = create_proposal_matrix_quantum_exact(h, J, gamma, t)
        elif P == "layden":
            P = create_proposal_matrix_quantum_layden(h, J)
        else:
            raise ValueError(f"Transition matrix '{P}' is not known. Allowed values are: uniform, local1, local2, local3, qemc, layden.")

    Q = P * A
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, 1.0 - Q.sum(axis=0))

    return Q
