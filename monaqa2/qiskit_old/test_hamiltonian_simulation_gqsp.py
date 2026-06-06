import pytest
import numpy as np
import scipy as sc
from qiskit import QuantumCircuit

from monaqa2.qiskit.hamiltonian_simulation_gqsp_gate import HamiltonianSimulationGQSP
from monaqa2.qiskit.hamiltonian_simulation_gqsp_symbolic import (
    hamiltonian_simulation_gqsp_number_qubits,
    hamiltonian_simulation_gqsp_nc_depth,
    hamiltonian_simulation_gqsp_t_count,
    hamiltonian_simulation_gqsp_rz_count,
    hamiltonian_simulation_gqsp_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import get_unitary, get_nc_depth, get_t_count, get_toffoli_count, get_rz_count
from monaqa2.qiskit.utils_numpy import kron, ket, bra
from monaqa2.qiskit.utils_numpy import ( _ising_alpha, _ising_hamiltonian,)


def _make_hsim_cases():
    return [
        (1, np.array([1.0]), np.array([[0.0]]), np.array([1.0]), 0.05, 1e-2),
        (1, np.array([1.0]), np.array([[0.0]]), np.array([1.0]), 0.05, 1e-8),
        (1, np.array([-0.75]), np.array([[0.0]]), np.array([0.25]), 0.10, 1e-3),
        (1, np.array([-0.75]), np.array([[0.0]]), np.array([0.25]), 0.10, 1e-8),
        (1, np.array([1.0]), np.array([[0.0]]), np.array([1.25]), 0.20, 1e-4),
        (1, np.array([1.0]), np.array([[0.0]]), np.array([1.25]), 0.20, 1e-8),
        (2, np.array([1.0, 0.0]), np.zeros((2, 2)), np.array([0.25, 0.0]), 0.04, 1e-2),
        (2, np.array([1.0, 0.0]), np.zeros((2, 2)), np.array([0.25, 0.0]), 0.04, 1e-8),
        (2, np.array([0.0, -1.0]), np.array([[0.0, 0.5], [0.5, 0.0]]), np.array([0.0, 0.0]), 0.08, 1e-3),
        (2, np.array([0.0, -1.0]), np.array([[0.0, 0.5], [0.5, 0.0]]), np.array([0.0, 0.0]), 0.08, 1e-8),
        (2, np.array([0.5, 0.0]), np.zeros((2, 2)), np.array([-0.25, 0.0]), 0.15, 1e-4),
        (2, np.array([0.5, 0.0]), np.zeros((2, 2)), np.array([-0.25, 0.0]), 0.15, 1e-8),
        (2, np.array([0.0, 0.75]), np.array([[0.0, -0.4], [-0.4, 0.0]]), np.array([0.0, 0.0]), 0.25, 1e-6),
        (2, np.array([0.0, 0.75]), np.array([[0.0, -0.4], [-0.4, 0.0]]), np.array([0.0, 0.0]), 0.25, 1e-8),
    ]

def _make_symbolic_cases():
    return [
        (1, np.array([1.0]), np.array([[0.0]]), np.array([0.1]), 0.05, 0.5),
        (2, np.array([1.0, 0.0]), np.zeros((2, 2)), np.array([0.25, 0.0]), 0.04, 0.5),
        (3, np.array([1.0, 0.0, -0.5]), np.array([[0.0, 0.0, 0.75], [0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]), np.zeros(3), 0.03, 0.5),
        (4, np.array([0.0, -1.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]]), np.zeros(4), 0.02, 0.5),
    ]


def _matrix_poly(coeffs: np.ndarray, W: np.ndarray) -> np.ndarray:
    out = np.zeros_like(W, dtype=complex)
    power = np.eye(W.shape[0], dtype=complex)

    for c in coeffs:
        out += c * power
        power = power @ W

    return out


@pytest.mark.parametrize("n,h,J,gamma,t,eps", _make_hsim_cases())
def test_hamiltonian_simulation_gqsp_polynomial_block(n, h, J, gamma, t, eps):
    hsim = HamiltonianSimulationGQSP(n, h, J, gamma, t, eps, mocked_reflection=True)
    qc = QuantumCircuit(hsim.num_qubits)
    qc.append(hsim, list(range(hsim.num_qubits)))
    U = get_unitary(qc, big_endian=True)

    qc_w = QuantumCircuit(hsim.qubitization.num_qubits)
    qc_w.append(hsim.qubitization, list(range(hsim.qubitization.num_qubits)))
    W = get_unitary(qc_w, big_endian=True)

    n_work = hsim.num_qubits - 1 - hsim.qubitization.num_qubits
    dim_control_and_w = 2 ** (1 + hsim.qubitization.num_qubits)

    if n_work > 0:
        U = kron(np.eye(dim_control_and_w, dtype=complex), bra(0, n_work)) @ U @ kron(np.eye(dim_control_and_w, dtype=complex), ket(0, n_work))

    dim_w = W.shape[0]
    actual = kron(bra(0, 1), np.eye(dim_w, dtype=complex)) @ U @ kron(ket(0, 1), np.eye(dim_w, dtype=complex))
    expected = np.linalg.matrix_power(W, -hsim.degree) @ _matrix_poly(hsim.poly_coeffs, W)

    np.testing.assert_allclose(actual, expected, atol=1e-8)


@pytest.mark.parametrize("n,h,J,gamma,t,eps", _make_hsim_cases())
def test_hamiltonian_simulation_gqsp_system_block(n, h, J, gamma, t, eps):
    hsim = HamiltonianSimulationGQSP(n, h, J, gamma, t, eps, mocked_reflection=True)
    qc = QuantumCircuit(hsim.num_qubits)
    qc.append(hsim, list(range(hsim.num_qubits)))
    U = get_unitary(qc, big_endian=True)

    n_work = hsim.num_qubits - 1 - hsim.qubitization.num_qubits
    dim_control_and_w = 2 ** (1 + hsim.qubitization.num_qubits)

    if n_work > 0:
        U = kron(np.eye(dim_control_and_w, dtype=complex), bra(0, n_work)) @ U @ kron(np.eye(dim_control_and_w, dtype=complex), ket(0, n_work))

    dim_w = 2 ** hsim.qubitization.num_qubits
    gqsp_block = kron(bra(0, 1), np.eye(dim_w, dtype=complex)) @ U @ kron(ket(0, 1), np.eye(dim_w, dtype=complex))

    n_system = len(hsim.qubitization.layout["system"])
    n_ancilla = hsim.qubitization.num_qubits - n_system
    actual_system = kron(np.eye(2**n_system, dtype=complex), bra(0, n_ancilla)) @ gqsp_block @ kron(np.eye(2**n_system, dtype=complex), ket(0, n_ancilla))

    H = _ising_hamiltonian(n, h, J, gamma)
    expected_system = sc.linalg.expm(-1j * H * t)

    np.testing.assert_allclose(hsim._alpha(), _ising_alpha(n, h, J, gamma), atol=1e-12)
    np.testing.assert_allclose(actual_system, expected_system, atol=eps)


@pytest.mark.parametrize("n,h,J,gamma,t,eps", _make_symbolic_cases())
def test_hamiltonian_simulation_gqsp_symbolic_bounds(n, h, J, gamma, t, eps):
    hsim = HamiltonianSimulationGQSP(n, h, J, gamma, t, eps, mocked_angles=True)
    qc = QuantumCircuit(hsim.num_qubits)
    qc.append(hsim, list(range(hsim.num_qubits)))

    n_terms = hsim.qubitization.lcu.n_terms
    alpha = hsim._alpha()

    sym_num_qubits = int(hamiltonian_simulation_gqsp_number_qubits(n, n_terms, alpha, t, eps))
    sym_nc_depth = int(hamiltonian_simulation_gqsp_nc_depth(n, n_terms, alpha, t, eps))
    sym_t_count = int(hamiltonian_simulation_gqsp_t_count(n, n_terms, alpha, t, eps))
    sym_rz_count = int(hamiltonian_simulation_gqsp_rz_count(n, n_terms, alpha, t, eps))
    sym_toffoli_count = int(hamiltonian_simulation_gqsp_toffoli_count(n, n_terms, alpha, t, eps))

    actual_num_qubits = qc.num_qubits
    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)

    assert sym_num_qubits == actual_num_qubits
    assert sym_nc_depth >= actual_nc_depth
    assert sym_t_count >= actual_t_count
    assert sym_rz_count >= actual_rz_count
    assert sym_toffoli_count >= actual_toffoli_count