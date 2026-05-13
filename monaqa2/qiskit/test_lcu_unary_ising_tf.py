import pytest
import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from monaqa2.qiskit.lcu_unary_ising_tf_gate import LcuUnaryIsingTF
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)
from monaqa2.qiskit.utils_numpy import (
    _ising_alpha, _ising_hamiltonian, kron, ket, bra, ketbra, X, Z, I,
)
from monaqa2.qiskit.lcu_unary_ising_tf_symbolic import (
    lcu_unary_number_qubits,
    lcu_unary_nc_depth,
    lcu_unary_t_count,
    lcu_unary_rz_count,
    lcu_unary_toffoli_count,
)


def _make_hamiltonians() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (2, np.array([1.0, 0.0]), np.array([[0.0, -2.0], [-2.0, 0.0]]), np.array([0.0, 0.5])),
        (2, np.array([0.0, -1.5]), np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([2.0, 0.0])),
        (3, np.array([1.0, 0.0, -0.5]), np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([0.0, -1.25, 0.0])),
        (3, np.array([0.0, 0.0, 2.0]), np.array([[0.0, -0.75, 0.0], [-0.75, 0.0, 1.5], [0.0, 1.5, 0.0]]), np.array([0.0, 0.0, 0.0])),
        (4, np.array([1.0, 0.0, 0.0, -2.0]), np.array([[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, -1.0], [0.5, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.75, 0.0])),
        (4, np.array([0.0, 0.0, 0.0, 0.0]), np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.4], [0.0, 0.0, -0.4, 0.0]]), np.array([0.5, 0.0, -1.5, 0.0])),
    ]


@pytest.mark.parametrize("n,h,J,gamma", _make_hamiltonians())
def test_lcu_unary_evolution(n, h, J, gamma):

    # Instantiate circuit
    lcu = LcuUnaryIsingTF(n, h, J, gamma)
    qc = QuantumCircuit(lcu.num_qubits)
    qc.append(lcu, list(range(lcu.num_qubits)))

    # Evolve unitary, using big-endian ordering.
    U_qiskit = get_unitary(qc, big_endian=True)

    # Project all LCU ancillas, tree + unary prepare, onto |0...0>.
    n_system = len(lcu.layout["system"])
    n_ancilla = lcu.num_qubits - n_system

    I_system = np.eye(2**n_system, dtype=complex)
    ancilla_ket_zero = ket(0, n_ancilla)
    ancilla_bra_zero = bra(0, n_ancilla)

    left_projector = kron(I_system, ancilla_bra_zero)
    right_projector = kron(I_system, ancilla_ket_zero)

    U_effective = left_projector @ U_qiskit @ right_projector

    # The block encoding should satisfy:
    #     <0_anc| U |0_anc> = H / alpha.
    alpha = _ising_alpha(n, h, J, gamma)
    H_actual = alpha * U_effective
    H_expected = _ising_hamiltonian(n, h, J, gamma)

    np.testing.assert_allclose(H_actual, H_expected, atol=1e-10)


@pytest.mark.parametrize("n,h,J,gamma", _make_hamiltonians())
def test_lcu_unary_symbolic_bounds(n, h, J, gamma):

    # Instantiate circuit
    lcu = LcuUnaryIsingTF(n, h, J, gamma)
    qc = QuantumCircuit(lcu.num_qubits)
    qc.append(lcu, list(range(lcu.num_qubits)))

    sym_num_qubits = int(lcu_unary_number_qubits(n, lcu.n_terms))
    sym_nc_depth = int(lcu_unary_nc_depth(n, lcu.n_terms))
    sym_t_count = int(lcu_unary_t_count(n, lcu.n_terms))
    sym_rz_count = int(lcu_unary_rz_count(n, lcu.n_terms))
    sym_toffoli_count = int(lcu_unary_toffoli_count(n, lcu.n_terms))

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