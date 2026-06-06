import pytest
import numpy as np
from qiskit import QuantumCircuit

from monaqa2.qiskit.qubitized_ising_tf_gate import (
    QubitizedOperatorIsingTF,
    ControlledQubitizedOperatorIsingTF,
)
from monaqa2.qiskit.qubitized_ising_tf_symbolic import (
    qubitized_ising_tf_toffoli_count,
    qubitized_ising_tf_rz_count,
    qubitized_ising_tf_t_count,
    qubitized_ising_tf_nc_depth,
    qubitized_ising_tf_number_qubits,
    controlled_qubitized_ising_tf_number_qubits,
    controlled_qubitized_ising_tf_nc_depth,
    controlled_qubitized_ising_tf_t_count,
    controlled_qubitized_ising_tf_rz_count,
    controlled_qubitized_ising_tf_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)
from monaqa2.qiskit.utils_numpy import bra, ket, kron, X, Z, I, _ising_hamiltonian, _ising_alpha



def _make_hamiltonians() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (2, np.array([1.0, -2.0]), np.array([[0.0, 3.0], [3.0, 0.0]]), np.array([-0.5, 0.0])),
        (2, np.array([0.0, 2.0]), np.array([[0.0, -1.0], [-1.0, 0.0]]), np.array([1.5, -0.25])),
        (3, np.array([1.0, 0.0, -0.5]), np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0])),
        (3, np.array([0.0, -1.5, 0.0]), np.array([[0.0, 0.75, 0.0], [0.75, 0.0, -2.0], [0.0, -2.0, 0.0]]), np.array([0.0, 0.0, 0.0])),
        (4, np.array([1.0, 0.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0, -0.75], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.75, 0.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0, 0.0])),
        (4, np.array([0.0, 0.0, 0.0, -2.0]), np.array([[0.0, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0, 0.0])),
    ]


def _make_controlled_hamiltonians() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (2, np.array([1.0, 0.0]), np.zeros((2, 2)), np.array([0.1, 0.0])),
        (2, np.array([1.0, 0.0]), np.zeros((2, 2)), np.array([0.25, 0.0])),
        (3, np.array([1.0, 0.0, 0.0]), np.zeros((3, 3)), np.array([0.0, 0.0, 1.0])),
    ]


def _make_symbolic_hamiltonians() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (2, np.array([1.0, 0.0]), np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([0.0, 0.0])),
        (2, np.array([1.0, 0.0]), np.zeros((2, 2)), np.array([0.25, 0.0])),
        (2, np.array([1.0, -2.0]), np.array([[0.0, 3.0], [3.0, 0.0]]), np.array([-0.5, 0.0])),
        (2, np.array([-0.75, 1.25]), np.array([[0.0, -0.4], [-0.4, 0.0]]), np.array([0.3, -0.6])),
        (2, np.array([0.2, -0.1]), np.array([[0.0, 0.8], [0.8, 0.0]]), np.array([1.0, -1.5])),

        (3, np.array([1.0, 0.0, -0.5]), np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.zeros(3)),
        (3, np.array([0.0, -1.0, 0.0]), np.array([[0.0, 0.4, 0.0], [0.4, 0.0, -0.6], [0.0, -0.6, 0.0]]), np.array([0.0, 0.0, 0.25])),
        (3, np.array([0.5, -0.75, 1.25]), np.array([[0.0, 0.2, -0.3], [0.2, 0.0, 0.4], [-0.3, 0.4, 0.0]]), np.array([-0.2, 0.6, -1.0])),
        (3, np.array([-1.2, 0.8, 0.3]), np.array([[0.0, -0.7, 0.5], [-0.7, 0.0, -0.25], [0.5, -0.25, 0.0]]), np.array([0.9, 0.0, -0.4])),
        (3, np.array([0.1, 0.2, 0.3]), np.array([[0.0, 1.0, 1.1], [1.0, 0.0, 1.2], [1.1, 1.2, 0.0]]), np.array([0.4, 0.5, 0.6])),

        (4, np.array([1.0, 0.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0, -0.75], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.75, 0.0, 0.0, 0.0]]), np.zeros(4)),
        (4, np.array([0.5, -1.0, 0.0, 0.75]), np.array([[0.0, 0.2, 0.0, -0.4], [0.2, 0.0, 0.6, 0.0], [0.0, 0.6, 0.0, -0.8], [-0.4, 0.0, -0.8, 0.0]]), np.array([0.0, 0.25, -0.5, 0.0])),
        (4, np.array([-0.3, 0.7, -1.1, 0.2]), np.array([[0.0, -0.2, 0.3, -0.4], [-0.2, 0.0, 0.5, -0.6], [0.3, 0.5, 0.0, 0.7], [-0.4, -0.6, 0.7, 0.0]]), np.array([1.0, -0.8, 0.6, -0.4])),
        (4, np.array([0.0, 0.0, 1.0, -1.0]), np.array([[0.0, 1.0, 0.0, 1.5], [1.0, 0.0, -0.5, 0.0], [0.0, -0.5, 0.0, 0.75], [1.5, 0.0, 0.75, 0.0]]), np.array([0.2, 0.0, -0.2, 0.4])),

        (5, np.array([0.0, -1.0, 0.0, 0.0, 0.5]), np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.25, 0.0, 0.0], [0.0, 1.25, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.4], [0.0, 0.0, 0.0, -0.4, 0.0]]), np.zeros(5)),
        (5, np.array([1.0, -0.5, 0.0, 0.75, -1.25]), np.array([[0.0, 0.2, -0.3, 0.0, 0.4], [0.2, 0.0, 0.5, -0.6, 0.0], [-0.3, 0.5, 0.0, 0.7, -0.8], [0.0, -0.6, 0.7, 0.0, 0.9], [0.4, 0.0, -0.8, 0.9, 0.0]]), np.array([0.3, 0.0, -0.2, 0.5, -0.7])),
        (5, np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([[0.0, 0.1, 0.2, 0.3, 0.4], [0.1, 0.0, 0.5, 0.6, 0.7], [0.2, 0.5, 0.0, 0.8, 0.9], [0.3, 0.6, 0.8, 0.0, 1.0], [0.4, 0.7, 0.9, 1.0, 0.0]]), np.array([-0.1, -0.2, -0.3, -0.4, -0.5])),

        (6, np.array([0.0, 0.0, 2.0, 0.0, 0.0, -0.75]), np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.zeros(6)),
        (6, np.array([1.0, -0.5, 0.25, 0.0, -0.75, 1.25]), np.array([[0.0, 0.2, 0.0, -0.3, 0.4, 0.0], [0.2, 0.0, 0.5, 0.0, -0.6, 0.7], [0.0, 0.5, 0.0, 0.8, 0.0, -0.9], [-0.3, 0.0, 0.8, 0.0, 1.0, 0.0], [0.4, -0.6, 0.0, 1.0, 0.0, -1.1], [0.0, 0.7, -0.9, 0.0, -1.1, 0.0]]), np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])),
        (6, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.0, 0.6, 0.7, 0.8, 0.9], [0.2, 0.6, 0.0, 1.0, 1.1, 1.2], [0.3, 0.7, 1.0, 0.0, 1.3, 1.4], [0.4, 0.8, 1.1, 1.3, 0.0, 1.5], [0.5, 0.9, 1.2, 1.4, 1.5, 0.0]]), np.array([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1])),

        (7, np.array([0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.25]), np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.zeros(7)),
        (7, np.array([1.0, -0.5, 0.25, -0.75, 0.0, 0.5, -1.25]), np.array([[0.0, 0.2, 0.0, -0.3, 0.4, 0.0, -0.5], [0.2, 0.0, 0.6, 0.0, -0.7, 0.8, 0.0], [0.0, 0.6, 0.0, 0.9, 0.0, -1.0, 1.1], [-0.3, 0.0, 0.9, 0.0, 1.2, 0.0, -1.3], [0.4, -0.7, 0.0, 1.2, 0.0, 1.4, 0.0], [0.0, 0.8, -1.0, 0.0, 1.4, 0.0, -1.5], [-0.5, 0.0, 1.1, -1.3, 0.0, -1.5, 0.0]]), np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])),
        (7, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.0, 0.7, 0.8, 0.9, 1.0, 1.1], [0.2, 0.7, 0.0, 1.2, 1.3, 1.4, 1.5], [0.3, 0.8, 1.2, 0.0, 1.6, 1.7, 1.8], [0.4, 0.9, 1.3, 1.6, 0.0, 1.9, 2.0], [0.5, 1.0, 1.4, 1.7, 1.9, 0.0, 2.1], [0.6, 1.1, 1.5, 1.8, 2.0, 2.1, 0.0]]), np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])),
    ]


@pytest.mark.parametrize("n,h,J,gamma", _make_hamiltonians())
def test_qubitized_ising_tf_spectrum(n, h, J, gamma):
    qub = QubitizedOperatorIsingTF(n, h, J, gamma, mocked_reflection=True)
    qc = QuantumCircuit(qub.num_qubits)
    qc.append(qub, list(range(qub.num_qubits)))

    Q_actual = get_unitary(qc, big_endian=True)
    q_eigs = np.linalg.eigvals(Q_actual)

    H = _ising_hamiltonian(n, h, J, gamma)
    alpha = _ising_alpha(n, h, J, gamma)
    h_eigs = np.linalg.eigvalsh(H)

    for E in h_eigs:
        x = np.clip(E / alpha, -1.0, 1.0)
        theta = np.arccos(x)
        expected_plus = np.exp(1j * theta)
        expected_minus = np.exp(-1j * theta)

        assert np.min(np.abs(q_eigs - expected_plus)) < 1e-8
        assert np.min(np.abs(q_eigs - expected_minus)) < 1e-8


@pytest.mark.parametrize("n,h,J,gamma", _make_symbolic_hamiltonians())
def test_qubitized_ising_tf_symbolic_bounds(n, h, J, gamma):
    qub = QubitizedOperatorIsingTF(n, h, J, gamma)
    qc = QuantumCircuit(qub.num_qubits)
    qc.append(qub, list(range(qub.num_qubits)))

    coeffs = np.concatenate([h, J[np.triu_indices(n, k=1)], gamma])
    n_terms = sum(np.abs(coeffs) >= 1e-8)

    sym_num_qubits = int(qubitized_ising_tf_number_qubits(n, n_terms))
    sym_nc_depth = int(qubitized_ising_tf_nc_depth(n, n_terms))
    sym_t_count = int(qubitized_ising_tf_t_count(n, n_terms))
    sym_rz_count = int(qubitized_ising_tf_rz_count(n, n_terms))
    sym_toffoli_count = int(qubitized_ising_tf_toffoli_count(n, n_terms))

    actual_num_qubits = qc.num_qubits
    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)

    assert sym_num_qubits >= actual_num_qubits
    assert sym_nc_depth >= actual_nc_depth
    assert sym_t_count >= actual_t_count
    assert sym_rz_count >= actual_rz_count
    assert sym_toffoli_count >= actual_toffoli_count


@pytest.mark.parametrize("n,h,J,gamma", _make_controlled_hamiltonians())
def test_controlled_qubitized_ising_tf_unitary(n, h, J, gamma):
    ctrl = ControlledQubitizedOperatorIsingTF(n, h, J, gamma)
    qc_ctrl = QuantumCircuit(ctrl.num_qubits)
    qc_ctrl.append(ctrl, list(range(ctrl.num_qubits)))
    U_ctrl = get_unitary(qc_ctrl, big_endian=True)

    qub = QubitizedOperatorIsingTF(n, h, J, gamma, mocked_reflection=False)
    qc_w = QuantumCircuit(qub.num_qubits)
    qc_w.append(qub, list(range(qub.num_qubits)))
    W = get_unitary(qc_w, big_endian=True)

    n_ctrl_work = len(ctrl.layout.get("prepare_control_work", [])) + len(ctrl.layout["reflection_work"])
    dim_control_lcu = 2 ** (1 + ctrl.lcu.num_qubits)

    if n_ctrl_work > 0:
        U_ctrl = (
            kron(np.eye(dim_control_lcu, dtype=complex), bra(0, n_ctrl_work))
            @ U_ctrl
            @ kron(np.eye(dim_control_lcu, dtype=complex), ket(0, n_ctrl_work))
        )

    n_w_work = len(qub.layout["reflection_work"])
    dim_lcu = 2 ** qub.lcu.num_qubits

    if n_w_work > 0:
        W = (
            kron(np.eye(dim_lcu, dtype=complex), bra(0, n_w_work))
            @ W
            @ kron(np.eye(dim_lcu, dtype=complex), ket(0, n_w_work))
        )

    zero = np.zeros((dim_lcu, dim_lcu), dtype=complex)
    expected = np.block([
        [np.eye(dim_lcu, dtype=complex), zero],
        [zero, W],
    ])

    np.testing.assert_allclose(U_ctrl, expected, atol=1e-10)


@pytest.mark.parametrize("n,h,J,gamma", _make_symbolic_hamiltonians())
def test_controlled_qubitized_ising_tf_symbolic_bounds(n, h, J, gamma):
    ctrl = ControlledQubitizedOperatorIsingTF(n, h, J, gamma)
    qc = QuantumCircuit(ctrl.num_qubits)
    qc.append(ctrl, list(range(ctrl.num_qubits)))

    n_terms = ctrl.lcu.n_terms

    sym_num_qubits = int(controlled_qubitized_ising_tf_number_qubits(n, n_terms))
    sym_nc_depth = int(controlled_qubitized_ising_tf_nc_depth(n, n_terms))
    sym_t_count = int(controlled_qubitized_ising_tf_t_count(n, n_terms))
    sym_rz_count = int(controlled_qubitized_ising_tf_rz_count(n, n_terms))
    sym_toffoli_count = int(controlled_qubitized_ising_tf_toffoli_count(n, n_terms))

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
