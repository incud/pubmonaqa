import numpy as np
import pytest
from qiskit import QuantumCircuit
from scipy.linalg import expm

from monaqa2.qiskit.trotterized_ising_tf_gate import (
    TrotterizedOperatorIsingTF,
    ControlledTrotterizedOperatorIsingTF,
)
from monaqa2.qiskit.trotterized_ising_tf_symbolic import (
    trotterized_ising_tf_number_qubits,
    trotterized_ising_tf_nc_depth,
    trotterized_ising_tf_t_count,
    trotterized_ising_tf_rz_count,
    trotterized_ising_tf_toffoli_count,
    controlled_trotterized_ising_tf_number_qubits,
    controlled_trotterized_ising_tf_nc_depth,
    controlled_trotterized_ising_tf_t_count,
    controlled_trotterized_ising_tf_rz_count,
    controlled_trotterized_ising_tf_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)
from monaqa2.qiskit.utils_numpy import bra, ket, kron, X, Z, I


def _make_hamiltonians() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (2, np.array([1.0, -0.5]), np.array([[0.0, 0.75], [0.75, 0.0]]), np.array([0.25, -0.4])),
        (3, np.array([1.0, 0.0, -0.5]), np.array([[0.0, 0.3, 0.0], [0.3, 0.0, -0.7], [0.0, -0.7, 0.0]]), np.array([0.2, 0.0, -0.6])),
        (4, np.array([0.0, -1.0, 0.5, 0.0]), np.array([[0.0, 0.2, 0.0, -0.4], [0.2, 0.0, 0.6, 0.0], [0.0, 0.6, 0.0, -0.8], [-0.4, 0.0, -0.8, 0.0]]), np.array([0.1, -0.3, 0.0, 0.4])),
    ]


def _make_symbolic_hamiltonians() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (2, np.array([1.0, 0.0]), np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.25])),
        (3, np.array([1.0, 0.0, -0.5]), np.array([[0.0, 0.3, 0.0], [0.3, 0.0, -0.7], [0.0, -0.7, 0.0]]), np.array([0.2, 0.0, -0.6])),
        (5, np.array([0.1, -0.2, 0.0, 0.4, -0.5]), np.array([[0.0, 0.1, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, -0.7], [0.0, 0.5, 0.0, 0.9, 0.0], [0.3, 0.0, 0.9, 0.0, 1.1], [0.0, -0.7, 0.0, 1.1, 0.0]]), np.array([0.6, 0.0, -0.4, 0.0, 0.2])),
    ]


def _op_on_n(pauli: np.ndarray, i: int, n: int) -> np.ndarray:
    ops = [I] * n
    ops[i] = pauli

    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)

    return out


def _two_op_on_n(pauli_a: np.ndarray, i: int, pauli_b: np.ndarray, j: int, n: int) -> np.ndarray:
    ops = [I] * n
    ops[i] = pauli_a
    ops[j] = pauli_b

    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)

    return out


def _ising_z_hamiltonian(n: int, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    H = np.zeros((2**n, 2**n), dtype=complex)

    for i, coeff in enumerate(h):
        if abs(coeff) >= 1e-8:
            H += coeff * _op_on_n(Z, i, n)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) >= 1e-8:
                H += J[i, j] * _two_op_on_n(Z, i, Z, j, n)

    return H


def _ising_x_hamiltonian(n: int, gamma: np.ndarray) -> np.ndarray:
    H = np.zeros((2**n, 2**n), dtype=complex)

    for i, coeff in enumerate(gamma):
        if abs(coeff) >= 1e-8:
            H += coeff * _op_on_n(X, i, n)

    return H


def _expected_first_order_trotter(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, time: float, r: int) -> np.ndarray:
    dt = time / r
    Hz = _ising_z_hamiltonian(n, h, J)
    Hx = _ising_x_hamiltonian(n, gamma)
    step = expm(-1j * dt * Hx) @ expm(-1j * dt * Hz)
    return np.linalg.matrix_power(step, r)


def _term_counts(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray) -> tuple[int, int, int]:
    n_terms_z = int(np.sum(np.abs(h) >= 1e-8))
    n_terms_zz = int(np.sum(np.abs(J[np.triu_indices(n, k=1)]) >= 1e-8))
    n_terms_x = int(np.sum(np.abs(gamma) >= 1e-8))
    return n_terms_z, n_terms_zz, n_terms_x


@pytest.mark.parametrize("n,h,J,gamma", _make_hamiltonians())
def test_trotterized_ising_tf_unitary(n, h, J, gamma):
    time = 0.37
    r = 3
    gate = TrotterizedOperatorIsingTF(n, h, J, gamma, time=time, num_trotter_steps=r)
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))

    actual = get_unitary(qc, big_endian=True)
    expected = _expected_first_order_trotter(n, h, J, gamma, time, r)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("n,h,J,gamma", _make_hamiltonians())
def test_controlled_trotterized_ising_tf_unitary(n, h, J, gamma):
    time = 0.21
    r = 2
    ctrl = ControlledTrotterizedOperatorIsingTF(n, h, J, gamma, time=time, num_trotter_steps=r)
    qc_ctrl = QuantumCircuit(ctrl.num_qubits)
    qc_ctrl.append(ctrl, list(range(ctrl.num_qubits)))
    U_ctrl = get_unitary(qc_ctrl, big_endian=True)

    n_work = len(ctrl.layout["control_work"])
    dim_control_system = 2 ** (1 + n)

    if n_work > 0:
        U_ctrl = (
            kron(np.eye(dim_control_system, dtype=complex), bra(0, n_work))
            @ U_ctrl
            @ kron(np.eye(dim_control_system, dtype=complex), ket(0, n_work))
        )

    U = _expected_first_order_trotter(n, h, J, gamma, time, r)
    zero = np.zeros_like(U)
    expected = np.block([
        [np.eye(2**n, dtype=complex), zero],
        [zero, U],
    ])

    np.testing.assert_allclose(U_ctrl, expected, atol=1e-10)


def test_trotterized_ising_tf_eps_sets_num_trotter_steps_from_commutator_bound():
    n = 2
    h = np.array([1.0, -0.5])
    J = np.array([[0.0, 0.75], [0.75, 0.0]])
    gamma = np.array([0.25, -0.4])
    time = 0.9
    eps = 0.05

    gate = TrotterizedOperatorIsingTF(n, h, J, gamma, time=time, eps=eps)
    comm_bound = TrotterizedOperatorIsingTF.first_order_commutator_bound(n, h, J, gamma)
    expected_r = max(1, int(np.ceil((abs(time) ** 2) * comm_bound / (2.0 * eps))))

    assert gate.num_trotter_steps == expected_r


@pytest.mark.parametrize("n,h,J,gamma", _make_symbolic_hamiltonians())
def test_trotterized_ising_tf_symbolic_bounds(n, h, J, gamma):
    r = 2
    gate = TrotterizedOperatorIsingTF(n, h, J, gamma, num_trotter_steps=r)
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))

    n_terms_z, n_terms_zz, n_terms_x = _term_counts(n, h, J, gamma)

    sym_num_qubits = int(trotterized_ising_tf_number_qubits(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_nc_depth = int(trotterized_ising_tf_nc_depth(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_t_count = int(trotterized_ising_tf_t_count(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_rz_count = int(trotterized_ising_tf_rz_count(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_toffoli_count = int(trotterized_ising_tf_toffoli_count(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))

    assert sym_num_qubits == qc.num_qubits
    assert sym_nc_depth >= get_nc_depth(qc)
    assert sym_t_count >= get_t_count(qc)
    assert sym_rz_count >= get_rz_count(qc)
    assert sym_toffoli_count >= get_toffoli_count(qc)


@pytest.mark.parametrize("n,h,J,gamma", _make_symbolic_hamiltonians())
def test_controlled_trotterized_ising_tf_symbolic_bounds(n, h, J, gamma):
    r = 2
    ctrl = ControlledTrotterizedOperatorIsingTF(n, h, J, gamma, num_trotter_steps=r)
    qc = QuantumCircuit(ctrl.num_qubits)
    qc.append(ctrl, list(range(ctrl.num_qubits)))

    n_terms_z, n_terms_zz, n_terms_x = _term_counts(n, h, J, gamma)

    sym_num_qubits = int(controlled_trotterized_ising_tf_number_qubits(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_nc_depth = int(controlled_trotterized_ising_tf_nc_depth(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_t_count = int(controlled_trotterized_ising_tf_t_count(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_rz_count = int(controlled_trotterized_ising_tf_rz_count(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))
    sym_toffoli_count = int(controlled_trotterized_ising_tf_toffoli_count(n, r, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, n_terms_x=n_terms_x))

    assert sym_num_qubits == qc.num_qubits
    assert sym_nc_depth >= get_nc_depth(qc)
    assert sym_t_count >= get_t_count(qc)
    assert sym_rz_count >= get_rz_count(qc)
    assert sym_toffoli_count >= get_toffoli_count(qc)


def test_controlled_trotterized_ising_tf_zz_schedule_is_linear_depth():
    n = 8
    h = np.zeros(n)
    J = np.ones((n, n)) - np.eye(n)
    gamma = np.zeros(n)
    ctrl = ControlledTrotterizedOperatorIsingTF(n, h, J, gamma, num_trotter_steps=1)

    # The round-robin edge coloring uses n - 1 rounds for even n and n rounds
    # for odd n, hence O(n) ZZ scheduling depth for all-to-all J.
    matchings = TrotterizedOperatorIsingTF._matchings(n)

    assert len(matchings) == n - 1
    assert all(len({q for edge in matching for q in edge}) == 2 * len(matching) for matching in matchings)
    assert max(len(matching) for matching in matchings) <= len(ctrl.layout["control_work"])
