import numpy as np
import pytest
import scipy.linalg as la
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.glauber_arithmetic_gate import GlauberArithmetic
from monaqa2.qiskit.glauber_arithmetic_symbolic import (
    glauber_arithmetic_degree,
    glauber_arithmetic_number_qubits,
    glauber_arithmetic_nc_depth,
    glauber_arithmetic_t_count,
    glauber_arithmetic_rz_count,
    glauber_arithmetic_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_rz_count,
    get_toffoli_count,
)
from monaqa2.qiskit.utils_numpy import kron, ket, bra


def _make_implementation_cases():
    return [
        (1, np.array([0.5]), np.array([[0.0]]), 0.40, 1e-8, 1.0),
        (1, np.array([0.5]), np.array([[0.0]]), 0.40, 1e-8, 5.0),
        (1, np.array([0.5]), np.array([[0.0]]), 0.40, 1e-8, 10.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-8, 1.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-8, 5.0),
        (2, np.array([1.0, 0.0]), np.array([[0.0, -0.35], [-0.35, 0.0]]), 0.20, 1e-8, 10.0),
    ]


def _make_symbolic_gate_cases():
    return [
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-3, 1.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-3, 5.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-3, 10.0),
        (3, np.array([0.5, -0.25, 0.75]), np.array([[0.0, 0.2, 0.0], [0.2, 0.0, -0.4], [0.0, -0.4, 0.0]]), 0.20, 1e-4, 1.0),
        (3, np.array([0.5, -0.25, 0.75]), np.array([[0.0, 0.2, 0.0], [0.2, 0.0, -0.4], [0.0, -0.4, 0.0]]), 0.20, 1e-4, 5.0),
        (4, np.array([0.5, 0.0, -0.25, 0.75]), np.array([[0.0, 0.2, 0.0, -0.1], [0.2, 0.0, -0.4, 0.0], [0.0, -0.4, 0.0, 0.3], [-0.1, 0.0, 0.3, 0.0]]), 0.15, 1e-5, 10.0),
        (5, np.array([0.5, -0.25, 0.0, 0.75, -0.5]), np.array([[0.0, 0.2, 0.0, -0.1, 0.0], [0.2, 0.0, -0.4, 0.0, 0.25], [0.0, -0.4, 0.0, 0.3, 0.0], [-0.1, 0.0, 0.3, 0.0, -0.2], [0.0, 0.25, 0.0, -0.2, 0.0]]), 0.10, 1e-6, 5.0),
    ]


def _make_large_symbolic_cases():
    return [
        (8, 5.0, 0.20, 1e-8, 8, 6, 1.0),
        (8, 5.0, 0.20, 1e-8, 8, 6, 5.0),
        (8, 5.0, 0.20, 1e-8, 8, 6, 10.0),
        (16, 8.0, 0.15, 1e-10, 12, 20, 1.0),
        (16, 8.0, 0.15, 1e-10, 12, 20, 5.0),
        (32, 12.0, 0.10, 1e-12, 20, 64, 10.0),
        (64, 16.0, 0.08, 1e-14, 26, 128, 5.0),
        (128, 24.0, 0.05, 1e-16, 26, 256, 10.0),
    ]


def _make_gate(n, h, J, beta, eps, a, mocked_circuit, mocked_angles):
    return GlauberArithmetic(n, h, J, beta, eps, a=a, mocked_circuit=mocked_circuit, mocked_angles=mocked_angles)


def _matrix_poly(coeffs: np.ndarray, W: np.ndarray) -> np.ndarray:
    out = np.zeros_like(W, dtype=complex)
    power = np.eye(W.shape[0], dtype=complex)

    for coeff in coeffs:
        out += coeff * power
        power = power @ W

    return out


def _energy(bits: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    z = 1 - 2 * bits
    value = float(np.dot(h, z))

    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            value += J[i, j] * z[i] * z[j]

    return value


def _target_signal_matrix(n: int, h: np.ndarray, J: np.ndarray, beta: float, a: float) -> np.ndarray:
    diag = []

    for basis_index in range(2 ** (2 * n)):
        bits = np.array([int(c) for c in np.binary_repr(basis_index, width=2 * n)], dtype=int)
        x_bits = bits[:n]
        y_bits = bits[n:]
        delta = _energy(y_bits, h, J) - _energy(x_bits, h, J)
        diag.append((1.0 / (1.0 + np.exp(beta * delta))) ** (1.0 / (2.0 * a)))

    return np.diag(np.array(diag, dtype=complex))


def _project_control_work(U: np.ndarray, gate: GlauberArithmetic) -> np.ndarray:
    n_extra = gate.num_qubits - 1 - gate.qubitization.num_qubits
    dim_control_w = 2 ** (1 + gate.qubitization.num_qubits)

    if n_extra == 0:
        return U

    left = kron(np.eye(dim_control_w, dtype=complex), bra(0, n_extra))
    right = kron(np.eye(dim_control_w, dtype=complex), ket(0, n_extra))
    return left @ U @ right


def _control_zero_block(U: np.ndarray, gate: GlauberArithmetic) -> np.ndarray:
    dim_w = 2 ** gate.qubitization.num_qubits
    return kron(bra(0, 1), np.eye(dim_w, dtype=complex)) @ U @ kron(ket(0, 1), np.eye(dim_w, dtype=complex))


def _signal_good_block(U_walk: np.ndarray, gate: GlauberArithmetic) -> np.ndarray:
    n_signal = 2 * gate.n
    n_aux = gate.qubitization.num_qubits - n_signal
    dim_signal = 2**n_signal

    left = kron(np.eye(dim_signal, dtype=complex), bra(0, n_aux))
    right = kron(np.eye(dim_signal, dtype=complex), ket(0, n_aux))
    return left @ U_walk @ right


def _bits_from_basis_index(basis_index: int, width: int) -> np.ndarray:
    return np.array([int(c) for c in np.binary_repr(basis_index, width=width)], dtype=int)


def _qiskit_index(signal_qubits: list[int], bits: np.ndarray) -> int:
    return sum(int(bit) << int(q) for bit, q in zip(bits, signal_qubits))


def _same_signal_norm(state: np.ndarray, signal_qubits: list[int], bits: np.ndarray) -> float:
    total = 0.0

    for idx, amp in enumerate(state):
        if all(((idx >> q) & 1) == int(bit) for bit, q in zip(bits, signal_qubits)):
            total += abs(amp) ** 2

    return float(np.sqrt(total))


def _active_counts(h: np.ndarray, J: np.ndarray, tol: float = 1e-8) -> tuple[int, int]:
    n_terms_z = int(np.count_nonzero(np.abs(h) > tol))
    upper = np.triu(J, k=1)
    n_terms_zz = int(np.count_nonzero(np.abs(upper) > tol))
    return n_terms_z, n_terms_zz


def _as_float(expr) -> float:
    value = complex(expr.evalf() if hasattr(expr, "evalf") else expr)
    assert abs(value.imag) <= 1e-9
    return float(value.real)


def _as_int(expr) -> int:
    value = _as_float(expr)
    assert np.isfinite(value)
    assert abs(value - round(value)) <= 1e-9
    return int(round(value))


@pytest.mark.parametrize("n,h,J,beta,eps,a", _make_implementation_cases())
def test_glauber_arithmetic_delta_hamiltonian_structure(n, h, J, beta, eps, a):
    """
    Implementation structural check.

    This verifies that the gate builds the correct two-register delta Hamiltonian:
    register A carries -E(x), register B carries +E(y), there are no cross-register
    ZZ couplings, and all X/transverse-field coefficients are zero.

    The parameter a changes only the scalar acceptance function, not the encoded
    delta-energy Hamiltonian.
    """
    gate = _make_gate(n, h, J, beta, eps, a, mocked_circuit=True, mocked_angles=False)

    expected_h = np.concatenate([-h, h])
    expected_J = np.zeros((2 * n, 2 * n), dtype=float)
    expected_J[:n, :n] = -J
    expected_J[n:, n:] = J

    np.testing.assert_allclose(gate.h, expected_h, atol=1e-12)
    np.testing.assert_allclose(gate.J, expected_J, atol=1e-12)
    np.testing.assert_allclose(gate.gamma, np.zeros(2 * n), atol=1e-12)
    np.testing.assert_allclose(gate.J[:n, n:], np.zeros((n, n)), atol=1e-12)
    np.testing.assert_allclose(gate.J[n:, :n], np.zeros((n, n)), atol=1e-12)
    assert gate.a == float(a)


@pytest.mark.parametrize("n,h,J,beta,eps,a", _make_implementation_cases())
def test_glauber_arithmetic_polynomial_block(n, h, J, beta, eps, a):
    """
    Implementation GQSP-block check.

    With mocked_circuit=True and real GQSP angles, this extracts the good-control
    block of the full arithmetic gate and verifies that it equals

        W^{-degree} P(W),

    where W is the compact qubitized delta-energy walk and P is the generalized
    Glauber polynomial generated by the class.
    """
    gate = _make_gate(n, h, J, beta, eps, a, mocked_circuit=True, mocked_angles=False)

    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))
    U = _project_control_work(get_unitary(qc, big_endian=True), gate)

    qc_w = QuantumCircuit(gate.qubitization.num_qubits)
    qc_w.append(gate.qubitization, list(range(gate.qubitization.num_qubits)))
    W = get_unitary(qc_w, big_endian=True)

    actual = _control_zero_block(U, gate)
    expected = np.linalg.matrix_power(W, -gate.degree) @ _matrix_poly(gate.poly_coeffs, W)

    np.testing.assert_allclose(actual, expected, atol=max(1000 * eps, 1e-6))


@pytest.mark.parametrize("n,h,J,beta,eps,a", _make_implementation_cases())
def test_glauber_arithmetic_matches_generalized_glauber_acceptance_signal_block(n, h, J, beta, eps, a):
    """
    Implementation generalized-acceptance amplitude check.

    This projects the GQSP control and qubitization ancillas to the all-zero good
    state, leaving only the two input registers. The resulting diagonal block must
    approximate

        (1 / (1 + exp(beta * (E(y) - E(x))))) ** (1 / (2a)).

    For a = 1 this is the usual square-root Glauber acceptance amplitude.
    """
    gate = _make_gate(n, h, J, beta, eps, a, mocked_circuit=True, mocked_angles=False)

    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))
    U = _project_control_work(get_unitary(qc, big_endian=True), gate)

    walk_block = _control_zero_block(U, gate)
    actual_signal = _signal_good_block(walk_block, gate)
    expected_signal = _target_signal_matrix(n, h, J, beta, a)

    assert la.norm(actual_signal - expected_signal, ord=2) <= max(1000 * eps, 1e-6)


@pytest.mark.parametrize("n,h,J,beta,eps,a", _make_implementation_cases())
def test_glauber_arithmetic_basis_states_preserve_registers_statevector(n, h, J, beta, eps, a):
    """
    Implementation basis-state check.

    For selected computational-basis pairs |x>|y>, this verifies the block-encoding
    form

        |x>|y>|0...0> -> |x>|y>(
            g_a(x, y)|0...0> + |perp_{x,y}>
        ),

    where g_a is the generalized Glauber acceptance amplitude. The bad branch is
    not required to be a clean single-qubit flag.
    """
    gate = _make_gate(n, h, J, beta, eps, a, mocked_circuit=True, mocked_angles=False)
    signal_qubits = gate.layout["system"]
    target = _target_signal_matrix(n, h, J, beta, a)

    basis_indices = list(range(2 ** (2 * n))) if n <= 2 else [0, 1, 2 ** (2 * n) - 2, 2 ** (2 * n) - 1]

    max_amp_error = 0.0
    max_bot_norm_error = 0.0
    max_leakage = 0.0

    for basis_index in basis_indices:
        bits = _bits_from_basis_index(basis_index, 2 * n)

        qc = QuantumCircuit(gate.num_qubits)

        for bit, q in zip(bits, signal_qubits):
            if bit:
                qc.x(q)

        qc.append(gate, list(range(gate.num_qubits)))

        state = Statevector.from_int(0, 2**gate.num_qubits).evolve(qc).data
        good_index = _qiskit_index(signal_qubits, bits)

        realized_amp = state[good_index]
        same_signal_norm = _same_signal_norm(state, signal_qubits, bits)
        realized_bot_norm = np.sqrt(max(0.0, same_signal_norm**2 - abs(realized_amp) ** 2))

        expected_amp = target[basis_index, basis_index]
        expected_bot_norm = np.sqrt(max(0.0, 1.0 - abs(expected_amp) ** 2))

        max_amp_error = max(max_amp_error, abs(realized_amp - expected_amp))
        max_bot_norm_error = max(max_bot_norm_error, abs(realized_bot_norm - expected_bot_norm))
        max_leakage = max(max_leakage, np.sqrt(max(0.0, 1.0 - same_signal_norm**2)))

    assert max_amp_error <= max(1000 * eps, 1e-6)
    assert max_bot_norm_error <= max(1000 * eps, 1e-6)
    assert max_leakage <= max(1000 * eps, 1e-6)


@pytest.mark.parametrize("n,h,J,beta,eps,a", _make_symbolic_gate_cases())
def test_glauber_arithmetic_symbolic_static_values_match_gate(n, h, J, beta, eps, a):
    """
    Symbolic static-value check against a constructed full circuit.

    This uses mocked_circuit=False and mocked_angles=True: the qubitized circuit
    structure is the full symbolic target, but GQSP angle synthesis is skipped.

    It checks that symbolic degree and qubit count match the constructed gate for
    different generalized Glauber exponents a.
    """
    gate = _make_gate(n, h, J, beta, eps, a, mocked_circuit=False, mocked_angles=True)
    n_terms_z, n_terms_zz = _active_counts(h, J, gate.tol)

    degree = glauber_arithmetic_degree(gate.alpha, beta, eps, a)
    num_qubits = glauber_arithmetic_number_qubits(n, gate.alpha, beta, eps, n_terms_z, n_terms_zz, a)

    assert _as_int(degree) == gate.degree
    assert _as_int(num_qubits) == gate.num_qubits


@pytest.mark.parametrize("n,h,J,beta,eps,a", _make_symbolic_gate_cases())
def test_glauber_arithmetic_symbolic_bounds_actual_counters(n, h, J, beta, eps, a):
    """
    Symbolic resource-bound check against static circuit counters.

    This appends the full non-mocked circuit with mocked angles to a QuantumCircuit
    and compares symbolic resource formulas against the existing static counters.
    It does not execute the circuit and does not extract a dense unitary.

    The parameter a changes only the GQSP polynomial degree, so the resource
    formulas must scale through degree while preserving the same delta-Hamiltonian
    term-count structure.
    """
    gate = _make_gate(n, h, J, beta, eps, a, mocked_circuit=False, mocked_angles=True)
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))

    n_terms_z, n_terms_zz = _active_counts(h, J, gate.tol)

    sym_num_qubits = _as_int(glauber_arithmetic_number_qubits(n, gate.alpha, beta, eps, n_terms_z, n_terms_zz, a))
    sym_nc_depth = _as_int(glauber_arithmetic_nc_depth(n, gate.alpha, beta, eps, n_terms_z, n_terms_zz, a))
    sym_t_count = _as_int(glauber_arithmetic_t_count(n, gate.alpha, beta, eps, n_terms_z, n_terms_zz, a))
    sym_rz_count = _as_int(glauber_arithmetic_rz_count(n, gate.alpha, beta, eps, n_terms_z, n_terms_zz, a))
    sym_toffoli_count = _as_int(glauber_arithmetic_toffoli_count(n, gate.alpha, beta, eps, n_terms_z, n_terms_zz, a))

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


@pytest.mark.parametrize("n,alpha,beta,eps,n_terms_z,n_terms_zz,a", _make_large_symbolic_cases())
def test_glauber_arithmetic_large_symbolic_formulas_are_well_formed(n, alpha, beta, eps, n_terms_z, n_terms_zz, a):
    """
    Large symbolic-only resource sanity check.

    This evaluates the symbolic formulas for larger systems and multiple values
    of a without constructing or executing any circuit. The goal is to catch
    invalid expressions, non-real values, negative resources, or accidental
    dependence on dense simulation.
    """
    resources = [
        glauber_arithmetic_degree(alpha, beta, eps, a),
        glauber_arithmetic_number_qubits(n, alpha, beta, eps, n_terms_z, n_terms_zz, a),
        glauber_arithmetic_nc_depth(n, alpha, beta, eps, n_terms_z, n_terms_zz, a),
        glauber_arithmetic_t_count(n, alpha, beta, eps, n_terms_z, n_terms_zz, a),
        glauber_arithmetic_rz_count(n, alpha, beta, eps, n_terms_z, n_terms_zz, a),
        glauber_arithmetic_toffoli_count(n, alpha, beta, eps, n_terms_z, n_terms_zz, a),
    ]

    values = [_as_int(resource) for resource in resources]

    assert all(value >= 0 for value in values)
    assert values[0] >= 1
    assert values[1] > 2 * n
    assert values[2] > 0
    assert values[4] > 0
    assert values[5] > 0