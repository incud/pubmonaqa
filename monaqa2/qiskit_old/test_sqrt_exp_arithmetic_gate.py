import pytest
import numpy as np
import scipy.linalg as la
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.sqrt_exp_arithmetic_gate import SqrtExpArithmetic
from monaqa2.qiskit.utils_qiskit import get_unitary
from monaqa2.qiskit.utils_numpy import kron, ket, bra


def _make_cases():
    return [
        (2, 0.25, 0.75, 1e-4),
        (2, 0.40, 1.25, 1e-4),
        (3, 1.1, 1.12, 1e-8),
        (4, 1.0, 1.0, 1e-8),
    ]


def _make_matrix_cases():
    return _make_cases()


def _basis_indices(b: int) -> list[int]:
    if b <= 4:
        return list(range(2**b))

    return sorted(set([0, 1, 2, 2 ** (b - 1) - 1, 2 ** (b - 1), 2**b - 2, 2**b - 1]))


def _make_gate(b: int, beta: float, normalization: float, eps: float) -> SqrtExpArithmetic:
    return SqrtExpArithmetic(b, beta, normalization, eps, mocked_circuit=True, mocked_angles=False)


def _matrix_poly(coeffs: np.ndarray, W: np.ndarray) -> np.ndarray:
    out = np.zeros_like(W, dtype=complex)
    power = np.eye(W.shape[0], dtype=complex)

    for coeff in coeffs:
        out += coeff * power
        power = power @ W

    return out


def _signal_max_value(b: int) -> float:
    # even though actually the biggest value is 1 - 2^{1-b},
    # pretending to go slighlty higher allows us to be sure to
    # satify the constrains of GQSP
    return 1.0 + 2.0 ** (1 - b)


def _hz_diag(b: int) -> np.ndarray:
    diag = []

    for basis_index in range(2**b):
        bits = np.array([int(c) for c in np.binary_repr(basis_index, width=b)], dtype=int)
        z = 1 - 2 * bits
        value = 0.5 * z[-1] - sum((2.0 ** (j - b)) * z[j] for j in range(b - 1))
        diag.append(value)

    return np.array(diag, dtype=float)


def _target_signal_matrix(b: int, beta: float, normalization: float) -> np.ndarray:
    lam = beta * normalization
    shift = 2.0 ** (-b) + _signal_max_value(b)
    diag = np.exp(lam * (_hz_diag(b) - shift))
    return np.diag(diag.astype(complex))


def _project_control_work(U: np.ndarray, gate: SqrtExpArithmetic) -> np.ndarray:
    n_extra = gate.num_qubits - 1 - gate.qubitization.num_qubits
    dim_control_w = 2 ** (1 + gate.qubitization.num_qubits)

    if n_extra == 0:
        return U

    left = kron(np.eye(dim_control_w, dtype=complex), bra(0, n_extra))
    right = kron(np.eye(dim_control_w, dtype=complex), ket(0, n_extra))
    return left @ U @ right


def _control_zero_block(U: np.ndarray, gate: SqrtExpArithmetic) -> np.ndarray:
    dim_w = 2 ** gate.qubitization.num_qubits
    return kron(bra(0, 1), np.eye(dim_w, dtype=complex)) @ U @ kron(ket(0, 1), np.eye(dim_w, dtype=complex))


def _signal_good_block(U_walk: np.ndarray, gate: SqrtExpArithmetic) -> np.ndarray:
    n_signal = gate.b
    n_aux = gate.qubitization.num_qubits - n_signal
    dim_signal = 2**n_signal

    left = kron(np.eye(dim_signal, dtype=complex), bra(0, n_aux))
    right = kron(np.eye(dim_signal, dtype=complex), ket(0, n_aux))
    return left @ U_walk @ right


def _bits_from_basis_index(basis_index: int, b: int) -> np.ndarray:
    return np.array([int(c) for c in np.binary_repr(basis_index, width=b)], dtype=int)


def _qiskit_index(signal_qubits: list[int], bits: np.ndarray) -> int:
    return sum(int(bit) << int(q) for bit, q in zip(bits, signal_qubits))


def _same_signal_norm(state: np.ndarray, signal_qubits: list[int], bits: np.ndarray) -> float:
    total = 0.0

    for idx, amp in enumerate(state):
        if all(((idx >> q) & 1) == int(bit) for bit, q in zip(bits, signal_qubits)):
            total += abs(amp) ** 2

    return float(np.sqrt(total))


@pytest.mark.parametrize("b,beta,normalization,eps", _make_cases())
def test_sqrt_exp_arithmetic_signal_target_diagonal(b, beta, normalization, eps):
    """
    Structural sanity check.

    This verifies that SqrtExpArithmetic encodes the intended diagonal signal
    Hamiltonian

        H_z = 1/2 Z_sign - sum_j 2^{j-b} Z_j,

    with no ZZ terms and no X terms. It also fixes the deliberately enlarged
    reference value

        y_ref_max = 1 + 2^{1-b}.

    This test does not check GQSP, amplitudes, or the final proposal map.
    """
    gate = _make_gate(b, beta, normalization, eps)

    expected_h = np.zeros(b, dtype=float)
    expected_h[-1] = 0.5

    for j in range(b - 1):
        expected_h[j] = -(2.0 ** (j - b))

    np.testing.assert_allclose(gate.h, expected_h, atol=1e-12)
    np.testing.assert_allclose(gate.J, np.zeros((b, b)), atol=1e-12)
    np.testing.assert_allclose(gate.gamma, np.zeros(b), atol=1e-12)
    np.testing.assert_allclose(gate._signal_max_value(), 1.0 + 2.0 ** (1 - b), atol=1e-12)


@pytest.mark.parametrize("b,beta,normalization,eps", _make_cases())
def test_sqrt_exp_arithmetic_coefficients_are_bounded(b, beta, normalization, eps):
    """
    Polynomial sanity check.

    This verifies that the truncated shifted Laurent polynomial used by GQSP has
    the expected size, finite coefficients, and sup coefficient scale compatible
    with a bounded amplitude transform.

    This does not check that the circuit realizes the polynomial.
    """
    gate = _make_gate(b, beta, normalization, eps)

    assert gate.degree >= 1
    assert gate.poly_coeffs.shape == (2 * gate.degree + 1,)
    assert np.all(np.isfinite(gate.poly_coeffs))
    assert np.max(np.abs(gate.poly_coeffs)) <= 1.0 + 1e-12


@pytest.mark.parametrize("b,beta,normalization,eps", _make_matrix_cases())
def test_sqrt_exp_arithmetic_polynomial_block(b, beta, normalization, eps):
    """
    GQSP realization check.

    This extracts the GQSP good-control block and verifies that it equals

        W^{-degree} P_shifted(W),

    where W is the qubitized signal walk and P_shifted is the ordinary
    polynomial obtained by shifting the Laurent series.

    This checks the GQSP/qubitization composition, but not yet the projected
    signal-register scalar function.
    """
    gate = _make_gate(b, beta, normalization, eps)

    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))
    U = _project_control_work(get_unitary(qc, big_endian=True), gate)

    qc_w = QuantumCircuit(gate.qubitization.num_qubits)
    qc_w.append(gate.qubitization, list(range(gate.qubitization.num_qubits)))
    W = get_unitary(qc_w, big_endian=True)

    actual = _control_zero_block(U, gate)
    expected = np.linalg.matrix_power(W, -gate.degree) @ _matrix_poly(gate.poly_coeffs, W)

    np.testing.assert_allclose(actual, expected, atol=max(500 * eps, 1e-7))


@pytest.mark.parametrize("b,beta,normalization,eps", _make_matrix_cases())
def test_sqrt_exp_arithmetic_matches_implemented_signal_target_matrix(b, beta, normalization, eps):
    """
    Principal signal-block check.

    This projects the GQSP control and qubitization ancillas to the all-zero
    good state, leaving only the signal register. The resulting diagonal block
    must equal

        exp(beta * normalization * (H_z - shift I)),

    with

        shift = 2^{-b} + y_ref_max,
        y_ref_max = 1 + 2^{1-b}.

    This is the matrix-level check of the implemented amplitude function.
    """
    gate = _make_gate(b, beta, normalization, eps)

    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))
    U = _project_control_work(get_unitary(qc, big_endian=True), gate)

    walk_block = _control_zero_block(U, gate)
    actual_signal = _signal_good_block(walk_block, gate)
    expected_signal = _target_signal_matrix(b, beta, normalization)

    assert la.norm(actual_signal - expected_signal, ord=2) <= max(500 * eps, 1e-7)


@pytest.mark.parametrize("b,beta,normalization,eps", _make_cases())
def test_sqrt_exp_arithmetic_basis_states_preserve_signal_statevector(b, beta, normalization, eps):
    """
    Basis-state good-branch amplitude check.

    For each selected computational-basis signal state |tilde{x}>, this checks
    that the circuit maps

        |tilde{x}> |0...0>_anc

    to

        |tilde{x}> (
            a(tilde{x}) |0...0>_anc
            + |bot_{tilde{x}}>_anc
        ),

    where

        a(tilde{x}) = exp(beta * normalization * (h_z(tilde{x}) - shift)).

    It checks three things:
      1. the good-ancilla amplitude is correct;
      2. the norm of the orthogonal branch is sqrt(1 - |a|^2);
      3. the signal register does not leak to a different basis value.

    It does not require the bad branch to be a clean single-qubit |1> state;
    the bad branch may occupy an arbitrary orthogonal ancilla state.
    """
    gate = _make_gate(b, beta, normalization, eps)
    signal_qubits = gate.layout["signal"]
    expected_signal = _target_signal_matrix(b, beta, normalization)

    max_amp_error = 0.0
    max_bot_norm_error = 0.0
    max_leakage = 0.0

    for basis_index in _basis_indices(b):
        bits = _bits_from_basis_index(basis_index, b)

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

        expected_amp = expected_signal[basis_index, basis_index]
        expected_bot_norm = np.sqrt(max(0.0, 1.0 - abs(expected_amp) ** 2))

        max_amp_error = max(max_amp_error, abs(realized_amp - expected_amp))
        max_bot_norm_error = max(max_bot_norm_error, abs(realized_bot_norm - expected_bot_norm))
        max_leakage = max(max_leakage, np.sqrt(max(0.0, 1.0 - same_signal_norm**2)))

    assert max_amp_error <= max(500 * eps, 1e-7)
    assert max_bot_norm_error <= max(500 * eps, 1e-7)
    assert max_leakage <= max(500 * eps, 1e-7)