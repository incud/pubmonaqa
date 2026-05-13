import math

import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit

from monaqa2.qiskit.sqrt_exp_arithmetic_gate import SqrtExpArithmetic
from monaqa2.qiskit.sqrt_exp_arithmetic_symbolic import (
    sqrt_exp_arithmetic_alpha_signal,
    sqrt_exp_arithmetic_lambda,
    sqrt_exp_arithmetic_mu,
    sqrt_exp_arithmetic_degree,
    sqrt_exp_arithmetic_n_terms_z,
    sqrt_exp_arithmetic_number_qubits,
    sqrt_exp_arithmetic_nc_depth,
    sqrt_exp_arithmetic_t_count,
    sqrt_exp_arithmetic_rz_count,
    sqrt_exp_arithmetic_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_nc_depth,
    get_t_count,
    get_rz_count,
    get_toffoli_count,
)


def _make_symbolic_cases():
    return [
        (1, 0.10, 0.50, 1e-2),
        (1, 0.25, 0.75, 1e-4),
        (2, 0.25, 0.75, 1e-4),
        (3, 1.10, 1.12, 1e-8),
        (4, 1.00, 1.00, 1e-8),
        (5, 0.75, 1.50, 1e-10),
        (6, 0.60, 1.25, 1e-8),
        (8, 0.50, 2.00, 1e-12),
        (16, 0.25, 4.00, 1e-16),
        (32, 0.10, 8.00, 1e-18),
        (64, 0.05, 16.00, 1e-20),
        (128, 0.025, 32.00, 1e-24),
    ]


def _make_instantiation_cases():
    # b = 1 has only one active Z term, and the non-mocked unary LCU requires
    # at least two active terms. Keep b = 1 in formula-only tests.
    return [case for case in _make_symbolic_cases() if case[0] >= 2]


def _make_counter_cases():
    # Static resource counters are cheap compared with unitary simulation, but
    # keep this set moderate because it still expands composite gate structure.
    return [
        (2, 0.25, 0.75, 1e-4),
        (3, 1.10, 1.12, 1e-8),
        (4, 1.00, 1.00, 1e-8),
        (5, 0.75, 1.50, 1e-10),
        (6, 0.60, 1.25, 1e-8),
        (8, 0.50, 2.00, 1e-12),
    ]


def _as_float(expr) -> float:
    value = complex(sp.N(expr))
    assert abs(value.imag) <= 1e-9
    return float(value.real)


def _as_int(expr) -> int:
    value = _as_float(expr)
    assert np.isfinite(value)
    assert abs(value - round(value)) <= 1e-9
    return int(round(value))


def _expected_active_z_terms(b: int, tol: float = 1e-8) -> int:
    """
    Active Z terms in SqrtExpArithmetic after LcuUnaryIsingTF pruning.

    h[-1] = 1/2 is always active.
    For j = 0, ..., b - 2, the coefficient is 2^{j-b}.
    It is active iff 2^{j-b} > tol.
    """
    j_min = max(0, math.floor(b + math.log2(tol)) + 1)
    return 1 + max(0, (b - 1) - j_min)


@pytest.mark.parametrize("b,beta,normalization,eps", _make_symbolic_cases())
def test_sqrt_exp_arithmetic_symbolic_scalars(b, beta, normalization, eps):
    """
    Formula-only scalar check.

    This test does not instantiate the gate. It is valid even for b = 1, where
    the non-mocked unary LCU circuit is intentionally not constructible because
    it has only one active term.

    It checks:
      * alpha = 1 - 2^{-b};
      * lambda = beta * normalization;
      * mu = lambda * alpha;
      * degree = ceil(e * mu + log(4 / eps) + 1);
      * active Z-term count after the 1e-8 LCU pruning threshold.
    """
    alpha = sqrt_exp_arithmetic_alpha_signal(b)
    lam = sqrt_exp_arithmetic_lambda(beta, normalization)
    mu = sqrt_exp_arithmetic_mu(b, beta, normalization)
    degree = sqrt_exp_arithmetic_degree(b, beta, normalization, eps)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)

    expected_alpha = 1.0 - 2.0 ** (-b)
    expected_lambda = beta * normalization
    expected_mu = expected_lambda * expected_alpha
    expected_degree = int(np.ceil(np.e * expected_mu + np.log(4.0 / eps) + 1.0))

    np.testing.assert_allclose(_as_float(alpha), expected_alpha, atol=1e-12)
    np.testing.assert_allclose(_as_float(lam), expected_lambda, atol=1e-12)
    np.testing.assert_allclose(_as_float(mu), expected_mu, atol=1e-12)

    assert _as_int(degree) == expected_degree
    assert _as_int(n_terms_z) == _expected_active_z_terms(b)


@pytest.mark.parametrize("b,beta,normalization,eps", _make_symbolic_cases())
def test_sqrt_exp_arithmetic_symbolic_resources_are_well_formed(b, beta, normalization, eps):
    """
    Formula-only resource sanity check.

    This evaluates all symbolic resource methods, including b = 1 and large b,
    without constructing or executing any circuit. The goal is to catch invalid
    expressions such as zoo/nan, negative resources, or non-real symbolic values.
    """
    resources = [
        sqrt_exp_arithmetic_number_qubits(b, beta, normalization, eps),
        sqrt_exp_arithmetic_nc_depth(b, beta, normalization, eps),
        sqrt_exp_arithmetic_t_count(b, beta, normalization, eps),
        sqrt_exp_arithmetic_rz_count(b, beta, normalization, eps),
        sqrt_exp_arithmetic_toffoli_count(b, beta, normalization, eps),
    ]

    values = [_as_int(resource) for resource in resources]

    assert all(value >= 0 for value in values)
    assert values[0] > b
    assert values[1] >= 0
    assert values[2] >= 0
    assert values[3] >= 0
    assert values[4] >= 0


@pytest.mark.parametrize("b,beta,normalization,eps", _make_instantiation_cases())
def test_sqrt_exp_arithmetic_symbolic_static_values_match_gate(b, beta, normalization, eps):
    """
    Static gate-layout check.

    This instantiates SqrtExpArithmetic with mocked GQSP angles only. It does
    not simulate the circuit or extract the unitary.

    It checks:
      * symbolic degree equals the class degree;
      * symbolic active Z-term count equals the actual active LCU term count;
      * symbolic qubit count equals the actual gate qubit count.

    The arithmetic Hamiltonian is Z-only, so the refined symbolic formulas must
    use n_terms_z = active_terms, n_terms_zz = 0, n_terms_x = 0.
    """
    gate = SqrtExpArithmetic(b, beta, normalization, eps, mocked_circuit=False, mocked_angles=True)

    degree = sqrt_exp_arithmetic_degree(b, beta, normalization, eps)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)
    num_qubits = sqrt_exp_arithmetic_number_qubits(b, beta, normalization, eps)

    assert _as_int(degree) == gate.degree
    assert _as_int(n_terms_z) == gate.qubitization.lcu.n_terms
    assert _as_int(n_terms_z) == gate.controlled_qubitization.lcu.n_terms
    assert _as_int(num_qubits) == gate.num_qubits


@pytest.mark.parametrize("b,beta,normalization,eps", _make_counter_cases())
def test_sqrt_exp_arithmetic_symbolic_bounds_actual_counters(b, beta, normalization, eps):
    """
    Static circuit-counter check.

    This appends the gate to a QuantumCircuit and runs the existing static
    resource counters. It does not execute the circuit and does not extract any
    dense unitary.

    Since SqrtExpArithmetic has only Z terms in its Ising-TF Hamiltonian, the
    controlled SELECT symbolic depth should use the O(1) Z-layer case, not the
    generic O(n) ZZ-matching case.
    """
    gate = SqrtExpArithmetic(b, beta, normalization, eps, mocked_circuit=False, mocked_angles=True)
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, list(range(gate.num_qubits)))

    sym_num_qubits = _as_int(sqrt_exp_arithmetic_number_qubits(b, beta, normalization, eps))
    sym_nc_depth = _as_int(sqrt_exp_arithmetic_nc_depth(b, beta, normalization, eps))
    sym_t_count = _as_int(sqrt_exp_arithmetic_t_count(b, beta, normalization, eps))
    sym_rz_count = _as_int(sqrt_exp_arithmetic_rz_count(b, beta, normalization, eps))
    sym_toffoli_count = _as_int(sqrt_exp_arithmetic_toffoli_count(b, beta, normalization, eps))

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


@pytest.mark.parametrize("b,beta,normalization,eps", _make_instantiation_cases())
def test_sqrt_exp_arithmetic_symbolic_mocked_circuit_qubits_are_no_larger(b, beta, normalization, eps):
    """
    Compact-circuit sanity check.

    mocked_circuit=True uses the compact binary LCU path in both qubitized
    operators. This test only checks the direction one expects from that mode:
    it should not use more qubits than the full non-mocked circuit.

    The symbolic methods in this file model the full non-mocked circuit, so this
    test intentionally compares mocked_circuit against the symbolic full-circuit
    qubit count as an upper bound.
    """
    compact_gate = SqrtExpArithmetic(b, beta, normalization, eps, mocked_circuit=True, mocked_angles=True)
    full_num_qubits = _as_int(sqrt_exp_arithmetic_number_qubits(b, beta, normalization, eps))

    assert compact_gate.num_qubits <= full_num_qubits