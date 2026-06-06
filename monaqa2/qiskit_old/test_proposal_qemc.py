import math
import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from monaqa2.qiskit.proposal_qemc_gate import ProposalQemc
from monaqa2.qiskit.proposal_qemc_symbolic import (
    proposal_qemc_number_qubits,
    proposal_qemc_nc_depth,
    proposal_qemc_t_count,
    proposal_qemc_rz_count,
    proposal_qemc_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import get_nc_depth, get_t_count, get_rz_count, get_toffoli_count


def _make_unitary_cases():
    return [
        (1, np.array([0.40]), np.array([[0.0]]), np.array([0.25]), 0.10, 1e-3),
        (1, np.array([-0.30]), np.array([[0.0]]), np.array([0.15]), 0.07, 1e-3),
        (1, np.array([1.0]), np.array([[0.0]]), np.array([1.0]), 0.05, 1e-2),
        (1, np.array([-0.75]), np.array([[0.0]]), np.array([0.25]), 0.10, 1e-3),
    ]


def _make_resource_cases():
    return [
        (1, np.array([0.40]), np.array([[0.0]]), np.array([0.25]), 0.10, 2e-1),
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.08, 2e-1),
    ]


def _active_term_count(h: np.ndarray, J: np.ndarray, gamma: np.ndarray, tol: float = 1e-12) -> int:
    return int(np.count_nonzero(np.abs(h) > tol) + np.count_nonzero(np.abs(np.triu(J, k=1)) > tol) + np.count_nonzero(np.abs(gamma) > tol))


def _alpha(h: np.ndarray, J: np.ndarray, gamma: np.ndarray) -> float:
    return float(np.sum(np.abs(h)) + np.sum(np.abs(np.triu(J, k=1))) + np.sum(np.abs(gamma)))


def _as_upper_int(expr) -> int:
    value = complex(sp.N(expr))
    assert abs(value.imag) <= 1e-9
    real = float(value.real)
    assert np.isfinite(real)
    return int(math.ceil(real - 1e-12))


def _replace_bits(index: int, qubits: list[int], local_index: int) -> int:
    out = int(index)

    for local_q, global_q in enumerate(qubits):
        if (local_index >> local_q) & 1:
            out |= 1 << global_q
        else:
            out &= ~(1 << global_q)

    return out


def _extract_bits(index: int, qubits: list[int]) -> int:
    return sum(((index >> global_q) & 1) << local_q for local_q, global_q in enumerate(qubits))


def _copy_a_into_b_index(index: int, proposal: ProposalQemc) -> int:
    out = int(index)

    for a, b in zip(proposal.layout["A"], proposal.layout["B"]):
        if (index >> a) & 1:
            out ^= 1 << b

    return out


def _expected_hsim_qubits(proposal: ProposalQemc) -> list[int]:
    qargs = [None] * proposal.hsim.num_qubits
    system = [1 + q for q in proposal.hsim.qubitization.layout["system"]]

    for local_q, global_q in zip(system, proposal.layout["B"]):
        qargs[local_q] = global_q

    aux_iter = iter(proposal.layout["aux"])

    return [next(aux_iter) if q is None else q for q in qargs]


def _expected_proposal_qemc_unitary(proposal: ProposalQemc) -> np.ndarray:
    dim = 2**proposal.num_qubits
    hsim_qubits = _expected_hsim_qubits(proposal)
    hsim_u = Operator(proposal.hsim).data
    expected = np.zeros((dim, dim), dtype=complex)

    for col in range(dim):
        after_copy = _copy_a_into_b_index(col, proposal)
        local_in = _extract_bits(after_copy, hsim_qubits)

        for local_out in range(2**proposal.hsim.num_qubits):
            amp = hsim_u[local_out, local_in]

            if abs(amp) > 1e-14:
                row = _replace_bits(after_copy, hsim_qubits, local_out)
                expected[row, col] += amp

    return expected


@pytest.mark.parametrize("n,h,J,gamma,t,eps", _make_unitary_cases())
def test_proposal_qemc_unitary_matches_linear_algebra(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, t: float, eps: float) -> None:
    """
    Check the actual ProposalQemc unitary against a direct linear-algebra construction.

    The reference unitary is constructed as:
      1. the reversible copy layer B <- B xor A;
      2. the HamiltonianSimulationGQSP unitary embedded on B plus aux according to
         the expected ProposalQemc layout.

    This checks the ProposalQemc qubit mapping and composition without reusing
    ProposalQemc.definition as the reference.
    """
    proposal = ProposalQemc(n, h, J, gamma, t, eps, mocked_reflection=True, mocked_angles=False)

    actual = Operator(proposal).data
    expected = _expected_proposal_qemc_unitary(proposal)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("n,h,J,gamma,t,eps", _make_resource_cases())
def test_proposal_qemc_symbolic_bounds_non_mocked_circuit(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, t: float, eps: float) -> None:
    """
    Check symbolic upper bounds against the actual non-mocked ProposalQemc circuit.
    """
    proposal = ProposalQemc(n, h, J, gamma, t, eps, mocked_reflection=False, mocked_angles=True)
    qc = QuantumCircuit(proposal.num_qubits)
    qc.append(proposal, list(range(proposal.num_qubits)))

    n_terms = sp.Integer(_active_term_count(h, J, gamma))
    alpha = sp.Float(_alpha(h, J, gamma))
    kwargs = {"n": sp.Integer(n), "n_terms": n_terms, "alpha": alpha, "t": sp.Float(t), "eps": sp.Float(eps)}

    symbolic_num_qubits = _as_upper_int(proposal_qemc_number_qubits(**kwargs))
    symbolic_nc_depth = _as_upper_int(proposal_qemc_nc_depth(**kwargs))
    symbolic_t_count = _as_upper_int(proposal_qemc_t_count(**kwargs))
    symbolic_rz_count = _as_upper_int(proposal_qemc_rz_count(**kwargs))
    symbolic_toffoli_count = _as_upper_int(proposal_qemc_toffoli_count(**kwargs))

    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)

    assert proposal.num_qubits <= symbolic_num_qubits
    assert actual_nc_depth <= symbolic_nc_depth
    assert actual_t_count <= symbolic_t_count
    assert actual_rz_count <= symbolic_rz_count
    assert actual_toffoli_count <= symbolic_toffoli_count

