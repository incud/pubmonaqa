import numpy as np
import pytest
from qiskit import QuantumCircuit

from monaqa2.qiskit.proposal_uniform_gate import ProposalUniform
from monaqa2.qiskit.proposal_uniform_symbolic import (
    proposal_uniform_number_qubits,
    proposal_uniform_nc_depth,
    proposal_uniform_t_count,
    proposal_uniform_rz_count,
    proposal_uniform_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)
from monaqa2.qiskit.utils_numpy import kron


@pytest.mark.parametrize("n", [1, 2, 3])
def test_proposal_uniform_unitary(n):
    """Test ProposalUniform unitary matches I_A \\otimes H_B^{\\otimes n}."""
    proposal = ProposalUniform(n=n)
    qc = QuantumCircuit(proposal.num_qubits)
    qc.append(proposal, list(range(proposal.num_qubits)))

    U_qiskit = get_unitary(qc, big_endian=True)

    H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    H_n = kron(*([H] * n))
    U_expected = kron(np.eye(2**n, dtype=complex), H_n)

    np.testing.assert_allclose(U_qiskit, U_expected, atol=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_proposal_uniform_symbolic_bounds(n):
    """Test that ProposalUniform symbolic bounds upper bound the actual circuit costs."""
    proposal = ProposalUniform(n=n)
    qc = QuantumCircuit(proposal.num_qubits)
    qc.append(proposal, list(range(proposal.num_qubits)))

    sym_num_qubits = proposal_uniform_number_qubits(n)
    sym_nc_depth = proposal_uniform_nc_depth(n)
    sym_t_count = proposal_uniform_t_count(n)
    sym_rz_count = proposal_uniform_rz_count(n)
    sym_toffoli_count = proposal_uniform_toffoli_count(n)

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
