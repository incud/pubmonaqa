import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.proposal_local_gate import ProposalLocal
from monaqa2.qiskit.proposal_local_symbolic import (
    proposal_local_number_qubits,
    proposal_local_nc_depth,
    proposal_local_t_count,
    proposal_local_rz_count,
    proposal_local_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def expected_proposal_local_state(n: int, k: int, x: int) -> np.ndarray:
    psi = np.zeros(2 ** (2 * n), dtype=complex)
    normalization = 1.0 / np.sqrt(math.comb(n, k))

    for y in range(2**n):
        if hamming_distance(x, y) == k:
            index = x + (y << n)
            psi[index] = normalization

    return psi


@pytest.mark.parametrize("n,k", [
    (2, 1),
    (3, 1), (3, 2),
    (4, 1), (4, 2), (4, 3),
])
def test_proposal_local_evolution(n: int, k: int):
    """
    Test that ProposalLocal maps |x>|0> to the uniform superposition over
    |x>|y> with Hamming distance(x, y) = k.
    """
    proposal = ProposalLocal(n, k)

    for x in range(2**n):
        qc = QuantumCircuit(2 * n)
        for bit in range(n):
            if (x >> bit) & 1:
                qc.x(bit)

        qc.append(proposal, list(range(2 * n)))

        sv_qiskit = Statevector.from_instruction(qc).data
        sv_expected = expected_proposal_local_state(n, k, x)
        np.testing.assert_allclose(sv_qiskit, sv_expected, atol=1e-10)


@pytest.mark.parametrize("n,k", [
    (2, 1),
    (3, 1), (3, 2),
    (4, 1), (4, 2), (4, 3),
])
def test_proposal_local_symbolic_bounds(n: int, k: int):
    """
    Test that ProposalLocal symbolic bounds upper bound the actual circuit costs.
    """
    proposal = ProposalLocal(n, k)
    qc = QuantumCircuit(2 * n)
    qc.append(proposal, list(range(2 * n)))

    sym_num_qubits = int(proposal_local_number_qubits(n, k))
    sym_nc_depth = int(proposal_local_nc_depth(n, k))
    sym_t_count = int(proposal_local_t_count(n, k))
    sym_rz_count = int(proposal_local_rz_count(n, k))
    sym_toffoli_count = int(proposal_local_toffoli_count(n, k))

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
