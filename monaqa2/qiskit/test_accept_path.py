import numpy as np
import pytest
from qiskit import QuantumCircuit

from monaqa2.qiskit.accept_path_gate import AcceptPath
from monaqa2.qiskit.accept_path_symbolic import (
    accept_path_number_qubits,
    accept_path_nc_depth,
    accept_path_t_count,
    accept_path_rz_count,
    accept_path_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import get_unitary, get_nc_depth, get_t_count, get_toffoli_count, get_rz_count
from monaqa2.qiskit.utils_numpy import kron, ket, bra, SWAP, apply_unitary, sequence


@pytest.mark.parametrize("n, coins", [(2, 3), (3, 3), (3, 4)])
def test_accept_path_unitary(n, coins):
    """Test AcceptPath unitary against the ideal conditional swap on A and B if coins == 0."""
    accept_path = AcceptPath(n=n, coins=coins)
    qc = QuantumCircuit(accept_path.num_qubits)
    qc.append(accept_path, list(range(accept_path.num_qubits)))

    U_qiskit = get_unitary(qc, big_endian=True)

    # Project work ancillas into |0...0> before and after the circuit.
    ancillas = accept_path.num_qubits - (2 * n + coins)
    active_dim = 2 ** (2 * n + coins)
    U_actual = kron(np.eye(active_dim, dtype=complex), bra(0, ancillas)) @ U_qiskit @ kron(np.eye(active_dim, dtype=complex), ket(0, ancillas))

    # Expected: conditional swap of A and B if coins == 0.
    # F = (SWAP_{AB} \otimes |0..0><0..0|_C) + (I_A \otimes I_B \otimes (I - |0...0><0..0|)_C)
    I_A = np.eye(2**n)
    I_B = np.eye(2**n)
    I_C = np.eye(2**coins)
    P_C = ket(0, coins) @ bra(0, coins)
    SWAP_n = sequence(*(apply_unitary(SWAP, [a, a + n], 2 * n) for a in range(n)))
    U_expected = kron(SWAP_n, P_C) + kron(I_A, I_B, I_C - P_C)

    np.testing.assert_allclose(U_actual, U_expected, atol=1e-10)


@pytest.mark.parametrize("n, coins", [(2, 3), (3, 3), (3, 4)])
def test_accept_path_symbolic_bounds(n, coins):
    """Test that AcceptPath symbolic bounds upper bound the actual circuit costs."""
    accept_path = AcceptPath(n=n, coins=coins)
    qc = QuantumCircuit(accept_path.num_qubits)
    qc.append(accept_path, list(range(accept_path.num_qubits)))

    sym_num_qubits = accept_path_number_qubits(n, coins)
    sym_nc_depth = accept_path_nc_depth(n, coins)
    sym_t_count = accept_path_t_count(n, coins)
    sym_rz_count = accept_path_rz_count(n, coins)
    sym_toffoli_count = accept_path_toffoli_count(n, coins)

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
