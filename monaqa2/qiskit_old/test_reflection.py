import numpy as np
import pytest
from qiskit import QuantumCircuit

from monaqa2.qiskit.reflection_gate import Reflection
from monaqa2.qiskit.reflection_symbolic import (
    reflection_number_qubits,
    reflection_nc_depth,
    reflection_t_count,
    reflection_rz_count,
    reflection_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import get_unitary, get_nc_depth, get_t_count, get_toffoli_count, get_rz_count
from monaqa2.qiskit.utils_numpy import kron, ket, bra, ketbra


@pytest.mark.parametrize("n, coins", [(2, 2), (3, 2), (3, 2), (3, 3)])
def test_reflection_unitary(n, coins):
    """Test Reflection unitary against the ideal reflection on B+coins."""
    reflection = Reflection(n=n, coins=coins)
    qc = QuantumCircuit(reflection.num_qubits)
    qc.append(reflection, list(range(reflection.num_qubits)))

    U_qiskit = get_unitary(qc, big_endian=True)

    # Project work ancillas into |0...0> before and after the circuit.
    ancillas = reflection.num_qubits - (2 * n + coins)
    active_dim = 2 ** (2 * n + coins)
    U_actual = kron(np.eye(active_dim, dtype=complex), bra(0, ancillas)) @ U_qiskit @ kron(np.eye(active_dim, dtype=complex), ket(0, ancillas))

    # Expected reflection: I_A ⊗ (2|0...0><0...0| - I)_{B+coins}
    expected_reflection = kron(
        np.eye(2**n, dtype=complex),
        2 * ketbra(0, n + coins) - np.eye(2**(n + coins), dtype=complex),
    )

    np.testing.assert_allclose(U_actual, expected_reflection, atol=1e-10)


@pytest.mark.parametrize("n, coins", [(2, 2), (3, 2), (3, 2), (3, 3)])
def test_reflection_symbolic_bounds(n, coins):
    """Test that Reflection symbolic bounds upper bound the actual circuit costs."""
    reflection = Reflection(n=n, coins=coins)
    qc = QuantumCircuit(reflection.num_qubits)
    qc.append(reflection, list(range(reflection.num_qubits)))

    sym_num_qubits = reflection_number_qubits(n, coins)
    sym_nc_depth = reflection_nc_depth(n, coins)
    sym_t_count = reflection_t_count(n, coins)
    sym_rz_count = reflection_rz_count(n, coins)
    sym_toffoli_count = reflection_toffoli_count(n, coins)

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
