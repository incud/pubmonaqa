import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit

from monaqa2.qiskit.multi_controlled_not_gate import MultiControlledNot
from monaqa2.qiskit.multi_controlled_not_symbolic import (
    mcx_number_qubits,
    mcx_nc_depth,
    mcx_t_count,
    mcx_rz_count,
    mcx_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)
from monaqa2.qiskit.utils_numpy import (
    kron, ket, bra, ketbra, X, I,
)


@pytest.mark.parametrize("controls", [2, 3, 4, 5])
def test_mcx_unitary(controls):
    """Test that MultiControlledNot unitary matches the expected MCX definition."""
    mcx = MultiControlledNot(controls)
    qc = QuantumCircuit(mcx.num_qubits)
    qc.append(mcx, list(range(mcx.num_qubits)))

    # Get unitary from Qiskit
    U_qiskit = get_unitary(qc, big_endian=True)

    # Reduce ancilla subsystem by projecting it into |0...0>
    ancillas = mcx.num_qubits - controls - 1
    I_ct = np.eye(2**(controls + 1))
    U_actual = kron(I_ct, bra(0, ancillas)) @ U_qiskit @ kron(I_ct, ket(0, ancillas))

    # Calculate expected unitary on controls + target only
    c = controls
    U_expected = kron(ketbra(2**c - 1, c), X) + kron(np.eye(2**c) - ketbra(2**c - 1, c), I)

    np.testing.assert_allclose(U_actual, U_expected, atol=1e-10)


@pytest.mark.parametrize("controls", [2, 3, 4, 5])
def test_symbolic_bounds(controls):
    """Test that symbolic methods upper bound the actual quantities."""
    mcx = MultiControlledNot(controls)
    qc = QuantumCircuit(mcx.num_qubits)
    qc.append(mcx, list(range(mcx.num_qubits)))

    # Evaluate symbolic expressions
    sym_num_qubits = mcx_number_qubits(controls)
    sym_nc_depth = mcx_nc_depth(controls)
    sym_t_count = mcx_t_count(controls)
    sym_rz_count = mcx_rz_count(controls)
    sym_toffoli_count = mcx_toffoli_count(controls)

    # Get actual quantities
    actual_num_qubits = qc.num_qubits
    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)

    # Check upper bounds
    assert sym_num_qubits >= actual_num_qubits
    assert sym_nc_depth >= actual_nc_depth
    assert sym_t_count >= actual_t_count
    assert sym_rz_count >= actual_rz_count
    assert sym_toffoli_count >= actual_toffoli_count
