import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.dicke_preparation_gate import DickePreparation
from monaqa2.qiskit.dicke_preparation_symbolic import (
    dicke_preparation_number_qubits,
    dicke_preparation_nc_depth,
    dicke_preparation_t_count,
    dicke_preparation_rz_count,
    dicke_preparation_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import (
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)


def dicke_state_vector(n: int, k: int) -> np.ndarray:
    """
    Construct the Dicke state D(n, k) as a state vector.
    
    D(n, k) is the uniform superposition of all n-qubit states with Hamming weight k.
    Uses little-endian qubit ordering (as per Qiskit convention).
    
    :param n: Number of qubits.
    :param k: Hamming weight.
    :return: State vector of shape (2^n,) representing D(n, k).
    """
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")
    
    psi = np.zeros(2**n, dtype=complex)
    normalization = 1.0 / np.sqrt(math.comb(n, k))
    
    # Iterate over all basis states
    for i in range(2**n):
        # Count number of ones in binary representation (Hamming weight)
        if bin(i).count('1') == k:
            psi[i] = normalization
    
    return psi


@pytest.mark.parametrize("n,k", [
    (2, 1),
    (3, 1), (3, 2),
    (4, 1), (4, 2), (4, 3),
    (5, 1), (5, 2), (5, 3), (5, 4),
])
def test_dicke_preparation_unitary(n: int, k: int):
    """
    Test DickePreparation produces correct Dicke state when applied to |0...0>.
    
    The output state should be D(n, k), the uniform superposition of all
    n-qubit states with Hamming weight k.
    """
    dicke = DickePreparation(n, k)
    qc = QuantumCircuit(n)
    qc.append(dicke, list(range(n)))
    
    # Get the output state vector
    sv_qiskit = Statevector.from_instruction(qc).data
    
    # Construct expected Dicke state
    sv_expected = dicke_state_vector(n, k)
    
    # Compare (Qiskit uses little-endian)
    np.testing.assert_allclose(sv_qiskit, sv_expected, atol=1e-10)


@pytest.mark.parametrize("n,k", [
    (2, 1),
    (3, 1), (3, 2),
    (4, 1), (4, 2), (4, 3),
    (5, 1), (5, 2), (5, 3), (5, 4),
])
def test_dicke_preparation_symbolic_bounds(n: int, k: int):
    """
    Test that DickePreparation symbolic bounds upper bound the actual circuit costs.
    """
    dicke = DickePreparation(n, k)
    qc = QuantumCircuit(n)
    qc.append(dicke, list(range(n)))
    
    sym_num_qubits = int(dicke_preparation_number_qubits(n, k))
    sym_nc_depth = int(dicke_preparation_nc_depth(n, k))
    sym_t_count = int(dicke_preparation_t_count(n, k))
    sym_rz_count = int(dicke_preparation_rz_count(n, k))
    sym_toffoli_count = int(dicke_preparation_toffoli_count(n, k))
    
    actual_num_qubits = qc.num_qubits
    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)
    
    assert sym_num_qubits >= actual_num_qubits
    assert sym_nc_depth >= actual_nc_depth
    assert sym_t_count >= actual_t_count
    assert sym_toffoli_count >= actual_toffoli_count
    assert sym_rz_count >= actual_rz_count
