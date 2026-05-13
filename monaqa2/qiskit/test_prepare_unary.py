import pytest
import numpy as np
from qiskit import QuantumCircuit
from monaqa2.qiskit.prepare_unary_gate import PrepareUnary
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
from monaqa2.qiskit.prepare_unary_symbolic import (
    prepare_unary_number_qubits,
    prepare_unary_nc_depth,
    prepare_unary_t_count,
    prepare_unary_rz_count,
    prepare_unary_toffoli_count
)


def _make_coeffs() -> list[np.ndarray]:
    return [
        np.array([1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 3.0]),
        np.array([1.0, 0.0, 3.0]),
        np.array([1.0, 2.0, 0.0, 4.0]),
        np.array([0.5, 1.5, 2.0, 3.0, 0.0]),
    ]


@pytest.mark.parametrize("coeffs", _make_coeffs())
def test_prepare_unary_evolution(coeffs: np.ndarray):

    # Instantiate
    prepare = PrepareUnary(coeffs)
    qc = QuantumCircuit(prepare.num_qubits)
    qc.append(prepare, list(range(prepare.num_qubits)))

    # Evolve unitary by ket zero (note we enforce big endian)
    U_qiskit = get_unitary(qc, big_endian=True)
    actual_vector = U_qiskit @ ket(0, prepare.num_qubits)

    n_tree = len(prepare.layout["tree"])
    n_terms = len(prepare.layout["leaf"])

    tree_vec = ket(0, n_tree)
    coeffs_vec = np.zeros_like(ket(0, n_terms), dtype=complex)

    for l, coeff in enumerate(prepare.coeffs):
        amp = np.sqrt(coeff / prepare.alpha)

        # Big-endian one-hot convention:
        # l = 0 -> |100...0>
        # l = 1 -> |010...0>
        # ...
        basis_index = 1 << (n_terms - 1 - l)
        coeffs_vec += amp * ket(basis_index, n_terms)

    expected_vector = kron(tree_vec, coeffs_vec)

    np.testing.assert_allclose(actual_vector, expected_vector, atol=1e-10)
    

@pytest.mark.parametrize("coeffs", _make_coeffs())
def test_prepare_symbolic_bounds(coeffs):
    """Test that ProposalUniform symbolic bounds upper bound the actual circuit costs."""
    
    
    prepare = PrepareUnary(coeffs=coeffs)
    qc = QuantumCircuit(prepare.num_qubits)
    qc.append(prepare, list(range(prepare.num_qubits)))

    n = len(coeffs)
    sym_num_qubits = prepare_unary_number_qubits(n)
    sym_nc_depth = prepare_unary_nc_depth(n)
    sym_t_count = prepare_unary_t_count(n)
    sym_rz_count = prepare_unary_rz_count(n)
    sym_toffoli_count = prepare_unary_toffoli_count(n)

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
