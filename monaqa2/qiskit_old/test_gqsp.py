from monaqa2.qiskit.qubitized_ising_tf_gate import ControlledQubitizedOperatorIsingTF, QubitizedOperatorIsingTF
import pytest
import numpy as np
from qiskit import QuantumCircuit

from monaqa2.qiskit.gqsp_gate import GQSP
from monaqa2.qiskit.gqsp_symbolic import (
    gqsp_toffoli_count,
    gqsp_rz_count,
    gqsp_t_count,
    gqsp_nc_depth,
    gqsp_number_qubits,
)
from monaqa2.qiskit.utils_qiskit import (
    get_unitary,
    get_nc_depth,
    get_t_count,
    get_toffoli_count,
    get_rz_count,
)
from monaqa2.qiskit.utils_numpy import kron, X, Z, I, ket, bra


def _poly_eval(coeffs: np.ndarray, z: complex) -> complex:
    out = 0.0j
    power = 1.0 + 0.0j

    for coeff in coeffs:
        out += coeff * power
        power *= z

    return out


def _make_gqsp_cases():
    return [
        (2, np.array([1.0, 0.0]), np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([0.0, 0.0]), np.array([0.1, 0.5]), 0),
        (2, np.array([1.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([0.25, 0.0]), np.array([0.1, 0.2, -0.05]), 1),
        (2, np.array([0.0, -1.0]), np.array([[0.0, -0.5], [0.0, 0.0]]), np.array([0.0, 0.0]), np.array([0.2, -0.3]), 0),
        (2, np.array([0.5, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([-0.75, 0.0]), np.array([0.05, 0.15, 0.1]), 1),
        (3, np.array([1.0, 0.0, 0.0]), np.array([[0.0, 0.0, -0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]), np.array([0.1, -0.2]), 0),
        (3, np.array([0.0, 0.0, -1.0]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, 0.0]]), np.array([0.25, 0.0, 0.0]), np.array([0.1, 0.2, -0.05]), 1),
        (3, np.array([0.0, 1.0, 0.0]), np.zeros((3, 3)), np.array([0.0, -0.5, 0.0]), np.array([0.05, -0.1, 0.15, -0.02]), 2),
        (4, np.array([1.0, 0.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]), np.zeros(4), np.array([0.1, 0.25]), 0),
    ]

@pytest.mark.parametrize("n,h,J,gamma,poly_coeffs,laurent_negative_power", _make_gqsp_cases())
def test_gqsp_polynomial_transforms_qubitized_eigenvalues(n, h, J, gamma, poly_coeffs, laurent_negative_power):
    
    qubitization = QubitizedOperatorIsingTF(n, h, J, gamma, mocked_reflection=True)
    ctrl_qubitization = ControlledQubitizedOperatorIsingTF(n, h, J, gamma, mocked_reflection=True)

    gqsp = GQSP(
        qubitization=qubitization,
        controlled_qubitization=ctrl_qubitization,
        poly_coeffs=poly_coeffs,
        mocked_angles=False,
        laurent_negative_power=laurent_negative_power,
    )

    qc_w = QuantumCircuit(qubitization.num_qubits)
    qc_w.append(qubitization, list(range(qubitization.num_qubits)))
    W = get_unitary(qc_w, big_endian=True)

    qc_gqsp = QuantumCircuit(gqsp.num_qubits)
    qc_gqsp.append(gqsp, list(range(gqsp.num_qubits)))
    U_gqsp = get_unitary(qc_gqsp, big_endian=True)

    dim_w = 2**qubitization.num_qubits
    I_w = np.eye(dim_w, dtype=complex)

    left = kron(bra(0, 1), I_w)
    right = kron(ket(0, 1), I_w)
    transformed_block = left @ U_gqsp @ right

    w_eigs = np.linalg.eigvals(W)
    block_eigs = np.linalg.eigvals(transformed_block)

    expected_eigs = np.array([
        _poly_eval(poly_coeffs, z) * z ** (-laurent_negative_power)
        for z in w_eigs
    ])

    for expected in expected_eigs:
        assert np.min(np.abs(block_eigs - expected)) < 1e-8


@pytest.mark.parametrize("n,h,J,gamma,poly_coeffs,laurent_negative_power", _make_gqsp_cases())
def test_gqsp_symbolic_bounds_with_real_angles(n, h, J, gamma, poly_coeffs, laurent_negative_power):

    qubitization = QubitizedOperatorIsingTF(n, h, J, gamma, mocked_reflection=False)
    ctrl_qubitization = ControlledQubitizedOperatorIsingTF(n, h, J, gamma, mocked_reflection=False)

    gqsp = GQSP(
        qubitization=qubitization,
        controlled_qubitization=ctrl_qubitization,
        poly_coeffs=poly_coeffs,
        mocked_angles=False,
        laurent_negative_power=laurent_negative_power,
    )

    qc = QuantumCircuit(gqsp.num_qubits)
    qc.append(gqsp, list(range(gqsp.num_qubits)))

    degree = len(poly_coeffs) - 1
    n_terms = qubitization.lcu.n_terms
    print(f"{n=} {type(n)=} {n_terms=} {type(n_terms)=} {degree=} {type(degree)=} {laurent_negative_power=} {type(laurent_negative_power)=}")

    sym_num_qubits = int(gqsp_number_qubits(n, n_terms))
    sym_nc_depth = int(gqsp_nc_depth(n, n_terms, degree, laurent_negative_power))
    sym_t_count = int(gqsp_t_count(n, n_terms, degree, laurent_negative_power))
    sym_rz_count = int(gqsp_rz_count(n, n_terms, degree, laurent_negative_power))
    sym_toffoli_count = int(gqsp_toffoli_count(n, n_terms, degree, laurent_negative_power))

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
