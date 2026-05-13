import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator


def qiskit_to_clifford_rz(qc: QuantumCircuit, seed: int=1234) -> QuantumCircuit:
    qc_u = transpile(
        qc,
        basis_gates=["cx", "h", "s", "sdg", "t", "tdg", "x", "z", "rz", "ccx"],
        optimization_level=0,
        seed_transpiler=seed,
        output_name=""
    )
    return qc_u


def get_unitary(qc: QuantumCircuit, big_endian: bool = True) -> np.ndarray:
    """Return the unitary matrix for the circuit.

    Qiskit uses little-endian ordering for qubits, while numpy kron
    typically assumes big-endian ordering. If `big_endian` is True,
    this function returns the matrix with both row and column indices
    bit-reversed so that it matches the usual numpy kron ordering.
    """
    if big_endian:
        qc = qc.reverse_bits()
    return Operator(qc).data


def get_nc_depth(qc: QuantumCircuit, transpile: bool = True) -> int:
    if transpile:
        qc = qiskit_to_clifford_rz(qc)
    filter = lambda inst: inst.operation.name in ["t", "tdg", "rz", "ccx"]
    return int(qc.depth(filter_function=filter))


def get_t_count(qc: QuantumCircuit, transpile: bool = True) -> int:
    if transpile:
        qc = qiskit_to_clifford_rz(qc)
    ops = qc.count_ops()
    return int(ops.get("t", 0) + ops.get("tdg", 0))


def get_toffoli_count(qc: QuantumCircuit, transpile: bool = True) -> int:
    if transpile:
        qc = qiskit_to_clifford_rz(qc)
    ops = qc.count_ops()
    return int(ops.get("ccx", 0))


def get_rz_count(qc: QuantumCircuit, transpile: bool = True) -> int:
    if transpile:
        qc = qiskit_to_clifford_rz(qc)
    ops = qc.count_ops()
    return int(ops.get("rz", 0))

