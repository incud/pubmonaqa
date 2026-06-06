import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator


def qiskit_to_clifford_rz(qc: QuantumCircuit, seed: int = 1234, opt: int = 0) -> QuantumCircuit:
    qc_u = transpile(qc, basis_gates=["cx", "h", "s", "sdg", "t", "tdg", "x", "z", "rz", "ccx"], optimization_level=opt, seed_transpiler=seed, output_name="")
    out = QuantumCircuit(*qc_u.qregs, *qc_u.cregs, name=qc_u.name)
    out.global_phase = qc_u.global_phase

    for inst in qc_u.data:
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits

        # Replace numerical Rz(k*pi/2) gates by Clifford phase gates.
        if op.name == "rz" and len(op.params) == 1:
            try:
                theta = float(op.params[0])
            except TypeError:
                out.append(op, qargs, cargs)
                continue

            # k is the nearest integer such that theta ≈ k*pi/2.
            k = int(np.round(2 * theta / np.pi))

            if np.isclose(theta, k * np.pi / 2, atol=1e-10, rtol=0):
                # Qiskit Rz(theta) = exp(-i theta Z/2), while S/Z/Sdg differ by global phases.
                out.global_phase += -k * np.pi / 4

                # Only k modulo 4 matters: 0 -> I, 1 -> S, 2 -> Z, 3 -> Sdg.
                if k % 4 == 1:
                    out.s(qargs[0])
                elif k % 4 == 2:
                    out.z(qargs[0])
                elif k % 4 == 3:
                    out.sdg(qargs[0])

                continue

        out.append(op, qargs, cargs)

    return out


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

