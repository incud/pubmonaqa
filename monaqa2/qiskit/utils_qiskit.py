import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator


import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Gate, Instruction


_CLIFFORD_RZ_BASIS = {
    "cx", "h", "s", "sdg", "t", "tdg", "x", "z", "rz",
    "barrier", "measure", "id",
}


def _replace_clifford_rz(qc: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    out.global_phase = qc.global_phase

    for inst in qc.data:
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits

        if op.name == "rz" and len(op.params) == 1:
            try:
                theta = float(op.params[0])
            except TypeError:
                out.append(op, qargs, cargs)
                continue

            k = int(np.round(2 * theta / np.pi))

            if np.isclose(theta, k * np.pi / 2, atol=1e-10, rtol=0):
                out.global_phase += -k * np.pi / 4

                if k % 4 == 1:
                    out.s(qargs[0])
                elif k % 4 == 2:
                    out.z(qargs[0])
                elif k % 4 == 3:
                    out.sdg(qargs[0])

                continue

        out.append(op, qargs, cargs)

    return out


def _append_recursively_decomposed(
    out: QuantumCircuit,
    op: Instruction,
    qargs,
    cargs,
    basis: set[str],
) -> None:
    if op.name in basis:
        out.append(op, qargs, cargs)
        return

    if op.definition is not None:
        definition = op.definition

        # A global phase in the definition is part of the decomposition.
        out.global_phase += definition.global_phase

        for subinst in definition.data:
            subop = subinst.operation
            subqargs = [qargs[definition.find_bit(q).index] for q in subinst.qubits]
            subcargs = [cargs[definition.find_bit(c).index] for c in subinst.clbits]

            _append_recursively_decomposed(
                out=out,
                op=subop,
                qargs=subqargs,
                cargs=subcargs,
                basis=basis,
            )

        return

    # Last-resort fallback: synthesize only this one opaque operation,
    # not the whole circuit.
    tmp = QuantumCircuit(len(qargs), len(cargs))
    tmp.append(op, range(len(qargs)), range(len(cargs)))

    tmp_u = transpile(
        tmp,
        basis_gates=sorted(basis - {"barrier", "measure", "id"}),
        optimization_level=0,
        seed_transpiler=1234,
        output_name="",
    )

    out.global_phase += tmp_u.global_phase

    for subinst in tmp_u.data:
        subop = subinst.operation
        subqargs = [qargs[tmp_u.find_bit(q).index] for q in subinst.qubits]
        subcargs = [cargs[tmp_u.find_bit(c).index] for c in subinst.clbits]

        _append_recursively_decomposed(
            out=out,
            op=subop,
            qargs=subqargs,
            cargs=subcargs,
            basis=basis,
        )


def recursively_decompose_to_clifford_rz(
    qc: QuantumCircuit,
    basis: set[str] | None = None,
) -> QuantumCircuit:
    basis = _CLIFFORD_RZ_BASIS if basis is None else basis

    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    out.global_phase = qc.global_phase

    for inst in qc.data:
        _append_recursively_decomposed(
            out=out,
            op=inst.operation,
            qargs=inst.qubits,
            cargs=inst.clbits,
            basis=basis,
        )

    return out


def qiskit_to_clifford_rz(qc: QuantumCircuit, seed: int = 1234, opt: int = 0) -> QuantumCircuit:
    qc_u = recursively_decompose_to_clifford_rz(qc)
    return _replace_clifford_rz(qc_u)


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
    filter = lambda inst: inst.operation.name in ["t", "tdg", "rz"]
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

