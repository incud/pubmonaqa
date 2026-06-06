import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.kogge_stone_in_place_adder_gate import KoggeStoneInPlaceAdder
from monaqa2.qiskit.kogge_stone_in_place_adder_symbolic import (
    kogge_stone_in_place_adder_number_qubits,
    kogge_stone_in_place_adder_nc_depth,
    kogge_stone_in_place_adder_t_count,
    kogge_stone_in_place_adder_rz_count,
    kogge_stone_in_place_adder_toffoli_count,
)


def _int_to_le_bits(x: int, n: int) -> list[int]:
    """Little-endian bit decomposition."""
    return [(x >> i) & 1 for i in range(n)]


def _bits_to_index(bits: list[int]) -> int:
    """Convert a little-endian full-register bitstring to a Qiskit basis index."""
    return sum(int(bit) << i for i, bit in enumerate(bits))


def _apply_classical_circuit(qc, bits: list[int]) -> list[int]:
    """
    Evolve a computational-basis bitstring through a circuit containing only
    classical reversible gates X, CX, and CCX.
    """
    state = bits[:]

    for instruction in qc.data:
        name = instruction.operation.name
        qargs = [qc.find_bit(q).index for q in instruction.qubits]

        if name == "x":
            (target,) = qargs
            state[target] ^= 1

        elif name == "cx":
            control, target = qargs
            if state[control]:
                state[target] ^= 1

        elif name == "ccx":
            c0, c1, target = qargs
            if state[c0] and state[c1]:
                state[target] ^= 1

        else:
            raise AssertionError(f"Unexpected non-classical gate in definition: {name}")

    return state


def _expected_output_bits(gate: KoggeStoneInPlaceAdder, a_val: int, b_val: int) -> list[int]:
    """
    Expected output bitstring in the documented register layout:
        [a, b, prefix_ancillas, carry_copy, optional carry_out].
    """
    n = gate.n
    a_bits = _int_to_le_bits(a_val, n)

    if gate.with_carry_out:
        s_val = a_val + b_val
        b_bits = _int_to_le_bits(s_val, n)
        carry_bits = [(s_val >> n) & 1]
    else:
        s_val = (a_val + b_val) % (1 << n)
        b_bits = _int_to_le_bits(s_val, n)
        carry_bits = []

    prefix_bits = [0] * len(gate.layout["prefix"])
    carry_copy_bits = [0] * len(gate.layout["carry_copy"])

    return a_bits + b_bits + prefix_bits + carry_copy_bits + carry_bits


def _ccx_layer_depth(qc: QuantumCircuit) -> int:
    """Count DAG layers containing at least one CCX gate."""
    dag = circuit_to_dag(qc)

    return sum(
        1
        for layer in dag.layers()
        if any(node.name == "ccx" for node in layer["graph"].op_nodes())
    )


def _rotation_layer_depth(qc: QuantumCircuit) -> int:
    """
    Count DAG layers containing explicit non-Clifford rotation-style gates used
    by the mocked Draper/QFT adder.
    """
    rotation_names = {"cp", "p", "rz"}
    dag = circuit_to_dag(qc)

    return sum(
        1
        for layer in dag.layers()
        if any(node.name in rotation_names for node in layer["graph"].op_nodes())
    )


def _as_int(expr) -> int:
    value = complex(sp.N(expr))
    assert abs(value.imag) <= 1e-9
    real = float(value.real)
    assert np.isfinite(real)
    assert abs(real - round(real)) <= 1e-9
    return int(round(real))


@pytest.mark.parametrize("with_carry_out", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_kogge_stone_in_place_adder_classical_action(n: int, with_carry_out: bool) -> None:
    """
    Check the exact classical action of the full Kogge-Stone implementation.

    The non-mocked definition should contain only X, CX, and CCX gates, so basis
    states can be evolved exactly by a classical reversible simulator. The test
    verifies

        |a>|b>|0...0> -> |a>|a+b mod 2^n>|0...0>

    and, when requested, the final carry-out bit.
    """
    gate = KoggeStoneInPlaceAdder(n=n, with_carry_out=with_carry_out, mocked_adder=False)

    allowed = {"x", "cx", "ccx"}
    seen = {instruction.operation.name for instruction in gate.definition.data}
    assert seen <= allowed

    for a_val in range(1 << n):
        for b_val in range(1 << n):
            input_bits = (
                _int_to_le_bits(a_val, n)
                + _int_to_le_bits(b_val, n)
                + [0] * (gate.num_qubits - 2 * n)
            )

            actual = _apply_classical_circuit(gate.definition, input_bits)
            expected = _expected_output_bits(gate, a_val, b_val)

            assert actual == expected, (
                f"Mismatch for n={n}, with_carry_out={with_carry_out}, "
                f"a={a_val}, b={b_val}"
            )


@pytest.mark.parametrize("with_carry_out", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_kogge_stone_mocked_adder_statevector_action(n: int, with_carry_out: bool) -> None:
    """
    Check the mocked ancilla-free adder on basis states.

    mocked_adder=True uses a QFT/Draper-style adder with H and controlled-phase
    gates, so it is not a classical X/CX/CCX circuit. This test uses statevector
    evolution and verifies that each computational-basis input is mapped to the
    expected computational-basis output with probability one.
    """
    gate = KoggeStoneInPlaceAdder(n=n, with_carry_out=with_carry_out, mocked_adder=True)

    assert len(gate.layout["prefix"]) == 0
    assert len(gate.layout["carry_copy"]) == 0

    for a_val in range(1 << n):
        for b_val in range(1 << n):
            input_bits = (
                _int_to_le_bits(a_val, n)
                + _int_to_le_bits(b_val, n)
                + [0] * int(with_carry_out)
            )
            expected_bits = _expected_output_bits(gate, a_val, b_val)

            qc = QuantumCircuit(gate.num_qubits)

            for q, bit in enumerate(input_bits):
                if bit:
                    qc.x(q)

            qc.append(gate, list(range(gate.num_qubits)))

            state = Statevector.from_int(0, 2**gate.num_qubits).evolve(qc).data
            expected_index = _bits_to_index(expected_bits)

            probabilities = np.abs(state) ** 2
            assert np.argmax(probabilities) == expected_index
            assert probabilities[expected_index] >= 1.0 - 1e-10
            assert np.sum(probabilities) <= 1.0 + 1e-10


@pytest.mark.parametrize("with_carry_out", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 8, 10, 16])
def test_kogge_stone_in_place_adder_symbolic_bounds_full(n: int, with_carry_out: bool) -> None:
    """
    Check symbolic resources for the full Kogge-Stone implementation.

    The full circuit is CCX-based. The symbolic formulas should match or
    upper-bound:
      * exact number of qubits;
      * zero explicit RZ/rotation count;
      * Toffoli count;
      * T-count under the 7-T-per-CCX cost model;
      * non-Clifford depth measured as CCX-containing DAG layers.
    """
    gate = KoggeStoneInPlaceAdder(n=n, with_carry_out=with_carry_out, mocked_adder=False)
    qc = gate.definition
    counts = qc.count_ops()

    actual_number_qubits = gate.num_qubits
    actual_toffoli_count = int(counts.get("ccx", 0))
    actual_t_count = 7 * actual_toffoli_count
    actual_rz_count = int(counts.get("rz", 0)) + int(counts.get("p", 0)) + int(counts.get("cp", 0))
    actual_nc_depth = _ccx_layer_depth(qc)

    n_sym = sp.symbols("n", integer=True, positive=True)
    subs = {n_sym: n}

    symbolic_number_qubits = _as_int(
        kogge_stone_in_place_adder_number_qubits(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=False,
        ).subs(subs)
    )
    symbolic_toffoli_count = _as_int(
        kogge_stone_in_place_adder_toffoli_count(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=False,
        ).subs(subs)
    )
    symbolic_t_count = _as_int(
        kogge_stone_in_place_adder_t_count(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=False,
        ).subs(subs)
    )
    symbolic_rz_count = _as_int(
        kogge_stone_in_place_adder_rz_count(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=False,
        ).subs(subs)
    )
    symbolic_nc_depth = _as_int(
        kogge_stone_in_place_adder_nc_depth(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=False,
        ).subs(subs)
    )

    assert actual_number_qubits == symbolic_number_qubits
    assert actual_toffoli_count <= symbolic_toffoli_count
    assert actual_t_count <= symbolic_t_count
    assert actual_rz_count <= symbolic_rz_count
    assert actual_nc_depth <= symbolic_nc_depth


@pytest.mark.parametrize("with_carry_out", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 8, 10, 16])
def test_kogge_stone_in_place_adder_symbolic_bounds_mocked(n: int, with_carry_out: bool) -> None:
    """
    Check symbolic resources for the mocked ancilla-free adder.

    The mocked circuit is QFT/Draper-style: it has no Toffoli gates and no T
    gates in the CCX cost model, but it does use controlled-phase rotations.
    The symbolic RZ-count is interpreted as a rotation-count upper bound for
    explicit CP/P/RZ-style gates.
    """
    gate = KoggeStoneInPlaceAdder(n=n, with_carry_out=with_carry_out, mocked_adder=True)
    qc = gate.definition
    counts = qc.count_ops()

    actual_number_qubits = gate.num_qubits
    actual_toffoli_count = int(counts.get("ccx", 0))
    actual_t_count = 7 * actual_toffoli_count
    actual_rz_count = int(counts.get("rz", 0)) + int(counts.get("p", 0)) + int(counts.get("cp", 0))
    actual_nc_depth = _rotation_layer_depth(qc)

    n_sym = sp.symbols("n", integer=True, positive=True)
    subs = {n_sym: n}

    symbolic_number_qubits = _as_int(
        kogge_stone_in_place_adder_number_qubits(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=True,
        ).subs(subs)
    )
    symbolic_toffoli_count = _as_int(
        kogge_stone_in_place_adder_toffoli_count(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=True,
        ).subs(subs)
    )
    symbolic_t_count = _as_int(
        kogge_stone_in_place_adder_t_count(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=True,
        ).subs(subs)
    )
    symbolic_rz_count = _as_int(
        kogge_stone_in_place_adder_rz_count(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=True,
        ).subs(subs)
    )
    symbolic_nc_depth = _as_int(
        kogge_stone_in_place_adder_nc_depth(
            n_sym,
            with_carry_out=with_carry_out,
            mocked_adder=True,
        ).subs(subs)
    )

    assert actual_number_qubits == symbolic_number_qubits
    assert actual_toffoli_count <= symbolic_toffoli_count
    assert actual_t_count <= symbolic_t_count
    assert actual_rz_count <= symbolic_rz_count
    assert actual_nc_depth <= symbolic_nc_depth