import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from monaqa2.qiskit.metropolis_hastings_energy_gate import MetropolisHastingsEnergy
from monaqa2.qiskit.metropolis_hastings_energy_symbolic import (
    metropolis_hastings_energy_number_qubits,
    metropolis_hastings_energy_nc_depth,
    metropolis_hastings_energy_t_count,
    metropolis_hastings_energy_rz_count,
    metropolis_hastings_energy_toffoli_count,
)


def _flatten_to_classical_circuit(obj) -> QuantumCircuit:
    """
    Recursively inline a circuit or gate down to X, CX, and CCX.

    This is valid because MetropolisHastingsEnergy uses the non-mocked
    KoggeStoneInPlaceAdder, which is an X/CX/CCX circuit.
    """
    circ = obj if isinstance(obj, QuantumCircuit) else obj.definition
    out = QuantumCircuit(circ.num_qubits)

    def inline_into(dst: QuantumCircuit, src: QuantumCircuit, qmap: list[int]) -> None:
        for inst in src.data:
            name = inst.operation.name
            local_qargs = [src.find_bit(q).index for q in inst.qubits]
            mapped = [qmap[i] for i in local_qargs]

            if name == "x":
                dst.x(mapped[0])
            elif name == "cx":
                dst.cx(mapped[0], mapped[1])
            elif name == "ccx":
                dst.ccx(mapped[0], mapped[1], mapped[2])
            else:
                subdef = inst.operation.definition
                if subdef is None:
                    raise AssertionError(f"Unsupported opaque gate in flattening: {name}")
                inline_into(dst, subdef, mapped)

    inline_into(out, circ, list(range(circ.num_qubits)))
    return out


def _apply_classical_circuit(qc: QuantumCircuit, bits: list[int]) -> list[int]:
    """
    Exact computational-basis evolution for X/CX/CCX circuits.
    """
    state = bits[:]

    for inst in qc.data:
        name = inst.operation.name
        qargs = [qc.find_bit(q).index for q in inst.qubits]

        if name == "x":
            state[qargs[0]] ^= 1

        elif name == "cx":
            control, target = qargs

            if state[control]:
                state[target] ^= 1

        elif name == "ccx":
            c0, c1, target = qargs

            if state[c0] and state[c1]:
                state[target] ^= 1

        else:
            raise AssertionError(f"Unexpected gate in flattened circuit: {name}")

    return state


def _bits_of_int_le(value: int, n: int) -> list[int]:
    return [(value >> i) & 1 for i in range(n)]


def _bits_from_scaled(scaled: int, width: int) -> list[int]:
    unsigned = scaled % (1 << width)
    return [(unsigned >> j) & 1 for j in range(width)]


def _scaled_normalized_energy_diff(
    gate: MetropolisHastingsEnergy,
    x_bits: list[int],
    y_bits: list[int],
) -> int:
    """
    Expected scaled integer before clipping.

    This is computed from the exact quantized terms used by the implementation,
    not from floating-point energies.
    """
    total = 0

    for term in gate.terms:
        bits = x_bits if term["side"] == "A" else y_bits

        if term["arity"] == 1:
            active = bits[term["idxs"][0]] == 1
        else:
            i, j = term["idxs"]
            active = bits[i] == 1 and bits[j] == 1

        if active:
            total += term["scaled"]

    return total


def _expected_wide_and_signal_bits(
    gate: MetropolisHastingsEnergy,
    x_bits: list[int],
    y_bits: list[int],
) -> tuple[list[int], list[int]]:
    """
    Expected wide shifted output and final narrow Q1.f signal.
    """
    raw_scaled = _scaled_normalized_energy_diff(gate, x_bits, y_bits)
    clipped_scaled = min(0, raw_scaled)
    const_scaled = (1 << gate.fractional_bits) - 1
    shifted_scaled = clipped_scaled + const_scaled

    wide_bits = _bits_from_scaled(shifted_scaled, gate.acc_word_bits)
    signal_bits = wide_bits[: gate.fractional_bits] + [wide_bits[-1]]

    return wide_bits, signal_bits


def _make_cases() -> list[tuple[int, np.ndarray, np.ndarray, float, list[tuple[int, int]]]]:
    cases = []

    J = np.array([[0.0, 0.8], [0.8, 0.0]])
    h = np.array([0.5, -0.25])
    cases.append((2, h, J, 0.25, [(0, 0), (1, 0), (0, 3), (2, 1), (3, 2)]))

    h = np.array([0.25, 0.0, 0.5])
    J = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -0.75], [0.0, -0.75, 0.0]])
    cases.append((3, h, J, 0.2, [(0, 0), (1, 2), (5, 3), (7, 4), (6, 1), (3, 7)]))

    h = np.array([-0.25, 0.5, -0.75])
    J = np.array([[0.0, 1.0, -0.5], [1.0, 0.0, 0.25], [-0.5, 0.25, 0.0]])
    cases.append((3, h, J, 0.1, [(0, 0), (1, 1), (3, 5), (7, 2), (4, 6), (6, 7)]))

    J = np.array(
        [
            [0.0, 0.4, -0.2, 0.1],
            [0.4, 0.0, 0.3, -0.5],
            [-0.2, 0.3, 0.0, 0.6],
            [0.1, -0.5, 0.6, 0.0],
        ]
    )
    h = np.array([0.1, -0.2, 0.3, -0.1])
    cases.append((4, h, J, 0.15, [(0, 0), (1, 8), (3, 12), (5, 10), (15, 0), (9, 6)]))

    J = np.zeros((4, 4))
    h = np.array([0.2, -0.4, 0.6, -0.8])
    cases.append((4, h, J, 0.3, [(0, 0), (1, 2), (3, 4), (7, 8), (15, 5)]))

    return cases


def _as_int(expr) -> int:
    value = complex(sp.N(expr))
    assert abs(value.imag) <= 1e-9

    real = float(value.real)

    assert np.isfinite(real)
    assert abs(real - round(real)) <= 1e-9

    return int(round(real))


@pytest.mark.parametrize("n,h,J,eps,samples", _make_cases())
def test_metropolis_hastings_energy_classical_action(
    n: int,
    h: np.ndarray,
    J: np.ndarray,
    eps: float,
    samples: list[tuple[int, int]],
) -> None:
    """
    Check the exact reversible classical action.

    The flattened circuit must contain only X/CX/CCX gates. The test evolves
    computational-basis inputs classically and verifies:
      * A and B are preserved;
      * all non-output work qubits return to zero;
      * output_wide stores the shifted clipped fixed-point value;
      * signal stores the final Q1.f narrow encoding.
    """
    gate = MetropolisHastingsEnergy(n, h, J, eps)
    flat = _flatten_to_classical_circuit(gate)

    seen = {inst.operation.name for inst in flat.data}
    assert seen <= {"x", "cx", "ccx"}

    output_wide = gate.layout["output_wide"]
    signal = gate.layout["signal"]
    output_qubits = set(output_wide) | set(signal)
    non_output_aux = [
        q
        for q in range(2 * gate.n, gate.num_qubits)
        if q not in output_qubits
    ]

    sum_abs = float(np.sum(np.abs(h))) + float(np.sum(np.abs(np.triu(J, k=1))))
    expected_upper_bound = 2.0 * sum_abs

    if np.isclose(expected_upper_bound, 0.0):
        expected_upper_bound = 1.0

    expected_normalization = expected_upper_bound / (
        2.0 - 2.0 ** (-gate.fractional_bits)
    )

    assert np.isclose(gate.upper_bound_energy_diff, expected_upper_bound)
    assert np.isclose(gate.normalization, expected_normalization)

    for x_val, y_val in samples:
        x_bits = _bits_of_int_le(x_val, gate.n)
        y_bits = _bits_of_int_le(y_val, gate.n)

        input_bits = x_bits + y_bits + [0] * (gate.num_qubits - 2 * gate.n)
        actual = _apply_classical_circuit(flat, input_bits)

        assert actual[: gate.n] == x_bits
        assert actual[gate.n : 2 * gate.n] == y_bits

        for q in non_output_aux:
            assert actual[q] == 0

        expected_wide, expected_signal = _expected_wide_and_signal_bits(
            gate,
            x_bits,
            y_bits,
        )

        assert [actual[q] for q in output_wide] == expected_wide, (
            f"Wide-output mismatch for n={gate.n}, eps={eps}, "
            f"x={x_val}, y={y_val}"
        )
        assert [actual[q] for q in signal] == expected_signal, (
            f"Signal mismatch for n={gate.n}, eps={eps}, "
            f"x={x_val}, y={y_val}"
        )


@pytest.mark.parametrize("n,h,J,eps,_samples", _make_cases())
def test_metropolis_hastings_energy_symbolic_bounds(
    n: int,
    h: np.ndarray,
    J: np.ndarray,
    eps: float,
    _samples,
) -> None:
    """
    Check symbolic resource upper bounds.

    The circuit is flattened to X/CX/CCX and counted directly. Symbolic methods
    should upper-bound:
      * qubit count;
      * RZ count, which is zero;
      * T count under the 7-T-per-Toffoli model;
      * Toffoli count;
      * non-Clifford depth as CCX-containing DAG layers.
    """
    gate = MetropolisHastingsEnergy(n, h, J, eps)
    flat = _flatten_to_classical_circuit(gate)
    counts = flat.count_ops()

    actual_number_qubits = flat.num_qubits
    actual_rz_count = int(counts.get("rz", 0))
    actual_toffoli_count = int(counts.get("ccx", 0))
    actual_t_count = 7 * actual_toffoli_count

    dag = circuit_to_dag(flat)
    actual_nc_depth = sum(
        1
        for layer in dag.layers()
        if any(node.name == "ccx" for node in layer["graph"].op_nodes())
    )

    sum_abs = float(np.sum(np.abs(h))) + float(np.sum(np.abs(np.triu(J, k=1))))

    n_sym = sp.Integer(n)
    sum_abs_sym = sp.Float(sum_abs)
    eps_sym = sp.Float(eps)

    symbolic_number_qubits = _as_int(
        metropolis_hastings_energy_number_qubits(n_sym, sum_abs_sym, eps_sym)
    )
    symbolic_rz_count = _as_int(
        metropolis_hastings_energy_rz_count(n_sym, sum_abs_sym, eps_sym)
    )
    symbolic_t_count = _as_int(
        metropolis_hastings_energy_t_count(n_sym, sum_abs_sym, eps_sym)
    )
    symbolic_toffoli_count = _as_int(
        metropolis_hastings_energy_toffoli_count(n_sym, sum_abs_sym, eps_sym)
    )
    symbolic_nc_depth = _as_int(
        metropolis_hastings_energy_nc_depth(n_sym, sum_abs_sym, eps_sym)
    )

    assert actual_number_qubits <= symbolic_number_qubits
    assert actual_rz_count <= symbolic_rz_count
    assert actual_t_count <= symbolic_t_count
    assert actual_toffoli_count <= symbolic_toffoli_count
    assert actual_nc_depth <= symbolic_nc_depth