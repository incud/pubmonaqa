import numpy as np
import scipy as sc
from qiskit.circuit import Gate, QuantumCircuit
import matplotlib.pyplot as plt
from monaqa2.qiskit.arithmetic_fully_phase import ReflectionZero, ControlledReflectionZero
from monaqa2.qiskit.gqsp_gate_generic import GQSP
from monaqa2.qiskit.primitives import Ccx, Cry, Ccry, GivensRotation
from monaqa2.qiskit.utils_qiskit import get_unitary
from monaqa2.qiskit.utils_numpy import ket, bra, kron
from qiskit.synthesis.multi_controlled import synth_mcx_2_clean_kg24


class ThreeTwoCompressor(Gate):

    def __init__(self, label=None) -> None:
        super().__init__(name="3to2 compressor", num_qubits=5, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"a": [0], "b": [1], "c": [2], "s": [3], "k": [4]}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(5, name=self.name)
        a, b, c, s, k = range(5)
        qc.cx(a, s)
        qc.cx(b, s)
        qc.cx(c, s)
        qc.append(Ccx(), [a, b, k])
        qc.append(Ccx(), [a, c, k])
        qc.append(Ccx(), [b, c, k])
        return qc


class Majority(Gate):

    def __init__(self, num_controls: int, label=None) -> None:
        self.num_controls = num_controls
        super().__init__(name=f"majority", num_qubits=num_controls + 1, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"x": list(range(self.num_controls)), "y": [self.num_controls]}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        controls = list(range(self.num_controls))
        target = self.num_controls
        for i in range(len(controls)):
            for j in range(i + 1, len(controls)):
                qc.append(Ccx(), [controls[i], controls[j], target])
        return qc


class WallaceTreeAdder(Gate):

    def __init__(self, M: int, W: int, label=None):
        """
        Clean Wallace-tree adder for M >= 2 terms of W bits.

        Layout:
            inputs:  M * W qubits
            output:  W qubits
            scratch: (2W - 1)(M - 2) Wallace qubits + (W - 1) final carries

        Extra qubits beyond the inputs:
            (2W - 1)(M - 1)

        Semantics:
            |x_0, ..., x_{M-1}> |y> |0...0> -> |x_0, ..., x_{M-1}> |y xor ((x_0 + ... + x_{M-1}) mod 2^W)> |0...0>
        """
        self.M = M
        self.W = W
        super().__init__(name=f"WallaceTree({M},{W})", num_qubits=M * W + (2 * W - 1) * (M - 1), params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict:
        M, W = self.M, self.W
        output = M * W
        wallace = output + W
        carries = wallace + (2 * W - 1) * (M - 2)
        inputs = {f"in^{{({idx})}}": list(range((idx - 1) * W, idx * W)) for idx in range(1, M + 1)}
        return {**inputs, "output": list(range(output, output + W)), "wallace": list(range(wallace, carries)), "carries": list(range(carries, carries + W - 1)), "scratch": list(range(wallace, self.num_qubits))}

    def _maj(self, qc: QuantumCircuit, xs: list[int], t: int) -> None:
        if len(xs) >= 2:
            qc.append(Majority(len(xs)), xs + [t])

    def _row3to2(self, qc: QuantumCircuit, a: list[int | None], b: list[int | None], c: list[int | None], s: list[int], k: list[int]) -> None:
        """Compress three W-bit rows into s and shifted carry row [None] + k."""
        for j in range(self.W):
            xs = [q for q in (a[j], b[j], c[j]) if q is not None]
            if j < self.W - 1 and len(xs) == 3:
                qc.append(ThreeTwoCompressor(), [xs[0], xs[1], xs[2], s[j], k[j]])
            else:
                for q in xs:
                    qc.cx(q, s[j])
                if j < self.W - 1:
                    self._maj(qc, xs, k[j])

    def _build_definition(self) -> QuantumCircuit:
        M, W = self.M, self.W
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        output = M * W
        wallace = output + W
        carries = wallace + (2 * W - 1) * (M - 2)

        rows = [[i * W + j for j in range(W)] for i in range(M)]
        ops = []
        free = wallace

        while len(rows) > 2:
            new_rows = []
            for i in range(0, len(rows) - 2, 3):
                a, b, c = rows[i], rows[i + 1], rows[i + 2]
                s = list(range(free, free + W))
                k = list(range(free + W, free + 2 * W - 1))
                free += 2 * W - 1
                self._row3to2(qc, a, b, c, s, k)
                ops.append((a, b, c, s, k))
                new_rows += [s, [None] + k]
            new_rows += rows[3 * (len(rows) // 3):]
            rows = new_rows
        qc.barrier()

        r0, r1 = rows

        for j in range(W - 1):
            xs = [q for q in (r0[j], r1[j]) if q is not None]
            if j > 0:
                xs.append(carries + j - 1)
            self._maj(qc, xs, carries + j)
        qc.barrier()

        for j in range(W):
            for q in (r0[j], r1[j]):
                if q is not None:
                    qc.cx(q, output + j)
            if j > 0:
                qc.cx(carries + j - 1, output + j)
        qc.barrier()

        for j in reversed(range(W - 1)):
            xs = [q for q in (r0[j], r1[j]) if q is not None]
            if j > 0:
                xs.append(carries + j - 1)
            self._maj(qc, xs, carries + j)
        qc.barrier()

        for a, b, c, s, k in reversed(ops):
            self._row3to2(qc, a, b, c, s, k)

        return qc
    

class ConditionalTermsLoader(Gate):

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, F: int, invert_coefficients: bool, label=None):
        """
        Layout:
            x register:      n qubits, indices 0 ... n - 1
            terms register:  M * W qubits, indices n ... n + M * W - 1
            flags register:  2 * binom(n, 2) qubits, indices n + M * W ... n + M * W + 2 * binom(n, 2) - 1

        Here:
            W = F + 1
            M = n + binom(n, 2)

        The spin encoding is s_i = (-1)^x_i.
        The h_i term is always loaded as either +h_i or -h_i, depending on x_i.
        Each J_ij term uses two temporary flags: a copy of x_i and a copy of x_j.
        One copy is temporarily overwritten with x_i xor x_j, which selects +J_ij or -J_ij.
        Coefficients are normalized by alpha and encoded as signed Q1.F two's-complement words.
        """
        self.n = n
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.F = F
        self.W = F + 1
        self.M = n + n * (n - 1) // 2
        self.alpha = self._alpha(self.h, self.J)
        self.invert_coefficients = invert_coefficients
        super().__init__(name=f"ConditionalTermsLoader({n},{F})", num_qubits=n + self.M * self.W + n * (n - 1), params=[], label=label)
        self.definition = self._build_definition()

    @staticmethod
    def _alpha(h: np.ndarray, J: np.ndarray) -> float:
        return 2 * (np.sum(np.abs(h)) + np.sum(np.abs(np.triu(J, 1))))

    def _bits(self, value: float) -> list[int]:
        value = 0.0 if self.alpha == 0 else value / self.alpha
        value = -value if self.invert_coefficients else value
        z = int(np.round(value * (1 << self.F))) % (1 << self.W)
        return [(z >> j) & 1 for j in range(self.W)]

    def _build_definition(self) -> QuantumCircuit:
        n, W = self.n, self.W
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        terms = n
        flags = terms + self.M * W
        free = flags

        def term(m: int, j: int) -> int:
            return terms + m * W + j

        def load_selected(control: int, m: int, value: float) -> None:
            bits_plus = self._bits(value)
            bits_minus = self._bits(-value)

            for j, bit in enumerate(bits_plus):
                if bit:
                    qc.x(term(m, j))

            for j, bit in enumerate([a ^ b for a, b in zip(bits_plus, bits_minus)]):
                if bit:
                    qc.cx(control, term(m, j))

        def fanout(src: int, targets: list[int]) -> list[tuple[int, int]]:
            ops = []
            known = [src]
            done = 0
            while done < len(targets):
                for q in known[:]:
                    if done == len(targets):
                        break
                    qc.cx(q, targets[done])
                    ops.append((q, targets[done]))
                    known.append(targets[done])
                    done += 1
            return ops

        copies = [[] for _ in range(n)]
        pair_data = []
        m = n

        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = free, free + 1
                free += 2
                copies[i].append(fi)
                copies[j].append(fj)
                pair_data.append((i, j, fi, fj, m))
                m += 1

        fanout_ops = []
        for i in range(n):
            fanout_ops += fanout(i, copies[i])

        for i in range(n):
            load_selected(i, i, self.h[i])

        for i, j, fi, fj, m in pair_data:
            qc.cx(fj, fi)
            load_selected(fi, m, self.J[i, j])

        for i, j, fi, fj, m in reversed(pair_data):
            qc.cx(fj, fi)

        for src, dst in reversed(fanout_ops):
            qc.cx(src, dst)

        return qc
    

class DeltaEnergy(Gate):

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, F: int, label=None):
        """
        Layout:
            x register:              n qubits
            y register:              n qubits
            x terms register:        M * W qubits
            y terms register:        M * W qubits
            delta register:          W qubits
            work register:           (2W - 1)(2M - 1) - W qubits

        Here:
            W = F + 1
            M = n + binom(n, 2)

        The work register is reused as:
            loader flags for E(x),
            loader flags for -E(y),
            Wallace-tree scratch.

        The gate computes:
            delta ^= E(x) - E(y)

        The x/y term registers and the work register are cleaned back to zero.
        """
        self.n = n
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.F = F
        self.W = F + 1
        self.M = n + n * (n - 1) // 2
        self.P = n * (n - 1) // 2
        self.work = (2 * self.W - 1) * (2 * self.M - 1) - self.W

        super().__init__(name=f"DeltaEnergy({n},{F})", num_qubits=2 * n + 2 * self.M * self.W + self.W + self.work, params=[], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        n, W, M, P = self.n, self.W, self.M, self.P
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        x = list(range(n))
        y = list(range(n, 2 * n))

        x_terms_start = 2 * n
        y_terms_start = x_terms_start + M * W
        delta_start = y_terms_start + M * W
        work_start = delta_start + W

        x_terms = list(range(x_terms_start, x_terms_start + M * W))
        y_terms = list(range(y_terms_start, y_terms_start + M * W))
        delta = list(range(delta_start, delta_start + W))
        work = list(range(work_start, self.num_qubits))

        loader_flags = work[:2 * P]
        wallace_scratch = work

        x_loader = ConditionalTermsLoader(n, self.h, self.J, self.F, invert_coefficients=False)
        y_loader = ConditionalTermsLoader(n, self.h, self.J, self.F, invert_coefficients=True)
        adder = WallaceTreeAdder(2 * M, W)

        qc.append(x_loader, x + x_terms + loader_flags)
        qc.append(y_loader, y + y_terms + loader_flags)
        qc.append(adder, x_terms + y_terms + delta + wallace_scratch)
        qc.append(y_loader.inverse(), y + y_terms + loader_flags)
        qc.append(x_loader.inverse(), x + x_terms + loader_flags)

        return qc


class PrepareOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        self.lam = float(np.sum(np.abs(self.h)))
        super().__init__(name="prepare_one_body", num_qubits=b, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(self.b))

    def _tree_layers(self, weights: np.ndarray) -> list[list[tuple[int, int, float, float]]]:
        layers = []

        def add(level: int, indices: list[int]) -> None:
            if len(indices) <= 1:
                return

            mid = (len(indices) + 1) // 2
            left = indices[:mid]
            right = indices[mid:]

            left_weight = float(np.sum(weights[left]))
            right_weight = float(np.sum(weights[right]))

            while len(layers) <= level:
                layers.append([])

            layers[level].append((left[0], right[0], left_weight, right_weight))
            add(level + 1, left)
            add(level + 1, right)

        add(0, list(range(self.b)))
        return layers

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        weights = np.abs(self.h)

        # Unary tree PREPARE: amplitudes proportional to sqrt(|h_i| / lambda).
        qc.x(self.selection[0])

        for layer in self._tree_layers(weights):
            for left, right, left_weight, right_weight in layer:
                qc.append(GivensRotation(left_weight, right_weight), [self.selection[left], self.selection[right]])

        return qc


class SelectOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        super().__init__(name="select_one_body", num_qubits=2 * b, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(self.b))

    @property
    def system(self) -> list[int]:
        return list(range(self.b, 2 * self.b))

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        # SELECT = sum_i |i><i| sign(h_i) Z_i.
        for i in range(self.b):
            if self.h[i] < 0.0:
                qc.z(self.selection[i])
            qc.cz(self.selection[i], self.system[i])

        return qc


class ControlledSelectOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        super().__init__(name="c_select_one_body", num_qubits=1 + 2 * b + (b - 1), params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def selection(self) -> list[int]:
        return list(range(1, 1 + self.b))

    @property
    def system(self) -> list[int]:
        return list(range(1 + self.b, 1 + 2 * self.b))

    @property
    def control_copies(self) -> list[int]:
        return [self.c] + list(range(1 + 2 * self.b, 1 + 3 * self.b - 1))

    @staticmethod
    def _ccz(qc: QuantumCircuit, a: int, b: int, target: int) -> None:
        qc.h(target)
        qc.append(Ccx(), [a, b, target])
        qc.h(target)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        # Fan out the external control so the controlled SELECT can be parallelized.
        for q in self.control_copies[1:]:
            qc.cx(self.c, q)

        for i in range(self.b):
            if self.h[i] < 0.0:
                qc.cz(self.control_copies[i], self.selection[i])
            self._ccz(qc, self.control_copies[i], self.selection[i], self.system[i])

        for q in reversed(self.control_copies[1:]):
            qc.cx(self.c, q)

        return qc


class QubitizedOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        self.lam = float(np.sum(np.abs(self.h)))
        super().__init__(name="qubitized_one_body", num_qubits=2 * b + 2, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(self.b))

    @property
    def system(self) -> list[int]:
        return list(range(self.b, 2 * self.b))

    @property
    def reflection_clean(self) -> list[int]:
        return [2 * self.b, 2 * self.b + 1]

    @property
    def all_qubits(self) -> list[int]:
        return self.selection + self.system

    @property
    def reflection_qubits(self) -> list[int]:
        return self.selection + self.reflection_clean

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"selection": self.selection, "system": self.system, "reflection_clean": self.reflection_clean}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = PrepareOneBodyHamiltonian(self.b, self.h)
        select = SelectOneBodyHamiltonian(self.b, self.h)
        qc.append(prepare, self.selection)
        qc.append(select, self.all_qubits)
        qc.append(prepare.inverse(), self.selection)
        qc.append(ReflectionZero(self.b), self.reflection_qubits)
        return qc


class ControlledQubitizedOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        super().__init__(name="c_qubitized_one_body", num_qubits=3 * b + 2, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(1, 1 + self.b))

    @property
    def system(self) -> list[int]:
        return list(range(1 + self.b, 1 + 2 * self.b))

    @property
    def reflection_clean(self) -> list[int]:
        return [1 + 2 * self.b, 2 + 2 * self.b]

    @property
    def fanout_clean(self) -> list[int]:
        return list(range(3 + 2 * self.b, 2 + 3 * self.b))

    @property
    def select_qubits(self) -> list[int]:
        return [0] + self.selection + self.system + self.fanout_clean

    @property
    def reflection_qubits(self) -> list[int]:
        return [0] + self.selection + self.reflection_clean

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = PrepareOneBodyHamiltonian(self.b, self.h)
        c_select = ControlledSelectOneBodyHamiltonian(self.b, self.h)
        qc.append(prepare, self.selection)
        qc.append(c_select, self.select_qubits)
        qc.append(prepare.inverse(), self.selection)
        qc.append(ControlledReflectionZero(self.b), self.reflection_qubits)
        return qc


class MockedQubitizedOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        self.lam = float(np.sum(np.abs(self.h)))
        super().__init__(name="mocked_qubitized_one_body", num_qubits=b, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def system(self) -> list[int]:
        return list(range(self.b))

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"system": self.system}

    @staticmethod
    def _energy(bits: int, h: np.ndarray) -> float:
        z = np.array([1.0 if ((bits >> i) & 1) == 0 else -1.0 for i in range(len(h))])
        return float(h @ z)

    @staticmethod
    def _phases(h: np.ndarray, lam: float) -> np.ndarray:
        phases = np.empty(2 ** len(h), dtype=complex)
        for bits in range(2 ** len(h)):
            x = np.clip(MockedQubitizedOneBodyHamiltonian._energy(bits, h) / lam, -1.0, 1.0)
            phases[bits] = np.exp(1j * np.arccos(x))
        return phases

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.unitary(np.diag(self._phases(self.h, self.lam)), self.system, label=self.name)
        return qc


class MockedControlledQubitizedOneBodyHamiltonian(Gate):

    def __init__(self, b: int, h: np.ndarray, label=None):
        self.b = b
        self.h = np.asarray(h, dtype=float)
        self.lam = float(np.sum(np.abs(self.h)))
        super().__init__(name="c_mocked_qubitized_one_body", num_qubits=1 + b, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def system(self) -> list[int]:
        return list(range(1, self.b + 1))

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        phases = MockedQubitizedOneBodyHamiltonian._phases(self.h, self.lam)
        controlled_phases = np.ones(2 ** (self.b + 1), dtype=complex)

        for bits, phase in enumerate(phases):
            controlled_phases[(bits << 1) | 1] = phase

        qc.unitary(np.diag(controlled_phases), list(range(self.b + 1)), label=self.name)
        return qc


class CutoffTail(Gate):

    def __init__(self, W: int, m: int, label=None):
        """
        Layout:
            delta:  W
            signal: W
            tail:   1
            aux:    2 if m >= 3 else 0

        Assumes delta is already nonnegative. If m=0, copies delta into signal.
        If m>0, computes tail=1 when delta >= 2^{-m}. If tail=0, writes
        signal = 2^m delta by a bit shift. If tail=1, writes signal = 1.
        The signal register, tail qubit, and auxiliary qubits are assumed clean.
        """
        self.W = int(W)
        self.F = self.W - 1
        self.m = max(0, min(int(m), self.F))
        self.num_aux = 2 if self.m >= 3 else 0
        super().__init__(name=f"cutoff_tail({self.W},{self.m})", num_qubits=2 * self.W + 1 + self.num_aux, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        W = self.W
        return {
            "delta": list(range(W)),
            "signal": list(range(W, 2 * W)),
            "tail": [2 * W],
            "aux": list(range(2 * W + 1, self.num_qubits)),
        }

    def _build_definition(self) -> QuantumCircuit:
        W, F, m = self.W, self.F, self.m
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        delta = self.layout["delta"]
        signal = self.layout["signal"]
        tail = self.layout["tail"][0]
        aux = self.layout["aux"]

        if m == 0:
            for j in range(W):
                qc.cx(delta[j], signal[j])
            return qc

        high = delta[F - m:F]

        qc.x(tail)
        for q in high:
            qc.x(q)
        if m == 1:
            qc.cx(high[0], tail)
        elif m == 2:
            qc.append(Ccx(), [high[0], high[1], tail])
        else:
            qc.append(synth_mcx_2_clean_kg24(m), high + [tail] + aux)
        for q in high:
            qc.x(q)

        qc.x(tail)
        for j in range(F - m):
            qc.append(Ccx(), [tail, delta[j], signal[j + m]])
        qc.x(tail)

        qc.cx(tail, signal[F])

        return qc
    

class SqrtExpArithmetic(GQSP):

    def __init__(self, b: int, beta: float, alpha: float, eps: float, eps_tail: float | None = None, degree: int | None = None, is_mocked_construction: bool = False, is_mocked_angles: bool = False, label=None):
        self.b = int(b)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.eps_tail = eps_tail

        self.tail_power = self._tail_power(self.beta, self.alpha, self.eps_tail, self.b - 1)
        self.tail_fraction = 2.0 ** (-self.tail_power)
        self.tail_delta = None if self.tail_power == 0 else self.alpha * self.tail_fraction
        self.effective_alpha = self.alpha * self.tail_fraction

        self.h = self._ising_h()
        self.kappa = 0.5 * self.beta * self.effective_alpha
        self.signal_norm = float(np.sum(np.abs(self.h)))
        self.signal_shift = 1.0 + 3.0 * 2.0 ** (-self.b)

        self.selected_degree = self._degree(self.kappa, self.eps) if degree is None else int(degree)
        self.poly_coeffs = self._poly()
        self.poly_scale = 1.0

        self.qubitization = MockedQubitizedOneBodyHamiltonian(self.b, self.h) if is_mocked_construction else QubitizedOneBodyHamiltonian(self.b, self.h)
        self.controlled_qubitization = MockedControlledQubitizedOneBodyHamiltonian(self.b, self.h) if is_mocked_construction else ControlledQubitizedOneBodyHamiltonian(self.b, self.h)

        super().__init__(qubitization=self.qubitization, controlled_qubitization=self.controlled_qubitization, poly_coeffs=self.poly_coeffs, mocked_angles=is_mocked_angles, laurent_negative_power=self.selected_degree, label=label)

    @staticmethod
    def _tail_power(beta: float, alpha: float, eps_tail: float | None, max_power: int) -> int:
        if eps_tail is None or beta <= 0.0 or alpha <= 0.0:
            return 0
        target = np.log(1.0 / eps_tail)
        if not np.isfinite(target) or target <= 0.0:
            return 0
        m = int(np.floor(np.log2(beta * alpha / target) - 1.0))
        return max(0, min(m, int(max_power)))

    def _ising_h(self) -> np.ndarray:
        h = np.zeros(self.b, dtype=float)
        h[-1] = 0.5
        for j in range(self.b - 1):
            h[j] = -(2.0 ** (j - self.b))
        return h

    @staticmethod
    def _degree(kappa: float, eps: float) -> int:
        return int(np.ceil(kappa + np.log(1.0 / eps) + 1.0))

    @property
    def layout(self) -> dict[str, list[int]]:
        base = super().layout
        signal = [1 + q for q in self.qubitization.layout["system"]]
        aux = [q for q in range(self.num_qubits) if q not in base["control"] + signal]
        base["signal"] = signal
        base["aux"] = aux
        return base

    def _poly(self) -> np.ndarray:
        d = self.selected_degree
        mu = self.kappa * self.signal_norm
        prefactor = np.exp(-self.kappa * self.signal_shift)
        return np.array([prefactor * sc.special.iv(abs(k), mu) for k in range(-d, d + 1)], dtype=complex)

    def exact_amplitude(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.exp(-0.5 * self.beta * self.alpha * x)
        return y if self.tail_power == 0 else np.where(x < self.tail_fraction, y, 0.0)

    def evaluate_polynomial_eigenvalue(self, y: np.ndarray | float) -> np.ndarray:
        theta = np.arccos(np.clip(np.asarray(y, dtype=float), -1.0, 1.0))
        return np.real_if_close(self._eval_laurent_theta(self.poly_coeffs, -theta)).real

    def evaluate_polynomial_signal(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        scaled_x = x / self.tail_fraction
        y = (self.signal_shift - scaled_x) / self.signal_norm
        out = self.evaluate_polynomial_eigenvalue(y)
        return out if self.tail_power == 0 else np.where(x < self.tail_fraction, out, 0.0)

    @staticmethod
    def _eval_laurent_theta(coeffs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=complex)
        theta = np.asarray(theta, dtype=float)
        d = (len(coeffs) - 1) // 2
        out = np.zeros_like(theta, dtype=complex)
        for k in range(-d, d + 1):
            out += coeffs[d + k] * np.exp(-1j * k * theta)
        return out
    

def plot_hybrid_phase_arithmetic(b: int, beta: float, alpha: float, eps: float, eps_tail: float | None = None, degree: int | None = None, is_mocked_angles: bool = False):

    gate = SqrtExpArithmetic(b, beta, alpha, eps=eps, eps_tail=eps_tail, degree=degree, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)
    qc = QuantumCircuit(b + 1)
    qc.append(gate, range(b + 1))
    U = get_unitary(qc, big_endian=True)

    def H(bits: int) -> float:
        z = np.array([1.0 if ((bits >> (b - 1 - i)) & 1) == 0 else -1.0 for i in range(b)])
        return float(gate.h @ z)

    u_min = gate.signal_shift - gate.signal_norm
    u_max = gate.signal_shift + gate.signal_norm
    cutoff = None if gate.eps_tail is None else gate.tail_fraction

    signals, circuit = [], []

    for bits in range(2**b):
        u = gate.signal_shift - H(bits)
        x = u if gate.eps_tail is None else gate.tail_fraction * u
        psi = kron(ket(0, 1), ket(bits, b))
        amp = (kron(bra(0, 1), bra(bits, b)) @ U @ psi)[0, 0]
        signals.append(x)
        circuit.append(float(np.real(np.real_if_close(amp))))

    signals = np.asarray(signals)
    circuit = np.asarray(circuit)

    if gate.eps_tail is None:
        x_ideal = np.linspace(u_min, u_max, 1000)
        x_poly = x_ideal
    else:
        x_ideal = np.linspace(0.0, max(cutoff, gate.tail_fraction * u_max), 1000)
        x_poly_min = gate.tail_fraction * u_min
        x_poly_max = min(cutoff, gate.tail_fraction * u_max)
        x_poly = np.linspace(x_poly_min, x_poly_max, 1000)

    plt.figure(figsize=(8, 5))
    plt.plot(x_ideal, np.exp(-0.5 * beta * alpha * x_ideal), color="red", linewidth=1.0, label=r"ideal $\exp\!\left(-\frac{\beta\alpha}{2}x\right)$")
    plt.plot(x_poly, gate.evaluate_polynomial_signal(x_poly), color="blue", linewidth=2.0, label="implemented GQSP polynomial")

    if cutoff is not None:
        plt.axvline(cutoff, color="gray", linestyle="--", linewidth=1.0, label="tail cutoff")
        tail_mask = signals >= cutoff
        if np.any(tail_mask):
            plt.scatter(signals[tail_mask], np.zeros_like(signals[tail_mask]), s=24, color="gray", alpha=0.45, edgecolors="none", label="clipped tail")
        if np.any(~tail_mask):
            plt.scatter(signals[~tail_mask], circuit[~tail_mask], s=24, color="lightskyblue", alpha=0.75, edgecolors="none", label="circuit implementation")
    else:
        plt.scatter(signals, circuit, s=24, color="lightskyblue", alpha=0.75, edgecolors="none", label="circuit implementation")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$\exp(-\beta\alpha x/2)$")
    plt.title(f"b={b}, degree={gate.selected_degree}, kappa={gate.kappa:.6g}, eps_tail={gate.eps_tail}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return gate


def plot_error_hybrid_phase_arithmetic(b: int, eps: float, kappas: list[float] | None = None, fixed_degree: int = 20, n_signal_samples: int = 4000, eps_tail: float | None = None, is_mocked_angles: bool = True):

    kappas = [1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if kappas is None else kappas

    alpha = 1.0
    err_fixed, err_selected, deg_selected = [], [], []

    for kappa in kappas:
        beta = 2.0 * kappa / alpha

        gate_selected_tmp = SqrtExpArithmetic(b, beta, alpha, eps, eps_tail=eps_tail, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)
        degree_selected = gate_selected_tmp.selected_degree
        deg_selected.append(degree_selected)

        gate_fixed = SqrtExpArithmetic(b, beta, alpha, eps, eps_tail=eps_tail, degree=fixed_degree, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)
        gate_selected = SqrtExpArithmetic(b, beta, alpha, eps, eps_tail=eps_tail, degree=degree_selected, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)

        u_min = gate_selected.signal_shift - gate_selected.signal_norm
        u_max = gate_selected.signal_shift + gate_selected.signal_norm

        if gate_selected.eps_tail is None:
            x_min = u_min
            x_max = u_max
        else:
            x_min = gate_selected.tail_fraction * u_min
            x_max = min(gate_selected.tail_fraction, gate_selected.tail_fraction * u_max)

        x_grid = np.linspace(x_min, x_max, n_signal_samples)
        exact = np.exp(-0.5 * beta * alpha * x_grid)

        err_fixed.append(float(np.max(np.abs(gate_fixed.evaluate_polynomial_signal(x_grid) - exact))))
        err_selected.append(float(np.max(np.abs(gate_selected.evaluate_polynomial_signal(x_grid) - exact))))

    selected_label = r"$d=\lceil \kappa_{\mathrm{eff}}+\log(1/\varepsilon)+1\rceil$"

    plt.figure(figsize=(7, 4.5))
    plt.scatter(kappas, err_fixed, color="red", label=rf"Constant degree $d={fixed_degree}$")
    plt.plot(kappas, err_fixed, color="red", linewidth=1.0, alpha=0.5)
    plt.scatter(kappas, err_selected, color="blue", label=selected_label)
    plt.plot(kappas, err_selected, color="blue", linewidth=1.0, alpha=0.5)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(r"$\kappa=\beta\alpha/2$")
    plt.ylabel("max absolute error (outside discarded tail)")
    plt.title("Hybrid phase arithmetic polynomial error")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    return {"kappa": np.asarray(kappas), "fixed_degree_error": np.asarray(err_fixed), "selected_degree_error": np.asarray(err_selected), "selected_degree": np.asarray(deg_selected)}

class PositivePartSelector(Gate):

    def __init__(self, W: int, label=None):
        """
        Layout:
            delta: W
            zero:  W
            signs: W

        If the sign bit of delta is 1, swaps delta with zero, so that delta
        becomes zero. If the sign bit is 0, leaves delta unchanged. The signs
        register stores a fanout copy of the sign bit and is uncomputed by the
        inverse gate.
        """
        self.W = W
        super().__init__(name=f"positive_part_selector({W})", num_qubits=3 * W, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        W = self.W
        return {
            "delta": list(range(W)),
            "zero": list(range(W, 2 * W)),
            "signs": list(range(2 * W, 3 * W)),
        }

    def _build_definition(self) -> QuantumCircuit:
        W = self.W
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        delta = self.layout["delta"]
        zero = self.layout["zero"]
        signs = self.layout["signs"]

        def fanout(src, targets):
            known, k = [src], 0
            while k < len(targets):
                for q in known[:]:
                    if k == len(targets):
                        break
                    qc.cx(q, targets[k])
                    known.append(targets[k])
                    k += 1

        qc.cx(delta[-1], signs[0])
        fanout(signs[0], signs[1:])

        for c, d, z in zip(signs, delta, zero):
            qc.cx(z, d)
            qc.append(Ccx(), [c, d, z])
            qc.cx(z, d)

        return qc


class HybridPhaseArithmetic(Gate):

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, F: int, beta: float, eps: float, degree: int | None = None, is_mocked_construction: bool = False, is_mocked_angles: bool = False, label=None):
        """
        Layout:
            qsp qubit:    1
            x register:   n
            y register:   n
            delta:        W
            zero:         W
            sign flags:   W
            work:         max(DeltaEnergy auxiliary, CutoffTail + SqrtExpArithmetic auxiliary)

        Computes sqrt(A_yx) with DeltaE = E(y) - E(x), using max(DeltaE, 0).
        If the square-root exponential uses a power-of-two tail cutoff, the
        positive delta is saturated and shifted before entering the phase arithmetic:
            x <  2^{-m}: signal = 2^m x,
            x >= 2^{-m}: signal = 1.
        The square-root exponential arithmetic is applied unconditionally.
        """
        self.n = n
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.F = F
        self.W = F + 1
        self.M = n + n * (n - 1) // 2

        self.delta_energy = DeltaEnergy(n, self.h, self.J, F)
        self.positive_selector = PositivePartSelector(self.W)
        self.sqrt_exp = SqrtExpArithmetic(self.W, beta, ConditionalTermsLoader._alpha(self.h, self.J), eps/2, eps_tail=eps/2, degree=degree, is_mocked_construction=is_mocked_construction, is_mocked_angles=is_mocked_angles)
        self.cutoff_tail = CutoffTail(self.W, self.sqrt_exp.tail_power)

        self.delta_aux = self.delta_energy.num_qubits - 2 * n - self.W
        self.sqrt_aux = self.sqrt_exp.num_qubits - self.W - 1
        self.tail_aux = 0 if self.sqrt_exp.tail_power == 0 else self.W + 1 + self.cutoff_tail.num_aux
        self.work = max(self.delta_aux, self.tail_aux + self.sqrt_aux)

        super().__init__(name=f"HybridPhaseArithmetic({n},{F})", num_qubits=1 + 2 * n + 3 * self.W + self.work, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        n, W = self.n, self.W
        return {
            "control": [0],
            "x": list(range(1, 1 + n)),
            "y": list(range(1 + n, 1 + 2 * n)),
            "delta": list(range(1 + 2 * n, 1 + 2 * n + W)),
            "zero": list(range(1 + 2 * n + W, 1 + 2 * n + 2 * W)),
            "signs": list(range(1 + 2 * n + 2 * W, 1 + 2 * n + 3 * W)),
            "work": list(range(1 + 2 * n + 3 * W, self.num_qubits)),
        }

    def _build_definition(self) -> QuantumCircuit:
        n, W, M = self.n, self.W, self.M
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        qsp = self.layout["control"][0]
        x = self.layout["x"]
        y = self.layout["y"]
        delta = self.layout["delta"]
        zero = self.layout["zero"]
        signs = self.layout["signs"]
        work = self.layout["work"]

        delta_wires = y + x + work[:2 * M * W] + delta + work[2 * M * W:self.delta_aux]
        selector_wires = delta + zero + signs

        qc.append(self.delta_energy, delta_wires)
        qc.append(self.positive_selector, selector_wires)

        if self.sqrt_exp.tail_power == 0:
            def sqrt_wires():
                layout = self.sqrt_exp.layout
                wires = [None] * self.sqrt_exp.num_qubits
                wires[layout["control"][0]] = qsp
                for internal, external in zip(layout["signal"], delta):
                    wires[internal] = external
                aux = iter(work[:self.sqrt_aux])
                return [next(aux) if q is None else q for q in wires]

            qc.append(self.sqrt_exp, sqrt_wires())

        else:
            tail_signal = work[:W]
            tail_flag = work[W]
            sqrt_work = work[W + 1:W + 1 + self.sqrt_aux]
            cutoff_aux = work[W + 1 + self.sqrt_aux:W + 1 + self.sqrt_aux + self.cutoff_tail.num_aux]

            def sqrt_wires():
                layout = self.sqrt_exp.layout
                wires = [None] * self.sqrt_exp.num_qubits
                wires[layout["control"][0]] = qsp
                for internal, external in zip(layout["signal"], tail_signal):
                    wires[internal] = external
                aux = iter(sqrt_work)
                return [next(aux) if q is None else q for q in wires]

            cutoff_wires = delta + tail_signal + [tail_flag] + cutoff_aux

            qc.append(self.cutoff_tail, cutoff_wires)
            qc.append(self.sqrt_exp, sqrt_wires())
            qc.append(self.cutoff_tail.inverse(), cutoff_wires)

        qc.append(self.positive_selector.inverse(), selector_wires)
        qc.append(self.delta_energy.inverse(), delta_wires)

        return qc