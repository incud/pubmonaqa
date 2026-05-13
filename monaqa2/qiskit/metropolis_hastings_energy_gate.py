import numpy as np
import sympy as sp
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.kogge_stone_in_place_adder_gate import KoggeStoneInPlaceAdder


class MetropolisHastingsEnergy(Gate):
    r"""
    Reversible fixed-point computation of a clipped normalized energy difference.

    Input registers:
        A: x bits
        B: y bits

    Energy convention:
        E(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j,

    where z_i are computational-basis bits, not Ising spin variables.

    Let

        U = 2 * (sum_i |h_i| + sum_{i<j} |J_ij|),
        D = U / (2 - 2^{-f}).

    The circuit computes a quantized version of

        raw(x, y) = (E(y) - E(x)) / D,

    clips positive values to zero,

        clipped(x, y) = min(0, raw(x, y)),

    and outputs the shifted Q1.f signal

        signal(x, y) = (1 - 2^{-f}) + clipped(x, y).

    Output registers:
        output_wide:
            wide signed fixed-point representation of the shifted clipped value;

        signal:
            narrow Q1.f representation, consisting of the low f fractional bits
            plus the wide sign bit.

    All work registers except output_wide and signal are returned to |0>.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        eps: float,
        label=None,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        assert h.shape == (n,)
        assert J.shape == (n, n)

        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.eps = float(eps)

        sum_abs = self._sum_abs(self.h, self.J)

        self._upper_bound_energy_diff = float(2.0 * sum_abs)
        if np.isclose(self._upper_bound_energy_diff, 0.0):
            self._upper_bound_energy_diff = 1.0

        self._scaled_eps = float(self.eps / self._upper_bound_energy_diff)
        self.fractional_bits = int(
            sp.simplify(self._fractional_bits(self.n, self._scaled_eps))
        )

        ulp = 2.0 ** (-self.fractional_bits)
        self._normalization = float(self._upper_bound_energy_diff / (2.0 - ulp))

        self.acc_integer_bits = 2
        self.acc_word_bits = int(
            sp.simplify(
                self._bit_precision(
                    self.acc_integer_bits,
                    self.n,
                    self._scaled_eps,
                )
            )
        )

        self.signal_bits = 1 + self.fractional_bits

        self.one_terms = self._one_terms(
            self.h,
            self.J,
            normalization=self._normalization,
            word_bits=self.acc_word_bits,
            fractional_bits=self.fractional_bits,
        )
        self.terms = self._terms(
            self.h,
            self.J,
            normalization=self._normalization,
            word_bits=self.acc_word_bits,
            fractional_bits=self.fractional_bits,
        )

        self.use_counts = [0] * self.n
        for term in self.one_terms:
            if term["arity"] == 1:
                self.use_counts[term["idxs"][0]] += 1
            else:
                i, j = term["idxs"]
                self.use_counts[i] += 1
                self.use_counts[j] += 1

        self.num_copy_ancillas_per_side = sum(
            max(0, count - 1) for count in self.use_counts
        )
        self.num_pair_flags = sum(term["arity"] == 2 for term in self.terms)
        self.num_bit_copies = sum(max(0, term["hw"] - 1) for term in self.terms)
        self.num_sign_copies = max(0, self.acc_word_bits - 2)

        adder = KoggeStoneInPlaceAdder(self.acc_word_bits, with_carry_out=False)
        self.adder_ancilla_bits = adder.num_qubits - 2 * self.acc_word_bits

        self.num_tree_adders = len(self.terms) // 2
        self.max_parallel_adders = self.num_tree_adders + 1

        num_qubits = (
            2 * self.n
            + 2 * self.num_copy_ancillas_per_side
            + len(self.terms) * self.acc_word_bits
            + self.num_pair_flags
            + self.num_bit_copies
            + self.max_parallel_adders * self.adder_ancilla_bits
            + self.num_sign_copies
            + self.acc_word_bits
            + self.acc_word_bits
            + self.signal_bits
        )

        super().__init__("MetropolisHastingsEnergy", num_qubits, [], label=label)
        self.definition = self._build_definition()

    @property
    def upper_bound_energy_diff(self) -> float:
        return self._upper_bound_energy_diff

    @property
    def normalization(self) -> float:
        return self._normalization

    @property
    def word_bits(self) -> int:
        return self.acc_word_bits

    @property
    def layout(self) -> dict[str, list[int] | list[list[int]]]:
        start = 0

        A = list(range(start, start + self.n))
        start += self.n

        B = list(range(start, start + self.n))
        start += self.n

        A_copy = list(range(start, start + self.num_copy_ancillas_per_side))
        start += self.num_copy_ancillas_per_side

        B_copy = list(range(start, start + self.num_copy_ancillas_per_side))
        start += self.num_copy_ancillas_per_side

        term_registers = []
        for _ in range(len(self.terms)):
            term_registers.append(list(range(start, start + self.acc_word_bits)))
            start += self.acc_word_bits

        pair_flags = list(range(start, start + self.num_pair_flags))
        start += self.num_pair_flags

        bit_copies = list(range(start, start + self.num_bit_copies))
        start += self.num_bit_copies

        adder_blocks = []
        for _ in range(self.max_parallel_adders):
            adder_blocks.append(list(range(start, start + self.adder_ancilla_bits)))
            start += self.adder_ancilla_bits

        sign_copies = list(range(start, start + self.num_sign_copies))
        start += self.num_sign_copies

        constant = list(range(start, start + self.acc_word_bits))
        start += self.acc_word_bits

        output_wide = list(range(start, start + self.acc_word_bits))
        start += self.acc_word_bits

        signal = list(range(start, start + self.signal_bits))

        return {
            "A": A,
            "B": B,
            "A_copy": A_copy,
            "B_copy": B_copy,
            "term_registers": term_registers,
            "pair_flags": pair_flags,
            "bit_copies": bit_copies,
            "adder_blocks": adder_blocks,
            "sign_copies": sign_copies,
            "constant": constant,
            "output_wide": output_wide,
            "signal": signal,
        }

    def _layout(self):
        """
        Backward-compatible tuple layout for older tests.
        """
        layout = self.layout

        return (
            layout["A"],
            layout["B"],
            layout["A_copy"],
            layout["B_copy"],
            layout["term_registers"],
            layout["pair_flags"],
            layout["bit_copies"],
            layout["adder_blocks"],
            layout["sign_copies"],
            layout["constant"],
            layout["output_wide"],
            layout["signal"],
        )

    @staticmethod
    def _fields(h: np.ndarray) -> list[tuple[int, float]]:
        return [
            (i, float(coeff))
            for i, coeff in enumerate(h)
            if not np.isclose(float(coeff), 0.0)
        ]

    @staticmethod
    def _pairs(J: np.ndarray) -> list[tuple[int, int, float]]:
        n = J.shape[0]

        return [
            (i, j, float(J[i, j]))
            for i in range(n)
            for j in range(i + 1, n)
            if not np.isclose(float(J[i, j]), 0.0)
        ]

    @staticmethod
    def _sum_abs_h(h: np.ndarray) -> float:
        return float(np.sum(np.abs(h)))

    @staticmethod
    def _sum_abs_J(J: np.ndarray) -> float:
        return float(np.sum(np.abs(np.triu(J, k=1))))

    @staticmethod
    def _sum_abs(h: np.ndarray, J: np.ndarray) -> float:
        return MetropolisHastingsEnergy._sum_abs_h(h) + MetropolisHastingsEnergy._sum_abs_J(J)

    @staticmethod
    def _fractional_bits(n: int | sp.Expr, scaled_eps: float | sp.Expr) -> sp.Expr:
        n = sp.sympify(n)
        scaled_eps = sp.sympify(scaled_eps)
        return sp.ceiling(sp.log(n * (n + 1) / (scaled_eps**2), 2))

    @staticmethod
    def _bit_precision(
        integer_bits: int | sp.Expr,
        n: int | sp.Expr,
        scaled_eps: float | sp.Expr,
    ) -> sp.Expr:
        n = sp.sympify(n)
        integer_bits = sp.sympify(integer_bits)

        return 1 + integer_bits + MetropolisHastingsEnergy._fractional_bits(
            n,
            scaled_eps,
        )

    @staticmethod
    def _quantize(
        value: float,
        *,
        word_bits: int,
        fractional_bits: int,
    ) -> int:
        scaled = int(round(value * (2**fractional_bits)))
        lo = -(1 << (word_bits - 1))
        hi = (1 << (word_bits - 1)) - 1

        if not (lo <= scaled <= hi):
            raise ValueError(
                f"fixed-point overflow: scaled={scaled}, range=[{lo}, {hi}]"
            )

        return scaled

    @staticmethod
    def _bits(
        scaled: int,
        *,
        word_bits: int,
    ) -> list[int]:
        unsigned = scaled % (1 << word_bits)
        return [(unsigned >> j) & 1 for j in range(word_bits)]

    @staticmethod
    def _one_terms(
        h: np.ndarray,
        J: np.ndarray,
        *,
        normalization: float,
        word_bits: int,
        fractional_bits: int,
    ) -> list[dict]:
        out = []

        for i, coeff in MetropolisHastingsEnergy._fields(h):
            scaled = MetropolisHastingsEnergy._quantize(
                float(coeff) / normalization,
                word_bits=word_bits,
                fractional_bits=fractional_bits,
            )

            if scaled != 0:
                bits = MetropolisHastingsEnergy._bits(
                    scaled,
                    word_bits=word_bits,
                )
                out.append(
                    {
                        "idxs": (i,),
                        "scaled": scaled,
                        "bits": bits,
                        "arity": 1,
                        "hw": sum(bits),
                    }
                )

        for i, j, coeff in MetropolisHastingsEnergy._pairs(J):
            scaled = MetropolisHastingsEnergy._quantize(
                float(coeff) / normalization,
                word_bits=word_bits,
                fractional_bits=fractional_bits,
            )

            if scaled != 0:
                bits = MetropolisHastingsEnergy._bits(
                    scaled,
                    word_bits=word_bits,
                )
                out.append(
                    {
                        "idxs": (i, j),
                        "scaled": scaled,
                        "bits": bits,
                        "arity": 2,
                        "hw": sum(bits),
                    }
                )

        return out

    @staticmethod
    def _terms(
        h: np.ndarray,
        J: np.ndarray,
        *,
        normalization: float,
        word_bits: int,
        fractional_bits: int,
    ) -> list[dict]:
        one_terms = MetropolisHastingsEnergy._one_terms(
            h,
            J,
            normalization=normalization,
            word_bits=word_bits,
            fractional_bits=fractional_bits,
        )

        out = []

        for term in one_terms:
            out.append({"side": "B", **term})

        for term in one_terms:
            scaled = -term["scaled"]
            bits = MetropolisHastingsEnergy._bits(
                scaled,
                word_bits=word_bits,
            )

            out.append(
                {
                    "side": "A",
                    "idxs": term["idxs"],
                    "scaled": scaled,
                    "bits": bits,
                    "arity": term["arity"],
                    "hw": sum(bits),
                }
            )

        return out

    @staticmethod
    def _fanout(
        qc: QuantumCircuit,
        source: int,
        ancillas: list[int],
    ) -> tuple[list[int], list[list[tuple[int, int]]]]:
        lines = [source]

        if not ancillas:
            return lines, []

        frontier = [source]
        rounds: list[list[tuple[int, int]]] = []
        cursor = 0

        while cursor < len(ancillas):
            round_ops = []
            new_frontier = []

            for parent in frontier:
                if cursor >= len(ancillas):
                    break

                child = ancillas[cursor]
                cursor += 1

                qc.cx(parent, child)
                round_ops.append((parent, child))
                new_frontier.append(child)
                lines.append(child)

            rounds.append(round_ops)
            frontier = frontier + new_frontier

        return lines, rounds

    @staticmethod
    def _unfanout(
        qc: QuantumCircuit,
        rounds: list[list[tuple[int, int]]],
    ) -> None:
        for round_ops in reversed(rounds):
            for parent, child in reversed(round_ops):
                qc.cx(parent, child)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        A = self.layout["A"]
        B = self.layout["B"]
        A_copy = self.layout["A_copy"]
        B_copy = self.layout["B_copy"]
        term_registers = self.layout["term_registers"]
        pair_flags = self.layout["pair_flags"]
        bit_copies = self.layout["bit_copies"]
        adder_blocks = self.layout["adder_blocks"]
        sign_copies = self.layout["sign_copies"]
        constant = self.layout["constant"]
        output_wide = self.layout["output_wide"]
        signal = self.layout["signal"]

        A_lines = []
        B_lines = []
        A_rounds = []
        B_rounds = []

        a_cursor = 0
        b_cursor = 0

        for i, count in enumerate(self.use_counts):
            need = max(0, count - 1)

            lines, rounds = self._fanout(
                qc,
                A[i],
                A_copy[a_cursor : a_cursor + need],
            )
            A_lines.append(lines)
            A_rounds.append(rounds)
            a_cursor += need

            lines, rounds = self._fanout(
                qc,
                B[i],
                B_copy[b_cursor : b_cursor + need],
            )
            B_lines.append(lines)
            B_rounds.append(rounds)
            b_cursor += need

        A_use = [0] * self.n
        B_use = [0] * self.n
        flag_cursor = 0
        bit_copy_cursor = 0
        term_data = []

        for term, target in zip(self.terms, term_registers):
            controls = []

            if term["side"] == "A":
                for i in term["idxs"]:
                    controls.append(A_lines[i][A_use[i]])
                    A_use[i] += 1
            else:
                for i in term["idxs"]:
                    controls.append(B_lines[i][B_use[i]])
                    B_use[i] += 1

            flag = None

            if term["arity"] == 2:
                flag = pair_flags[flag_cursor]
                flag_cursor += 1

            copies = bit_copies[
                bit_copy_cursor : bit_copy_cursor + max(0, term["hw"] - 1)
            ]
            bit_copy_cursor += max(0, term["hw"] - 1)

            term_data.append(
                {
                    "arity": term["arity"],
                    "controls": controls,
                    "flag": flag,
                    "copies": copies,
                    "bits": [j for j, bit in enumerate(term["bits"]) if bit],
                    "target": target,
                }
            )

        for data in term_data:
            if data["arity"] == 2:
                qc.ccx(data["controls"][0], data["controls"][1], data["flag"])

        load_rounds = []

        for data in term_data:
            source = data["controls"][0] if data["arity"] == 1 else data["flag"]
            lines, rounds = self._fanout(qc, source, data["copies"])
            load_rounds.append(rounds)

            for line, bit in zip(lines, data["bits"]):
                qc.cx(line, data["target"][bit])

        for rounds in reversed(load_rounds):
            self._unfanout(qc, rounds)

        for data in reversed(term_data):
            if data["arity"] == 2:
                qc.ccx(data["controls"][0], data["controls"][1], data["flag"])

        adder = KoggeStoneInPlaceAdder(
            self.acc_word_bits,
            with_carry_out=False,
        )

        levels: list[list[tuple[list[int], list[int], list[int]]]] = []
        live = term_registers[:]

        while len(live) > 1:
            level = []
            nxt = []
            block_cursor = 0

            for i in range(0, len(live) - 1, 2):
                left = live[i]
                right = live[i + 1]
                block = adder_blocks[block_cursor]
                block_cursor += 1

                qc.append(adder, left + right + block)

                level.append((left, right, block))
                nxt.append(right)

            if len(live) % 2 == 1:
                nxt.append(live[-1])

            levels.append(level)
            live = nxt

        final_sum = live[0] if live else None

        if final_sum is not None:
            sign = final_sum[-1]

            qc.cx(sign, output_wide[-1])

            lines, rounds = self._fanout(qc, sign, sign_copies)

            for line, bit in zip(lines, range(self.acc_word_bits - 1)):
                qc.ccx(line, final_sum[bit], output_wide[bit])

            self._unfanout(qc, rounds)

        for j in range(self.fractional_bits):
            qc.x(constant[j])

        qc.append(adder, constant + output_wide + adder_blocks[-1])

        for j in range(self.fractional_bits):
            qc.x(constant[j])

        for j in range(self.fractional_bits):
            qc.cx(output_wide[j], signal[j])

        qc.cx(output_wide[-1], signal[-1])

        for level in reversed(levels):
            for left, right, block in reversed(level):
                qc.append(adder.inverse(), left + right + block)

        for data in term_data:
            if data["arity"] == 2:
                qc.ccx(data["controls"][0], data["controls"][1], data["flag"])

        unload_rounds = []

        for data in term_data:
            source = data["controls"][0] if data["arity"] == 1 else data["flag"]
            lines, rounds = self._fanout(qc, source, data["copies"])
            unload_rounds.append(rounds)

            for line, bit in zip(lines, data["bits"]):
                qc.cx(line, data["target"][bit])

        for rounds in reversed(unload_rounds):
            self._unfanout(qc, rounds)

        for data in reversed(term_data):
            if data["arity"] == 2:
                qc.ccx(data["controls"][0], data["controls"][1], data["flag"])

        for rounds in reversed(A_rounds):
            self._unfanout(qc, rounds)

        for rounds in reversed(B_rounds):
            self._unfanout(qc, rounds)

        return qc