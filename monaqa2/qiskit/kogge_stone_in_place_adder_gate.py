import math
from typing import Optional

from qiskit.circuit import Gate, QuantumCircuit


class KoggeStoneInPlaceAdder(Gate):
    """
    In-place adder.

    Little-endian convention:
        a[0], b[0] are the least significant bits.

    Full path:
        Kogge-Stone-style prefix adder with clean prefix/carry-copy ancillas.

    mocked_adder=True:
        Ancilla-free Draper/QFT-style adder. It uses only the input/output
        registers, plus the carry_out qubit if with_carry_out=True.

    Action:
        with_carry_out=False:
            |a>|b> -> |a>|a+b mod 2^n>

        with_carry_out=True:
            |a>|b>|0_carry> -> |a>|a+b as an (n+1)-bit value>
    """

    def __init__(
        self,
        n: int,
        with_carry_out: bool = False,
        mocked_adder: bool = False,
        label=None,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")

        self.n = int(n)
        self.with_carry_out = bool(with_carry_out)
        self.mocked_adder = bool(mocked_adder)
        self.num_stages = 0 if self.n <= 1 else math.ceil(math.log2(self.n))

        if self.mocked_adder:
            self.num_prefix_ancillas = 0
            self.num_carry_copy_ancillas = 0
        else:
            self.num_prefix_ancillas = 2 * self.n + 2 * sum(
                self.n - (1 << j) for j in range(self.num_stages)
            )
            self.num_carry_copy_ancillas = max(0, self.n - 1)

        num_qubits = (
            2 * self.n
            + self.num_prefix_ancillas
            + self.num_carry_copy_ancillas
            + int(self.with_carry_out)
        )

        super().__init__(
            name="kogge_stone_in_place_adder",
            num_qubits=num_qubits,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        a = list(range(0, self.n))
        b = list(range(self.n, 2 * self.n))

        prefix_start = 2 * self.n
        prefix_stop = prefix_start + self.num_prefix_ancillas
        prefix = list(range(prefix_start, prefix_stop))

        carry_start = prefix_stop
        carry_stop = carry_start + self.num_carry_copy_ancillas
        carry_copy = list(range(carry_start, carry_stop))

        carry_out = [carry_stop] if self.with_carry_out else []

        return {
            "a": a,
            "b": b,
            "prefix": prefix,
            "carry_copy": carry_copy,
            "carry_out": carry_out,
        }

    @staticmethod
    def _ccx_stage(
        qc: QuantumCircuit,
        triples: list[tuple[int, int, int, int]],
        distance: int,
    ) -> None:
        for color in (0, 1):
            for i, c0, c1, target in triples:
                if ((i // distance) & 1) == color:
                    qc.ccx(c0, c1, target)

    def _prefix_forward(
        self,
        qc: QuantumCircuit,
        p_leaf: list[int],
        g_leaf: list[int],
        workspace: list[int],
    ):
        p_layers = [p_leaf.copy()]
        g_layers = [g_leaf.copy()]
        stage_allocs = []

        cursor = 0
        distance = 1

        while distance < self.n:
            prev_p = p_layers[-1]
            prev_g = g_layers[-1]

            new_p = prev_p.copy()
            new_g = prev_g.copy()
            alloc = []

            for i in range(distance, self.n):
                p_new = workspace[cursor]
                g_new = workspace[cursor + 1]
                cursor += 2
                new_p[i] = p_new
                new_g[i] = g_new
                alloc.append((i, p_new, g_new))

            self._ccx_stage(
                qc,
                [(i, prev_p[i], prev_p[i - distance], p_new) for i, p_new, _ in alloc],
                distance,
            )

            for i, _, g_new in alloc:
                qc.cx(prev_g[i], g_new)

            self._ccx_stage(
                qc,
                [(i, prev_p[i], prev_g[i - distance], g_new) for i, _, g_new in alloc],
                distance,
            )

            p_layers.append(new_p)
            g_layers.append(new_g)
            stage_allocs.append((distance, alloc))
            distance <<= 1

        if cursor != len(workspace):
            raise RuntimeError(
                f"Prefix ancilla accounting mismatch: used {cursor}, allocated {len(workspace)}."
            )

        return p_layers, g_layers, stage_allocs

    def _prefix_backward(
        self,
        qc: QuantumCircuit,
        p_layers: list[list[int]],
        g_layers: list[list[int]],
        stage_allocs,
    ) -> None:
        for layer_idx in reversed(range(len(stage_allocs))):
            distance, alloc = stage_allocs[layer_idx]
            prev_p = p_layers[layer_idx]
            prev_g = g_layers[layer_idx]

            self._ccx_stage(
                qc,
                [(i, prev_p[i], prev_g[i - distance], g_new) for i, _, g_new in alloc],
                distance,
            )

            for i, _, g_new in alloc:
                qc.cx(prev_g[i], g_new)

            self._ccx_stage(
                qc,
                [(i, prev_p[i], prev_p[i - distance], p_new) for i, p_new, _ in alloc],
                distance,
            )

    @staticmethod
    def _qft_no_swaps(qc: QuantumCircuit, qubits: list[int]) -> None:
        for j in reversed(range(len(qubits))):
            qc.h(qubits[j])
            for k in reversed(range(j)):
                qc.cp(math.pi / (2 ** (j - k)), qubits[k], qubits[j])

    @staticmethod
    def _iqft_no_swaps(qc: QuantumCircuit, qubits: list[int]) -> None:
        for j in range(len(qubits)):
            for k in range(j):
                qc.cp(-math.pi / (2 ** (j - k)), qubits[k], qubits[j])
            qc.h(qubits[j])

    def _apply_mocked_adder(self, qc: QuantumCircuit) -> None:
        a = self.layout["a"]
        b = self.layout["b"]
        carry_out = self.layout["carry_out"]
        target = b + carry_out

        self._qft_no_swaps(qc, target)

        for i, aq in enumerate(a):
            for j in range(i, len(target)):
                qc.cp(math.pi / (2 ** (j - i)), aq, target[j])

        self._iqft_no_swaps(qc, target)

    def _apply_kogge_stone_adder(self, qc: QuantumCircuit) -> None:
        a = self.layout["a"]
        b = self.layout["b"]
        prefix = self.layout["prefix"]
        carry_copy = self.layout["carry_copy"]
        carry_out: Optional[int] = self.layout["carry_out"][0] if self.with_carry_out else None

        p_leaf = prefix[: self.n]
        g_leaf = prefix[self.n : 2 * self.n]
        workspace = prefix[2 * self.n :]

        for i in range(self.n):
            qc.cx(a[i], p_leaf[i])
        for i in range(self.n):
            qc.cx(b[i], p_leaf[i])

        for i in range(self.n):
            qc.ccx(a[i], b[i], g_leaf[i])

        p_layers, g_layers, stage_allocs = self._prefix_forward(qc, p_leaf, g_leaf, workspace)
        final_g = g_layers[-1]

        for i in range(1, self.n):
            qc.cx(final_g[i - 1], carry_copy[i - 1])

        if carry_out is not None:
            qc.cx(final_g[self.n - 1], carry_out)

        qc.cx(a[0], b[0])

        for i in range(1, self.n):
            qc.cx(a[i], b[i])
            qc.cx(carry_copy[i - 1], b[i])

        self._prefix_backward(qc, p_layers, g_layers, stage_allocs)

        for i in range(self.n):
            qc.x(p_leaf[i])
        for i in range(self.n):
            qc.ccx(a[i], p_leaf[i], g_leaf[i])
        for i in range(self.n):
            qc.x(p_leaf[i])

        qc.cx(b[0], p_leaf[0])

        for i in range(1, self.n):
            qc.cx(b[i], p_leaf[i])
            qc.cx(carry_copy[i - 1], p_leaf[i])

        for i in range(self.n):
            qc.x(p_leaf[i])
        for i in range(self.n):
            qc.cx(a[i], p_leaf[i])
        for i in range(self.n):
            qc.cx(b[i], p_leaf[i])

        for i in range(self.n):
            qc.x(b[i])
        for i in range(self.n):
            qc.ccx(a[i], b[i], g_leaf[i])
        for i in range(self.n):
            qc.x(b[i])

        p_layers2, g_layers2, stage_allocs2 = self._prefix_forward(qc, p_leaf, g_leaf, workspace)
        final_g2 = g_layers2[-1]

        for i in range(1, self.n):
            qc.cx(final_g2[i - 1], carry_copy[i - 1])

        self._prefix_backward(qc, p_layers2, g_layers2, stage_allocs2)

        for i in range(self.n):
            qc.x(b[i])
        for i in range(self.n):
            qc.ccx(a[i], b[i], g_leaf[i])
        for i in range(self.n):
            qc.x(b[i])

        for i in range(self.n):
            qc.cx(b[i], p_leaf[i])
        for i in range(self.n):
            qc.cx(a[i], p_leaf[i])
        for i in range(self.n):
            qc.x(p_leaf[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        if self.mocked_adder:
            self._apply_mocked_adder(qc)
        else:
            self._apply_kogge_stone_adder(qc)

        return qc