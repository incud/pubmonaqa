from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx
from monaqa2.qiskit2.reversible_carry_lookahead_adder import ReversibleCarryLookaheadAdder


class ReversibleWallaceAdder(Gate):
    """output ^= sum(inputs) mod 2**W; all input words preserved and work cleaned.

    Three-to-two compression takes O(log M) rounds; the final sum takes O(log W).
    """

    def __init__(self, M: int, W: int, label=None):
        self.M, self.W = int(M), int(W)
        if min(self.M, self.W) < 1:
            raise ValueError("M and W must be positive.")
        self.final_adder = ReversibleCarryLookaheadAdder(self.W)
        self.scratch = 0 if self.M == 1 else (
            (self.M - 2) * (2 * self.W - 1) + 2 * self.W + self.final_adder.scratch)
        super().__init__("reversible_wallace", (self.M + 1) * self.W + self.scratch, [], label=label)
        self.definition = self._build_definition()

    def _compress(self, qc, a, b, c, sums, carries):
        for bit in range(self.W):
            inputs = [row[bit] for row in (a, b, c) if row[bit] is not None]
            for source in inputs:
                qc.cx(source, sums[bit])
            if bit < self.W - 1:
                # Majority(a,b,c) = ab XOR ac XOR bc. Carries move one column left.
                for i in range(len(inputs)):
                    for j in range(i + 1, len(inputs)):
                        qc.append(Ccx(), [inputs[i], inputs[j], carries[bit]])

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        inputs, output, work = (self.layout[key] for key in ("inputs", "output", "work"))
        W = self.W
        if self.M == 1:
            qc.cx(inputs, output)
            return qc
        prepare = QuantumCircuit(self.num_qubits)
        rows = [inputs[i * W:(i + 1) * W] for i in range(self.M)]
        free = 0
        while len(rows) > 2:
            next_rows = []
            for i in range(0, len(rows) - 2, 3):
                sums = work[free:free + W]
                carries = work[free + W:free + 2 * W - 1]
                free += 2 * W - 1
                self._compress(prepare, rows[i], rows[i + 1], rows[i + 2], sums, carries)
                next_rows.extend([sums, [None] + carries])
            rows = next_rows + rows[3 * (len(rows) // 3):]

        # Copy the two remaining rows, filling absent carry bits with clean zeros.
        a, b = work[free:free + W], work[free + W:free + 2 * W]
        prefix_work = work[free + 2 * W:]
        for row, target in zip(rows, (a, b)):
            for source, bit in zip(row, target):
                if source is not None:
                    prepare.cx(source, bit)
        qc.compose(prepare, inplace=True)
        qc.append(self.final_adder, a + b + output + prefix_work)
        qc.compose(prepare.inverse(), inplace=True)
        return qc

    @property
    def layout(self):
        start = self.M * self.W
        return {"inputs": list(range(start)), "output": list(range(start, start + self.W)),
                "work": list(range(start + self.W, self.num_qubits))}
