from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx


class ReversibleCarryLookaheadAdder(Gate):
    """output ^= (a+b) mod 2**W; inputs preserved, scratch returned to zero.

    Fresh prefix levels give O(log W) depth and O(W log W) scratch.
    An extra propagate copy prevents shared controls from serializing a level.
    """

    def __init__(self, W: int, label=None):
        self.W = int(W)
        if self.W < 1:
            raise ValueError("W must be positive.")
        self.levels = (self.W - 1).bit_length()
        self.scratch = (2 * (self.levels + 1) + 1) * self.W
        super().__init__("reversible_carry_lookahead", 3 * self.W + self.scratch, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = QuantumCircuit(self.num_qubits)
        a, b, output, work = (self.layout[key] for key in ("a", "b", "output", "work"))
        W = self.W
        propagate = [work[2 * level * W:(2 * level + 1) * W] for level in range(self.levels + 1)]
        generate = [work[(2 * level + 1) * W:(2 * level + 2) * W] for level in range(self.levels + 1)]
        copies = work[2 * (self.levels + 1) * W:]

        # A bit propagates a carry when a XOR b = 1, and generates one when a AND b = 1.
        prepare.cx(a, propagate[0])
        prepare.cx(b, propagate[0])
        for i in range(W):
            prepare.append(Ccx(), [a[i], b[i], generate[0][i]])

        for level in range(self.levels):
            distance = 2**level
            old_p, old_g = propagate[level], generate[level]
            new_p, new_g = propagate[level + 1], generate[level + 1]
            prepare.cx(old_p, copies)
            prepare.cx(old_g, new_g)
            prepare.cx(old_p[:distance], new_p[:distance])
            for i in range(distance, W):
                prepare.append(Ccx(), [old_p[i], copies[i - distance], new_p[i]])
            # Generate and propagate are disjoint, so this XOR implements the carry OR.
            for i in range(distance, W):
                prepare.append(Ccx(), [old_p[i], old_g[i - distance], new_g[i]])
            prepare.cx(old_p, copies)

        qc.compose(prepare, inplace=True)
        qc.cx(propagate[0], output)
        if W > 1:
            qc.cx(generate[-1][:-1], output[1:])
        qc.compose(prepare.inverse(), inplace=True)
        return qc

    @property
    def layout(self):
        W = self.W
        return {"a": list(range(W)), "b": list(range(W, 2 * W)),
                "output": list(range(2 * W, 3 * W)), "work": list(range(3 * W, self.num_qubits))}
