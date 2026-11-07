from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx
from monaqa2.qiskit2.reversible_fanout import ReversibleFanout


class ReversiblePositivePart(Gate):
    """positive ^= max(signed(delta),0); delta preserved and work cleaned."""

    def __init__(self, W: int, label=None):
        self.W = int(W)
        if self.W < 1:
            raise ValueError("W must be positive.")
        self.scratch = self.W - 1
        super().__init__("reversible_positive_part", 2 * self.W + self.scratch, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        if self.W == 1:
            return qc
        delta, positive, work = (self.layout[key] for key in ("delta", "positive", "work"))
        prepare = QuantumCircuit(self.num_qubits)
        prepare.cx(delta[-1], work[0])
        prepare.append(ReversibleFanout(len(work)), work)
        prepare.x(work)
        qc.compose(prepare, inplace=True)
        for bit, ctrl_1 in enumerate(work):
            qc.append(Ccx(), [ctrl_1, delta[bit], positive[bit]])
        qc.compose(prepare.inverse(), inplace=True)
        return qc

    @property
    def layout(self):
        W = self.W
        return {"delta": list(range(W)), "positive": list(range(W, 2 * W)),
                "work": list(range(2 * W, self.num_qubits))}
