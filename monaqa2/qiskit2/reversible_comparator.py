from qiskit.circuit import Gate, QuantumCircuit


class ReversibleComparator(Gate):
    """negative ^= (signed(delta) < 0), reading the two's-complement sign bit."""

    def __init__(self, W: int, label=None):
        self.W = int(W)
        if self.W < 1:
            raise ValueError("W must be positive.")
        super().__init__("reversible_comparator", self.W + 1, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.cx(self.W - 1, self.W)
        return qc

    @property
    def layout(self):
        return {"delta": list(range(self.W)), "negative": [self.W]}
