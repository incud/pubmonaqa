from qiskit.circuit import Gate, QuantumCircuit


class ReversibleFanout(Gate):
    """Copy a computational-basis control into clean wires in ceil(log2(n)) depth."""

    def __init__(self, n: int, label=None):
        self.n = int(n)
        if self.n < 1:
            raise ValueError("n must be positive.")
        super().__init__("reversible_fanout", self.n, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.n, name=self.name)
        width = 1
        while width < self.n:
            for i in range(min(width, self.n - width)):
                qc.cx(i, i + width)
            width *= 2
        return qc

    @property
    def layout(self):
        return {"control": [0], "copies": list(range(1, self.n))}
