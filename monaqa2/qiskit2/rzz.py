from qiskit.circuit import Gate, QuantumCircuit


class Rzz(Gate):
    """RZZ(theta) = exp(-i theta Z(anc_1) Z(anc_2) / 2)."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="rzz", num_qubits=2, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name=self.name)
        anc_1, anc_2 = 0, 1
        qc.cx(anc_1, anc_2)
        qc.rz(self.theta, anc_2)
        qc.cx(anc_1, anc_2)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"anc_1": 0, "anc_2": 1}

