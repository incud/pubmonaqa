from qiskit.circuit import Gate, QuantumCircuit


class Crz(Gate):
    """Exact controlled RZ; two RZ gates and two CNOTs."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="crz", num_qubits=2, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name=self.name)
        ctrl_1, target = 0, 1
        qc.rz(self.theta / 2, target)
        qc.cx(ctrl_1, target)
        qc.rz(-self.theta / 2, target)
        qc.cx(ctrl_1, target)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "target": 1}

