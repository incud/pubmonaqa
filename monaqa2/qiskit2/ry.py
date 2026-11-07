from qiskit.circuit import Gate, QuantumCircuit


class Ry(Gate):
    """RY(theta), explicitly in Clifford+RZ."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="ry", num_qubits=1, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(1, name=self.name)
        target = 0
        qc.sdg(target)
        qc.h(target)
        qc.rz(self.theta, target)
        qc.h(target)
        qc.s(target)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"target": 0}

