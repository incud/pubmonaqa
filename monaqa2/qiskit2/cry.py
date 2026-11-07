from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ry import Ry


class Cry(Gate):
    """Exact controlled RY, using explicit Ry half-steps and two CNOTs."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="cry", num_qubits=2, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name=self.name)
        ctrl_1, target = 0, 1
        qc.append(Ry(self.theta / 2), [target])
        qc.cx(ctrl_1, target)
        qc.append(Ry(-self.theta / 2), [target])
        qc.cx(ctrl_1, target)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "target": 1}

