from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx


class Ccrz(Gate):
    """Exact doubly controlled RZ; two RZ gates and two exact Ccx gates."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="ccrz", num_qubits=3, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        ctrl_1, ctrl_2, target = 0, 1, 2
        qc.rz(self.theta / 2, target)
        qc.append(Ccx(), [ctrl_1, ctrl_2, target])
        qc.rz(-self.theta / 2, target)
        qc.append(Ccx(), [ctrl_1, ctrl_2, target])
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "ctrl_2": 1, "target": 2}

