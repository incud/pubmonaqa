from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccrz import Ccrz


class Ccrx(Gate):
    """Exact doubly controlled RX, using explicit Ccrz."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="ccrx", num_qubits=3, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        ctrl_1, ctrl_2, target = 0, 1, 2
        qc.h(target)
        qc.append(Ccrz(self.theta), [ctrl_1, ctrl_2, target])
        qc.h(target)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "ctrl_2": 1, "target": 2}

