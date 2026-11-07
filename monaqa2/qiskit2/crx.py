from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.crz import Crz


class Crx(Gate):
    """Exact controlled RX, from a target basis change and explicit Crz."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="crx", num_qubits=2, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name=self.name)
        ctrl_1, target = 0, 1
        qc.h(target)
        qc.append(Crz(self.theta), [ctrl_1, target])
        qc.h(target)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "target": 1}

