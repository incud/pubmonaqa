from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccrz import Ccrz


class Ccrzz(Gate):
    """Exact doubly controlled RZZ; uses the explicit Ccrz decomposition."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="ccrzz", num_qubits=4, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(4, name=self.name)
        ctrl_1, ctrl_2, anc_1, anc_2 = 0, 1, 2, 3
        qc.cx(anc_1, anc_2)
        qc.append(Ccrz(self.theta), [ctrl_1, ctrl_2, anc_2])
        qc.cx(anc_1, anc_2)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "ctrl_2": 1, "anc_1": 2, "anc_2": 3}

