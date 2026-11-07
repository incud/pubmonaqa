from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.crz import Crz


class Crzz(Gate):
    """Exact controlled RZZ; only the middle RZ is controlled."""

    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="crzz", num_qubits=3, params=[theta], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        ctrl_1, anc_1, anc_2 = 0, 1, 2
        qc.cx(anc_1, anc_2)
        qc.append(Crz(self.theta), [ctrl_1, anc_2])
        qc.cx(anc_1, anc_2)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "anc_1": 1, "anc_2": 2}

