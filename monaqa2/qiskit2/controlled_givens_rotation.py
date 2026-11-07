from monaqa2.qiskit2.ccry import Ccry
from monaqa2.qiskit2.ccx import Ccx
from monaqa2.qiskit2.givens_rotation import GivensRotation
from qiskit.circuit import Gate, QuantumCircuit


class ControlledGivensRotation(Gate):
    def __init__(self, left_weight: float, right_weight: float, label=None) -> None:
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.theta = GivensRotation._givens_angle_from_weights(left_weight, right_weight)
        super().__init__(name="c_givens", num_qubits=3, params=[left_weight, right_weight], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, int]:
        return {"control": 0, "l": 1, "r": 2}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        qc.append(Ccx(), [0, 1, 2])
        qc.append(Ccry(-2.0 * self.theta), [0, 2, 1])
        qc.append(Ccx(), [0, 1, 2])
        return qc
