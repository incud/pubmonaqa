import math
from monaqa2.qiskit2.cry import Cry
from qiskit.circuit import Gate, QuantumCircuit


class GivensRotation(Gate):
    def __init__(self, left_weight: float, right_weight: float, label=None) -> None:
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.theta = self._givens_angle_from_weights(left_weight, right_weight)
        super().__init__(name="givens", num_qubits=2, params=[left_weight, right_weight], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, int]:
        return {"l": 0, "r": 1}

    @staticmethod
    def _givens_angle_from_weights(left_weight: float, right_weight: float) -> float:
        safe_sqrt = lambda x: math.sqrt(max(0.0, min(1.0, x)))
        total = left_weight + right_weight
        if total == 0.0:
            return 0.0
        return math.atan2(safe_sqrt(right_weight / total), safe_sqrt(left_weight / total))

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name=self.name)
        qc.cx(0, 1)
        qc.append(Cry(-2.0 * self.theta), [1, 0])
        qc.cx(0, 1)
        return qc

