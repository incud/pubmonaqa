import math
from qiskit.circuit import Gate, QuantumCircuit


class Ccx(Gate):
    """Exact CCX/Toffoli gate with 7 T gates and T-depth 3."""

    def __init__(self, label=None) -> None:
        super().__init__(name="ccx", num_qubits=3, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, int]:
        return {"control_0": 0, "control_1": 1, "target": 2}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)

        # Target qubit is 2, Controls are 0 and 1
        qc.h(2)
        
        # --- T-Layer 1 ---
        qc.t(0)
        qc.t(1)
        qc.t(2)
        
        qc.cx(1, 0)
        qc.cx(2, 1)
        
        # --- T-Layer 2 ---
        qc.tdg(0)
        qc.t(1)
        qc.tdg(2)
        
        qc.cx(1, 0)
        qc.cx(2, 1)
        
        # --- T-Layer 3 ---
        qc.tdg(1)
        
        qc.cx(0, 2)
        qc.cx(2, 1)
        qc.cx(0, 1)
        
        qc.h(2)

        return qc


class Cry(Gate):
    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="cry", num_qubits=2, params=[theta], label=label)
        self.definition = self._build_definition()

    def _ry_rz_clifford(self, qc: QuantumCircuit, angle: float, target: int) -> None:
        qc.sdg(target)
        qc.h(target)
        qc.rz(angle, target)
        qc.h(target)
        qc.s(target)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name=self.name)
        self._ry_rz_clifford(qc, self.theta / 2.0, 1)
        qc.cx(0, 1)
        self._ry_rz_clifford(qc, -self.theta / 2.0, 1)
        qc.cx(0, 1)
        return qc


class Ccry(Gate):
    def __init__(self, theta: float, label=None) -> None:
        self.theta = theta
        super().__init__(name="ccry", num_qubits=3, params=[theta], label=label)
        self.definition = self._build_definition()

    def _ry_rz_clifford(self, qc: QuantumCircuit, angle: float, target: int) -> None:
        qc.sdg(target)
        qc.h(target)
        qc.rz(angle, target)
        qc.h(target)
        qc.s(target)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        self._ry_rz_clifford(qc, self.theta / 2.0, 2)
        qc.append(Ccx(), [0, 1, 2])
        self._ry_rz_clifford(qc, -self.theta / 2.0, 2)
        qc.append(Ccx(), [0, 1, 2])
        return qc


class GivensRotation(Gate):
    def __init__(self, left_weight: float, right_weight: float, label=None) -> None:
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.theta = self._givens_angle_from_weights(left_weight, right_weight)
        super().__init__(name="givens", num_qubits=2, params=[left_weight, right_weight], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
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


class ControlledGivensRotation(Gate):
    def __init__(self, left_weight: float, right_weight: float, label=None) -> None:
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.theta = GivensRotation._givens_angle_from_weights(left_weight, right_weight)
        super().__init__(name="c_givens", num_qubits=3, params=[left_weight, right_weight], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"control": 0, "l": 1, "r": 2}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        qc.append(Ccx(), [0, 1, 2])
        qc.append(Ccry(-2.0 * self.theta), [0, 2, 1])
        qc.append(Ccx(), [0, 1, 2])
        return qc