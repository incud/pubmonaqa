import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit.gqsp_angles import poly_to_gqsp_angles


class GQSP(Gate):
    """
    Generalized quantum signal processing over any qubitized operator gate.

    Expected layout:
        control: one GQSP control qubit at position 0
        qubitization: the full uncontrolled qubitized operator on positions 1, ..., q
        clean: optional clean ancillas used only by the controlled qubitization

    The controlled qubitization gate must use layout:
        [control] + qubitization_qubits + clean_ancillas
    """

    def __init__(self, qubitization: Gate, controlled_qubitization: Gate, poly_coeffs: np.ndarray, mocked_angles: bool = False, laurent_negative_power: int = 0, label=None) -> None:
        if laurent_negative_power < 0:
            raise ValueError("laurent_negative_power must be non-negative.")
        if controlled_qubitization.num_qubits < qubitization.num_qubits + 1:
            raise ValueError("controlled_qubitization must have at least one more qubit than qubitization.")

        self.qubitization = qubitization
        self.controlled_qubitization = controlled_qubitization
        self.num_clean_ancillas = controlled_qubitization.num_qubits - qubitization.num_qubits - 1
        self.poly_coeffs = np.asarray(poly_coeffs, dtype=complex).reshape(-1)
        self.mocked_angles = mocked_angles
        self.laurent_negative_power = laurent_negative_power
        self.degree = len(self.poly_coeffs) - 1

        if self.degree < 0:
            raise ValueError("GQSP requires at least one polynomial coefficient.")

        super().__init__(name="GQSP", num_qubits=controlled_qubitization.num_qubits, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        q = self.qubitization.num_qubits
        return {
            "control": [0],
            "qubitization": list(range(1, 1 + q)),
            "clean": list(range(1 + q, self.controlled_qubitization.num_qubits)),
            "controlled_qubitization": list(range(self.controlled_qubitization.num_qubits)),
        }

    def _angles(self) -> np.ndarray:
        if self.mocked_angles:
            return np.random.random((self.degree + 1, 3))
        return np.stack(poly_to_gqsp_angles(self.poly_coeffs), axis=1)

    @staticmethod
    def _phase_block(qc: QuantumCircuit, control: int, theta: float, phi: float, lamb: float) -> None:
        qc.x(control)
        qc.rz(lamb, control)
        qc.sdg(control)
        qc.h(control)
        qc.rz(2.0 * theta, control)
        qc.h(control)
        qc.s(control)
        qc.rz(phi, control)
        qc.x(control)
        qc.z(control)
        qc.global_phase += (phi + lamb) / 2.0

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        control = self.layout["control"][0]
        qubitization_reg = self.layout["qubitization"]
        controlled_qubitization_reg = self.layout["controlled_qubitization"]
        angles = self._angles()

        if angles.ndim != 2 or angles.shape[1] != 3:
            raise ValueError("GQSP angles must have shape (degree + 1, 3).")

        self._phase_block(qc, control, *angles[0])

        for theta, phi, lamb in angles[1:]:
            qc.x(control)
            qc.append(self.controlled_qubitization, controlled_qubitization_reg)
            qc.x(control)
            self._phase_block(qc, control, theta, phi, lamb)

        inverse_qubitization = self.qubitization.inverse()

        for _ in range(self.laurent_negative_power):
            qc.append(inverse_qubitization, qubitization_reg)

        return qc