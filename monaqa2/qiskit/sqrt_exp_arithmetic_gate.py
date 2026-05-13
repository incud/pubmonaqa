import numpy as np
import scipy as sc
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.gqsp_gate import GQSP
from monaqa2.qiskit.qubitized_ising_tf_gate import QubitizedOperatorIsingTF, ControlledQubitizedOperatorIsingTF


class SqrtExpArithmetic(Gate):

    def __init__(self, b: int, beta: float, normalization: float, eps: float, mocked_circuit: bool = False, mocked_angles: bool = False, label=None):
        if b < 1:
            raise ValueError("b must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if beta * normalization < 0.0:
            raise ValueError("beta * normalization must be non-negative.")

        self.b = int(b)
        self.beta = float(beta)
        self.normalization = float(normalization)
        self.eps = float(eps)
        self.mocked_circuit = bool(mocked_circuit)
        self.mocked_angles = bool(mocked_angles)

        self.h = self._ising_h()
        self.J = np.zeros((self.b, self.b), dtype=float)
        self.gamma = np.zeros(self.b, dtype=float)

        self.qubitization = QubitizedOperatorIsingTF(self.b, self.h, self.J, self.gamma, mocked_all=self.mocked_circuit)
        self.controlled_qubitization = ControlledQubitizedOperatorIsingTF(self.b, self.h, self.J, self.gamma, mocked_all=self.mocked_circuit)
        self.degree = self._degree()
        self.poly_coeffs = self._poly()
        self.gqsp = GQSP(self.qubitization, self.controlled_qubitization, self.poly_coeffs, mocked_angles=self.mocked_angles, laurent_negative_power=self.degree)

        super().__init__("SqrtExpArithmetic", self.gqsp.num_qubits, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        signal = [1 + q for q in self.qubitization.layout["system"]]
        aux = [q for q in range(self.num_qubits) if q not in [0] + signal]
        return {"control": [0], "signal": signal, "qubitization": self.gqsp.layout["qubitization"], "aux": aux}

    def _ising_h(self) -> np.ndarray:
        h = np.zeros(self.b, dtype=float)
        h[-1] = 0.5

        for j in range(self.b - 1):
            h[j] = -(2.0 ** (j - self.b))

        return h

    def _signal_max_value(self) -> float:
        return 1.0 + 2.0 ** (1 - self.b)

    def _alpha_signal(self) -> float:
        return float(np.sum(np.abs(self.h)))

    def _shift(self) -> float:
        return 2.0 ** (-self.b) + self._signal_max_value()

    def _lambda(self) -> float:
        return 0.5 * self.beta * self.normalization

    def _mu(self) -> float:
        return self._lambda() * self._alpha_signal()

    def _degree(self) -> int:
        lam = self._lambda()
        time_deg = np.e * lam
        correction = np.log(4.0 / self.eps)
        return int(np.ceil(time_deg + correction + 1.0))

    def _poly(self) -> np.ndarray:
        d = self.degree
        lam = self._lambda()
        mu = self._mu()
        shift = self._shift()
        powers = np.arange(-d, d + 1)
        return np.array([np.exp(-lam * shift) * sc.special.iv(abs(int(k)), mu) for k in powers], dtype=complex)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(self.gqsp, list(range(self.num_qubits)))
        return qc
