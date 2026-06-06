import numpy as np
import scipy as sc
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.gqsp_gate import GQSP
from monaqa2.qiskit.qubitized_ising_tf_gate import QubitizedOperatorIsingTF, ControlledQubitizedOperatorIsingTF


class HamiltonianSimulationGQSP(Gate):
    """
    Hamiltonian simulation by GQSP over the qubitized transverse-field Ising LCU.

    Uses the Jacobi-Anger expansion

        exp(-i H t) ≈ sum_{k=-d}^{d} (-i)^k J_k(alpha t) W^k

    and shifts the Laurent polynomial by d:

        sum_{k=-d}^{d} c_k W^k = W^{-d} sum_{r=0}^{2d} c_{r-d} W^r.
    """

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, t: float, eps: float, mocked_reflection: bool = False, mocked_angles: bool = False, label=None) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        assert h.shape == (n,) and J.shape == (n, n) and gamma.shape == (n,)

        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.t = float(t)
        self.eps = float(eps)

        self.mocked_reflection = mocked_reflection
        self.mocked_angles = mocked_angles

        self.qubitization = QubitizedOperatorIsingTF(self.n, self.h, self.J, self.gamma, mocked_reflection=self.mocked_reflection)
        self.controlled_qubitization = ControlledQubitizedOperatorIsingTF(self.n, self.h, self.J, self.gamma, mocked_reflection=self.mocked_reflection)

        self.degree = self._degree()
        self.poly_coeffs = self._poly()
        self.gqsp = GQSP(self.qubitization, self.controlled_qubitization, self.poly_coeffs, mocked_angles=self.mocked_angles, laurent_negative_power=self.degree)

        super().__init__("HamiltonianSimulationGQSP", self.gqsp.num_qubits, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return self.gqsp.layout

    def _alpha(self) -> float:
        return float(
            np.sum(np.abs(self.h))
            + np.sum(np.abs(self.gamma))
            + np.sum(np.abs(self.J[np.triu_indices(self.n, k=1)]))
        )
    
    def _degree(self) -> int:
        tau = abs(self._alpha() * self.t)
        if tau == 0.0:
            return 1

        time_deg = np.e * tau
        correction = np.log(4.0 / self.eps)
        return int(np.ceil(time_deg + correction + 1.0))

    def _poly(self) -> np.ndarray:
        d = self.degree
        tau = self._alpha() * self.t
        return np.array([(-1j) ** int(k) * sc.special.jv(int(k), tau) for k in np.arange(-d, d + 1)], dtype=complex)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(self.gqsp, list(range(self.num_qubits)))
        return qc
