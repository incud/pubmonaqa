import numpy as np
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.gqsp_gate import GQSP
from monaqa2.qiskit.qubitized_ising_tf_gate import (
    QubitizedOperatorIsingTF,
    ControlledQubitizedOperatorIsingTF,
)


class GlauberArithmetic(Gate):
    r"""
    GQSP block-encoding of the generalized Glauber acceptance amplitude

        g(x, y) = (1 / (1 + exp(beta * DeltaE(x, y)))) ** (1 / (2 * a)),

    where

        DeltaE(x, y) = E(y) - E(x)

    and

        E(z) = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.

    For a = 1 this reduces to the usual square-root Glauber acceptance

        g(x, y) = sqrt(1 / (1 + exp(beta * DeltaE(x, y)))).

    The good GQSP block acts as

        |x>|y>|0...0> -> |x>|y>(
            g(x, y)|0...0> + |perp_{x,y}>
        ),

    with |perp_{x,y}> orthogonal to the all-zero ancilla subspace.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        beta: float,
        eps: float,
        a: float = 1.0,
        mocked_circuit: bool = False,
        mocked_angles: bool = False,
        label=None,
        tol: float = 1e-8,
    ) -> None:
        if n < 1:
            raise ValueError("n must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if a <= 0.0:
            raise ValueError("a must be positive.")

        assert h.shape == (n,)
        assert J.shape == (n, n)

        self.n = int(n)
        self.h_single = np.asarray(h, dtype=float)
        self.J_single = np.asarray(J, dtype=float)
        self.beta = float(beta)
        self.eps = float(eps)
        self.a = float(a)
        self.mocked_circuit = bool(mocked_circuit)
        self.mocked_angles = bool(mocked_angles)
        self.tol = float(tol)

        self.h = np.concatenate([-self.h_single, self.h_single])
        self.J = np.zeros((2 * self.n, 2 * self.n), dtype=float)
        self.J[:self.n, :self.n] = -self.J_single
        self.J[self.n:, self.n:] = self.J_single
        self.gamma = np.zeros(2 * self.n, dtype=float)

        if (
            np.all(np.abs(self.h) <= self.tol)
            and np.all(np.abs(np.triu(self.J, k=1)) <= self.tol)
        ):
            raise ValueError("The delta-energy Hamiltonian has no active terms.")

        self.qubitization = QubitizedOperatorIsingTF(
            2 * self.n,
            self.h,
            self.J,
            self.gamma,
            mocked_all=self.mocked_circuit,
        )
        self.controlled_qubitization = ControlledQubitizedOperatorIsingTF(
            2 * self.n,
            self.h,
            self.J,
            self.gamma,
            mocked_all=self.mocked_circuit,
        )

        self.alpha = float(self.qubitization.lcu.alpha)
        self.degree = self._degree()
        self.poly_coeffs = self._poly()
        self.gqsp = GQSP(
            self.qubitization,
            self.controlled_qubitization,
            self.poly_coeffs,
            mocked_angles=self.mocked_angles,
            laurent_negative_power=self.degree,
        )

        super().__init__("GlauberArithmetic", self.gqsp.num_qubits, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        system = [1 + q for q in self.qubitization.layout["system"]]
        A = system[:self.n]
        B = system[self.n:]
        aux = [q for q in range(self.num_qubits) if q not in [0] + system]

        return {
            "control": [0],
            "A": A,
            "B": B,
            "system": system,
            "qubitization": self.gqsp.layout["qubitization"],
            "aux": aux,
        }

    def _target(self, x: np.ndarray) -> np.ndarray:
        base = 1.0 / (1.0 + np.exp(self.beta * self.alpha * x))
        return base ** (1.0 / (2.0 * self.a))

    def _degree(self) -> int:
        scale = self.beta * self.alpha

        if scale <= 1e-15:
            return 1

        tau = np.pi / scale
        rho = tau + np.sqrt(1.0 + tau * tau)

        # The nearest singularities are unchanged by a. The algebraic branch
        # strength changes from -1/2 to -1/(2a), so only the prefactor changes.
        branch_prefactor = 1.0 + 1.0 / (2.0 * self.a)
        return max(1, int(np.ceil(np.log(8.0 * branch_prefactor / self.eps) / np.log(rho))) + 2)

    def _poly(self) -> np.ndarray:
        d = self.degree
        n_samples = d + 1

        j = np.arange(n_samples)
        theta = np.pi * (j + 0.5) / n_samples
        x = np.cos(theta)
        y = self._target(x)

        cheb = np.zeros(d + 1, dtype=float)
        cheb[0] = np.sum(y) / n_samples

        for k in range(1, d + 1):
            cheb[k] = 2.0 * np.sum(y * np.cos(k * theta)) / n_samples

        coeffs = np.zeros(2 * d + 1, dtype=complex)
        coeffs[d] = cheb[0]

        for k in range(1, d + 1):
            coeffs[d - k] = 0.5 * cheb[k]
            coeffs[d + k] = 0.5 * cheb[k]

        return coeffs

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(self.gqsp, list(range(self.num_qubits)))
        return qc
