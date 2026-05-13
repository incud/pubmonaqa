import numpy as np
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.gqsp_gate import GQSP
from monaqa2.qiskit.qubitized_ising_tf_gate import (
    QubitizedOperatorIsingTF,
    ControlledQubitizedOperatorIsingTF,
)
from monaqa2.qiskit.trotterized_ising_tf_gate import (
    TrotterizedOperatorIsingTF,
    ControlledTrotterizedOperatorIsingTF,
)


class GlauberArithmetic(Gate):
    r"""
    GQSP implementation of the generalized Glauber acceptance amplitude.

        g(x, y) = (1 / (1 + exp(beta * DeltaE(x, y)))) ** (1 / (2 * a)),

    where

        DeltaE(x, y) = E(y) - E(x)

    and

        E(z) = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.

    For a = 1 this reduces to the usual square-root Glauber acceptance.
    For a = np.inf this uses the Metropolis-Hastings target

        g(x, y) = sqrt(min(1, exp(-beta * DeltaE(x, y)))).

    The delta-energy Hamiltonian is built as

        H_Delta = -H_E^(A) + H_E^(B),

    by using coefficients [-h, h] and block couplings [-J, +J] on the two
    registers A and B.

    signal="qubitization":
        Use the qubitized walk signal. The polynomial is a Chebyshev/Laurent
        polynomial in x = DeltaE / alpha.

    signal="trotter":
        Use direct phase arithmetic with a Trotterized simulation of
        exp(-i H_Delta phase_time). The polynomial is a Fourier/Laurent
        polynomial in z = exp(-i DeltaE phase_time). Since H_Delta is diagonal,
        the product formula is exact with one Trotter step.

    The good GQSP block acts as

        |x>|y>|0...0> -> |x>|y>(g(x, y)|0...0> + |perp_{x,y}>),

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
        signal: str = "qubitization",
        phase_time: float | None = None,
        phase_safety: float = 0.99999,
        num_trotter_steps: int = 1,
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
        if not (np.isinf(a) or a > 0.0):
            raise ValueError("a must be positive or np.inf.")
        if signal not in {"qubitization", "trotter"}:
            raise ValueError("signal must be either 'qubitization' or 'trotter'.")
        if phase_time is not None and phase_time <= 0.0:
            raise ValueError("phase_time must be positive when provided.")
        if not (0.0 < phase_safety <= 1.0):
            raise ValueError("phase_safety must satisfy 0 < phase_safety <= 1.")
        if num_trotter_steps <= 0:
            raise ValueError("num_trotter_steps must be positive.")

        assert h.shape == (n,)
        assert J.shape == (n, n)

        self.n = int(n)
        self.h_single = np.asarray(h, dtype=float)
        self.J_single = np.asarray(J, dtype=float)
        self.beta = float(beta)
        self.eps = float(eps)
        self.a = float(a)
        self.signal = str(signal)
        self.phase_time = None if phase_time is None else float(phase_time)
        self.phase_safety = float(phase_safety)
        self.num_trotter_steps = int(num_trotter_steps)
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

        self.alpha = self._delta_energy_bound()
        self._setup_signal_operator()
        self.degree = self._degree()
        self.poly_coeffs = self._poly()
        self.gqsp = GQSP(
            self.signal_operator,
            self.controlled_signal_operator,
            self.poly_coeffs,
            mocked_angles=self.mocked_angles,
            laurent_negative_power=self.degree,
        )

        super().__init__("GlauberArithmetic", self.gqsp.num_qubits, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        system = [1 + q for q in self.signal_operator.layout["system"]]
        A = system[:self.n]
        B = system[self.n:]
        aux = [q for q in range(self.num_qubits) if q not in [0] + system]

        return {
            "control": [0],
            "A": A,
            "B": B,
            "system": system,
            "signal": self.gqsp.layout["qubitization"],
            "qubitization": self.gqsp.layout["qubitization"],
            "aux": aux,
        }

    def _delta_energy_bound(self) -> float:
        return float(
            2.0 * np.sum(np.abs(self.h_single))
            + 2.0 * np.sum(np.abs(self.J_single[np.triu_indices(self.n, k=1)]))
        )

    def _setup_signal_operator(self) -> None:
        if self.signal == "qubitization":
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
            self.signal_operator = self.qubitization
            self.controlled_signal_operator = self.controlled_qubitization
            self.alpha = float(self.qubitization.lcu.alpha)
            return

        if self.phase_time is None:
            self.phase_time = self.phase_safety * np.pi / self.alpha

        # The delta-energy Hamiltonian is diagonal, so gamma is exactly zero and
        # one product-formula step is exact. Passing scalar 0.0 also avoids any
        # array truth-value checks in older controlled-Trotter implementations.
        gamma_zero = np.asarray(0.0, dtype=float)

        self.trotterized = TrotterizedOperatorIsingTF(
            2 * self.n,
            self.h,
            self.J,
            gamma_zero,
            time=self.phase_time,
            num_trotter_steps=self.num_trotter_steps,
        )
        self.controlled_trotterized = ControlledTrotterizedOperatorIsingTF(
            2 * self.n,
            self.h,
            self.J,
            gamma_zero,
            time=self.phase_time,
            num_trotter_steps=self.num_trotter_steps,
        )
        self.signal_operator = self.trotterized
        self.controlled_signal_operator = self.controlled_trotterized

    def _acceptance_amplitude(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)

        if self.beta <= 1e-15:
            if np.isinf(self.a):
                return np.ones_like(delta, dtype=float)
            return np.full_like(delta, 2.0 ** (-1.0 / (2.0 * self.a)), dtype=float)

        if np.isinf(self.a):
            log_acceptance = np.minimum(0.0, -self.beta * delta)
            return np.exp(0.5 * log_acceptance)

        base = 1.0 / (1.0 + np.exp(self.beta * delta))
        return base ** (1.0 / (2.0 * self.a))

    def _target_qubitization(self, x: np.ndarray) -> np.ndarray:
        return self._acceptance_amplitude(self.alpha * x)

    def _target_phase(self, theta: np.ndarray) -> np.ndarray:
        return self._acceptance_amplitude(theta / self.phase_time)

    def _degree(self) -> int:
        if np.isinf(self.a):
            return max(1, int(np.ceil(8.0 / self.eps))) + 2

        if self.beta <= 1e-15:
            return 1

        if self.signal == "qubitization":
            scale = self.beta * self.alpha
            tau = np.pi / scale
            rho = tau + np.sqrt(1.0 + tau * tau)
            branch_prefactor = 1.0 + 1.0 / (2.0 * self.a)
            return max(1, int(np.ceil(np.log(8.0 * branch_prefactor / self.eps) / np.log(rho))) + 2)

        strip_width = np.pi * self.phase_time / self.beta
        branch_prefactor = 1.0 + 1.0 / (2.0 * self.a)
        return max(1, int(np.ceil(np.log(8.0 * branch_prefactor / self.eps) / strip_width)) + 2)

    def _poly(self) -> np.ndarray:
        if self.signal == "qubitization":
            return self._poly_qubitization()

        return self._poly_trotter_phase()

    def _poly_qubitization(self) -> np.ndarray:
        d = self.degree
        n_samples = d + 1

        j = np.arange(n_samples)
        theta = np.pi * (j + 0.5) / n_samples
        x = np.cos(theta)
        y = self._target_qubitization(x)

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

    def _poly_trotter_phase(self) -> np.ndarray:
        d = self.degree
        n_samples = max(256, 8 * d + 1)

        j = np.arange(n_samples)
        theta = -np.pi + 2.0 * np.pi * (j + 0.5) / n_samples
        y = self._target_phase(theta)

        coeffs = np.zeros(2 * d + 1, dtype=complex)

        for k in range(-d, d + 1):
            coeffs[d + k] = np.sum(y * np.exp(1j * k * theta)) / n_samples

        return coeffs

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(self.gqsp, list(range(self.num_qubits)))
        return qc
