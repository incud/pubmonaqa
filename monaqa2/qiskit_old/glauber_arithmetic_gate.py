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

    where DeltaE(x, y) = E(y) - E(x). For a = np.inf this uses the
    Metropolis-Hastings target sqrt(min(1, exp(-beta DeltaE))).

    signal="qubitization":
        Use a normalized Hermitian signal x = DeltaE / alpha. The polynomial
        is a Chebyshev/Laurent polynomial.

    signal="trotter":
        Use direct phase arithmetic with exp(-i H_Delta phase_time). Since
        H_Delta is diagonal, one Trotter step is exact. The polynomial is a
        Fourier/Laurent polynomial with a smooth buffer outside the physical arc.
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
        degree: int | None = None,
        phase_time: float | None = None,
        phase_buffer: float | None = 2.0,
        phase_safety: float = 0.99999,
        num_trotter_steps: int = 1,
        mocked_circuit: bool = False,
        mocked_angles: bool = False,
        label=None,
        tol: float = 1e-8,
        n_poly_samples: int | None = None,
        n_check_samples: int | None = None,
        contractive_margin: float = 1e-8,
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
        if degree is not None and int(degree) < 1:
            raise ValueError("degree must be positive when provided.")
        if phase_time is not None and phase_time <= 0.0:
            raise ValueError("phase_time must be positive when provided.")
        if phase_buffer is not None and phase_buffer <= 1.0:
            raise ValueError("phase_buffer must be > 1 when provided.")
        if not (0.0 < phase_safety <= 1.0):
            raise ValueError("phase_safety must satisfy 0 < phase_safety <= 1.")
        if num_trotter_steps <= 0:
            raise ValueError("num_trotter_steps must be positive.")
        if contractive_margin <= 0.0:
            raise ValueError("contractive_margin must be positive.")

        h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)

        if h.shape != (n,):
            raise ValueError(f"h must have shape ({n},), got {h.shape}.")
        if J.shape != (n, n):
            raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}.")

        self.n = int(n)
        self.h_single = h
        self.J_single = J
        self.beta = float(beta)
        self.eps = float(eps)
        self.a = float(a)
        self.signal = str(signal)
        self.degree_override = None if degree is None else int(degree)
        self.phase_time = None if phase_time is None else float(phase_time)
        self.phase_buffer = None if phase_buffer is None else float(phase_buffer)
        self.phase_safety = float(phase_safety)
        self.num_trotter_steps = int(num_trotter_steps)
        self.mocked_circuit = bool(mocked_circuit)
        self.mocked_angles = bool(mocked_angles)
        self.tol = float(tol)
        self.n_poly_samples = None if n_poly_samples is None else int(n_poly_samples)
        self.n_check_samples = None if n_check_samples is None else int(n_check_samples)
        self.contractive_margin = float(contractive_margin)

        self.h = np.concatenate([-self.h_single, self.h_single])
        self.J = np.zeros((2 * self.n, 2 * self.n), dtype=float)
        self.J[:self.n, :self.n] = -self.J_single
        self.J[self.n:, self.n:] = self.J_single
        self.gamma = np.zeros(2 * self.n, dtype=float)

        if np.all(np.abs(self.h) <= self.tol) and np.all(np.abs(np.triu(self.J, k=1)) <= self.tol):
            raise ValueError("The delta-energy Hamiltonian has no active terms.")

        self.alpha = self.delta_energy_bound()
        self._setup_signal_operator()

        self.degree = self._resolve_degree()
        self.poly_coeffs = self.make_polynomial(self.degree)
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

    def delta_energy_bound(self) -> float:
        return float(2.0 * np.sum(np.abs(self.h_single)) + 2.0 * np.sum(np.abs(self.J_single[np.triu_indices(self.n, k=1)])))

    def physical_delta_bound(self) -> float:
        return float(self.alpha)

    def signal_plot_bound(self) -> float:
        if self.signal == "trotter":
            return float(np.pi / self.phase_time)
        return float(self.alpha)

    def _setup_signal_operator(self) -> None:
        if self.signal == "qubitization":
            self.qubitization = QubitizedOperatorIsingTF(2 * self.n, self.h, self.J, self.gamma, mocked_all=self.mocked_circuit)
            self.controlled_qubitization = ControlledQubitizedOperatorIsingTF(2 * self.n, self.h, self.J, self.gamma, mocked_all=self.mocked_circuit)
            self.signal_operator = self.qubitization
            self.controlled_signal_operator = self.controlled_qubitization
            self.alpha = float(self.qubitization.lcu.alpha)
            return

        if self.phase_time is None:
            if self.phase_buffer is not None:
                self.phase_time = np.pi / (self.phase_buffer * self.alpha)
            else:
                self.phase_time = self.phase_safety * np.pi / self.alpha

        if self.phase_time * self.alpha >= np.pi:
            raise ValueError("phase_time is too large: the physical energy-difference range wraps around the unit circle.")

        gamma_zero = np.asarray(0.0, dtype=float)
        self.trotterized = TrotterizedOperatorIsingTF(2 * self.n, self.h, self.J, gamma_zero, time=self.phase_time, num_trotter_steps=self.num_trotter_steps)
        self.controlled_trotterized = ControlledTrotterizedOperatorIsingTF(2 * self.n, self.h, self.J, gamma_zero, time=self.phase_time, num_trotter_steps=self.num_trotter_steps)
        self.signal_operator = self.trotterized
        self.controlled_signal_operator = self.controlled_trotterized

    def exact_amplitude(self, delta: np.ndarray | float) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)

        if self.beta <= 1e-15:
            if np.isinf(self.a):
                return np.ones_like(delta, dtype=float)
            return np.full_like(delta, 2.0 ** (-1.0 / (2.0 * self.a)), dtype=float)

        if np.isinf(self.a):
            log_acceptance = np.minimum(0.0, -self.beta * delta)
            return np.exp(0.5 * log_acceptance)

        log_denom = np.logaddexp(0.0, self.beta * delta)
        return np.exp(-log_denom / (2.0 * self.a))

    def target_qubitization(self, x: np.ndarray | float) -> np.ndarray:
        return self.exact_amplitude(self.alpha * np.asarray(x, dtype=float))

    def target_phase(self, theta: np.ndarray | float) -> np.ndarray:
        return self.exact_amplitude(np.asarray(theta, dtype=float) / self.phase_time)

    def _resolve_degree(self) -> int:
        if self.degree_override is not None:
            return self.degree_override
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

        theta_phys = self.phase_time * self.alpha
        return max(1, int(np.ceil(np.log(8.0 / self.eps) / max(1e-12, np.pi - theta_phys))) + 2)

    def make_polynomial(self, degree: int | None = None) -> np.ndarray:
        degree = self.degree if degree is None else int(degree)
        if degree < 1:
            raise ValueError("degree must be positive.")
        if self.signal == "qubitization":
            return self._make_polynomial_qubitization(degree)
        return self._make_polynomial_trotter_phase(degree)

    def _make_polynomial_qubitization(self, degree: int) -> np.ndarray:
        d = int(degree)
        n_samples = max(d + 1, self.n_poly_samples or 0)
        j = np.arange(n_samples)
        theta = np.pi * (j + 0.5) / n_samples
        x = np.cos(theta)
        y = self.target_qubitization(x)

        cheb = np.zeros(d + 1, dtype=float)
        cheb[0] = np.sum(y) / n_samples
        for k in range(1, d + 1):
            cheb[k] = 2.0 * np.sum(y * np.cos(k * theta)) / n_samples

        coeffs = np.zeros(2 * d + 1, dtype=complex)
        coeffs[d] = cheb[0]
        for k in range(1, d + 1):
            coeffs[d - k] = 0.5 * cheb[k]
            coeffs[d + k] = 0.5 * cheb[k]
        return self._enforce_global_contractivity(coeffs)

    def _make_polynomial_trotter_phase(self, degree: int) -> np.ndarray:
        if np.isinf(self.a):
            return self._make_polynomial_trotter_smooth_buffer(degree)
        return self._make_polynomial_trotter_hermite_buffer(degree)

    def _make_polynomial_trotter_smooth_buffer(self, degree: int) -> np.ndarray:
        d = int(degree)
        theta_phys = self.phase_time * self.alpha
        n_samples = self.n_poly_samples or max(4096, 128 * d + 1)
        theta = -np.pi + 2.0 * np.pi * (np.arange(n_samples) + 0.5) / n_samples
        y = np.empty_like(theta, dtype=float)

        inside = np.abs(theta) <= theta_phys
        y[inside] = self.target_phase(theta[inside])
        y_left = float(np.asarray(self.exact_amplitude(-self.alpha)).item())
        y_right = float(np.asarray(self.exact_amplitude(+self.alpha)).item())

        theta_mod = np.mod(theta[~inside], 2.0 * np.pi)
        s = (theta_mod - theta_phys) / (2.0 * np.pi - 2.0 * theta_phys)
        s = np.clip(s, 0.0, 1.0)
        smooth = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
        y[~inside] = (1.0 - smooth) * y_right + smooth * y_left

        coeffs = np.zeros(2 * d + 1, dtype=complex)
        for k in range(-d, d + 1):
            coeffs[d + k] = np.sum(y * np.exp(1j * k * theta)) / n_samples
        return self._enforce_global_contractivity(coeffs)

    def _target_with_phase_derivatives(self, delta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.isinf(self.a):
            raise ValueError("The derivative target is not used for a=np.inf.")
        delta = np.asarray(delta, dtype=float)
        p = 1.0 / (2.0 * self.a)
        log_denom = np.logaddexp(0.0, self.beta * delta)
        L = np.exp(-log_denom)
        g = np.exp(-p * log_denom)
        d1_delta = -p * self.beta * (1.0 - L) * g
        d2_delta = ((p * self.beta * (1.0 - L)) ** 2 - p * self.beta**2 * L * (1.0 - L)) * g
        return g, d1_delta / self.phase_time, d2_delta / (self.phase_time**2)

    def _make_polynomial_trotter_hermite_buffer(self, degree: int) -> np.ndarray:
        d = int(degree)
        theta_phys = self.phase_time * self.alpha
        buffer_length = 2.0 * np.pi - 2.0 * theta_phys
        n_samples = self.n_poly_samples or max(4096, 128 * d + 1)
        theta = -np.pi + 2.0 * np.pi * (np.arange(n_samples) + 0.5) / n_samples
        y = np.empty_like(theta, dtype=float)

        inside = np.abs(theta) <= theta_phys
        y[inside] = self._target_with_phase_derivatives(theta[inside] / self.phase_time)[0]

        y_left, d1_left, d2_left = [float(np.asarray(v).item()) for v in self._target_with_phase_derivatives(-self.alpha)]
        y_right, d1_right, d2_right = [float(np.asarray(v).item()) for v in self._target_with_phase_derivatives(+self.alpha)]

        theta_mod = np.mod(theta[~inside], 2.0 * np.pi)
        u = (theta_mod - theta_phys) / buffer_length
        u = np.clip(u, 0.0, 1.0)

        y0 = y_right
        y1 = y_left
        m0 = d1_right * buffer_length
        m1 = d1_left * buffer_length
        s0 = d2_right * buffer_length**2
        s1 = d2_left * buffer_length**2

        a0 = y0
        a1 = m0
        a2 = 0.5 * s0
        A = y1 - a0 - a1 - a2
        B = m1 - a1 - 2.0 * a2
        C = s1 - 2.0 * a2
        a3 = 10.0 * A - 4.0 * B + 0.5 * C
        a4 = -15.0 * A + 7.0 * B - C
        a5 = 6.0 * A - 3.0 * B + 0.5 * C
        y[~inside] = a0 + a1 * u + a2 * u**2 + a3 * u**3 + a4 * u**4 + a5 * u**5

        coeffs = np.zeros(2 * d + 1, dtype=complex)
        for k in range(-d, d + 1):
            coeffs[d + k] = np.sum(y * np.exp(1j * k * theta)) / n_samples
        return self._enforce_global_contractivity(coeffs)

    def _enforce_global_contractivity(self, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=complex).copy()
        d = (len(coeffs) - 1) // 2
        n_check = self.n_check_samples or max(20000, 256 * d)
        theta = np.linspace(-np.pi, np.pi, n_check, endpoint=False)
        values = self._eval_laurent_theta(coeffs, theta)
        max_abs = float(np.max(np.abs(values)))
        if max_abs >= 1.0 - self.contractive_margin:
            coeffs *= (1.0 - self.contractive_margin) / max_abs
        return coeffs

    @staticmethod
    def _eval_laurent_theta(coeffs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=complex)
        theta = np.asarray(theta, dtype=float)
        d = (len(coeffs) - 1) // 2
        out = np.zeros_like(theta, dtype=complex)
        for k in range(-d, d + 1):
            out += coeffs[d + k] * np.exp(-1j * k * theta)
        return out

    def evaluate_polynomial_delta(self, delta: np.ndarray | float) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if self.signal == "qubitization":
            x = np.clip(delta / self.alpha, -1.0, 1.0)
            theta = np.arccos(x)
            return np.real_if_close(self._eval_laurent_theta(self.poly_coeffs, -theta)).real
        theta = self.phase_time * delta
        return np.real_if_close(self._eval_laurent_theta(self.poly_coeffs, theta)).real

    def polynomial_error_on_delta(self, delta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        exact = self.exact_amplitude(delta)
        approx = self.evaluate_polynomial_delta(delta)
        return approx, exact, approx - exact

    @staticmethod
    def _bits(index: int, n: int) -> np.ndarray:
        return np.array([(index >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)

    def energy(self, index: int) -> float:
        z = 1 - 2 * self._bits(index, self.n)
        value = float(np.dot(self.h_single, z))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                value += float(self.J_single[i, j]) * int(z[i]) * int(z[j])
        return value

    def energies(self) -> np.ndarray:
        return np.array([self.energy(x) for x in range(2**self.n)], dtype=float)

    def delta_energies(self) -> np.ndarray:
        E = self.energies()
        d = 2**self.n
        return np.array([E[y] - E[x] for x in range(d) for y in range(d)], dtype=float)

    @staticmethod
    def _ket(index: int, n: int) -> np.ndarray:
        if index < 0 or index >= 2**n:
            raise ValueError(f"index={index} is outside the {n}-qubit basis.")
        out = np.zeros(2**n, dtype=complex)
        out[index] = 1.0
        return out

    @staticmethod
    def _kron_all(*vectors: np.ndarray) -> np.ndarray:
        out = np.array([1.0 + 0.0j])
        for vector in vectors:
            out = np.kron(out, vector)
        return out

    def good_block_state(self, x: int, y: int) -> np.ndarray:
        n_aux = len(self.layout["aux"])
        return self._kron_all(self._ket(0, 1), self._ket(x, self.n), self._ket(y, self.n), self._ket(0, n_aux))

    def good_block_amplitudes_from_unitary(self, unitary: np.ndarray) -> np.ndarray:
        unitary = np.asarray(unitary, dtype=complex)
        d = 2**self.n
        out = []
        for x in range(d):
            for y in range(d):
                state = self.good_block_state(x, y)
                amp = (state.conj().T @ unitary @ state).item()
                out.append(float(np.real(np.real_if_close(amp))))
        return np.array(out, dtype=float)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(self.gqsp, list(range(self.num_qubits)))
        return qc
