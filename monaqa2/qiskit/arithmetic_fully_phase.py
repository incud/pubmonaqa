from dataclasses import dataclass
import numpy as np
from numpy.polynomial import chebyshev as ncheb
from monaqa2.qiskit.gqsp_gate_generic import GQSP
from monaqa2.qiskit.primitives import Ccx, ControlledGivensRotation, Cry, Ccry, GivensRotation
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.synthesis.multi_controlled.mcx_synthesis import synth_mcx_2_clean_kg24


class PrepareDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.m = 2 * n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        self.h_delta, self.J_delta = self._build_delta_hamiltonian()
        self.lambda_Z = np.sum(np.abs(self.h_delta))
        self.lambda_J = np.sum(np.abs(np.triu(self.J_delta, 1)))
        self.lam = self.lambda_Z + self.lambda_J
        super().__init__(name="prepare_delta_hamiltonian", num_qubits=3 * self.m, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def uZ(self) -> list[int]:
        return list(range(0, self.m))

    @property
    def uJ(self) -> list[int]:
        return list(range(self.m, 2 * self.m))

    @property
    def vJ(self) -> list[int]:
        return list(range(2 * self.m, 3 * self.m))

    def _build_delta_hamiltonian(self) -> tuple[np.ndarray, np.ndarray]:
        h_delta = np.concatenate([-self.h, self.h])
        J_delta = np.zeros((self.m, self.m), dtype=float)
        J_delta[:self.n, :self.n] = -self.J
        J_delta[self.n:, self.n:] = self.J
        return h_delta, J_delta

    def _append_unary_ladder(self, qc: QuantumCircuit, weights: np.ndarray, register: list[int]) -> None:
        tails = np.cumsum(weights[::-1])[::-1]
        for k in range(len(register) - 1):
            qc.append(GivensRotation(weights[k], tails[k + 1]), [register[k], register[k + 1]])

    def _prepare_branch_no_ancilla(self, qc: QuantumCircuit) -> None:
        qc.x(self.uZ[0])
        qc.append(GivensRotation(self.lambda_Z, self.lambda_J), [self.uZ[0], self.uJ[0]])

    def _prepare_Z_terms(self, qc: QuantumCircuit) -> None:
        self._append_unary_ladder(qc, np.abs(self.h_delta), self.uZ)

    def _prepare_ZZ_terms(self, qc: QuantumCircuit) -> None:
        abs_J = np.abs(np.triu(self.J_delta, 1))
        R = np.zeros(self.m)
        for i in range(self.m - 1):
            R[i] = np.sum(abs_J[i, i + 1:])
        self._append_unary_ladder(qc, R, self.uJ)
        for i in range(self.m - 1):
            qc.cx(self.uJ[i], self.vJ[i + 1])
        T = np.zeros((self.m, self.m))
        for i in range(self.m):
            running = 0.0
            for j in range(self.m - 1, -1, -1):
                running += abs_J[i, j]
                T[i, j] = running
        for d in range(1, self.m - 1):
            for parity in (0, 1):
                for i in range(self.m):
                    if i % 2 != parity:
                        continue
                    j = i + d
                    if j >= self.m - 1 or abs_J[i, j] + T[i, j + 1] == 0.0:
                        continue
                    qc.append(ControlledGivensRotation(abs_J[i, j], T[i, j + 1]), [self.uJ[i], self.vJ[j], self.vJ[j + 1]])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3 * self.m, name=self.name)
        self._prepare_branch_no_ancilla(qc)
        self._prepare_Z_terms(qc)
        self._prepare_ZZ_terms(qc)
        return qc


class SelectDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.m = 2 * n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        self.h_delta, self.J_delta = self._build_delta_hamiltonian()
        super().__init__(name="select_delta_hamiltonian", num_qubits=4 * self.m, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def uZ(self) -> list[int]:
        return list(range(0, self.m))

    @property
    def uJ(self) -> list[int]:
        return list(range(self.m, 2 * self.m))

    @property
    def vJ(self) -> list[int]:
        return list(range(2 * self.m, 3 * self.m))

    @property
    def q(self) -> list[int]:
        return list(range(3 * self.m, 4 * self.m))

    def _build_delta_hamiltonian(self) -> tuple[np.ndarray, np.ndarray]:
        h_delta = np.concatenate([-self.h, self.h])
        J_delta = np.zeros((self.m, self.m), dtype=float)
        J_delta[:self.n, :self.n] = -self.J
        J_delta[self.n:, self.n:] = self.J
        return h_delta, J_delta

    def _apply_signs(self, qc: QuantumCircuit) -> None:
        for i in range(self.m):
            if self.h_delta[i] < 0:
                qc.z(self.uZ[i])
        for color in range(self.m):
            for i in range(self.m):
                j = (color - i) % self.m
                if i < j and self.J_delta[i, j] < 0:
                    qc.cz(self.uJ[i], self.vJ[j])

    def _apply_paulis(self, qc: QuantumCircuit) -> None:
        for i in range(self.m):
            qc.cz(self.uZ[i], self.q[i])
        for i in range(self.m):
            qc.cz(self.uJ[i], self.q[i])
        for i in range(self.m):
            qc.cz(self.vJ[i], self.q[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(4 * self.m, name=self.name)
        self._apply_signs(qc)
        self._apply_paulis(qc)
        return qc


class ControlledSelectDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.m = 2 * n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        self.h_delta, self.J_delta = self._build_delta_hamiltonian()
        super().__init__(name="c_select_delta_hamiltonian", num_qubits=1 + 4 * self.m + (self.m - 1), params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def uZ(self) -> list[int]:
        return list(range(1, 1 + self.m))

    @property
    def uJ(self) -> list[int]:
        return list(range(1 + self.m, 1 + 2 * self.m))

    @property
    def vJ(self) -> list[int]:
        return list(range(1 + 2 * self.m, 1 + 3 * self.m))

    @property
    def q(self) -> list[int]:
        return list(range(1 + 3 * self.m, 1 + 4 * self.m))

    @property
    def control_copies(self) -> list[int]:
        return [self.c] + list(range(1 + 4 * self.m, 1 + 5 * self.m - 1))

    def _build_delta_hamiltonian(self) -> tuple[np.ndarray, np.ndarray]:
        h_delta = np.concatenate([-self.h, self.h])
        J_delta = np.zeros((self.m, self.m), dtype=float)
        J_delta[:self.n, :self.n] = -self.J
        J_delta[self.n:, self.n:] = self.J
        return h_delta, J_delta

    def _ccz(self, qc: QuantumCircuit, a: int, b: int, target: int) -> None:
        qc.h(target)
        qc.append(Ccx(), [a, b, target])
        qc.h(target)

    def _fanout_control(self, qc: QuantumCircuit) -> None:
        for copy in self.control_copies[1:]:
            qc.cx(self.c, copy)

    def _unfanout_control(self, qc: QuantumCircuit) -> None:
        for copy in reversed(self.control_copies[1:]):
            qc.cx(self.c, copy)

    def _apply_signs(self, qc: QuantumCircuit) -> None:
        controls = self.control_copies
        for i in range(self.m):
            if self.h_delta[i] < 0:
                qc.cz(controls[i], self.uZ[i])
        for color in range(self.m):
            for i in range(self.m):
                j = (color - i) % self.m
                if i < j and self.J_delta[i, j] < 0:
                    self._ccz(qc, controls[i], self.uJ[i], self.vJ[j])

    def _apply_paulis(self, qc: QuantumCircuit) -> None:
        controls = self.control_copies
        for i in range(self.m):
            self._ccz(qc, controls[i], self.uZ[i], self.q[i])
        for i in range(self.m):
            self._ccz(qc, controls[i], self.uJ[i], self.q[i])
        for i in range(self.m):
            self._ccz(qc, controls[i], self.vJ[i], self.q[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        self._fanout_control(qc)
        self._apply_signs(qc)
        self._apply_paulis(qc)
        self._unfanout_control(qc)
        return qc


class ReflectionZero(Gate):
    def __init__(self, m: int, label=None):
        self.m = m
        super().__init__(name="reflection_zero", num_qubits=m + 2, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(self.m))

    @property
    def clean(self) -> list[int]:
        return [self.m, self.m + 1]

    def _mcx(self, qc: QuantumCircuit, controls: list[int], target: int) -> None:
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        elif len(controls) == 2:
            qc.append(Ccx(), [controls[0], controls[1], target])
        else:
            qc.append(synth_mcx_2_clean_kg24(len(controls)), controls + [target] + self.clean)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.x(self.selection)
        qc.h(self.selection[-1])
        self._mcx(qc, self.selection[:-1], self.selection[-1])
        qc.h(self.selection[-1])
        qc.x(self.selection)
        return qc


class ControlledReflectionZero(Gate):
    def __init__(self, m: int, label=None):
        self.m = m
        super().__init__(name="c_reflection_zero", num_qubits=1 + m + 2, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def selection(self) -> list[int]:
        return list(range(1, 1 + self.m))

    @property
    def clean(self) -> list[int]:
        return [1 + self.m, 2 + self.m]

    def _mcx(self, qc: QuantumCircuit, controls: list[int], target: int) -> None:
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        elif len(controls) == 2:
            qc.append(Ccx(), [controls[0], controls[1], target])
        else:
            qc.append(synth_mcx_2_clean_kg24(len(controls)), controls + [target] + self.clean)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.x(self.selection)
        qc.h(self.selection[-1])
        self._mcx(qc, [self.c] + self.selection[:-1], self.selection[-1])
        qc.h(self.selection[-1])
        qc.x(self.selection)
        return qc


class QubitizedDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.m = 2 * n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        super().__init__(name="qubitized_delta_hamiltonian", num_qubits=4 * self.m + 2, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(0, 3 * self.m))

    @property
    def all_qubits(self) -> list[int]:
        return list(range(0, 4 * self.m))

    @property
    def reflection_qubits(self) -> list[int]:
        return self.selection + [4 * self.m, 4 * self.m + 1]

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = PrepareDeltaHamiltonian(self.n, self.h, self.J)
        select = SelectDeltaHamiltonian(self.n, self.h, self.J)
        qc.append(prepare, self.selection)
        qc.append(select, self.all_qubits)
        qc.append(prepare.inverse(), self.selection)
        qc.append(ReflectionZero(3 * self.m), self.reflection_qubits)
        return qc


class ControlledQubitizedDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.m = 2 * n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        super().__init__(name="c_qubitized_delta_hamiltonian", num_qubits=1 + 4 * self.m + (self.m - 1) + 2, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(1, 1 + 3 * self.m))

    @property
    def select_qubits(self) -> list[int]:
        return list(range(0, 1 + 4 * self.m + (self.m - 1)))

    @property
    def reflection_qubits(self) -> list[int]:
        return [0] + self.selection + [1 + 4 * self.m + (self.m - 1), 2 + 4 * self.m + (self.m - 1)]

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = PrepareDeltaHamiltonian(self.n, self.h, self.J)
        c_select = ControlledSelectDeltaHamiltonian(self.n, self.h, self.J)
        qc.append(prepare, self.selection)
        qc.append(c_select, self.select_qubits)
        qc.append(prepare.inverse(), self.selection)
        qc.append(ControlledReflectionZero(3 * self.m), self.reflection_qubits)
        return qc


class MockedQubitizedDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        self.h_delta, self.J_delta = self._build_delta_hamiltonian(n, h, J)
        self.lam = self._lcu_norm(self.h_delta, self.J_delta)
        if self.lam == 0.0:
            raise ValueError("The Delta Hamiltonian has zero LCU norm.")
        super().__init__(name="mocked_qubitized_delta_hamiltonian", num_qubits=2 * n, params=[], label=label)
        self.definition = self._build_definition()

    @staticmethod
    def _build_delta_hamiltonian(n: int, h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_delta = np.concatenate([-h, h])
        J_delta = np.zeros((2 * n, 2 * n), dtype=float)
        J_delta[:n, :n] = -J
        J_delta[n:, n:] = J
        return h_delta, J_delta

    @staticmethod
    def _lcu_norm(h_delta: np.ndarray, J_delta: np.ndarray) -> float:
        return np.sum(np.abs(h_delta)) + np.sum(np.abs(np.triu(J_delta, 1)))

    @staticmethod
    def _energy(bits: int, h_delta: np.ndarray, J_delta: np.ndarray) -> float:
        n_delta = len(h_delta)
        z = np.array([1.0 if ((bits >> i) & 1) == 0 else -1.0 for i in range(n_delta)])
        energy = np.dot(h_delta, z)
        for i in range(n_delta - 1):
            for j in range(i + 1, n_delta):
                energy += J_delta[i, j] * z[i] * z[j]
        return energy

    @staticmethod
    def _phases(h_delta: np.ndarray, J_delta: np.ndarray, lam: float) -> np.ndarray:
        n_delta = len(h_delta)
        phases = np.empty(2 ** n_delta, dtype=complex)
        for bits in range(2 ** n_delta):
            x = np.clip(MockedQubitizedDeltaHamiltonian._energy(bits, h_delta, J_delta) / lam, -1.0, 1.0)
            phases[bits] = np.exp(1j * np.arccos(x))
        return phases

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.unitary(np.diag(self._phases(self.h_delta, self.J_delta, self.lam)), list(range(2 * self.n)), label=self.name)
        return qc


class MockedControlledQubitizedDeltaHamiltonian(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        assert h.shape == (n,) and J.shape == (n, n)
        self.h_delta, self.J_delta = MockedQubitizedDeltaHamiltonian._build_delta_hamiltonian(n, h, J)
        self.lam = MockedQubitizedDeltaHamiltonian._lcu_norm(self.h_delta, self.J_delta)
        if self.lam == 0.0:
            raise ValueError("The Delta Hamiltonian has zero LCU norm.")
        super().__init__(name="c_mocked_qubitized_delta_hamiltonian", num_qubits=2 * n + 1, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def system(self) -> list[int]:
        return list(range(1, 2 * self.n + 1))

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        phases = MockedQubitizedDeltaHamiltonian._phases(self.h_delta, self.J_delta, self.lam)
        controlled_phases = np.ones(2 ** (2 * self.n + 1), dtype=complex)
        for bits, phase in enumerate(phases):
            controlled_phases[(bits << 1) | 1] = phase
        qc.unitary(np.diag(controlled_phases), list(range(2 * self.n + 1)), label=self.name)
        return qc
    




@dataclass(frozen=True)
class FullyPhaseArithmeticPolyOptions:
    fit: str = "projection"                 # How Chebyshev coefficients are fitted: "projection" uses Chebyshev-series projection; "least_squares" solves a sampled least-squares problem.
    zero_window: float = 0.0                 # Physical DeltaE window around zero where the MH kink is treated specially; units are energy, not normalized x=DeltaE/alpha.
    zero_window_strategy: str = "none"       # How to use zero_window: "none" ignores it, "mask" excludes the window from the fit, "hermite" fills it with a smooth Hermite bridge.
    jackson_damping: bool = False            # If True, applies Jackson damping to Chebyshev coefficients to reduce Gibbs oscillations/ringing.
    contractive_rescale: bool = True         # If True, rescales the Laurent polynomial so that |P(e^{i theta})| < 1 for GQSP angle synthesis.
    contractive_margin: float = 1e-8         # Safety margin used in contractive rescaling: target maximum modulus is 1 - contractive_margin.
    eps_tail: float | None = None            # Tail error used only for diagnostics/cutoff bookkeeping; if None, eps_ops/2 is used.
    n_poly_samples: int | None = None        # Number of sample points used to build the Chebyshev/Laurent polynomial; if None, an automatic value is used.
    n_check_samples: int | None = None       # Number of samples used to check contractivity on the unit circle; if None, an automatic value is used.

    @staticmethod
    def projection() -> "FullyPhaseArithmeticPolyOptions":
        # Use for smooth targets, especially finite-a Glauber; simplest and usually stable.
        return FullyPhaseArithmeticPolyOptions(fit="projection", zero_window_strategy="none", jackson_damping=False, contractive_rescale=True)

    @staticmethod
    def least_squares() -> "FullyPhaseArithmeticPolyOptions":
        # Use when projection gives visible global oscillations but no zero-window treatment is desired.
        return FullyPhaseArithmeticPolyOptions(fit="least_squares", zero_window_strategy="none", jackson_damping=False, contractive_rescale=True)

    @staticmethod
    def masked_window(zero_window: float) -> "FullyPhaseArithmeticPolyOptions":
        # Use for MH when you explicitly do not care about accuracy near DeltaE=0; fits only outside the window.
        return FullyPhaseArithmeticPolyOptions(fit="least_squares", zero_window=zero_window, zero_window_strategy="mask", jackson_damping=False, contractive_rescale=True)

    @staticmethod
    def hermite_window(zero_window: float) -> "FullyPhaseArithmeticPolyOptions":
        # Use for MH when you want to reduce ringing by replacing the ignored zero window with a smooth bridge.
        return FullyPhaseArithmeticPolyOptions(fit="projection", zero_window=zero_window, zero_window_strategy="hermite", jackson_damping=False, contractive_rescale=True)

    @staticmethod
    def hermite_jackson_window(zero_window: float) -> "FullyPhaseArithmeticPolyOptions":
        # Use for the smoothest MH plots: Hermite bridge near zero plus Jackson damping to suppress residual oscillations.
        return FullyPhaseArithmeticPolyOptions(fit="projection", zero_window=zero_window, zero_window_strategy="hermite", jackson_damping=True, contractive_rescale=True)

    @staticmethod
    def masked_jackson_window(zero_window: float) -> "FullyPhaseArithmeticPolyOptions":
        # Use when masking near zero helps but the outside fit still rings; stronger smoothing, but may round features.
        return FullyPhaseArithmeticPolyOptions(fit="least_squares", zero_window=zero_window, zero_window_strategy="mask", jackson_damping=True, contractive_rescale=True)

    @staticmethod
    def no_rescale_projection() -> "FullyPhaseArithmeticPolyOptions":
        # Use only for debugging polynomial quality before GQSP synthesis; angle synthesis may fail if |P| exceeds 1.
        return FullyPhaseArithmeticPolyOptions(fit="projection", zero_window_strategy="none", jackson_damping=False, contractive_rescale=False)
    



class FullyPhaseArithmetic(GQSP):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, beta: float, eps_ops: float, a: float = np.inf, degree: int | None = None, is_mocked_construction: bool = True, is_mocked_angles: bool = False, poly_options: FullyPhaseArithmeticPolyOptions | None = None, label=None):
        self.n = n
        self.h = h
        self.J = J
        self.beta = beta
        self.eps_ops = eps_ops
        self.a = a
        self.poly_options = FullyPhaseArithmeticPolyOptions.projection() if poly_options is None else poly_options
        assert h.shape == (n,) and J.shape == (n, n)
        if not (np.isinf(a) or a > 0.0):
            raise ValueError("a must be positive or np.inf.")
        if eps_ops <= 0.0:
            raise ValueError("eps_ops must be positive.")
        if self.poly_options.fit not in {"projection", "least_squares"}:
            raise ValueError("poly_options.fit must be 'projection' or 'least_squares'.")
        if self.poly_options.zero_window_strategy not in {"none", "mask", "hermite"}:
            raise ValueError("poly_options.zero_window_strategy must be 'none', 'mask', or 'hermite'.")
        if self.poly_options.zero_window < 0.0:
            raise ValueError("poly_options.zero_window must be non-negative.")
        if self.poly_options.contractive_margin <= 0.0:
            raise ValueError("poly_options.contractive_margin must be positive.")
        self.eps_tail = eps_ops / 2.0 if self.poly_options.eps_tail is None else float(self.poly_options.eps_tail)
        h_delta, J_delta = MockedQubitizedDeltaHamiltonian._build_delta_hamiltonian(n, h, J)
        self.alpha = MockedQubitizedDeltaHamiltonian._lcu_norm(h_delta, J_delta)
        self.zero_window = float(self.poly_options.zero_window)
        self.tail_cutoff_delta = np.inf if beta <= 1e-15 or not np.isinf(a) else 2.0 * np.log(1.0 / self.eps_tail) / beta
        self.selected_degree = self._degree(beta, self.alpha, eps_ops, a) if degree is None else int(degree)
        self.poly_coeffs, self.poly_scale = self._make_laurent_polynomial(beta, self.alpha, a, self.selected_degree, self.poly_options)
        qubitization = MockedQubitizedDeltaHamiltonian(n, h, J) if is_mocked_construction else QubitizedDeltaHamiltonian(n, h, J)
        controlled_qubitization = MockedControlledQubitizedDeltaHamiltonian(n, h, J) if is_mocked_construction else ControlledQubitizedDeltaHamiltonian(n, h, J)
        super().__init__(qubitization=qubitization, controlled_qubitization=controlled_qubitization, poly_coeffs=self.poly_coeffs, mocked_angles=is_mocked_angles, laurent_negative_power=self.selected_degree, label=label)

    @staticmethod
    def _degree(beta: float, alpha: float, eps_ops: float, a: float) -> int:
        if beta <= 1e-15:
            return 1
        if np.isinf(a):
            # MH rule
            kappa = beta * alpha / 2.0
            return max(1, int(np.ceil(np.sqrt(kappa * np.log(1.0 / eps_ops)) + np.log(1.0 / eps_ops))) + 2)
        else:
            # Generalized Glauber
            tau = np.pi / (beta * alpha)
            rho = tau + np.sqrt(1.0 + tau * tau)
            branch_prefactor = 1.0 + 1.0 / (2.0 * a)
            return max(1, int(np.ceil(np.log(8.0 * branch_prefactor / eps_ops) / np.log(rho))) + 2)

    @staticmethod
    def _exact_amplitude(delta: np.ndarray | float, beta: float, a: float) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if beta <= 1e-15:
            return np.ones_like(delta, dtype=float) if np.isinf(a) else np.full_like(delta, 2.0 ** (-1.0 / (2.0 * a)), dtype=float)
        if np.isinf(a):
            return np.exp(0.5 * np.minimum(0.0, -beta * delta))
        return np.exp(-np.logaddexp(0.0, beta * delta) / (2.0 * a))

    @staticmethod
    def _exact_amplitude_derivative(delta: np.ndarray | float, beta: float, a: float) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if beta <= 1e-15:
            return np.zeros_like(delta, dtype=float)
        if np.isinf(a):
            return np.where(delta <= 0.0, 0.0, -0.5 * beta * np.exp(-0.5 * beta * delta))
        p = 1.0 / (2.0 * a)
        g = FullyPhaseArithmetic._exact_amplitude(delta, beta, a)
        sigmoid = 1.0 / (1.0 + np.exp(-beta * delta))
        return -p * beta * sigmoid * g

    @staticmethod
    def _make_laurent_polynomial(beta: float, alpha: float, a: float, degree: int, options: FullyPhaseArithmeticPolyOptions) -> tuple[np.ndarray, float]:
        d = int(degree)
        n_samples = options.n_poly_samples or max(8192, 512 * d)
        theta = np.pi * (np.arange(n_samples) + 0.5) / n_samples
        x = np.cos(theta)
        delta = alpha * x
        y = FullyPhaseArithmetic._exact_amplitude(delta, beta, a)
        if options.zero_window_strategy == "hermite" and options.zero_window > 0.0:
            y = FullyPhaseArithmetic._smooth_window_target(delta, beta, a, options.zero_window)
        if options.zero_window_strategy == "mask" and options.zero_window > 0.0:
            mask = np.abs(delta) >= options.zero_window
            cheb = FullyPhaseArithmetic._chebyshev_lstsq(x[mask], y[mask], d)
        elif options.fit == "least_squares":
            cheb = FullyPhaseArithmetic._chebyshev_lstsq(x, y, d)
        else:
            cheb = FullyPhaseArithmetic._chebyshev_projection(theta, y, d)
        if options.jackson_damping:
            cheb = FullyPhaseArithmetic._jackson_damp_chebyshev(cheb)
        coeffs = FullyPhaseArithmetic._chebyshev_to_shifted_laurent(cheb)
        if options.contractive_rescale:
            return FullyPhaseArithmetic._enforce_global_contractivity(coeffs, options.n_check_samples, options.contractive_margin)
        return coeffs, 1.0

    @staticmethod
    def _chebyshev_projection(theta: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        d = int(degree)
        cheb = np.zeros(d + 1, dtype=float)
        cheb[0] = np.sum(y) / len(y)
        for k in range(1, d + 1):
            cheb[k] = 2.0 * np.sum(y * np.cos(k * theta)) / len(y)
        return cheb

    @staticmethod
    def _chebyshev_lstsq(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        V = ncheb.chebvander(x, int(degree))
        cheb, *_ = np.linalg.lstsq(V, y, rcond=None)
        return cheb

    @staticmethod
    def _smooth_window_target(delta: np.ndarray, beta: float, a: float, zero_window: float) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        y = FullyPhaseArithmetic._exact_amplitude(delta, beta, a)
        inside = np.abs(delta) < zero_window
        if not np.any(inside):
            return y
        w = float(zero_window)
        t = (delta[inside] + w) / (2.0 * w)
        y0 = float(FullyPhaseArithmetic._exact_amplitude(-w, beta, a))
        y1 = float(FullyPhaseArithmetic._exact_amplitude(+w, beta, a))
        dy0 = float(FullyPhaseArithmetic._exact_amplitude_derivative(-w, beta, a))
        dy1 = float(FullyPhaseArithmetic._exact_amplitude_derivative(+w, beta, a))
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        y[inside] = h00 * y0 + h10 * (2.0 * w * dy0) + h01 * y1 + h11 * (2.0 * w * dy1)
        return y

    @staticmethod
    def _jackson_damp_chebyshev(cheb: np.ndarray) -> np.ndarray:
        cheb = np.asarray(cheb, dtype=float).copy()
        d = len(cheb) - 1
        if d <= 0:
            return cheb
        theta = np.pi / (d + 2)
        for k in range(d + 1):
            cheb[k] *= ((d - k + 2) * np.cos(k * theta) + np.sin(k * theta) / np.tan(theta)) / (d + 2)
        return cheb

    @staticmethod
    def _chebyshev_to_shifted_laurent(cheb: np.ndarray) -> np.ndarray:
        d = len(cheb) - 1
        coeffs = np.zeros(2 * d + 1, dtype=complex)
        coeffs[d] = cheb[0]
        for k in range(1, d + 1):
            coeffs[d - k] = 0.5 * cheb[k]
            coeffs[d + k] = 0.5 * cheb[k]
        return coeffs

    @staticmethod
    def _enforce_global_contractivity(coeffs: np.ndarray, n_check_samples: int | None, contractive_margin: float) -> tuple[np.ndarray, float]:
        coeffs = np.asarray(coeffs, dtype=complex).copy()
        d = (len(coeffs) - 1) // 2
        theta = np.linspace(-np.pi, np.pi, n_check_samples or max(20000, 256 * d), endpoint=False)
        values = FullyPhaseArithmetic._eval_laurent_theta(coeffs, theta)
        scale = max(1.0, float(np.max(np.abs(values))) / (1.0 - contractive_margin))
        return coeffs / scale, scale

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
        x = np.clip(np.asarray(delta, dtype=float) / self.alpha, -1.0, 1.0)
        theta = np.arccos(x)
        return np.real_if_close(self._eval_laurent_theta(self.poly_coeffs, -theta)).real

    def exact_amplitude(self, delta: np.ndarray | float) -> np.ndarray:
        return self._exact_amplitude(delta, self.beta, self.a)


def plot_fully_phase_arithmetic(n: int, h: np.ndarray, J: np.ndarray, beta: float, eps_ops: float, a: float = np.inf, degree: int | None = None, poly_options: FullyPhaseArithmeticPolyOptions | None = None, is_mocked_angles: bool = False):
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit
    from monaqa2.qiskit.utils_qiskit import get_unitary
    from monaqa2.qiskit.utils_numpy import ket, bra, kron

    gate = FullyPhaseArithmetic(n, h, J, beta, eps_ops, a=a, degree=degree, poly_options=poly_options, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)
    qc = QuantumCircuit(2 * n + 1)
    qc.append(gate, range(2 * n + 1))
    U = get_unitary(qc, big_endian=True)

    def E(bits: int) -> float:
        z = np.array([1.0 if ((bits >> (n - 1 - i)) & 1) == 0 else -1.0 for i in range(n)])
        return h @ z + sum(J[i, j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n))

    dE, circuit = [], []
    for x in range(2**n):
        for y in range(2**n):
            delta = E(y) - E(x)
            psi = kron(ket(0, 1), ket(x, n), ket(y, n))
            amp = (kron(bra(0, 1), bra(x, n), bra(y, n)) @ U @ psi)[0, 0]
            dE.append(delta)
            circuit.append(float(np.real(np.real_if_close(amp))))

    dE = np.asarray(dE)
    circuit = np.asarray(circuit)
    dE_grid = np.linspace(dE.min(), dE.max(), 1000)
    ideal_label = "ideal sqrt MH acceptance" if np.isinf(a) else "ideal sqrt generalized Glauber"

    plt.figure(figsize=(8, 5))
    plt.plot(dE_grid, gate.exact_amplitude(dE_grid), color="red", linewidth=1.0, label=ideal_label)
    plt.plot(dE_grid, gate.evaluate_polynomial_delta(dE_grid), color="blue", linewidth=2.0, label="implemented GQSP polynomial")
    plt.scatter(dE, circuit, s=24, color="lightskyblue", alpha=0.75, edgecolors="none", label="circuit implementation")

    if gate.zero_window > 0.0:
        plt.axvline(-gate.zero_window, color="grey", linestyle="--", linewidth=1.0, label="zero-window boundary")
        plt.axvline(+gate.zero_window, color="grey", linestyle="--", linewidth=1.0)

    if np.isfinite(gate.tail_cutoff_delta) and dE.min() <= gate.tail_cutoff_delta <= dE.max():
        plt.axvline(gate.tail_cutoff_delta, color="black", linestyle=":", linewidth=1.0, label="positive tail cutoff")

    plt.xlabel(r"$\Delta E = E(y)-E(x)$")
    plt.ylabel(r"$\sqrt{A_{yx}}$")
    plt.title(f"degree={gate.selected_degree}, scale={gate.poly_scale:.6g}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return gate


def plot_error_fully_phase_arithmetic(n: int, h: np.ndarray, J: np.ndarray, eps_ops: float, a: float = np.inf, poly_options: FullyPhaseArithmeticPolyOptions | None = None, kappas: list[float] | None = None, fixed_degree: int = 20, n_delta_samples: int = 4000, is_mocked_angles: bool = True):
    import matplotlib.pyplot as plt

    kappas = [1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32, 64, 128] if kappas is None else kappas
    h_delta, J_delta = MockedQubitizedDeltaHamiltonian._build_delta_hamiltonian(n, h, J)
    alpha = MockedQubitizedDeltaHamiltonian._lcu_norm(h_delta, J_delta)

    err_fixed, err_linear, deg_linear = [], [], []
    delta_grid = np.linspace(-alpha, alpha, n_delta_samples)

    for kappa in kappas:
        beta = 2.0 * kappa / alpha

        L = np.log(1.0 / eps_ops)
        alpha_ = 2 * n
        # print(f"{n=} {alpha=} {alpha_=}")
        kappa_ = kappa * alpha_ / alpha
        zero_window = poly_options.zero_window
        deg_exp = int(np.ceil(1.0 * (np.sqrt(kappa_ * L) + L)))
        deg_window = int(np.ceil((alpha_ / zero_window) * np.log2(1 / eps_ops)))
        degree_linear = max(deg_window, deg_exp) + 2
        # print(kappa, "=> Degree linear: ", degree_linear, " | exp =", deg_exp, " ; win=", deg_window, " ; zero window: ", zero_window)
        deg_linear.append(degree_linear)

        gate_fixed = FullyPhaseArithmetic(n, h, J, beta, eps_ops, a=a, degree=fixed_degree, poly_options=poly_options, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)
        gate_linear = FullyPhaseArithmetic(n, h, J, beta, eps_ops, a=a, degree=degree_linear, poly_options=poly_options, is_mocked_construction=True, is_mocked_angles=is_mocked_angles)

        zero_window = max(getattr(gate_fixed, "zero_window", 0.0), getattr(gate_linear, "zero_window", 0.0))
        mask = np.abs(delta_grid) >= zero_window

        exact = gate_fixed.exact_amplitude(delta_grid[mask])
        err_fixed.append(float(np.max(np.abs(gate_fixed.evaluate_polynomial_delta(delta_grid[mask]) - exact))))
        err_linear.append(float(np.max(np.abs(gate_linear.evaluate_polynomial_delta(delta_grid[mask]) - exact))))

    plt.figure(figsize=(7, 4.5))
    plt.scatter(kappas, err_fixed, color="red", label=f"degree={fixed_degree}")
    plt.plot(kappas, err_fixed, color="red", linewidth=1.0, alpha=0.5)
    plt.scatter(kappas, err_linear, color="blue", label=r"degree=$2 + \max\{\sqrt{(\beta B/2) L}+L,(B/w)L\}$")
    plt.plot(kappas, err_linear, color="blue", linewidth=1.0, alpha=0.5)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(r"$\kappa=\beta\alpha/2$")
    plt.ylabel("max absolute error outside zero window")
    plt.title("FullyPhaseArithmetic polynomial error")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    return {"kappa": np.asarray(kappas), "fixed_degree_error": np.asarray(err_fixed), "linear_degree_error": np.asarray(err_linear), "linear_degree": np.asarray(deg_linear), "alpha": alpha}