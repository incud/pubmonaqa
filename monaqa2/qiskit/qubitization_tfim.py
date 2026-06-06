import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit



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
        qc.ccx(0, 1, 2)
        self._ry_rz_clifford(qc, -self.theta / 2.0, 2)
        qc.ccx(0, 1, 2)
        return qc


class GivensRotation(Gate):
    def __init__(self, left_weight: float, right_weight: float, label=None) -> None:
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.theta = self._givens_angle_from_weights(left_weight, right_weight)
        super().__init__(name="givens", num_qubits=2, params=[left_weight, right_weight], label=label)
        self.definition = self._build_definition()

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

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        qc.ccx(0, 1, 2)
        qc.append(Ccry(-2.0 * self.theta), [0, 2, 1])
        qc.ccx(0, 1, 2)
        return qc


class PrepareTFIM(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        self.gamma = gamma
        assert h.shape == gamma.shape == (n,) and J.shape == (n, n)

        # LCU normalization constants. Only the upper triangular part of J is used.
        self.lambda_Z = np.sum(np.abs(h))
        self.lambda_X = np.sum(np.abs(gamma))
        self.lambda_J = np.sum(np.abs(np.triu(J, 1)))
        self.lam = self.lambda_Z + self.lambda_X + self.lambda_J

        super().__init__(name="prepare_tfim", num_qubits=4 * n, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def uZ(self) -> list[int]:
        return list(range(0, self.n))

    @property
    def uX(self) -> list[int]:
        return list(range(self.n, 2 * self.n))

    @property
    def uJ(self) -> list[int]:
        return list(range(2 * self.n, 3 * self.n))

    @property
    def vJ(self) -> list[int]:
        return list(range(3 * self.n, 4 * self.n))

    def _append_unary_ladder(self, qc: QuantumCircuit, weights: np.ndarray, register: list[int]) -> None:
        # Starting from |e_1>, this prepares sum_i sqrt(weights[i] / sum(weights)) |e_i>.
        tails = np.cumsum(weights[::-1])[::-1]
        for k in range(len(register) - 1):
            qc.append(GivensRotation(weights[k], tails[k + 1]), [register[k], register[k + 1]])

    def _prepare_branch_no_ancilla(self, qc: QuantumCircuit) -> None:
        # Seed the Z branch as |e_1,0,0,0>.
        qc.x(self.uZ[0])

        # Split Z versus non-Z, i.e. ZZ + X.
        qc.append(GivensRotation(self.lambda_Z, self.lambda_J + self.lambda_X), [self.uZ[0], self.uJ[0]])

        # Split the non-Z amplitude into ZZ and X.
        qc.append(GivensRotation(self.lambda_J, self.lambda_X), [self.uJ[0], self.uX[0]])

    def _prepare_Z_terms(self, qc: QuantumCircuit) -> None:
        # Prepare sum_i sqrt(|h_i| / lambda_Z) |e_i> in uZ.
        self._append_unary_ladder(qc, np.abs(self.h), self.uZ)

    def _prepare_X_terms(self, qc: QuantumCircuit) -> None:
        # Prepare sum_i sqrt(|gamma_i| / lambda_X) |e_i> in uX.
        self._append_unary_ladder(qc, np.abs(self.gamma), self.uX)

    def _prepare_ZZ_terms(self, qc: QuantumCircuit) -> None:
        # Keep only the i < j part because the Hamiltonian has sum_{i<j} J_ij Z_i Z_j.
        abs_J = np.abs(np.triu(self.J, 1))

        # Row weights R_i = sum_{j>i} |J_ij|. First endpoint i is sampled with probability R_i / lambda_J.
        R = np.zeros(self.n)
        for i in range(self.n - 1):
            R[i] = np.sum(abs_J[i, i + 1:])

        # Prepare sum_i sqrt(R_i / lambda_J) |e_i> in uJ.
        self._append_unary_ladder(qc, R, self.uJ)

        # Given first endpoint i, initialize the second endpoint to j = i + 1.
        for i in range(self.n - 1):
            qc.cx(self.uJ[i], self.vJ[i + 1])

        # Tail sums T[i,j] = sum_{ell >= j} |J_iell|, used for the conditional row ladders.
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            running = 0.0
            for j in range(self.n - 1, -1, -1):
                running += abs_J[i, j]
                T[i, j] = running

        # Conditionally spread vJ over j > i so that amplitudes become sqrt(|J_ij| / R_i).
        for d in range(1, self.n - 1):
            for parity in (0, 1):
                for i in range(self.n):
                    if i % 2 != parity:
                        continue
                    j = i + d
                    if j >= self.n - 1:
                        continue
                    qc.append(ControlledGivensRotation(abs_J[i, j], T[i, j + 1]), [self.uJ[i], self.vJ[j], self.vJ[j + 1]])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(4 * self.n, name=self.name)

        # Prepare branch weights: Z, ZZ, X.
        self._prepare_branch_no_ancilla(qc)

        # Prepare the one-body and two-body unsigned magnitudes.
        self._prepare_Z_terms(qc)
        self._prepare_X_terms(qc)
        self._prepare_ZZ_terms(qc)

        return qc


class SelectTFIM(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        self.gamma = gamma
        assert h.shape == gamma.shape == (n,) and J.shape == (n, n)
        super().__init__(name="select_tfim", num_qubits=5 * n, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def uZ(self) -> list[int]:
        return list(range(0, self.n))

    @property
    def uX(self) -> list[int]:
        return list(range(self.n, 2 * self.n))

    @property
    def uJ(self) -> list[int]:
        return list(range(2 * self.n, 3 * self.n))

    @property
    def vJ(self) -> list[int]:
        return list(range(3 * self.n, 4 * self.n))

    @property
    def q(self) -> list[int]:
        return list(range(4 * self.n, 5 * self.n))

    def _apply_signs(self, qc: QuantumCircuit) -> None:
        for i in range(self.n):
            if self.h[i] < 0:
                qc.z(self.uZ[i])
            if self.gamma[i] < 0:
                qc.z(self.uX[i])
        for color in range(self.n):
            for i in range(self.n):
                j = (color - i) % self.n
                if i < j and self.J[i, j] < 0:
                    qc.cz(self.uJ[i], self.vJ[j])

    def _apply_paulis(self, qc: QuantumCircuit) -> None:
        for i in range(self.n):
            qc.cz(self.uZ[i], self.q[i])
        for i in range(self.n):
            qc.cx(self.uX[i], self.q[i])
        for i in range(self.n):
            qc.cz(self.uJ[i], self.q[i])
        for i in range(self.n):
            qc.cz(self.vJ[i], self.q[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(5 * self.n, name=self.name)
        self._apply_signs(qc)
        self._apply_paulis(qc)
        return qc


class ControlledSelectTFIM(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        self.gamma = gamma
        assert h.shape == gamma.shape == (n,) and J.shape == (n, n)
        super().__init__(name="c_select_tfim", num_qubits=1 + 5 * n, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def uZ(self) -> list[int]:
        return list(range(1, 1 + self.n))

    @property
    def uX(self) -> list[int]:
        return list(range(1 + self.n, 1 + 2 * self.n))

    @property
    def uJ(self) -> list[int]:
        return list(range(1 + 2 * self.n, 1 + 3 * self.n))

    @property
    def vJ(self) -> list[int]:
        return list(range(1 + 3 * self.n, 1 + 4 * self.n))

    @property
    def q(self) -> list[int]:
        return list(range(1 + 4 * self.n, 1 + 5 * self.n))

    def _ccz(self, qc: QuantumCircuit, a: int, b: int, target: int) -> None:
        qc.h(target)
        qc.ccx(a, b, target)
        qc.h(target)

    def _apply_signs(self, qc: QuantumCircuit) -> None:
        for i in range(self.n):
            if self.h[i] < 0:
                qc.cz(self.c, self.uZ[i])
            if self.gamma[i] < 0:
                qc.cz(self.c, self.uX[i])
        for color in range(self.n):
            for i in range(self.n):
                j = (color - i) % self.n
                if i < j and self.J[i, j] < 0:
                    self._ccz(qc, self.c, self.uJ[i], self.vJ[j])

    def _apply_paulis(self, qc: QuantumCircuit) -> None:
        for i in range(self.n):
            self._ccz(qc, self.c, self.uZ[i], self.q[i])
        for i in range(self.n):
            qc.ccx(self.c, self.uX[i], self.q[i])
        for i in range(self.n):
            self._ccz(qc, self.c, self.uJ[i], self.q[i])
        for i in range(self.n):
            self._ccz(qc, self.c, self.vJ[i], self.q[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(1 + 5 * self.n, name=self.name)
        self._apply_signs(qc)
        self._apply_paulis(qc)
        return qc
    


class ReflectionZero(Gate):
    def __init__(self, m: int, label=None):
        self.m = m
        super().__init__(name="reflection_zero", num_qubits=m, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(0, self.m))

    @property
    def target(self) -> int:
        return self.m - 1

    @property
    def controls(self) -> list[int]:
        return list(range(0, self.m - 1))

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.x(self.selection)
        qc.h(self.target)
        qc.mcx(self.controls, self.target)
        qc.h(self.target)
        qc.x(self.selection)
        return qc


class ControlledReflectionZero(Gate):
    def __init__(self, m: int, label=None):
        self.m = m
        super().__init__(name="c_reflection_zero", num_qubits=1 + m, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def selection(self) -> list[int]:
        return list(range(1, 1 + self.m))

    @property
    def target(self) -> int:
        return self.selection[-1]

    @property
    def controls(self) -> list[int]:
        return [self.c] + self.selection[:-1]
           

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.x(self.selection)
        qc.h(self.target)
        qc.mcx(self.controls, self.target)
        qc.h(self.target)
        qc.x(self.selection)
        return qc


class QubitizedTFIM(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        self.gamma = gamma
        assert h.shape == gamma.shape == (n,) and J.shape == (n, n)
        super().__init__(name="qubitized_tfim", num_qubits=5 * n, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def selection(self) -> list[int]:
        return list(range(0, 4 * self.n))

    @property
    def select_qubits(self) -> list[int]:
        return list(range(0, 5 * self.n))

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = PrepareTFIM(self.n, self.h, self.J, self.gamma)
        select = SelectTFIM(self.n, self.h, self.J, self.gamma)
        qc.append(prepare, self.selection)
        qc.append(select, self.select_qubits)
        qc.append(prepare.inverse(), self.selection)
        qc.append(ReflectionZero(4 * self.n), self.selection)
        return qc


class ControlledQubitizedTFIM(Gate):
    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, label=None):
        self.n = n
        self.h = h
        self.J = J
        self.gamma = gamma
        assert h.shape == gamma.shape == (n,) and J.shape == (n, n)
        super().__init__(name="c_qubitized_tfim", num_qubits=1 + 5 * n, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def c(self) -> int:
        return 0

    @property
    def selection(self) -> list[int]:
        return list(range(1, 1 + 4 * self.n))

    @property
    def select_qubits(self) -> list[int]:
        return list(range(0, 1 + 5 * self.n))

    @property
    def reflection_qubits(self) -> list[int]:
        return [self.c] + self.selection

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = PrepareTFIM(self.n, self.h, self.J, self.gamma)
        c_select = ControlledSelectTFIM(self.n, self.h, self.J, self.gamma)
        qc.append(prepare, self.selection)
        qc.append(c_select, self.select_qubits)
        qc.append(prepare.inverse(), self.selection)
        qc.append(ControlledReflectionZero(4 * self.n), self.reflection_qubits)
        return qc
