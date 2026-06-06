import numpy as np
from qiskit.circuit import Gate, QuantumCircuit


class TrotterizedOperatorIsingTF(Gate):
    r"""Second-order grouped Trotterization of exp(-i t H) for H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + sum_i gamma_i X_i, using S_2(dt)^r with dt=t/r and r=num_trotter_steps."""

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray | float, time: float = 1.0, num_trotter_steps: int = 1, atol: float = 1e-8, label=None) -> None:
        if n < 1:
            raise ValueError("n must be positive.")
        if not isinstance(num_trotter_steps, (int, np.integer)) or int(num_trotter_steps) <= 0:
            raise ValueError("num_trotter_steps must be a positive integer.")
        if atol < 0.0:
            raise ValueError("atol must be non-negative.")

        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.gamma = np.full(self.n, float(gamma)) if np.asarray(gamma).ndim == 0 else np.asarray(gamma, dtype=float)
        self.time = float(time)
        self.num_trotter_steps = int(num_trotter_steps)
        self.atol = float(atol)

        if self.h.shape != (self.n,):
            raise ValueError(f"h must have shape ({self.n},), got {self.h.shape}.")
        if self.J.shape != (self.n, self.n):
            raise ValueError(f"J must have shape ({self.n}, {self.n}), got {self.J.shape}.")
        if self.gamma.shape != (self.n,):
            raise ValueError(f"gamma must have shape ({self.n},), got {self.gamma.shape}.")

        self.z_terms = [(i, float(self.h[i])) for i in range(self.n) if self._is_nonzero(self.h[i])]
        self.zz_terms = [(i, j, float(self.J[i, j])) for i in range(self.n) for j in range(i + 1, self.n) if self._is_nonzero(self.J[i, j])]
        self.x_terms = [(i, float(self.gamma[i])) for i in range(self.n) if self._is_nonzero(self.gamma[i])]
        self.n_terms_z = len(self.z_terms)
        self.n_terms_zz = len(self.zz_terms)
        self.n_terms_x = len(self.x_terms)
        self.n_terms = self.n_terms_z + self.n_terms_zz + self.n_terms_x

        super().__init__("TrotterizedOperatorIsingTF", self.n, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"system": list(range(self.n))}

    @staticmethod
    def _matchings(n: int) -> list[list[tuple[int, int]]]:
        vertices: list[int | None] = list(range(n)) + ([] if n % 2 == 0 else [None])
        rounds = []
        for _ in range(len(vertices) - 1):
            rounds.append([tuple(sorted((a, b))) for a, b in zip(vertices[:len(vertices) // 2], reversed(vertices[len(vertices) // 2:])) if a is not None and b is not None])
            vertices = [vertices[0]] + [vertices[-1]] + vertices[1:-1]
        return rounds

    def _is_nonzero(self, x: float) -> bool:
        return bool(abs(float(x)) >= self.atol)

    @staticmethod
    def _append_rzz(qc: QuantumCircuit, angle: float, q0: int, q1: int) -> None:
        qc.cx(q0, q1)
        qc.rz(angle, q1)
        qc.cx(q0, q1)

    def _apply_z_layer(self, qc: QuantumCircuit, system: list[int], dt: float) -> None:
        for i, coeff in self.z_terms:
            qc.rz(2.0 * dt * coeff, system[i])

    def _apply_zz_layers(self, qc: QuantumCircuit, system: list[int], dt: float) -> None:
        coeff_by_edge = {(i, j): coeff for i, j, coeff in self.zz_terms}
        for matching in self._matchings(self.n):
            for i, j in matching:
                if (i, j) in coeff_by_edge:
                    self._append_rzz(qc, 2.0 * dt * coeff_by_edge[(i, j)], system[i], system[j])

    def _apply_x_layer(self, qc: QuantumCircuit, system: list[int], dt: float) -> None:
        for i, coeff in self.x_terms:
            qc.rx(2.0 * dt * coeff, system[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        system = self.layout["system"]
        dt = self.time / self.num_trotter_steps
        for _ in range(self.num_trotter_steps):
            if self.x_terms:
                self._apply_x_layer(qc, system, 0.5 * dt)
            self._apply_z_layer(qc, system, dt)
            self._apply_zz_layers(qc, system, dt)
            if self.x_terms:
                self._apply_x_layer(qc, system, 0.5 * dt)
        return qc


class ControlledTrotterizedOperatorIsingTF(Gate):
    r"""Controlled second-order grouped Trotterization of the transverse-field Ising evolution."""

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray | float, time: float = 1.0, num_trotter_steps: int = 1, atol: float = 1e-8, label=None) -> None:
        self.base = TrotterizedOperatorIsingTF(n, h, J, gamma, time=time, num_trotter_steps=num_trotter_steps, atol=atol)
        self.n = self.base.n
        self.h = self.base.h
        self.J = self.base.J
        self.gamma = self.base.gamma
        self.time = self.base.time
        self.atol = self.base.atol
        self.num_trotter_steps = self.base.num_trotter_steps
        self.z_terms = self.base.z_terms
        self.zz_terms = self.base.zz_terms
        self.x_terms = self.base.x_terms
        self.n_terms_z = self.base.n_terms_z
        self.n_terms_zz = self.base.n_terms_zz
        self.n_terms_x = self.base.n_terms_x
        self.n_terms = self.base.n_terms
        self.n_control_work = max(1, self.n)

        super().__init__("ControlledTrotterizedOperatorIsingTF", 1 + self.n + self.n_control_work, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"control": [0], "system": list(range(1, 1 + self.n)), "control_work": list(range(1 + self.n, 1 + self.n + self.n_control_work))}

    @staticmethod
    def _fanout_control(qc: QuantumCircuit, control: int, work: list[int]) -> None:
        for q in work:
            qc.cx(control, q)

    @staticmethod
    def _unfanout_control(qc: QuantumCircuit, control: int, work: list[int]) -> None:
        for q in reversed(work):
            qc.cx(control, q)

    @staticmethod
    def _append_controlled_rz(qc: QuantumCircuit, angle: float, control: int, target: int) -> None:
        qc.rz(0.5 * angle, target)
        qc.cx(control, target)
        qc.rz(-0.5 * angle, target)
        qc.cx(control, target)

    @classmethod
    def _append_controlled_rx(cls, qc: QuantumCircuit, angle: float, control: int, target: int) -> None:
        qc.h(target)
        cls._append_controlled_rz(qc, angle, control, target)
        qc.h(target)

    @classmethod
    def _append_controlled_rzz(cls, qc: QuantumCircuit, angle: float, control: int, q0: int, q1: int) -> None:
        qc.ccx(control, q0, q1)
        cls._append_controlled_rz(qc, angle, control, q1)
        qc.ccx(control, q0, q1)

    def _apply_terms(self, qc: QuantumCircuit, controls: list[int], terms: list, apply, dt: float) -> None:
        width = len(controls)
        for start in range(0, len(terms), width):
            for c, term in zip(controls, terms[start:start + width]):
                apply(qc, c, term, dt)

    def _apply_controlled_z_layer(self, qc: QuantumCircuit, controls: list[int], system: list[int], dt: float) -> None:
        self._apply_terms(qc, controls, self.z_terms, lambda qc_, c, term, dt_: self._append_controlled_rz(qc_, 2.0 * dt_ * term[1], c, system[term[0]]), dt)

    def _apply_controlled_zz_layers(self, qc: QuantumCircuit, controls: list[int], system: list[int], dt: float) -> None:
        coeff_by_edge = {(i, j): coeff for i, j, coeff in self.zz_terms}
        width = len(controls)
        for matching in TrotterizedOperatorIsingTF._matchings(self.n):
            active_edges = [(i, j, coeff_by_edge[(i, j)]) for i, j in matching if (i, j) in coeff_by_edge]
            for start in range(0, len(active_edges), width):
                for c, (i, j, coeff) in zip(controls, active_edges[start:start + width]):
                    self._append_controlled_rzz(qc, 2.0 * dt * coeff, c, system[i], system[j])

    def _apply_controlled_x_layer(self, qc: QuantumCircuit, controls: list[int], system: list[int], dt: float) -> None:
        self._apply_terms(qc, controls, self.x_terms, lambda qc_, c, term, dt_: self._append_controlled_rx(qc_, 2.0 * dt_ * term[1], c, system[term[0]]), dt)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        control, system, control_work = self.layout["control"][0], self.layout["system"], self.layout["control_work"]
        dt = self.time / self.num_trotter_steps
        self._fanout_control(qc, control, control_work)
        for _ in range(self.num_trotter_steps):
            if self.x_terms:
                self._apply_controlled_x_layer(qc, control_work, system, 0.5 * dt)
            self._apply_controlled_z_layer(qc, control_work, system, dt)
            self._apply_controlled_zz_layers(qc, control_work, system, dt)
            if self.x_terms:
                self._apply_controlled_x_layer(qc, control_work, system, 0.5 * dt)
        self._unfanout_control(qc, control, control_work)
        return qc


class ProposalQemc(Gate):
    r"""Trotter-only QEMC proposal: copy A into B, then apply TrotterizedOperatorIsingTF to B."""

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray | float, t: float, num_trotter_steps: int = 1, atol: float = 1e-8, label=None) -> None:
        if n < 1:
            raise ValueError("n must be a positive integer.")
        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.gamma = np.full(self.n, float(gamma)) if np.asarray(gamma).ndim == 0 else np.asarray(gamma, dtype=float)
        self.t = float(t)
        self.num_trotter_steps = int(num_trotter_steps)
        self.atol = float(atol)
        self.hsim = TrotterizedOperatorIsingTF(self.n, self.h, self.J, self.gamma, time=self.t, num_trotter_steps=self.num_trotter_steps, atol=self.atol)
        self.name = f"ProposalQemc(n={self.n},trotter_steps={self.num_trotter_steps})"

        super().__init__(self.name, 2 * self.n, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"A": list(range(self.n)), "B": list(range(self.n, 2 * self.n))}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        for a, b in zip(self.layout["A"], self.layout["B"]):
            qc.cx(a, b)
        qc.append(self.hsim, self.layout["B"])
        return qc
