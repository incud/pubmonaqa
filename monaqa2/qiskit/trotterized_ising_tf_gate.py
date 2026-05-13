import numpy as np
from qiskit.circuit import Gate, QuantumCircuit


class TrotterizedOperatorIsingTF(Gate):
    r"""
    Second-order grouped Trotterization of the transverse-field Ising evolution.

        U(t) = exp(-i t H),

    where

        H = H_Z + H_X,

        H_Z = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,
        H_X = sum_i gamma_i X_i.

    The implementation uses the second-order Strang formula

        S_2(dt) = exp(-i dt H_X / 2) exp(-i dt H_Z) exp(-i dt H_X / 2),

    and applies

        S_2(t / r)^r.

    The diagonal part exp(-i dt H_Z) is decomposed into commuting Z and ZZ
    rotations. The ZZ rotations are explicitly decomposed as

        RZZ(theta) = CX RZ(theta) CX,

    so the non-controlled circuit contains only cx, rz, and rx gates.

    If num_trotter_steps is None, r is chosen from a conservative second-order
    nested-commutator bound

        error <= |t|^3 C_2 / r^2,

    where

        C_2 =
            ||[H_X, [H_X, H_Z]]|| / 24
            +
            ||[H_Z, [H_Z, H_X]]|| / 12.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        gamma: np.ndarray,
        time: float = 1.0,
        eps: float | None = None,
        num_trotter_steps: int | None = None,
        atol: float = 1e-8,
        label=None,
    ) -> None:
        if n < 1:
            raise ValueError("n must be positive.")

        h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        if h.shape != (n,):
            raise ValueError(f"h must have shape ({n},), got {h.shape}.")
        if J.shape != (n, n):
            raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}.")
        if gamma.shape != (n,) and np.sum(np.abs(gamma)) > atol:
            raise ValueError(f"gamma must have shape ({n},), got {gamma.shape}.")
        if atol < 0.0:
            raise ValueError("atol must be non-negative.")

        self.n = int(n)
        self.h = h
        self.J = J
        self.gamma = gamma
        self.time = float(time)
        self.eps = None if eps is None else float(eps)
        self.atol = float(atol)
        self.num_trotter_steps = self._resolve_num_trotter_steps(num_trotter_steps)

        self.z_terms = self._z_terms()
        self.zz_terms = self._zz_terms()
        self.x_terms = self._x_terms()
        self.n_terms_z = len(self.z_terms)
        self.n_terms_zz = len(self.zz_terms)
        self.n_terms_x = len(self.x_terms)
        self.n_terms = self.n_terms_z + self.n_terms_zz + self.n_terms_x

        super().__init__("TrotterizedOperatorIsingTF", self.n, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"system": list(range(self.n))}

    def _resolve_num_trotter_steps(self, num_trotter_steps: int | None) -> int:
        if num_trotter_steps is not None:
            if not isinstance(num_trotter_steps, (int, np.integer)):
                raise TypeError("num_trotter_steps must be an integer or None.")
            if int(num_trotter_steps) <= 0:
                raise ValueError("num_trotter_steps must be positive.")
            return int(num_trotter_steps)

        if self.eps is None:
            raise ValueError("Pass either num_trotter_steps or eps.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive when num_trotter_steps is None.")

        # There is no X component, just skip it
        if np.sum(np.abs(self.gamma)) == 0:
            return 1

        bound = self.second_order_commutator_bound(self.n, self.h, self.J, self.gamma)

        if bound == 0.0 or self.time == 0.0:
            return 1

        r = int(np.ceil(np.sqrt((abs(self.time) ** 3) * bound / self.eps)))
        return max(1, r)


    @staticmethod
    def _pauli_anticommutes(a: dict[int, str], b: dict[int, str]) -> bool:
        count = 0

        for q in set(a) & set(b):
            if a[q] != b[q]:
                count += 1

        return bool(count % 2)

    @staticmethod
    def _pauli_product(a: dict[int, str], b: dict[int, str]) -> dict[int, str]:
        table = {
            ("X", "X"): None,
            ("Y", "Y"): None,
            ("Z", "Z"): None,
            ("X", "Y"): "Z",
            ("Y", "X"): "Z",
            ("X", "Z"): "Y",
            ("Z", "X"): "Y",
            ("Y", "Z"): "X",
            ("Z", "Y"): "X",
        }

        out = dict(a)

        for q, p in b.items():
            if q not in out:
                out[q] = p
            else:
                r = table[(out[q], p)]

                if r is None:
                    del out[q]
                else:
                    out[q] = r

        return out

    @staticmethod
    def _nested_commutator_bound(
        outer_terms: list[tuple[float, dict[int, str]]],
        left_terms: list[tuple[float, dict[int, str]]],
        right_terms: list[tuple[float, dict[int, str]]],
    ) -> float:
        bound = 0.0

        for a, pauli_a in outer_terms:
            for b, pauli_b in left_terms:
                for c, pauli_c in right_terms:
                    if not TrotterizedOperatorIsingTF._pauli_anticommutes(pauli_b, pauli_c):
                        continue

                    inner_pauli = TrotterizedOperatorIsingTF._pauli_product(pauli_b, pauli_c)

                    if TrotterizedOperatorIsingTF._pauli_anticommutes(pauli_a, inner_pauli):
                        bound += 4.0 * abs(float(a) * float(b) * float(c))

        return float(bound)

    @staticmethod
    def second_order_commutator_bound(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray) -> float:
        h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        x_terms = [(float(gamma[i]), {i: "X"}) for i in range(n) if gamma[i] != 0.0]

        z_terms = [(float(h[i]), {i: "Z"}) for i in range(n) if h[i] != 0.0]
        z_terms += [
            (float(J[i, j]), {i: "Z", j: "Z"})
            for i in range(n)
            for j in range(i + 1, n)
            if J[i, j] != 0.0
        ]

        x_x_z = TrotterizedOperatorIsingTF._nested_commutator_bound(x_terms, x_terms, z_terms)
        z_z_x = TrotterizedOperatorIsingTF._nested_commutator_bound(z_terms, z_terms, x_terms)

        return float(x_x_z / 24.0 + z_z_x / 12.0)

    @staticmethod
    def _matchings(n: int) -> list[list[tuple[int, int]]]:
        if n < 2:
            return []

        vertices: list[int | None] = list(range(n))

        if n % 2 == 1:
            vertices.append(None)

        m = len(vertices)
        rounds: list[list[tuple[int, int]]] = []

        for _ in range(m - 1):
            matching: list[tuple[int, int]] = []

            for k in range(m // 2):
                a = vertices[k]
                b = vertices[m - 1 - k]

                if a is None or b is None:
                    continue

                i, j = sorted((a, b))
                matching.append((i, j))

            rounds.append(matching)
            vertices = [vertices[0]] + [vertices[-1]] + vertices[1:-1]

        return rounds

    def _is_nonzero(self, x: float) -> bool:
        return bool(abs(float(x)) >= self.atol)

    def _z_terms(self) -> list[tuple[int, float]]:
        return [(i, float(self.h[i])) for i in range(self.n) if self._is_nonzero(self.h[i])]

    def _zz_terms(self) -> list[tuple[int, int, float]]:
        return [(i, j, float(self.J[i, j])) for i in range(self.n) for j in range(i + 1, self.n) if self._is_nonzero(self.J[i, j])]

    def _x_terms(self) -> list[tuple[int, float]]:
        if self.gamma is not None and np.sum(np.abs(self.gamma)) > 0:
            return [(i, float(self.gamma[i])) for i in range(self.n) if self._is_nonzero(self.gamma[i])]
        else:
            return []
        
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
                coeff = coeff_by_edge.get((i, j))

                if coeff is None:
                    continue

                self._append_rzz(qc, 2.0 * dt * coeff, system[i], system[j])

    def _apply_x_layer(self, qc: QuantumCircuit, system: list[int], dt: float) -> None:
        for i, coeff in self.x_terms:
            qc.rx(2.0 * dt * coeff, system[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        system = self.layout["system"]
        dt = self.time / self.num_trotter_steps

        for _ in range(self.num_trotter_steps):
            if np.sum(np.abs(self.gamma)) > 0:
                self._apply_x_layer(qc, system, 0.5 * dt)
            self._apply_z_layer(qc, system, dt)
            self._apply_zz_layers(qc, system, dt)
            if np.sum(np.abs(self.gamma)) > 0:
                self._apply_x_layer(qc, system, 0.5 * dt)

        return qc


class ControlledTrotterizedOperatorIsingTF(Gate):
    r"""
    Explicit controlled second-order Trotterized transverse-field Ising evolution.

    Acts as

        |0><0| \otimes I + |1><1| \otimes U_Trotter(t).

    The controlled circuit uses the same second-order Strang formula,

        exp(-i dt H_X / 2) exp(-i dt H_Z) exp(-i dt H_X / 2),

    controlled on the input control qubit.

    The control is fanned out to O(n) work qubits once, used throughout all
    Trotter steps, and unfanned at the end. The ZZ rotations are scheduled by a
    round-robin edge coloring of the complete graph, so each ZZ layer contains
    disjoint system-qubit pairs and distinct copied controls. Hence the ZZ depth
    per Trotter step is O(n), even for all-to-all J.

    The circuit uses only cx, ccx, h, rz, and rx-level decompositions. In
    particular, controlled RZ and controlled RX are decomposed into CNOTs and
    rotations, and controlled RZZ is decomposed into two CCX gates plus a
    controlled RZ on the parity qubit.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        gamma: np.ndarray,
        time: float = 1.0,
        eps: float | None = None,
        num_trotter_steps: int | None = None,
        atol: float = 1e-8,
        label=None,
    ) -> None:
        self.base = TrotterizedOperatorIsingTF(n, h, J, gamma, time=time, eps=eps, num_trotter_steps=num_trotter_steps, atol=atol)

        self.n = self.base.n
        self.h = self.base.h
        self.J = self.base.J
        self.gamma = self.base.gamma
        self.time = self.base.time
        self.eps = self.base.eps
        self.atol = self.base.atol
        self.num_trotter_steps = self.base.num_trotter_steps
        self.z_terms = self.base.z_terms
        self.zz_terms = self.base.zz_terms
        self.x_terms = self.base.x_terms
        self.n_terms_z = self.base.n_terms_z
        self.n_terms_zz = self.base.n_terms_zz
        self.n_terms_x = self.base.n_terms_x
        self.n_terms = self.base.n_terms
        self.n_control_work = self._num_control_work()

        super().__init__("ControlledTrotterizedOperatorIsingTF", 1 + self.n + self.n_control_work, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        system_start = 1
        work_start = system_start + self.n

        return {
            "control": [0],
            "system": list(range(system_start, work_start)),
            "control_work": list(range(work_start, work_start + self.n_control_work)),
        }

    def _num_control_work(self) -> int:
        return max(1, self.n)

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

    def _apply_controlled_z_layer(self, qc: QuantumCircuit, controls: list[int], system: list[int], dt: float) -> None:
        width = len(controls)

        for start in range(0, len(self.z_terms), width):
            batch = self.z_terms[start:start + width]

            for c, (i, coeff) in zip(controls, batch):
                self._append_controlled_rz(qc, 2.0 * dt * coeff, c, system[i])

    def _apply_controlled_zz_layers(self, qc: QuantumCircuit, controls: list[int], system: list[int], dt: float) -> None:
        coeff_by_edge = {(i, j): coeff for i, j, coeff in self.zz_terms}
        width = len(controls)

        for matching in TrotterizedOperatorIsingTF._matchings(self.n):
            active_edges = [(i, j, coeff_by_edge[(i, j)]) for i, j in matching if (i, j) in coeff_by_edge]

            for start in range(0, len(active_edges), width):
                batch = active_edges[start:start + width]

                for c, (i, j, coeff) in zip(controls, batch):
                    self._append_controlled_rzz(qc, 2.0 * dt * coeff, c, system[i], system[j])

    def _apply_controlled_x_layer(self, qc: QuantumCircuit, controls: list[int], system: list[int], dt: float) -> None:
        width = len(controls)

        for start in range(0, len(self.x_terms), width):
            batch = self.x_terms[start:start + width]

            for c, (i, coeff) in zip(controls, batch):
                self._append_controlled_rx(qc, 2.0 * dt * coeff, c, system[i])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        control = self.layout["control"][0]
        system = self.layout["system"]
        control_work = self.layout["control_work"]
        dt = self.time / self.num_trotter_steps

        self._fanout_control(qc, control, control_work)

        for _ in range(self.num_trotter_steps):
            if self.gamma > 0:
                self._apply_controlled_x_layer(qc, control_work, system, 0.5 * dt)
            self._apply_controlled_z_layer(qc, control_work, system, dt)
            self._apply_controlled_zz_layers(qc, control_work, system, dt)
            if self.gamma > 0:
                self._apply_controlled_x_layer(qc, control_work, system, 0.5 * dt)

        self._unfanout_control(qc, control, control_work)
        return qc
