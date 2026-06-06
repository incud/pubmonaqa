from monaqa2.qiskit.prepare_unary_gate import PrepareUnary
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import StatePreparation

from monaqa2.qiskit.lcu_unary_ising_tf_gate import LcuUnaryIsingTF
from monaqa2.qiskit.multi_controlled_not_gate import MultiControlledNot


class QubitizedOperatorIsingTF(Gate):
    """
    Qubitized walk operator for the transverse-field Ising LCU.

    Default:
        unary LCU with PrepareUnary and optional MultiControlledNot reflection work.

    mocked_prepare=True:
        compact binary LCU register with Qiskit StatePreparation.

    mocked_all=True:
        compact binary LCU register, StatePreparation, binary SELECT, and
        MCX reflection without work ancillas. Mutually exclusive with
        mocked_prepare and mocked_reflection.
    """

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, mocked_reflection: bool = False, mocked_prepare: bool = False, mocked_all: bool = False, label=None) -> None:
        if mocked_all and (mocked_reflection or mocked_prepare):
            raise ValueError("mocked_all is mutually exclusive with mocked_reflection and mocked_prepare.")

        self.lcu = LcuUnaryIsingTF(n, h, J, gamma)
        self.mocked_reflection = bool(mocked_reflection)
        self.mocked_prepare = bool(mocked_prepare)
        self.mocked_all = bool(mocked_all)

        self.n_binary_prepare = self._num_binary_prepare()
        self.n_reflection_ancillas = self.n_binary_prepare if self._uses_compact_lcu() else len(self.lcu.layout["tree"]) + len(self.lcu.layout["prepare"])
        self.n_reflection_controls = max(self.n_reflection_ancillas - 1, 0)
        self.n_reflection_work = self._num_reflection_work()

        num_qubits = self.lcu.n + self.n_binary_prepare + self.n_reflection_work if self._uses_compact_lcu() else self.lcu.num_qubits + self.n_reflection_work

        super().__init__("QubitizedOperatorIsingTF", num_qubits, [], label=label)
        self.definition = self._build_definition()

    def _uses_compact_lcu(self) -> bool:
        return self.mocked_prepare or self.mocked_all

    def _num_binary_prepare(self) -> int:
        if not self._uses_compact_lcu() or self.lcu.n_terms <= 1:
            return 0

        return int(np.ceil(np.log2(self.lcu.n_terms)))

    def _num_reflection_work(self) -> int:
        if self.mocked_reflection or self.mocked_all:
            return 0

        if self.n_reflection_ancillas <= 2:
            return 0

        return MultiControlledNot(self.n_reflection_controls).ancillas

    @property
    def layout(self) -> dict[str, list[int]]:
        if self._uses_compact_lcu():
            system = list(range(self.lcu.n))
            prepare = list(range(self.lcu.n, self.lcu.n + self.n_binary_prepare))
            work_start = self.lcu.n + self.n_binary_prepare
            return {"system": system, "tree": [], "prepare": prepare, "reflection_work": list(range(work_start, self.num_qubits))}

        base = self.lcu.layout
        work_start = self.lcu.num_qubits
        return {"system": base["system"], "tree": base["tree"], "prepare": base["prepare"], "reflection_work": list(range(work_start, work_start + self.n_reflection_work))}

    @staticmethod
    def _mcz(qc: QuantumCircuit, controls: list[int]) -> None:
        if len(controls) == 0:
            qc.global_phase += np.pi
            return

        if len(controls) == 1:
            qc.z(controls[0])
            return

        target = controls[-1]
        qc.h(target)
        qc.mcx(controls[:-1], target)
        qc.h(target)

    @staticmethod
    def _mcx(qc: QuantumCircuit, controls: list[int], target: int) -> None:
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        else:
            qc.mcx(controls, target)

    @staticmethod
    def _reflect_zero(qc: QuantumCircuit, qubits: list[int], work: list[int], mocked_reflection: bool) -> None:
        """
        Apply 2|0...0><0...0| - I on `qubits`.
        """
        if len(qubits) == 0:
            return

        if len(qubits) == 1:
            qc.z(qubits[0])
            return

        qc.global_phase += np.pi

        for q in qubits:
            qc.x(q)

        if len(qubits) == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            controls = qubits[:-1]
            target = qubits[-1]

            qc.h(target)

            if mocked_reflection:
                qc.mcx(controls, target)
            else:
                mcn = MultiControlledNot(len(controls))

                if len(work) < mcn.ancillas:
                    raise ValueError(f"Not enough reflection work qubits: needed {mcn.ancillas}, got {len(work)}.")

                qc.append(mcn, controls + [target] + work[:mcn.ancillas])

            qc.h(target)

        for q in reversed(qubits):
            qc.x(q)

    def _compact_state_preparation(self) -> StatePreparation | None:
        if self.n_binary_prepare == 0:
            return None

        alpha = float(np.sum(self.lcu.magnitudes))
        amplitudes = np.zeros(2**self.n_binary_prepare, dtype=complex)
        amplitudes[:self.lcu.n_terms] = np.sqrt(self.lcu.magnitudes / alpha)
        return StatePreparation(amplitudes)

    def _apply_compact_prepare(self, qc: QuantumCircuit, prepare: list[int], inverse: bool = False) -> None:
        prep = self._compact_state_preparation()

        if prep is None:
            return

        qc.append(prep.inverse() if inverse else prep, prepare)

    def _apply_with_binary_label(self, qc: QuantumCircuit, label: int, prepare: list[int], body) -> None:
        flipped = [q for k, q in enumerate(prepare) if ((label >> k) & 1) == 0]

        for q in flipped:
            qc.x(q)

        body()

        for q in reversed(flipped):
            qc.x(q)

    def _apply_compact_select(self, qc: QuantumCircuit, system: list[int], prepare: list[int]) -> None:
        for idx, (coeff, pauli) in enumerate(self.lcu.terms):
            def body(coeff=coeff, pauli=pauli):
                if coeff < 0.0:
                    self._mcz(qc, prepare)

                if pauli[0] == "Z":
                    target = system[pauli[1]]
                    qc.h(target)
                    self._mcx(qc, prepare, target)
                    qc.h(target)

                elif pauli[0] == "X":
                    self._mcx(qc, prepare, system[pauli[1]])

                elif pauli[0] == "ZZ":
                    _, i, j = pauli

                    qc.h(system[i])
                    self._mcx(qc, prepare, system[i])
                    qc.h(system[i])

                    qc.h(system[j])
                    self._mcx(qc, prepare, system[j])
                    qc.h(system[j])

                else:
                    raise ValueError(f"Unsupported Pauli term: {pauli}")

            self._apply_with_binary_label(qc, idx, prepare, body)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        system = self.layout["system"]
        prepare = self.layout["prepare"]
        work = self.layout["reflection_work"]

        if self._uses_compact_lcu():
            self._apply_compact_prepare(qc, prepare)
            self._apply_compact_select(qc, system, prepare)
            self._apply_compact_prepare(qc, prepare, inverse=True)
            self._reflect_zero(qc, prepare, work, self.mocked_reflection or self.mocked_all)
            return qc

        lcu_qubits = list(range(self.lcu.num_qubits))
        ancillas = self.layout["tree"] + self.layout["prepare"]

        qc.append(self.lcu, lcu_qubits)
        self._reflect_zero(qc, ancillas, work, self.mocked_reflection)

        return qc
    


class ControlledQubitizedOperatorIsingTF(Gate):
    """
    Explicit controlled qubitized walk operator.

    Acts as

        |0><0| ⊗ I + |1><1| ⊗ W.

    mocked_all=True uses a compact binary LCU:
        * ceil(log2(n_terms)) prepare qubits,
        * Qiskit StatePreparation,
        * multi-controlled SELECT,
        * MCX-based reflection without work ancillas.

    mocked_all is mutually exclusive with mocked_prepare and mocked_reflection.
    """

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, mocked_reflection: bool = False, mocked_prepare: bool = False, mocked_all: bool = False, label=None) -> None:
        if mocked_all and (mocked_reflection or mocked_prepare):
            raise ValueError("mocked_all is mutually exclusive with mocked_reflection and mocked_prepare.")

        self.lcu = LcuUnaryIsingTF(n, h, J, gamma)
        self.prepare_gate = PrepareUnary(self.lcu.magnitudes)
        self.mocked_reflection = bool(mocked_reflection)
        self.mocked_prepare = bool(mocked_prepare)
        self.mocked_all = bool(mocked_all)

        self.n_binary_prepare = self._num_binary_prepare()
        self.n_reflection_ancillas = len(self.lcu.layout["tree"]) + len(self.lcu.layout["prepare"])
        self.n_prepare_control_work = 0 if self.mocked_all else self._num_prepare_control_work()
        self.n_reflection_work = 0 if self.mocked_all else self._num_reflection_work()

        num_qubits = (1 + self.lcu.n + self.n_binary_prepare) if self.mocked_all else \
                     (1 + self.lcu.num_qubits + self.n_prepare_control_work + self.n_reflection_work)

        super().__init__("ControlledQubitizedOperatorIsingTF", num_qubits, [], label=label)
        self.definition = self._build_definition()

    def _num_binary_prepare(self) -> int:
        if self.lcu.n_terms <= 1:
            return 0

        return int(np.ceil(np.log2(self.lcu.n_terms)))

    def _num_prepare_control_work(self) -> int:
        if self.mocked_prepare:
            return 0

        if self.prepare_gate.n_terms <= 1:
            prepare_width = 0
        elif self.prepare_gate.n_terms == 2:
            prepare_width = 1
        else:
            prepare_width = max(len(nodes) for nodes in self.prepare_gate._levels())

        return max(self.lcu.n, prepare_width)

    def _num_reflection_work(self) -> int:
        if self.mocked_reflection:
            return 0

        if self.n_reflection_ancillas <= 1:
            return 0

        return MultiControlledNot(self.n_reflection_ancillas).ancillas

    @property
    def layout(self) -> dict[str, list[int]]:
        if self.mocked_all:
            system_start = 1
            prepare_start = system_start + self.lcu.n

            return {
                "control": [0],
                "system": list(range(system_start, prepare_start)),
                "tree": [],
                "prepare": list(range(prepare_start, prepare_start + self.n_binary_prepare)),
                "prepare_control_work": [],
                "reflection_work": [],
            }

        base = self.lcu.layout
        prepare_work_start = 1 + self.lcu.num_qubits
        reflection_work_start = prepare_work_start + self.n_prepare_control_work

        return {
            "control": [0],
            "system": [q + 1 for q in base["system"]],
            "tree": [q + 1 for q in base["tree"]],
            "prepare": [q + 1 for q in base["prepare"]],
            "prepare_control_work": list(range(prepare_work_start, reflection_work_start)),
            "reflection_work": list(range(reflection_work_start, self.num_qubits)),
        }

    @staticmethod
    def _ry_via_rz(qc: QuantumCircuit, theta: float, target: int) -> None:
        qc.sdg(target)
        qc.h(target)
        qc.rz(theta, target)
        qc.h(target)
        qc.s(target)

    @classmethod
    def _ccry(cls, qc: QuantumCircuit, theta: float, c0: int, c1: int, target: int) -> None:
        cls._ry_via_rz(qc, theta / 2.0, target)
        qc.ccx(c0, c1, target)
        cls._ry_via_rz(qc, -theta / 2.0, target)
        qc.ccx(c0, c1, target)

    @staticmethod
    def _ccz(qc: QuantumCircuit, c0: int, c1: int, target: int) -> None:
        qc.h(target)
        qc.ccx(c0, c1, target)
        qc.h(target)

    @staticmethod
    def _mcz(qc: QuantumCircuit, controls: list[int]) -> None:
        if len(controls) == 0:
            qc.global_phase += np.pi
            return

        if len(controls) == 1:
            qc.z(controls[0])
            return

        target = controls[-1]
        qc.h(target)
        qc.mcx(controls[:-1], target)
        qc.h(target)

    @staticmethod
    def _mcx(qc: QuantumCircuit, controls: list[int], target: int) -> None:
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        else:
            qc.mcx(controls, target)

    @staticmethod
    def _fanout_control(qc: QuantumCircuit, control: int, work: list[int]) -> None:
        for q in work:
            qc.cx(control, q)

    @staticmethod
    def _unfanout_control(qc: QuantumCircuit, control: int, work: list[int]) -> None:
        for q in reversed(work):
            qc.cx(control, q)

    def _compact_state_preparation(self) -> StatePreparation | None:
        m = self.n_binary_prepare

        if m == 0:
            return None

        alpha = float(np.sum(self.lcu.magnitudes))
        amplitudes = np.zeros(2**m, dtype=complex)
        amplitudes[: self.lcu.n_terms] = np.sqrt(self.lcu.magnitudes / alpha)

        return StatePreparation(amplitudes)

    def _apply_compact_controlled_prepare(self, qc: QuantumCircuit, control: int, prepare: list[int], inverse: bool = False) -> None:
        prep = self._compact_state_preparation()

        if prep is None:
            return

        gate = prep.inverse() if inverse else prep
        qc.append(gate.control(1), [control] + prepare)

    def _apply_with_binary_label(self, qc: QuantumCircuit, label: int, prepare: list[int], body) -> None:
        flipped = [q for k, q in enumerate(prepare) if ((label >> k) & 1) == 0]

        for q in flipped:
            qc.x(q)

        body()

        for q in reversed(flipped):
            qc.x(q)

    def _apply_compact_controlled_select(self, qc: QuantumCircuit, control: int, system: list[int], prepare: list[int]) -> None:
        for idx, (coeff, pauli) in enumerate(self.lcu.terms):
            controls = [control] + prepare

            def body(coeff=coeff, pauli=pauli, controls=controls):
                if coeff < 0.0:
                    self._mcz(qc, controls)

                if pauli[0] == "Z":
                    target = system[pauli[1]]
                    qc.h(target)
                    self._mcx(qc, controls, target)
                    qc.h(target)

                elif pauli[0] == "X":
                    self._mcx(qc, controls, system[pauli[1]])

                elif pauli[0] == "ZZ":
                    _, i, j = pauli

                    qc.h(system[i])
                    self._mcx(qc, controls, system[i])
                    qc.h(system[i])

                    qc.h(system[j])
                    self._mcx(qc, controls, system[j])
                    qc.h(system[j])

                else:
                    raise ValueError(f"Unsupported Pauli term: {pauli}")

            self._apply_with_binary_label(qc, idx, prepare, body)

    def _apply_compact_controlled_reflect_zero(self, qc: QuantumCircuit, control: int, prepare: list[int]) -> None:
        self._apply_controlled_reflect_zero(qc, control, prepare, [], True)

    def _controlled_prepare_ops(self, control: int, tree: list[int], prepare: list[int]) -> list[tuple]:
        local = tree + prepare

        if self.prepare_gate.n_terms == 1:
            return [("cx", local[self.prepare_gate.layout["leaf"][0]])]

        ops = [("cx", local[self.prepare_gate.root["q"]])]

        for nodes in self.prepare_gate._levels():
            for node in nodes:
                p = local[node["q"]]
                l = local[node["left"]["q"]]
                r = local[node["right"]["q"]]
                theta = self.prepare_gate._theta(node["left"]["w"], node["right"]["w"])
                ops.extend([("ccry", theta, p, r), ("ccx", p, l), ("ccx", r, l), ("ccx", l, p), ("ccx", r, p)])

        return ops

    def _apply_controlled_prepare_serial(self, qc: QuantumCircuit, control: int, tree: list[int], prepare: list[int], inverse: bool = False) -> None:
        ops = self._controlled_prepare_ops(control, tree, prepare)

        if inverse:
            ops = list(reversed(ops))

        for op in ops:
            if op[0] == "cx":
                qc.cx(control, op[1])
            elif op[0] == "ccx":
                qc.ccx(control, op[1], op[2])
            elif op[0] == "ccry":
                theta = -op[1] if inverse else op[1]
                self._ccry(qc, theta, control, op[2], op[3])
            else:
                raise ValueError(f"Unsupported controlled-prepare op: {op}")

    def _apply_prepare_level(self, qc: QuantumCircuit, nodes: list[dict], controls: list[int], local: list[int], inverse: bool = False) -> None:
        if not inverse:
            for c, node in zip(controls, nodes):
                p = local[node["q"]]
                r = local[node["right"]["q"]]
                theta = self.prepare_gate._theta(node["left"]["w"], node["right"]["w"])
                self._ccry(qc, theta, c, p, r)

            for c, node in zip(controls, nodes):
                qc.ccx(c, local[node["q"]], local[node["left"]["q"]])

            for c, node in zip(controls, nodes):
                qc.ccx(c, local[node["right"]["q"]], local[node["left"]["q"]])

            for c, node in zip(controls, nodes):
                qc.ccx(c, local[node["left"]["q"]], local[node["q"]])

            for c, node in zip(controls, nodes):
                qc.ccx(c, local[node["right"]["q"]], local[node["q"]])

            return

        for c, node in zip(controls, nodes):
            qc.ccx(c, local[node["right"]["q"]], local[node["q"]])

        for c, node in zip(controls, nodes):
            qc.ccx(c, local[node["left"]["q"]], local[node["q"]])

        for c, node in zip(controls, nodes):
            qc.ccx(c, local[node["right"]["q"]], local[node["left"]["q"]])

        for c, node in zip(controls, nodes):
            qc.ccx(c, local[node["q"]], local[node["left"]["q"]])

        for c, node in zip(controls, nodes):
            p = local[node["q"]]
            r = local[node["right"]["q"]]
            theta = self.prepare_gate._theta(node["left"]["w"], node["right"]["w"])
            self._ccry(qc, -theta, c, p, r)

    def _apply_controlled_prepare_fanout(self, qc: QuantumCircuit, control: int, tree: list[int], prepare: list[int], control_work: list[int], inverse: bool = False) -> None:
        local = tree + prepare

        if self.prepare_gate.n_terms == 1:
            qc.cx(control, local[self.prepare_gate.layout["leaf"][0]])
            return

        levels = self.prepare_gate._levels()
        levels = list(reversed(levels)) if inverse else levels

        if not inverse:
            qc.cx(control, local[self.prepare_gate.root["q"]])

        for nodes in levels:
            controls = control_work[:len(nodes)] if control_work else [control] * len(nodes)
            self._apply_prepare_level(qc, nodes, controls, local, inverse=inverse)

        if inverse:
            qc.cx(control, local[self.prepare_gate.root["q"]])

    def _apply_controlled_select_serial(self, qc: QuantumCircuit, control: int, system: list[int], prepare: list[int]) -> None:
        for idx, (coeff, pauli) in enumerate(self.lcu.terms):
            q = prepare[idx]

            if coeff < 0.0:
                qc.cz(control, q)

            if pauli[0] == "Z":
                self._ccz(qc, control, q, system[pauli[1]])
            elif pauli[0] == "X":
                qc.ccx(control, q, system[pauli[1]])
            elif pauli[0] == "ZZ":
                _, i, j = pauli
                self._ccz(qc, control, q, system[i])
                self._ccz(qc, control, q, system[j])
            else:
                raise ValueError(f"Unsupported Pauli term: {pauli}")

    def _apply_controlled_select_scheduled(self, qc: QuantumCircuit, control_work: list[int], system: list[int], prepare: list[int]) -> None:
        if len(control_work) == 0:
            raise ValueError("Scheduled controlled SELECT requires fanned-out control work qubits.")

        term_index = {pauli: idx for idx, (_, pauli) in enumerate(self.lcu.terms)}
        width = len(control_work)

        negative_terms = [idx for idx, (coeff, _) in enumerate(self.lcu.terms) if coeff < 0.0]
        for start in range(0, len(negative_terms), width):
            for c, idx in zip(control_work, negative_terms[start:start + width]):
                qc.cz(c, prepare[idx])

        z_terms = [("Z", i) for i in range(self.lcu.n) if ("Z", i) in term_index]
        for start in range(0, len(z_terms), width):
            for c, pauli in zip(control_work, z_terms[start:start + width]):
                self._ccz(qc, c, prepare[term_index[pauli]], system[pauli[1]])

        x_terms = [("X", i) for i in range(self.lcu.n) if ("X", i) in term_index]
        for start in range(0, len(x_terms), width):
            for c, pauli in zip(control_work, x_terms[start:start + width]):
                qc.ccx(c, prepare[term_index[pauli]], system[pauli[1]])

        for matching in self.lcu._matchings(self.lcu.n):
            active_edges = [(i, j) for i, j in matching if ("ZZ", i, j) in term_index]

            for start in range(0, len(active_edges), width):
                batch = active_edges[start:start + width]

                for c, (i, j) in zip(control_work, batch):
                    self._ccz(qc, c, prepare[term_index[("ZZ", i, j)]], system[i])

                for c, (i, j) in zip(control_work, batch):
                    self._ccz(qc, c, prepare[term_index[("ZZ", i, j)]], system[j])

    @staticmethod
    def _apply_controlled_reflect_zero(qc: QuantumCircuit, control: int, qubits: list[int], work: list[int], mocked_reflection: bool) -> None:
        if len(qubits) == 0:
            return

        if len(qubits) == 1:
            qc.cz(control, qubits[0])
            return

        qc.z(control)

        for q in qubits:
            qc.x(q)

        controls = [control] + qubits[:-1]
        target = qubits[-1]

        qc.h(target)

        if mocked_reflection:
            qc.mcx(controls, target)
        else:
            mcn = MultiControlledNot(len(controls))

            if len(work) < mcn.ancillas:
                raise ValueError(f"Not enough reflection work qubits: needed {mcn.ancillas}, got {len(work)}.")

            qc.append(mcn, controls + [target] + work[:mcn.ancillas])

        qc.h(target)

        for q in reversed(qubits):
            qc.x(q)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        control = self.layout["control"][0]
        system = self.layout["system"]
        tree = self.layout["tree"]
        prepare = self.layout["prepare"]

        if self.mocked_all:
            self._apply_compact_controlled_prepare(qc, control, prepare)
            self._apply_compact_controlled_select(qc, control, system, prepare)
            self._apply_compact_controlled_prepare(qc, control, prepare, inverse=True)
            self._apply_compact_controlled_reflect_zero(qc, control, prepare)
            return qc

        prepare_control_work = self.layout["prepare_control_work"]
        reflection_work = self.layout["reflection_work"]

        if self.mocked_prepare:
            self._apply_controlled_prepare_serial(qc, control, tree, prepare)
            self._apply_controlled_select_serial(qc, control, system, prepare)
            self._apply_controlled_prepare_serial(qc, control, tree, prepare, inverse=True)
        else:
            self._fanout_control(qc, control, prepare_control_work)
            self._apply_controlled_prepare_fanout(qc, control, tree, prepare, prepare_control_work)
            self._apply_controlled_select_scheduled(qc, prepare_control_work, system, prepare)
            self._apply_controlled_prepare_fanout(qc, control, tree, prepare, prepare_control_work, inverse=True)
            self._unfanout_control(qc, control, prepare_control_work)

        self._apply_controlled_reflect_zero(qc, control, tree + prepare, reflection_work, self.mocked_reflection)
        return qc