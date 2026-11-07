from qiskit.circuit import Gate, QuantumCircuit


class Ccx(Gate):
    """Exact Toffoli: seven T/Tdg gates, T-depth three, no work qubits."""

    def __init__(self, label=None) -> None:
        super().__init__(name="ccx", num_qubits=3, params=[], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name=self.name)
        ctrl_1, ctrl_2, target = 0, 1, 2
        qc.h(target)
        # T layer 1: phases of x, y, z.
        qc.t(ctrl_1)
        qc.t(ctrl_2)
        qc.t(target)
        qc.cx(ctrl_2, ctrl_1)
        qc.cx(ctrl_1, target)
        qc.cx(target, ctrl_2)
        # T layer 2: -(x xor y), -(x xor z), +(x xor y xor z).
        qc.tdg(ctrl_1)
        qc.tdg(ctrl_2)
        qc.t(target)
        qc.cx(target, ctrl_2)
        qc.cx(ctrl_2, ctrl_1)
        qc.cx(ctrl_1, target)
        # T layer 3: -(y xor z).
        qc.tdg(target)
        qc.cx(ctrl_2, target)
        qc.h(target)
        return qc

    @property
    def layout(self) -> dict[str, int]:
        return {"ctrl_1": 0, "ctrl_2": 1, "target": 2}

