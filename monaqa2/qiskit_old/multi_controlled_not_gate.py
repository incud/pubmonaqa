import numpy as np
from qiskit.circuit import QuantumCircuit, Gate


class MultiControlledNot(Gate):
    r"""
    This gate acts on the system

        controls, target, ancillas

    as the multi-controlled not

        MCX |b_1, ..., b_n>|psi> =
            |b_1, ..., b_n>(X |psi>) for b_1 = ... = b_n = 1
            |b_1, ..., b_n>(I |psi>) otherwise

    The definition is built with a balanced AND tree and ancillas for
    logarithmic depth.
    """

    def __init__(self, controls: int, label=None) -> None:
        if controls < 1:
            raise ValueError(f"`controls` must be an integer >= 2 (given {controls=}).")
        
        self.controls = int(controls)
        self.ancillas = max(0, int(controls - 2))
        num_qubits = self.controls + 1 + self.ancillas

        super().__init__(
            name="MultiControlledNot", 
            num_qubits=num_qubits, 
            params=[], 
            label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        controls = list(range(0, self.controls))
        target = self.controls
        ancillas = list(range(self.controls + 1, self.num_qubits))
        return {
            "controls": controls,
            "target": [target],
            "ancillas": ancillas,
        }

    def _and_tree_gates(
        self,
        leaves: list[int],
        work: list[int],
    ) -> tuple[int, list[tuple[int, int, int]], list[int]]:
        if len(leaves) == 1:
            return leaves[0], [], work

        mid = len(leaves) // 2
        left_root, left_gates, work = self._and_tree_gates(leaves[:mid], work)
        right_root, right_gates, work = self._and_tree_gates(leaves[mid:], work)

        out, work = work[0], work[1:]
        return out, left_gates + right_gates + [(left_root, right_root, out)], work

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        controls = self.layout["controls"]
        target = self.layout["target"][0]
        ancillas = self.layout["ancillas"]

        if self.controls == 1:
            qc.cx(controls[0], target)
        elif self.controls == 2:
            qc.ccx(controls[0], controls[1], target)
        else:
            root, gates, _ = self._and_tree_gates(controls[:-1], ancillas)
            for a, b, out in gates:
                qc.ccx(a, b, out)
            qc.ccx(root, controls[-1], target)
            for a, b, out in reversed(gates):
                qc.ccx(a, b, out)

        return qc