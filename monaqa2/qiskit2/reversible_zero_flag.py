from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx


class ReversibleZeroFlag(Gate):
    """flag ^= (bits == 0), with O(log m) depth and max(0,m-2) clean work bits."""

    def __init__(self, m: int, label=None):
        self.m = int(m)
        if self.m < 1:
            raise ValueError("m must be positive.")
        self.scratch = max(0, self.m - 2)
        super().__init__("reversible_zero_flag", self.m + 1 + self.scratch, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        bits, flag, work = (self.layout[key] for key in ("bits", "flag", "work"))
        qc.x(bits)
        nodes, edges, free = bits, [], 0
        while len(nodes) > 2:
            parents = []
            for i in range(0, len(nodes) - 1, 2):
                target = work[free]
                free += 1
                edge = [nodes[i], nodes[i + 1], target]
                qc.append(Ccx(), edge)
                edges.append(edge)
                parents.append(target)
            nodes = parents + nodes[2 * (len(nodes) // 2):]
        if self.m == 1:
            qc.cx(nodes[0], flag[0])
        else:
            qc.append(Ccx(), nodes + flag)
        for edge in reversed(edges):
            qc.append(Ccx(), edge)
        qc.x(bits)
        return qc

    @property
    def layout(self):
        return {"bits": list(range(self.m)), "flag": [self.m],
                "work": list(range(self.m + 1, self.num_qubits))}
