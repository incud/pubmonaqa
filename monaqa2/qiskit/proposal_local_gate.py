from qiskit.circuit import QuantumCircuit, Gate
from monaqa2.qiskit.dicke_preparation_gate import DickePreparation


class ProposalLocal(Gate):
    r"""
    This gate acts effectively on the system

        H_A \\otimes H_B 

    as 
    
        P_local |x> |0> = |x> |x \oplus dicke(n, k)>)
    """

    def __init__(self, n: int, k: int, label=None) -> None:
        if n < 1:
            raise ValueError("`n` must be a positive integer.")
        if not (1 <= k <= n):
            raise ValueError("`k` must satisfy 1 <= k <= n.")

        self.n = int(n)
        self.k = int(k)
        self.name = f"ProposalLocal(n={self.n},k={self.k})"
        super().__init__(
            name=self.name,
            num_qubits=2 * self.n,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {
            "A": list(range(0, self.n)),
            "B": list(range(self.n, 2 * self.n)),
        }

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(DickePreparation(self.n, self.k), self.layout["B"])
        for a, b in zip(self.layout["A"], self.layout["B"]):
            qc.cx(a, b)
        return qc
