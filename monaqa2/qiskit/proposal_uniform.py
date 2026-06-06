from qiskit.circuit import QuantumCircuit, Gate


class ProposalUniform(Gate):
    r"""
    This gate acts effectively on the system

        H_A \\otimes H_B 

    as 
    
        P_unif |x>|0> = |x>(H^{\\otimes n} |0>)
    """

    def __init__(self, n: int, label=None) -> None:
        if n < 1:
            raise ValueError("`n` must be a positive integer.")

        self.n = int(n)
        super().__init__(
            name="ProposalUniform",
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
        for q in self.layout["B"]:
            qc.h(q)
        return qc