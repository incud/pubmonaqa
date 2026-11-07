from qiskit.circuit import Gate, QuantumCircuit


class ProposalUniform(Gate):
    r"""Map |x>_A|0>_B to sum_y |x>_A|y>_B / sqrt(2**n).

    Uses 2*n qubits and constant depth.
    Registers: A = q[:n], B = q[n:].
    """

    def __init__(self, n: int, label=None) -> None:
        self.n = int(n)
        if n < 1:
            raise ValueError("n must be a positive integer.")

        super().__init__(name=f"ProposalUniform(n={self.n})", num_qubits=2 * self.n, params=[], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        for q in self.layout["B"]:
            qc.h(q)
        return qc

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"A": list(range(self.n)), "B": list(range(self.n, 2 * self.n))}