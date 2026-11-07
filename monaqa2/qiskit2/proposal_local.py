from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.givens_rotation import GivensRotation


class ProposalLocal(Gate):
    r"""Map |x>_A|0>_B to sum_i |x>_A|x xor 2**i>_B / sqrt(n).

    Uses 2*n qubits and O(log n) depth with all-to-all connectivity.
    Registers: A = q[:n], B = q[n:].
    """

    def __init__(self, n: int, label=None) -> None:
        self.n = int(n)
        if self.n < 1:
            raise ValueError("n must be a positive integer.")
        super().__init__(f"ProposalLocal(n={self.n})", 2 * self.n, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        def spread(lo, hi):
            if hi - lo <= 1:
                return
            mid = (lo + hi) // 2
            qc.append(GivensRotation(mid - lo, hi - mid), [lo, mid])
            # Disjoint subtrees run in parallel; weights count their leaves.
            spread(lo, mid)
            spread(mid, hi)

        qc.x(self.n)
        spread(self.n, 2 * self.n)
        for i in range(self.n):
            qc.cx(i, self.n + i)
        return qc

    @property
    def layout(self) -> dict[str, list[int]]:
        return {
            "A": list(range(self.n)),
            "B": list(range(self.n, 2 * self.n)),
        }