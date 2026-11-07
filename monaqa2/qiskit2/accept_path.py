from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx
from monaqa2.qiskit2.reversible_fanout import ReversibleFanout
from monaqa2.qiskit2.reversible_zero_flag import ReversibleZeroFlag


class AcceptPath(Gate):
    """Swap A,B iff all coins are zero; coins preserved and work returned to zero.

    A balanced zero test and flag fanout give O(log coins + log n) depth.
    """

    def __init__(self, n: int, coins: int = 1, label=None):
        self.n, self.coins = int(n), int(coins)
        if min(self.n, self.coins) < 1:
            raise ValueError("n and coins must be positive.")
        self.zero_flag = ReversibleZeroFlag(self.coins)
        self.fanout = ReversibleFanout(self.n)
        self.work = 1 + max(self.zero_flag.scratch, self.n - 1)
        super().__init__("accept_path", 2 * self.n + self.coins + self.work, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        a, b, coins, work = (self.layout[key] for key in ("A", "B", "coins", "work"))
        flag = work[0]
        zero_wires = coins + [flag] + work[1:1 + self.zero_flag.scratch]
        controls = [flag] + work[1:self.n]
        qc.append(self.zero_flag, zero_wires)
        qc.append(self.fanout, controls)
        # One Toffoli per controlled swap; distinct flag copies keep swaps parallel.
        for ctrl_1, anc_1, anc_2 in zip(controls, a, b):
            qc.cx(anc_2, anc_1)
            qc.append(Ccx(), [ctrl_1, anc_1, anc_2])
            qc.cx(anc_2, anc_1)
        qc.append(self.fanout.inverse(), controls)
        qc.append(self.zero_flag.inverse(), zero_wires)
        return qc

    @property
    def layout(self):
        n, end = self.n, 2 * self.n + self.coins
        work = list(range(end, self.num_qubits))
        return {"A": list(range(n)), "B": list(range(n, 2 * n)),
                "coins": list(range(2 * n, end)), "flag": work[:1],
                "fanout": work[1:n], "work": work}
