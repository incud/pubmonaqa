from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.reversible_zero_flag import ReversibleZeroFlag


class Reflection(Gate):
    """R = 2*(I_A tensor |0><0|_(B,coins,extra)) - I, with O(log(n+coins+extra)) depth.

    Extra registers include coherent proposal labels. Clean work is uncomputed.
    """

    def __init__(self, n: int, coins: int = 1, extra: int = 0, label=None):
        self.n, self.coins, self.extra = int(n), int(coins), int(extra)
        if min(self.n, self.coins) < 1 or self.extra < 0:
            raise ValueError("Require n,coins > 0 and extra >= 0.")
        self.zero_flag = ReversibleZeroFlag(self.n + self.coins + self.extra)
        self.work = 1 + self.zero_flag.scratch
        super().__init__("reflection", 2 * self.n + self.coins + self.extra + self.work, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        work = self.layout["work"]
        wires = self.layout["reflected_register"] + work
        qc.append(self.zero_flag, wires)
        # X Z X = -Z: phase +1 on flag=1, -1 on flag=0, with the exact global sign.
        qc.x(work[0])
        qc.s(work[0])
        qc.s(work[0])
        qc.x(work[0])
        qc.append(self.zero_flag.inverse(), wires)
        return qc

    @property
    def layout(self):
        n, end = self.n, 2 * self.n + self.coins
        b, coins = list(range(n, 2 * n)), list(range(2 * n, end))
        extra = list(range(end, end + self.extra))
        return {"A": list(range(n)), "B": b, "coins": coins, "extra": extra,
                "work": list(range(end + self.extra, self.num_qubits)),
                "reflected_register": b + coins + extra}
