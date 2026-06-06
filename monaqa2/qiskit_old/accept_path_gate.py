from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import SwapGate

from monaqa2.qiskit.multi_controlled_not_gate import MultiControlledNot


class AcceptPath(Gate):
    r"""
    This gate acts effectively on the system

        H_A \otimes H_B \otimes H_acc

    as the accept-path controlled swap

      F |x>|y>|coin> =
        |y>|x>|coin>    if coin = 0...0
        |x>|y>|coin>    otherwise.

    Non-mocked layout:

        A | B | coins | flag | fanout/work

    Mocked layout:

        A | B | coins

    In mocked mode, this is represented directly as n multi-controlled swap gates
    controlled on coins = 0...0, with no additional ancillas.
    """

    def __init__(
        self,
        n: int,
        coins: int = 3,
        mocked_circuit: bool = False,
        label=None,
    ) -> None:
        if n < 2:
            raise ValueError("`n` must be an integer >= 2.")
        if coins <= 0:
            raise ValueError("`coins` must be an integer >= 1.")

        self.n = int(n)
        self.coins = int(coins)
        self.mocked_circuit = bool(mocked_circuit)

        if self.mocked_circuit:
            self.n_mcx_ancillas = 0
            self.n_fanout = 0
            self.n_work = 0
        else:
            # For one and two controls, we implement the zero-flag toggle directly
            # using CX/CCX. This avoids constructing MultiControlledNot(1), which
            # may be invalid if that class computes a negative ancilla count.
            if self.coins <= 2:
                self.n_mcx_ancillas = 0
            else:
                self.n_mcx_ancillas = MultiControlledNot(self.coins).ancillas

            self.n_fanout = max(self.n - 1, self.n_mcx_ancillas)
            self.n_work = 1 + self.n_fanout

        super().__init__(
            "AcceptPath",
            2 * self.n + self.coins + self.n_work,
            [],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        a = list(range(0, self.n))
        b = list(range(self.n, 2 * self.n))
        coins = list(range(2 * self.n, 2 * self.n + self.coins))

        if self.mocked_circuit:
            return {
                "A": a,
                "B": b,
                "coins": coins,
                "flag": [],
                "fanout": [],
                "work": [],
            }

        flag = 2 * self.n + self.coins
        fanout = list(range(flag + 1, self.num_qubits))
        work = [flag] + fanout

        return {
            "A": a,
            "B": b,
            "coins": coins,
            "flag": [flag],
            "fanout": fanout,
            "work": work,
        }

    def _toggle_zero_flag(
        self,
        qc: QuantumCircuit,
        coins: list[int],
        flag: int,
        ancillas: list[int],
    ) -> None:
        """
        Toggle `flag` iff all coin qubits are in |0>.

        Implementation:
          1. X all coins, turning the all-zero condition into all-one.
          2. Apply a controlled-NOT onto `flag`.
          3. Undo the X gates.

        The one- and two-control cases are implemented directly with CX/CCX.
        Larger cases use MultiControlledNot.
        """
        if len(coins) == 0:
            raise ValueError("`coins` must contain at least one qubit index.")

        for q in coins:
            qc.x(q)

        if len(coins) == 1:
            qc.cx(coins[0], flag)

        elif len(coins) == 2:
            qc.ccx(coins[0], coins[1], flag)

        else:
            mcx = MultiControlledNot(len(coins))
            needed = mcx.ancillas

            if len(ancillas) < needed:
                raise ValueError(
                    f"Not enough MCX ancillas: needed {needed}, got {len(ancillas)}."
                )

            qc.append(mcx, coins + [flag] + ancillas[:needed])

        for q in reversed(coins):
            qc.x(q)

    def _build_mocked_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        a = self.layout["A"]
        b = self.layout["B"]
        coins = self.layout["coins"]

        zero_controlled_swap = SwapGate().control(
            num_ctrl_qubits=len(coins),
            ctrl_state=0,
        )

        for qa, qb in zip(a, b):
            qc.append(zero_controlled_swap, coins + [qa, qb])

        return qc

    def _build_full_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        a = self.layout["A"]
        b = self.layout["B"]
        coins = self.layout["coins"]
        flag = self.layout["flag"][0]
        fanout = self.layout["fanout"]

        mcx_ancillas = fanout[: self.n_mcx_ancillas]

        self._toggle_zero_flag(qc, coins, flag, mcx_ancillas)
        qc.barrier()

        controls = [flag]
        fanout_layers = []
        fanout_iter = iter(fanout)

        while len(controls) < self.n:
            layer = []

            for src in controls[: min(len(controls), self.n - len(controls))]:
                dst = next(fanout_iter)
                qc.cx(src, dst)
                layer.append((src, dst))
                controls.append(dst)

            fanout_layers.append(layer)

        for ctrl, qa, qb in zip(controls, a, b):
            qc.ccx(ctrl, qb, qa)
            qc.ccx(ctrl, qa, qb)
            qc.ccx(ctrl, qb, qa)

        for layer in reversed(fanout_layers):
            for src, dst in reversed(layer):
                qc.cx(src, dst)

        qc.barrier()
        self._toggle_zero_flag(qc, coins, flag, mcx_ancillas)

        return qc

    def _build_definition(self) -> QuantumCircuit:
        if self.mocked_circuit:
            return self._build_mocked_definition()

        return self._build_full_definition()