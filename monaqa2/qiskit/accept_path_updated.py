from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.synthesis.multi_controlled.mcx_synthesis import synth_mcx_2_clean_kg24

from monaqa2.qiskit.primitives import Ccx


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
            self.n_mcx_ancillas = 0 if self.coins <= 2 else 2
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
        return {
            "A": a,
            "B": b,
            "coins": coins,
            "flag": [flag],
            "fanout": fanout,
            "work": [flag] + fanout,
        }

    @staticmethod
    def _mcx(
        qc: QuantumCircuit,
        controls: list[int],
        target: int,
        clean: list[int],
    ) -> None:
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        elif len(controls) == 2:
            qc.append(Ccx(), [controls[0], controls[1], target])
        else:
            if len(clean) < 2:
                raise ValueError(
                    "Not enough clean ancillas for synth_mcx_2_clean_kg24: "
                    f"needed 2, got {len(clean)}."
                )
            qc.append(
                synth_mcx_2_clean_kg24(len(controls)),
                controls + [target] + clean[:2],
            )

    def _toggle_zero_flag(
        self,
        qc: QuantumCircuit,
        coins: list[int],
        flag: int,
        clean: list[int],
    ) -> None:
        """Toggle `flag` iff all coin qubits are in |0>."""
        if len(coins) == 0:
            raise ValueError("`coins` must contain at least one qubit index.")

        for q in coins:
            qc.x(q)

        self._mcx(qc, coins, flag, clean)

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
        mcx_clean = fanout[:self.n_mcx_ancillas]

        self._toggle_zero_flag(qc, coins, flag, mcx_clean)
        qc.barrier()

        controls = [flag]
        fanout_layers: list[list[tuple[int, int]]] = []
        fanout_iter = iter(fanout)

        while len(controls) < self.n:
            layer: list[tuple[int, int]] = []
            sources = controls[:min(len(controls), self.n - len(controls))]
            for src in sources:
                dst = next(fanout_iter)
                qc.cx(src, dst)
                layer.append((src, dst))
                controls.append(dst)
            fanout_layers.append(layer)

        for ctrl, qa, qb in zip(controls, a, b):
            qc.append(Ccx(), [ctrl, qb, qa])
            qc.append(Ccx(), [ctrl, qa, qb])
            qc.append(Ccx(), [ctrl, qb, qa])

        for layer in reversed(fanout_layers):
            for src, dst in reversed(layer):
                qc.cx(src, dst)

        qc.barrier()
        self._toggle_zero_flag(qc, coins, flag, mcx_clean)
        return qc

    def _build_definition(self) -> QuantumCircuit:
        if self.mocked_circuit:
            return self._build_mocked_definition()
        return self._build_full_definition()
