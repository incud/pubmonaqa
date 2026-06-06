import numpy as np
from qiskit.circuit import QuantumCircuit, Gate

from monaqa2.qiskit.multi_controlled_not_gate import MultiControlledNot


class Reflection(Gate):
    r"""
    This gate acts effectively on the system

        H_A \otimes H_B \otimes H_acc

    as the reflection unitary

        R = 2 (I_A \otimes |0^n><0^n|_B \otimes |0^m><0^m|_acc) - I.
    """

    def __init__(self, n: int, coins: int = 1, mocked_circuit: bool = False, label=None) -> None:
        if n <= 1:
            raise ValueError(f"`n` must be a positive integer >= 2 (given {n=}).")
        if coins <= 0:
            raise ValueError(f"`coins` must be a positive integer >= 1 (given {coins=}).")

        self.n = int(n)
        self.coins = int(coins)
        self.mocked_circuit = bool(mocked_circuit)

        controls = self.n + self.coins - 1

        if self.mocked_circuit:
            self.aux_size = 0
        else:
            self.aux_size = MultiControlledNot(controls).ancillas

        super().__init__(
            name="Reflection",
            num_qubits=2 * self.n + self.coins + self.aux_size,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        a = list(range(0, self.n))
        b = list(range(self.n, 2 * self.n))
        coins = list(range(2 * self.n, 2 * self.n + self.coins))
        work = list(range(2 * self.n + self.coins, self.num_qubits))

        return {
            "A": a,
            "B": b,
            "coins": coins,
            "work": work,
            "reflected_register": b + coins,
        }

    def _append_selective_phase_flip(self, qc: QuantumCircuit, controls: list[int], target: int, ancillas: list[int]) -> None:
        qc.h(target)

        if self.mocked_circuit:
            qc.mcx(controls, target)
        else:
            mcx = MultiControlledNot(len(controls))
            needed = mcx.ancillas

            if len(ancillas) < needed:
                raise ValueError(f"Not enough reflection ancillas: needed {needed}, got {len(ancillas)}.")

            qc.append(mcx, controls + [target] + ancillas[:needed])

        qc.h(target)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        reg = self.layout["reflected_register"]

        for q in reg:
            qc.x(q)

        qc.barrier()

        controls = reg[:-1]
        target = reg[-1]
        ancillas = self.layout["work"]

        self._append_selective_phase_flip(qc, controls, target, ancillas)

        qc.barrier()

        for q in reg:
            qc.x(q)

        qc.global_phase = np.pi

        return qc