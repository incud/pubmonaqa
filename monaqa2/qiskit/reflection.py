import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.synthesis.multi_controlled.mcx_synthesis import synth_mcx_2_clean_kg24


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
        self.aux_size = 0 if self.mocked_circuit or self.n + self.coins - 1 <= 2 else 2

        super().__init__(name="Reflection", num_qubits=2 * self.n + self.coins + self.aux_size, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        a = list(range(0, self.n))
        b = list(range(self.n, 2 * self.n))
        coins = list(range(2 * self.n, 2 * self.n + self.coins))
        work = list(range(2 * self.n + self.coins, self.num_qubits))
        return {"A": a, "B": b, "coins": coins, "work": work, "reflected_register": b + coins}

    @staticmethod
    def _mcx(qc: QuantumCircuit, controls: list[int], target: int, clean: list[int]) -> None:
        if len(controls) == 0:
            qc.x(target)
        elif len(controls) == 1:
            qc.cx(controls[0], target)
        elif len(controls) == 2:
            qc.ccx(controls[0], controls[1], target)
        else:
            if len(clean) < 2:
                raise ValueError(f"Not enough clean ancillas for synth_mcx_2_clean_kg24: needed 2, got {len(clean)}.")
            qc.append(synth_mcx_2_clean_kg24(len(controls)), controls + [target] + clean[:2])

    def _append_selective_phase_flip(self, qc: QuantumCircuit, controls: list[int], target: int, clean: list[int]) -> None:
        qc.h(target)
        if self.mocked_circuit:
            qc.mcx(controls, target)
        else:
            self._mcx(qc, controls, target, clean)
        qc.h(target)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        reg = self.layout["reflected_register"]

        for q in reg:
            qc.x(q)

        qc.barrier()
        self._append_selective_phase_flip(qc, reg[:-1], reg[-1], self.layout["work"])
        qc.barrier()

        for q in reg:
            qc.x(q)

        qc.global_phase = np.pi
        return qc
