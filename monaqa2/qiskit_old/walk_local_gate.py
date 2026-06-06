import numpy as np
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.proposal_local_gate import ProposalLocal
from monaqa2.qiskit.accept_path_gate import AcceptPath
from monaqa2.qiskit.reflection_gate import Reflection
from monaqa2.qiskit.metropolis_hastings_energy_gate import MetropolisHastingsEnergy
from monaqa2.qiskit.sqrt_exp_arithmetic_gate import SqrtExpArithmetic
from monaqa2.qiskit.glauber_arithmetic_gate import GlauberArithmetic
from monaqa2.qiskit.glauber_stateprep_arithmetic_gate import GlauberStateprepArithmetic


class WalkLocal(Gate):
    r"""
    Local-proposal quantum walk operator

        W = R_0 V^\dagger B^\dagger F B V

    with:
      * V = ProposalLocal(n, k), the Dicke/Hamming-distance-k local proposal;
      * B the coin unitary;
      * F the accept-path controlled swap;
      * R_0 the reflection about |0^n> on B and |0...0> on the coin register.

    Coin choices:
      * coin="mh": MetropolisHastingsEnergy -> SqrtExpArithmetic
          -> MetropolisHastingsEnergy^\dagger.
          eps is split evenly between fixed-point precision and sqrt-exp precision.
      * coin="glauber": GlauberArithmetic directly.
          The full eps is passed to GlauberArithmetic.
      * coin="stateprep": GlauberStateprepArithmetic directly.
          eps must be None. Parameters a and c are passed to GlauberStateprepArithmetic.
    """

    def __init__(
        self,
        n: int,
        k: int,
        h: np.ndarray,
        J: np.ndarray,
        beta: float,
        eps: float | None,
        coin: str = "mh",
        a: float = 1.0,
        c: int = 2,
        mocked_circuit: bool = False,
        mocked_angles: bool = False,
        label=None,
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2.")
        if not (1 <= k <= n):
            raise ValueError("k must satisfy 1 <= k <= n.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if coin not in {"mh", "glauber", "stateprep"}:
            raise ValueError("coin must be one of 'mh', 'glauber', or 'stateprep'.")

        # Keep backward-compatible default construction for mh/glauber.
        if coin != "stateprep" and c == 2:
            c = None

        if coin == "stateprep":
            assert eps is None
            assert c is not None
        else:
            assert c is None

        if coin == "stateprep" and not (np.isinf(a) or a >= 1.0):
            raise ValueError("For coin='stateprep', a must satisfy a >= 1 or a = np.inf.")

        if coin != "stateprep":
            if eps is None:
                raise ValueError("eps must be provided unless coin='stateprep'.")
            if eps <= 0.0:
                raise ValueError("eps must be positive.")
            if a <= 0.0:
                raise ValueError("a must be positive.")

        assert h.shape == (n,)
        assert J.shape == (n, n)

        self.n = int(n)
        self.k = int(k)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.beta = float(beta)
        self.eps = None if eps is None else float(eps)
        self.coin = str(coin)
        self.a = float(a)
        self.c = None if c is None else int(c)
        self.mocked_circuit = bool(mocked_circuit)
        self.mocked_angles = bool(mocked_angles)

        self.proposal = ProposalLocal(self.n, self.k)

        if self.coin == "mh":
            self.eps_fixed_point = 0.5 * self.eps
            self.eps_coin = 0.5 * self.eps

            self.energy = MetropolisHastingsEnergy(
                self.n,
                self.h,
                self.J,
                self.eps_fixed_point,
            )

            self.sqrt_exp = SqrtExpArithmetic(
                self.energy.signal_bits,
                self.beta,
                self.energy.normalization,
                self.eps_coin,
                mocked_circuit=self.mocked_circuit,
                mocked_angles=self.mocked_angles,
            )

            self.coin_gate = None
            self.n_coin_core = self._mh_coin_core_qubits()

        elif self.coin == "glauber":
            self.eps_fixed_point = None
            self.eps_coin = self.eps
            self.energy = None
            self.sqrt_exp = None

            self.coin_gate = GlauberArithmetic(
                self.n,
                self.h,
                self.J,
                self.beta,
                self.eps_coin,
                a=self.a
            )

            self.n_coin_core = self._direct_coin_core_qubits()

        elif self.coin == "stateprep":
            self.eps_fixed_point = None
            self.eps_coin = None
            self.energy = None
            self.sqrt_exp = None

            self.coin_gate = GlauberStateprepArithmetic(
                self.n,
                self.c,
                self.h,
                self.J,
                self.beta,
                a=self.a
            )

            self.n_coin_core = self._direct_coin_core_qubits()

        else:
            raise RuntimeError(f"Unhandled coin type: {self.coin}")

        self.coins = max(3, self.n_coin_core)

        self.accept_path = AcceptPath(self.n, self.coins, mocked_circuit=mocked_circuit)
        self.reflection = Reflection(self.n, self.coins, mocked_circuit=mocked_circuit)

        self.n_accept_work = 0 if mocked_circuit else self.accept_path.num_qubits - 2 * self.n - self.coins
        self.n_reflection_work = 0 if mocked_circuit else self.reflection.num_qubits - 2 * self.n - self.coins

        super().__init__(
            "WalkLocal",
            2 * self.n + self.coins + self.n_accept_work + self.n_reflection_work,
            [],
            label=label,
        )

        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        a = list(range(0, self.n))
        b = list(range(self.n, 2 * self.n))

        coins_start = 2 * self.n
        coins = list(range(coins_start, coins_start + self.coins))

        accept_start = coins_start + self.coins
        accept_work = list(range(accept_start, accept_start + self.n_accept_work))

        reflection_start = accept_start + self.n_accept_work
        reflection_work = list(
            range(reflection_start, reflection_start + self.n_reflection_work)
        )

        return {
            "A": a,
            "B": b,
            "coins": coins,
            "accept_work": accept_work,
            "reflection_work": reflection_work,
        }

    def _mh_coin_core_qubits(self) -> int:
        energy_work = self.energy.num_qubits - 2 * self.n
        sqrt_extra = self.sqrt_exp.num_qubits - len(self.sqrt_exp.layout["signal"])
        return energy_work + sqrt_extra

    def _direct_coin_core_qubits(self) -> int:
        return self.coin_gate.num_qubits - 2 * self.n

    def _energy_qargs(self) -> list[int]:
        return (
            self.layout["A"]
            + self.layout["B"]
            + self.layout["coins"][: self.energy.num_qubits - 2 * self.n]
        )

    def _sqrt_exp_qargs(self) -> list[int]:
        qargs = [None] * self.sqrt_exp.num_qubits

        energy_qargs = self._energy_qargs()
        signal_global = [energy_qargs[q] for q in self.energy.layout["signal"]]

        energy_work = self.energy.num_qubits - 2 * self.n
        sqrt_extra = self.sqrt_exp.num_qubits - len(self.sqrt_exp.layout["signal"])
        sqrt_coin = self.layout["coins"][energy_work : energy_work + sqrt_extra]

        qargs[self.sqrt_exp.layout["control"][0]] = sqrt_coin[0]

        for local, global_q in zip(self.sqrt_exp.layout["signal"], signal_global):
            qargs[local] = global_q

        for local, global_q in zip(self.sqrt_exp.layout["aux"], sqrt_coin[1:]):
            qargs[local] = global_q

        if any(q is None for q in qargs):
            raise RuntimeError("Incomplete SqrtExpArithmetic qubit mapping.")

        return qargs

    def _direct_coin_qargs(self) -> list[int]:
        qargs = [None] * self.coin_gate.num_qubits
        coins = self.layout["coins"]

        qargs[self.coin_gate.layout["control"][0]] = coins[0]

        for local, global_q in zip(self.coin_gate.layout["A"], self.layout["A"]):
            qargs[local] = global_q

        for local, global_q in zip(self.coin_gate.layout["B"], self.layout["B"]):
            qargs[local] = global_q

        for local, global_q in zip(self.coin_gate.layout["aux"], coins[1:]):
            qargs[local] = global_q

        if any(q is None for q in qargs):
            raise RuntimeError(f"Incomplete {self.coin} coin qubit mapping.")

        return qargs

    def _append_coin(self, qc: QuantumCircuit, inverse: bool = False) -> None:
        if self.coin in {"glauber", "stateprep"}:
            gate = self.coin_gate.inverse() if inverse else self.coin_gate
            qc.append(gate, self._direct_coin_qargs())
            return

        if self.coin == "mh":
            energy_qargs = self._energy_qargs()
            sqrt_qargs = self._sqrt_exp_qargs()

            qc.append(self.energy, energy_qargs)
            qc.append(self.sqrt_exp.inverse() if inverse else self.sqrt_exp, sqrt_qargs)
            qc.append(self.energy.inverse(), energy_qargs)
            return

        raise RuntimeError(f"Unhandled coin type: {self.coin}")

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        A = self.layout["A"]
        B = self.layout["B"]
        coins = self.layout["coins"]
        accept_work = self.layout["accept_work"]
        reflection_work = self.layout["reflection_work"]

        proposal_qargs = A + B
        accept_qargs = A + B + coins + accept_work
        reflection_qargs = A + B + coins + reflection_work

        qc.append(self.proposal, proposal_qargs)
        self._append_coin(qc, inverse=False)
        qc.append(self.accept_path, accept_qargs)
        self._append_coin(qc, inverse=True)
        qc.append(self.proposal.inverse(), proposal_qargs)
        qc.append(self.reflection, reflection_qargs)

        return qc
    