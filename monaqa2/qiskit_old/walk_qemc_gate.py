import numpy as np
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.proposal_qemc_gate import ProposalQemc
from monaqa2.qiskit.accept_path_gate import AcceptPath
from monaqa2.qiskit.reflection_gate import Reflection
from monaqa2.qiskit.metropolis_hastings_energy_gate import MetropolisHastingsEnergy
from monaqa2.qiskit.sqrt_exp_arithmetic_gate import SqrtExpArithmetic
from monaqa2.qiskit.glauber_arithmetic_gate import GlauberArithmetic
from monaqa2.qiskit.glauber_stateprep_arithmetic_gate import GlauberStateprepArithmetic


class WalkQemc(Gate):
    r"""
    QEMC-proposal quantum walk operator

        W = R_0 V^\dagger B^\dagger F B V

    with:
      * V = ProposalQemc;
      * B the acceptance coin unitary;
      * F the accept-path controlled swap on A and B;
      * R_0 the reflection about zero on B, the QEMC proposal auxiliary
        register, and the acceptance-coin register.

    Proposal choices:
        evolution="gqsp":
            Uses HamiltonianSimulationGQSP inside ProposalQemc.
            For this mode gamma must be an array of shape (n,), and eps is
            required for the proposal unless using coin="stateprep" is not
            enough to avoid the proposal approximation.

        evolution="trotter":
            Uses TrotterizedOperatorIsingTF inside ProposalQemc.
            For this mode gamma must be an array of shape (n,). Either eps or
            num_trotter_steps must be provided for the proposal.

        evolution="exact":
            Uses dense scipy.linalg.expm inside ProposalQemc.
            For this mode gamma must be a scalar mixing parameter.

    Coin choices:
        coin="mh":
            Uses MetropolisHastingsEnergy, SqrtExpArithmetic, and uncomputation.

        coin="glauber":
            Uses GQSP-based GlauberArithmetic.

        coin="stateprep":
            Uses exact dense GlauberStateprepArithmetic on c coin qubits.
            The accepted amplitude is loaded into |0...0>_c and the residual
            amplitude into |1...1>_c.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        gamma: np.ndarray | float,
        beta: float,
        t: float,
        eps: float | None = None,
        coin: str = "mh",
        a: float = 1.0,
        evolution: str = "gqsp",
        num_trotter_steps: int | None = None,
        c: int | None = None,
        mocked_reflection: bool = False,
        mocked_circuit: bool = False,
        mocked_angles: bool = False,
        label=None,
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if a <= 0.0:
            raise ValueError("a must be positive.")
        if coin not in {"mh", "glauber", "stateprep"}:
            raise ValueError("coin must be one of 'mh', 'glauber', or 'stateprep'.")
        if evolution not in {"gqsp", "trotter", "exact"}:
            raise ValueError("evolution must be one of 'gqsp', 'trotter', or 'exact'.")

        h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)

        if h.shape != (n,):
            raise ValueError(f"h must have shape ({n},), got {h.shape}.")
        if J.shape != (n, n):
            raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}.")

        if evolution == "exact":
            gamma_arr = np.asarray(gamma, dtype=float)
            if gamma_arr.ndim != 0:
                raise ValueError("For evolution='exact', gamma must be a scalar mixing parameter.")
            gamma_for_proposal = float(gamma_arr)
        else:
            gamma_for_proposal = np.asarray(gamma, dtype=float)
            if gamma_for_proposal.shape != (n,):
                raise ValueError(f"For evolution='{evolution}', gamma must have shape ({n},), got {gamma_for_proposal.shape}.")

        if coin == "stateprep":
            if c is None:
                raise ValueError("For coin='stateprep', c must be provided.")
            if c < 1:
                raise ValueError("c must be positive.")
            if not (np.isinf(a) or a >= 1.0):
                raise ValueError("For coin='stateprep', a must satisfy a >= 1 or a = np.inf.")
        elif c is not None:
            raise ValueError("c is only used for coin='stateprep'.")

        proposal_needs_eps = evolution == "gqsp" or (evolution == "trotter" and num_trotter_steps is None)
        coin_needs_eps = coin in {"mh", "glauber"}
        eps_is_used = proposal_needs_eps or coin_needs_eps

        if eps_is_used:
            if eps is None:
                raise ValueError("eps must be provided for the requested proposal/coin configuration.")
            if eps <= 0.0:
                raise ValueError("eps must be positive.")
        elif eps is not None:
            raise ValueError("eps is unused for this exact/stateprep configuration; pass eps=None.")

        self.n = int(n)
        self.h = h
        self.J = J
        self.gamma = gamma_for_proposal
        self.beta = float(beta)
        self.t = float(t)
        self.eps = float(eps) if eps is not None else None
        self.coin = str(coin)
        self.a = float(a)
        self.evolution = str(evolution)
        self.num_trotter_steps = num_trotter_steps
        self.c = int(c) if c is not None else None
        self.mocked_reflection = bool(mocked_reflection)
        self.mocked_circuit = bool(mocked_circuit)
        self.mocked_angles = bool(mocked_angles)

        self.eps_proposal = self._proposal_eps()
        self.eps_coin = self._coin_eps()

        self.proposal = ProposalQemc(
            self.n,
            self.h,
            self.J,
            gamma=self.gamma,
            t=self.t,
            eps=self.eps_proposal,
            evolution=self.evolution,
            num_trotter_steps=self.num_trotter_steps,
            mocked_reflection=self.mocked_reflection,
            mocked_angles=self.mocked_angles,
        )
        self.n_proposal_aux = self.proposal.num_qubits - 2 * self.n

        self.energy = None
        self.sqrt_exp = None
        self.coin_gate = None
        self.eps_fixed_point = None
        self.eps_sqrt_exp = None

        if self.coin == "mh":
            self.eps_fixed_point = 0.5 * self.eps_coin
            self.eps_sqrt_exp = 0.5 * self.eps_coin
            self.energy = MetropolisHastingsEnergy(self.n, self.h, self.J, self.eps_fixed_point)
            self.sqrt_exp = SqrtExpArithmetic(self.energy.signal_bits, self.beta, self.energy.normalization, self.eps_sqrt_exp, mocked_circuit=self.mocked_circuit, mocked_angles=self.mocked_angles)
            self.n_coin_core = self._mh_coin_core_qubits()

        elif self.coin == "glauber":
            self.coin_gate = GlauberArithmetic(self.n, self.h, self.J, self.beta, self.eps_coin, a=self.a, mocked_circuit=self.mocked_circuit, mocked_angles=self.mocked_angles)
            self.n_coin_core = self._glauber_coin_core_qubits()

        else:
            self.coin_gate = GlauberStateprepArithmetic(self.n, self.c, self.h, self.J, self.beta, a=self.a)
            self.n_coin_core = self.coin_gate.num_qubits - 2 * self.n

        self.coins = self.n_coin_core if self.coin == "stateprep" else max(3, self.n_coin_core)

        self.accept_path = AcceptPath(self.n, self.coins, mocked_circuit=self.mocked_circuit)
        self.reflection = Reflection(self.n, self.n_proposal_aux + self.coins, mocked_circuit=self.mocked_circuit)

        self.n_accept_work = self.accept_path.num_qubits - 2 * self.n - self.coins
        self.n_reflection_work = self.reflection.num_qubits - 2 * self.n - (self.n_proposal_aux + self.coins)

        super().__init__(
            "WalkQemc",
            2 * self.n + self.n_proposal_aux + self.coins + self.n_accept_work + self.n_reflection_work,
            [],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        a = list(range(0, self.n))
        b = list(range(self.n, 2 * self.n))

        proposal_aux_start = 2 * self.n
        proposal_aux = list(range(proposal_aux_start, proposal_aux_start + self.n_proposal_aux))

        coins_start = proposal_aux_start + self.n_proposal_aux
        coins = list(range(coins_start, coins_start + self.coins))

        accept_start = coins_start + self.coins
        accept_work = list(range(accept_start, accept_start + self.n_accept_work))

        reflection_start = accept_start + self.n_accept_work
        reflection_work = list(range(reflection_start, reflection_start + self.n_reflection_work))

        return {
            "A": a,
            "B": b,
            "proposal_aux": proposal_aux,
            "coins": coins,
            "accept_work": accept_work,
            "reflection_work": reflection_work,
        }

    def _proposal_eps(self) -> float | None:
        if self.evolution == "exact":
            return None

        if self.coin == "stateprep":
            return self.eps

        return 0.5 * self.eps

    def _coin_eps(self) -> float | None:
        if self.coin == "stateprep":
            return None

        if self.evolution == "exact":
            return self.eps

        return 0.5 * self.eps

    def _mh_coin_core_qubits(self) -> int:
        energy_work = self.energy.num_qubits - 2 * self.n
        sqrt_extra = self.sqrt_exp.num_qubits - len(self.sqrt_exp.layout["signal"])
        return energy_work + sqrt_extra

    def _glauber_coin_core_qubits(self) -> int:
        return self.coin_gate.num_qubits - 2 * self.n

    def _energy_qargs(self) -> list[int]:
        return self.layout["A"] + self.layout["B"] + self.layout["coins"][: self.energy.num_qubits - 2 * self.n]

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

    def _glauber_qargs(self) -> list[int]:
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
            raise RuntimeError("Incomplete GlauberArithmetic qubit mapping.")

        return qargs

    def _stateprep_qargs(self) -> list[int]:
        qargs = [None] * self.coin_gate.num_qubits

        for local, global_q in zip(self.coin_gate.layout["A"], self.layout["A"]):
            qargs[local] = global_q

        for local, global_q in zip(self.coin_gate.layout["B"], self.layout["B"]):
            qargs[local] = global_q

        for local, global_q in zip(self.coin_gate.layout["coin"], self.layout["coins"]):
            qargs[local] = global_q

        if any(q is None for q in qargs):
            raise RuntimeError("Incomplete GlauberStateprepArithmetic qubit mapping.")

        return qargs

    def _append_coin(self, qc: QuantumCircuit, inverse: bool = False) -> None:
        if self.coin == "glauber":
            gate = self.coin_gate.inverse() if inverse else self.coin_gate
            qc.append(gate, self._glauber_qargs())
            return

        if self.coin == "stateprep":
            gate = self.coin_gate.inverse() if inverse else self.coin_gate
            qc.append(gate, self._stateprep_qargs())
            return

        energy_qargs = self._energy_qargs()
        sqrt_qargs = self._sqrt_exp_qargs()

        qc.append(self.energy, energy_qargs)
        qc.append(self.sqrt_exp.inverse() if inverse else self.sqrt_exp, sqrt_qargs)
        qc.append(self.energy.inverse(), energy_qargs)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        A = self.layout["A"]
        B = self.layout["B"]
        proposal_aux = self.layout["proposal_aux"]
        coins = self.layout["coins"]
        accept_work = self.layout["accept_work"]
        reflection_work = self.layout["reflection_work"]

        proposal_qargs = A + B + proposal_aux
        accept_qargs = A + B + coins + accept_work
        reflection_qargs = A + B + proposal_aux + coins + reflection_work

        qc.append(self.proposal, proposal_qargs)
        self._append_coin(qc, inverse=False)
        qc.append(self.accept_path, accept_qargs)
        self._append_coin(qc, inverse=True)
        qc.append(self.proposal.inverse(), proposal_qargs)
        qc.append(self.reflection, reflection_qargs)

        return qc