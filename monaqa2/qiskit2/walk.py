import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.proposal_local import ProposalLocal
from monaqa2.qiskit2.proposal_uniform import ProposalUniform
from monaqa2.qiskit2.proposal_hamsim import ProposalHamSim
from monaqa2.qiskit2.proposal_hamsim_randomized import ProposalHamSimRandomized
from monaqa2.qiskit2.boltzmann_coin import BoltzmannCoin
from monaqa2.qiskit2.accept_path import AcceptPath
from monaqa2.qiskit2.reflection import Reflection


class Walk(Gate):
    r"""One step W = R P^dagger C^dagger F C P, with a local proposal by default.

    eps is epsilon_b in F=max(0,ceil(log2(beta*||H||_1*M/(2*eps)))).
    ||H||_1 is the sum of absolute Ising coefficients; M=n*(n+1)//2.
    A supplied proposal must preserve A and have symmetric transition amplitudes
    for each coherent label, as do the local, uniform and Hamiltonian proposals.
    All scratch returns to zero; B, coins and proposal labels are the walk space.
    k is the number of bits per randomized parameter register (2**k grid values).
    Hamiltonian defaults are t=1, gamma=0.5, trotter_steps=1; randomized scalar
    bounds mean (0,t) and (0,gamma). eps selects precision, not total walk error.
    """

    def __init__(self, h, J, eps: float, beta: float, proposal="local", k: int = 2,
                 *, t=1.0, gamma=0.5, trotter_steps: int = 1, label=None):
        self.h, self.J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
        self.n = len(self.h) if self.h.ndim == 1 else 0
        self.eps, self.beta = float(eps), float(beta)
        self.k = int(k)
        if self.n < 1 or self.J.shape != (self.n, self.n):
            raise ValueError("Expected a nonempty h.shape == (n,) and J.shape == (n,n).")
        if not np.all(np.isfinite([self.eps, self.beta])) or self.eps <= 0 or self.beta < 0:
            raise ValueError("Require finite eps > 0 and beta >= 0.")
        if self.k < 1:
            raise ValueError("k must be positive.")
        values = list(self.h) + [self.J[i, j] for i in range(self.n) for j in range(i + 1, self.n)]
        self.M, self.norm_1 = len(values), float(np.sum(np.abs(values)))
        if not np.all(np.isfinite(values)) or not np.isfinite(self.norm_1):
            raise ValueError("Ising coefficients and their absolute sum must be finite.")
        # Logs avoid overflow in beta*||H||_1*M/(2*eps); zero energy/beta needs no precision.
        self.F = 0
        if self.beta and self.norm_1:
            bits = np.log2(self.beta) + np.log2(self.norm_1) + np.log2(self.M) - 1 - np.log2(self.eps)
            self.F = max(0, int(np.ceil(bits)))
        if proposal == "local":
            self.proposal = ProposalLocal(self.n)
        elif proposal == "uniform":
            self.proposal = ProposalUniform(self.n)
        elif proposal == "hamsim":
            self.proposal = ProposalHamSim(self.n, self.h, self.J, t, gamma, trotter_steps)
        elif proposal == "hamsim_randomized":
            self.proposal = ProposalHamSimRandomized(self.n, self.h, self.J, t, gamma, trotter_steps, self.k)
        elif isinstance(proposal, Gate):
            self.proposal = proposal
        else:
            raise ValueError("proposal must be local, uniform, hamsim, hamsim_randomized, or a Gate.")
        if self.proposal.layout["A"] != list(range(self.n)) or self.proposal.layout["B"] != list(range(self.n, 2 * self.n)):
            raise ValueError("Proposal must use its first n qubits for A and next n for B.")
        self.extra = self.proposal.num_qubits - 2 * self.n
        self.coin = BoltzmannCoin(self.n, self.h, self.J, self.F, self.beta)
        self.coins = self.coin.num_coins
        self.accept = AcceptPath(self.n, self.coins)
        self.reflection = Reflection(self.n, self.coins, self.extra)
        self.coin_work = self.coin.num_qubits - 2 * self.n - self.coins
        self.work = max(self.coin_work, self.accept.work, self.reflection.work)
        super().__init__("walk", 2 * self.n + self.coins + self.extra + self.work, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        a, b, coins, extra, work = (self.layout[key] for key in ("A", "B", "coins", "proposal", "work"))
        proposal_wires = a + b + extra
        coin_wires = coins + a + b + work[:self.coin_work]
        qc.append(self.proposal, proposal_wires)
        qc.append(self.coin, coin_wires)
        qc.append(self.accept, a + b + coins + work[:self.accept.work])
        qc.append(self.coin.inverse(), coin_wires)
        qc.append(self.proposal.inverse(), proposal_wires)
        qc.append(self.reflection, a + b + coins + extra + work[:self.reflection.work])
        return qc

    @property
    def layout(self):
        n, end = self.n, 2 * self.n + self.coins
        return {"A": list(range(n)), "B": list(range(n, 2 * n)),
                "coins": list(range(2 * n, end)), "proposal": list(range(end, end + self.extra)),
                "work": list(range(end + self.extra, self.num_qubits))}
