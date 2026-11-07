from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.reversible_delta_energy import ReversibleDeltaEnergy
from monaqa2.qiskit2.reversible_positive_part import ReversiblePositivePart
from monaqa2.qiskit2.rotational_acceptance import RotationalAcceptance


class BoltzmannCoin(Gate):
    r"""All-zero coins have amplitude sqrt(min(1,exp(-beta*(E(y)-E(x))))).

    Reads both full states x,y, with any number of differing spins.
    E is the coefficient-rounded Ising Hamiltonian. Inputs x,y are preserved;
    delta, positive and work return to zero. Rejection may leave nonzero coins.
    Depth is O(log n + log W), at the logical rotation-gate level with all-to-all
    connectivity. Acceptance projects all coin qubits onto zero.
    """

    def __init__(self, n: int, h, J, F: int, beta: float, label=None):
        self.delta_energy = ReversibleDeltaEnergy(n, h, J, F)
        self.n, self.F, self.W = self.delta_energy.n, self.delta_energy.F, self.delta_energy.W
        self.energy_unit = self.delta_energy.energy_unit
        self.positive_part = ReversiblePositivePart(self.W)
        self.rotation = RotationalAcceptance(self.W, beta, self.energy_unit)
        self.num_coins = self.rotation.num_coins
        self.work = max(self.delta_energy.work, self.positive_part.scratch)
        super().__init__("boltzmann_coin", self.num_coins + 2 * self.n + 2 * self.W + self.work, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        coins, x, y, delta, positive, work = (self.layout[key] for key in (
            "coins", "x", "y", "delta", "positive", "work"))
        # Both full configurations feed E(y)-E(x); y is the proposed state.
        energy_wires = x + y + delta + work[:self.delta_energy.work]
        positive_wires = delta + positive + work[:self.positive_part.scratch]
        qc.append(self.delta_energy, energy_wires)
        qc.append(self.positive_part, positive_wires)
        qc.append(self.rotation, positive + coins)
        qc.append(self.positive_part.inverse(), positive_wires)
        qc.append(self.delta_energy.inverse(), energy_wires)
        return qc

    @property
    def layout(self):
        C, n, W = self.num_coins, self.n, self.W
        return {"coins": list(range(C)), "x": list(range(C, C + n)),
                "y": list(range(C + n, C + 2 * n)),
                "delta": list(range(C + 2 * n, C + 2 * n + W)),
                "positive": list(range(C + 2 * n + W, C + 2 * n + 2 * W)),
                "work": list(range(C + 2 * n + 2 * W, self.num_qubits))}
