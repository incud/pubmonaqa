import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.cry import Cry


class RotationalAcceptance(Gate):
    """All-zero coin amplitude = exp(-beta*energy_unit*positive/2).

    The W-bit positive word must be nonnegative, with its sign bit zero.
    Separate coin targets make all controlled rotations parallel. Angles are
    classical constants; no acceptance probability is stored or evaluated in qubits.
    """

    def __init__(self, W: int, beta: float, energy_unit: float, label=None):
        self.W, self.beta, self.energy_unit = int(W), float(beta), float(energy_unit)
        if self.W < 1 or min(self.beta, self.energy_unit) < 0 or not np.all(np.isfinite([self.beta, self.energy_unit])):
            raise ValueError("Require W > 0 and finite beta,energy_unit >= 0.")
        self.num_coins = max(1, self.W - 1)
        self.angles = []
        for bit in range(self.W - 1):
            exponent = self.beta * self.energy_unit * 2**bit
            # atan2/expm1 avoid cancellation when the acceptance is very close to one.
            self.angles.append(2 * np.arctan2(np.sqrt(-np.expm1(-exponent)), np.exp(-exponent / 2)))
        super().__init__("rotational_acceptance", self.W + self.num_coins, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        for bit, theta in enumerate(self.angles):
            if theta:
                qc.append(Cry(float(theta)), [bit, self.W + bit])
        return qc

    @property
    def layout(self):
        return {"positive": list(range(self.W)), "coins": list(range(self.W, self.num_qubits))}
