import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.reversible_conditional_loader import ReversibleConditionalLoader
from monaqa2.qiskit2.reversible_wallace_adder import ReversibleWallaceAdder


class ReversibleDeltaEnergy(Gate):
    """Compute the signed half-difference S; physical DeltaE = energy_unit*S.

    Quantize c_l as eta*q_l, eta=alpha/2**F, alpha=2*sum(abs(c_l)).
    S=sum_l q_l*(parity_l(x)-parity_l(y)); energy_unit=2*eta.
    This computes E(y)-E(x), with one loaded row per Ising term.
    delta is XOR-written; all other work returns to zero.
    """

    def __init__(self, n: int, h, J, F: int, label=None):
        self.n, self.F = int(n), int(F)
        self.h, self.J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
        if self.n < 1 or self.F < 0:
            raise ValueError("Require n > 0 and F >= 0.")
        if self.h.shape != (self.n,) or self.J.shape != (self.n, self.n):
            raise ValueError("Expected h.shape == (n,) and J.shape == (n,n).")
        values = list(self.h) + [self.J[i, j] for i in range(self.n) for j in range(i + 1, self.n)]
        if not np.all(np.isfinite(values)):
            raise ValueError("Ising coefficients must be finite.")
        self.alpha = 2 * float(np.sum(np.abs(values)))
        self.coefficient_unit = self.alpha * 2.0**(-self.F) if self.alpha else 1.0
        if not np.isfinite(self.coefficient_unit) or self.coefficient_unit <= 0:
            raise ValueError("Coefficient scale is not representable; reduce F or coefficient magnitudes.")
        self.coefficients = [int(np.rint(value / self.coefficient_unit)) for value in values]
        self.M = len(self.coefficients)
        self.bound = sum(abs(value) for value in self.coefficients)
        # Both signs, including the positive endpoint, must fit without wraparound.
        self.W = max(2, self.bound.bit_length() + 1)
        self.energy_unit = 2 * self.coefficient_unit
        self.loader = ReversibleConditionalLoader(self.n, self.coefficients, self.W)
        self.adder = ReversibleWallaceAdder(self.M, self.W)
        self.scratch = max(self.loader.scratch, self.adder.scratch)
        self.work = self.M * self.W + self.scratch
        super().__init__("reversible_delta_energy", 2 * self.n + self.W + self.work, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        x, y, delta, work = (self.layout[key] for key in ("x", "y", "delta", "work"))
        terms = work[:self.M * self.W]
        scratch = work[self.M * self.W:]
        loader_wires = x + y + terms + scratch[:self.loader.scratch]
        qc.append(self.loader, loader_wires)
        qc.append(self.adder, terms + delta + scratch[:self.adder.scratch])
        qc.append(self.loader.inverse(), loader_wires)
        return qc

    @property
    def layout(self):
        n, W = self.n, self.W
        return {"x": list(range(n)), "y": list(range(n, 2 * n)),
                "delta": list(range(2 * n, 2 * n + W)),
                "work": list(range(2 * n + W, self.num_qubits))}
