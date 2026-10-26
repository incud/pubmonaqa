"""Fixed-point energy arithmetic with a rotation-based Metropolis coin."""

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.arithmetic_hybrid import ConditionalTermsLoader, DeltaEnergy, PositivePartSelector
from monaqa2.qiskit.primitives import Cry
from monaqa2.qiskit.utils_numpy import boltzmann_rotation_angles


class RotationBoltzmannCoin(Gate):

    def __init__(self, W: int, beta: float, alpha: float, label=None):
        """Rotate one clean coin qubit per bit of a nonnegative Q1.F word.

        ``F = W - 1`` and the physical positive energy difference is
        ``alpha * sum(delta[j] * 2**(j - F))``. Acceptance means that every
        coin qubit is zero. The input word is preserved, including on a
        superposition. The former sign bit must be zero on valid inputs.
        """
        self.angles = boltzmann_rotation_angles(W, beta, alpha)
        self.W = int(W)
        self.F = self.W - 1
        self.beta = float(beta)
        self.alpha = float(alpha)
        super().__init__(name=f"RotationBoltzmannCoin({W})", num_qubits=2 * self.W, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"delta": list(range(self.W)), "coins": list(range(self.W, 2 * self.W))}

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        for delta, coin, theta in zip(self.layout["delta"], self.layout["coins"], self.angles):
            if theta != 0.0:
                qc.append(Cry(float(theta)), [delta, coin])
        return qc


class FixedPointArithmetic(Gate):

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, F: int, beta: float, label=None):
        """Compute the positive energy difference, rotate the coins, and uncompute.

        Layout: coins (W), x (n), y (n), delta (W), zero (W), signs (W),
        work (DeltaEnergy auxiliary). All registers except x/y start clean.
        Only the coins can remain nonzero after the gate. Pass their complete
        register to Reflection/AcceptPath, whose accepting subspace is zero.

        Coefficient normalization, rounding, little-endian Q1.F encoding,
        and modular arithmetic are exactly those of HybridPhaseArithmetic.
        As there, energy differences must fit the signed word; overflow,
        including the positive endpoint +1 of the normalized difference,
        is not representable. No QSP, polynomial, or tail cutoff is used.
        """
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer.")
        if not isinstance(F, (int, np.integer)) or isinstance(F, bool) or F < 0:
            raise ValueError("F must be a nonnegative integer.")
        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        if self.h.shape != (n,) or self.J.shape != (n, n):
            raise ValueError("Expected h.shape == (n,) and J.shape == (n, n).")
        if not np.all(np.isfinite(self.h)) or not np.all(np.isfinite(self.J)):
            raise ValueError("Hamiltonian coefficients must be finite.")
        self.F = int(F)
        self.W = self.F + 1
        self.M = self.n + self.n * (self.n - 1) // 2
        self.beta = float(beta)
        self.alpha = float(ConditionalTermsLoader._alpha(self.h, self.J))
        self.coin = RotationBoltzmannCoin(self.W, self.beta, self.alpha)
        self.delta_energy = DeltaEnergy(self.n, self.h, self.J, self.F)
        self.positive_selector = PositivePartSelector(self.W)
        self.delta_aux = self.delta_energy.num_qubits - 2 * self.n - self.W
        self.work = self.delta_aux
        super().__init__(name=f"FixedPointArithmetic({n},{F})", num_qubits=2 * self.n + 4 * self.W + self.work, params=[], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        n, W = self.n, self.W
        return {
            "coins": list(range(W)),
            "x": list(range(W, W + n)),
            "y": list(range(W + n, W + 2 * n)),
            "delta": list(range(W + 2 * n, 2 * W + 2 * n)),
            "zero": list(range(2 * W + 2 * n, 3 * W + 2 * n)),
            "signs": list(range(3 * W + 2 * n, 4 * W + 2 * n)),
            "work": list(range(4 * W + 2 * n, self.num_qubits)),
        }

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        layout = self.layout
        work = layout["work"]
        # DeltaEnergy computes its first energy minus its second: reverse x/y.
        delta_wires = layout["y"] + layout["x"] + work[:2 * self.M * self.W] + layout["delta"] + work[2 * self.M * self.W:self.delta_aux]
        selector_wires = layout["delta"] + layout["zero"] + layout["signs"]
        qc.append(self.delta_energy, delta_wires)
        qc.append(self.positive_selector, selector_wires)
        qc.append(self.coin, layout["delta"] + layout["coins"])
        qc.append(self.positive_selector.inverse(), selector_wires)
        qc.append(self.delta_energy.inverse(), delta_wires)
        return qc
