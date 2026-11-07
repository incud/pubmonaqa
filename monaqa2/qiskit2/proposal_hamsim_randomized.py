import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.rx import Rx
from monaqa2.qiskit2.crx import Crx
from monaqa2.qiskit2.ccrx import Ccrx
from monaqa2.qiskit2.crz import Crz
from monaqa2.qiskit2.ccrz import Ccrz
from monaqa2.qiskit2.rzz import Rzz
from monaqa2.qiskit2.crzz import Crzz
from monaqa2.qiskit2.ccrzz import Ccrzz


class ProposalHamSimRandomized(Gate):
    r"""Prepare 2**(-k) sum_{j,l} |x>_A U(t_j,gamma_l)|x>_B |j>_t |l>_gamma.

    Parameters t and gamma specify (low, high) bounds; scalars mean (0, value).
    Each register selects one of 2**k midpoint values, with bit 0 least significant.
    H = sum_i h[i]*Z_i + sum_{i<j} J[i,j]*Z_i*Z_j (upper triangle).
    U uses second-order Trotter splitting for (1-gamma)*H - gamma*sum_i X_i.
    Uses 2*n + 2*k*n qubits; parameter registers remain entangled with B.
    Work copies keep rotations parallel and return to zero after the evolution.
    With all-to-all connectivity, depth is O(trotter_steps*(k+1)**2*n + log(n)).
    """

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, t, gamma, trotter_steps: int, k: int, label=None) -> None:
        self.n, self.k, self.trotter_steps = int(n), int(k), int(trotter_steps)
        self.h, self.J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
        self.t = (0.0, float(t)) if np.isscalar(t) else tuple(map(float, t))
        self.gamma = (0.0, float(gamma)) if np.isscalar(gamma) else tuple(map(float, gamma))
        if min(self.n, self.k, self.trotter_steps) < 1:
            raise ValueError("n, k and trotter_steps must be positive integers.")
        if self.h.shape != (self.n,) or self.J.shape != (self.n, self.n):
            raise ValueError("Expected h.shape == (n,) and J.shape == (n, n).")
        if len(self.t) != 2 or not self.t[0] <= self.t[1]:
            raise ValueError("t must specify increasing interval bounds.")
        if len(self.gamma) != 2 or not 0 <= self.gamma[0] <= self.gamma[1] <= 1:
            raise ValueError("gamma must specify increasing bounds within [0, 1].")
        # Each register encodes its first midpoint plus weighted binary digits.
        # Time coefficients are already divided by the Trotter step count.
        self.time_spacing = (self.t[1] - self.t[0]) / (2**self.k * self.trotter_steps)
        self.time_base = self.t[0] / self.trotter_steps + self.time_spacing / 2
        self.gamma_spacing = (self.gamma[1] - self.gamma[0]) / 2**self.k
        self.gamma_base = self.gamma[0] + self.gamma_spacing / 2
        super().__init__(f"ProposalHamSimRandomized(n={self.n})", 2 * self.n + 2 * self.k * self.n, [], label=label)
        self.definition = self._build_definition()

    def _create_schedule(self) -> list[list[tuple[int, int]]]:
        q = list(range(self.n + self.n % 2))
        rounds = []
        for _ in range(len(q) - 1):
            rounds.append([(i, j) for i, j in zip(q, reversed(q)) if i < j < self.n])
            q = [q[0], q[-1], *q[1:-1]]
        return rounds

    def _x_layer(self, qc, copies, fraction):
        """Apply the transverse field: RX angles are -2*fraction*time*gamma."""
        layout = self.layout
        for spin, target in enumerate(layout["B"]):
            # Start with both parameter registers set to zero.
            theta = -2 * fraction * self.time_base * self.gamma_base
            if theta:
                qc.append(Rx(theta), [target])

            for p, time_bit in enumerate(layout["t"]):
                time_weight = self.time_spacing * 2**p
                ctrl_1 = copies[time_bit][spin]
                theta = -2 * fraction * time_weight * self.gamma_base
                if theta:
                    qc.append(Crx(theta), [ctrl_1, target])
                # Products of a time bit and a gamma bit need two controls.
                for q, gamma_bit in enumerate(layout["gamma"]):
                    gamma_weight = self.gamma_spacing * 2**q
                    ctrl_2 = copies[gamma_bit][spin]
                    theta = -2 * fraction * time_weight * gamma_weight
                    if theta:
                        qc.append(Ccrx(theta), [ctrl_1, ctrl_2, target])

            # Gamma-bit contributions multiplied by the first time midpoint.
            for q, gamma_bit in enumerate(layout["gamma"]):
                gamma_weight = self.gamma_spacing * 2**q
                ctrl_1 = copies[gamma_bit][spin]
                theta = -2 * fraction * self.time_base * gamma_weight
                if theta:
                    qc.append(Crx(theta), [ctrl_1, target])

    def _z_layer(self, qc, copies):
        """Apply the local Z fields, with angles 2*h[i]*time*(1-gamma)."""
        layout = self.layout
        for spin, target in enumerate(layout["B"]):
            coefficient = 2 * self.h[spin]
            theta = coefficient * self.time_base * (1 - self.gamma_base)
            if theta:
                qc.rz(theta, target)

            for p, time_bit in enumerate(layout["t"]):
                time_weight = self.time_spacing * 2**p
                ctrl_1 = copies[time_bit][spin]
                theta = coefficient * time_weight * (1 - self.gamma_base)
                if theta:
                    qc.append(Crz(theta), [ctrl_1, target])
                for q, gamma_bit in enumerate(layout["gamma"]):
                    gamma_weight = self.gamma_spacing * 2**q
                    ctrl_2 = copies[gamma_bit][spin]
                    # The gamma-dependent part of (1-gamma) has a minus sign.
                    theta = -coefficient * time_weight * gamma_weight
                    if theta:
                        qc.append(Ccrz(theta), [ctrl_1, ctrl_2, target])

            for q, gamma_bit in enumerate(layout["gamma"]):
                gamma_weight = self.gamma_spacing * 2**q
                ctrl_1 = copies[gamma_bit][spin]
                theta = -coefficient * self.time_base * gamma_weight
                if theta:
                    qc.append(Crz(theta), [ctrl_1, target])

    def _zz_layer(self, qc, copies, schedule):
        """Apply ZZ couplings, with disjoint pairs and distinct control copies."""
        layout = self.layout
        targets = layout["B"]
        for matching in schedule:
            for slot, (i, j) in enumerate(matching):
                anc_1, anc_2 = targets[i], targets[j]
                coefficient = 2 * self.J[i, j]
                theta = coefficient * self.time_base * (1 - self.gamma_base)
                if theta:
                    qc.append(Rzz(theta), [anc_1, anc_2])

                for p, time_bit in enumerate(layout["t"]):
                    time_weight = self.time_spacing * 2**p
                    ctrl_1 = copies[time_bit][slot]
                    theta = coefficient * time_weight * (1 - self.gamma_base)
                    if theta:
                        qc.append(Crzz(theta), [ctrl_1, anc_1, anc_2])
                    for q, gamma_bit in enumerate(layout["gamma"]):
                        gamma_weight = self.gamma_spacing * 2**q
                        ctrl_2 = copies[gamma_bit][slot]
                        theta = -coefficient * time_weight * gamma_weight
                        if theta:
                            qc.append(Ccrzz(theta), [ctrl_1, ctrl_2, anc_1, anc_2])

                for q, gamma_bit in enumerate(layout["gamma"]):
                    gamma_weight = self.gamma_spacing * 2**q
                    ctrl_1 = copies[gamma_bit][slot]
                    theta = -coefficient * self.time_base * gamma_weight
                    if theta:
                        qc.append(Crzz(theta), [ctrl_1, anc_1, anc_2])

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        a, b, t, g = (self.layout[key] for key in ("A", "B", "t", "gamma"))
        qc.cx(a, b)
        qc.h(t + g)
        work = self.layout["work"]
        # A distinct control copy for every spin or simultaneous matching edge.
        copies = {q: [q] + work[p * (self.n - 1):(p + 1) * (self.n - 1)]
                  for p, q in enumerate(t + g)}
        # Parallel binary-tree fanout; save the CNOTs for uncomputation.
        fanout, width = [], 1
        while width < self.n:
            for wires in copies.values():
                for i in range(min(width, self.n - width)):
                    ctrl_1, target = wires[i], wires[i + width]
                    qc.cx(ctrl_1, target)
                    fanout.append((ctrl_1, target))
            width *= 2

        schedule = self._create_schedule()
        # Second-order splitting: X/2, Ising, X, Ising, ..., X/2.
        self._x_layer(qc, copies, 0.5)
        for step in range(self.trotter_steps):
            self._z_layer(qc, copies)
            self._zz_layer(qc, copies, schedule)
            fraction = 0.5 if step == self.trotter_steps - 1 else 1.0
            self._x_layer(qc, copies, fraction)
        for ctrl_1, target in reversed(fanout):
            qc.cx(ctrl_1, target)
        return qc

    @property
    def layout(self) -> dict[str, list[int]]:
        n, k = self.n, self.k
        return {"A": list(range(n)), "B": list(range(n, 2 * n)),
                "t": list(range(2 * n, 2 * n + k)),
                "gamma": list(range(2 * n + k, 2 * n + 2 * k)),
                "work": list(range(2 * n + 2 * k, self.num_qubits))}
