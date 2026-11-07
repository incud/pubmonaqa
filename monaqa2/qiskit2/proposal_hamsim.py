import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.rx import Rx
from monaqa2.qiskit2.rzz import Rzz


class ProposalHamSim(Gate):
    r"""Map |x>_A|0>_B to |x>_A U(t)|x>_B using 2*n qubits.

    H = sum_i h[i]*Z_i + sum_{i<j} J[i,j]*Z_i*Z_j (upper triangle).
    U(t) approximates exp(-it*((1-gamma)*H - gamma*sum_i X_i))
    by second-order Trotter splitting with all-to-all connectivity.
    Registers: A = q[:n], B = q[n:].
    """

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, t: float, gamma: float, trotter_steps: int, label=None) -> None:
        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.t = float(t)
        self.gamma = float(gamma)
        self.trotter_steps = int(trotter_steps)
        if self.n < 1 or self.trotter_steps < 1:
            raise ValueError("n and trotter_steps must be positive integers.")
        if self.h.shape != (self.n,) or self.J.shape != (self.n, self.n):
            raise ValueError("Expected h.shape == (n,) and J.shape == (n, n).")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1].")
        super().__init__(f"ProposalHamSim(n={self.n})", 2 * self.n, [], label=label)
        self.definition = self._build_definition()

    def _create_schedule(self) -> list[list[tuple[int, int]]]:
        # Round-robin matchings; an extra vertex is a dummy for odd n.
        q = list(range(self.n + self.n % 2))
        rounds = []
        for _ in range(len(q) - 1):
            rounds.append([(i, j) for i, j in zip(q, reversed(q)) if i < j < self.n])
            q = [q[0], q[-1], *q[1:-1]]
        return rounds

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        a = self.layout["A"]
        b = self.layout["B"]
        qc.cx(a, b)
        
        schedule = self._create_schedule()
        dt = self.t / self.trotter_steps
        z, x = 2 * (1 - self.gamma) * dt, -2 * self.gamma * dt
        # x layer of the first step
        for target in b:
            qc.append(Rx(x / 2), [target])
        for step in range(self.trotter_steps):
            for i in range(self.n):
                qc.rz(z * self.h[i], b[i])
            for layer in schedule:
                for i, j in layer:
                    theta = z * self.J[i, j]
                    if theta:
                        qc.append(Rzz(theta), [b[i], b[j]])
            # Merge adjacent X half-steps, keeping the two endpoints half-sized.
            for target in b:
                qc.append(Rx(x / 2 if step == self.trotter_steps - 1 else x), [target])
        return qc

    @property
    def layout(self) -> dict[str, list[int]]:
        return {"A": list(range(self.n)), "B": list(range(self.n, 2 * self.n))}
