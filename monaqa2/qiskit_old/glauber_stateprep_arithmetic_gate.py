import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import StatePreparation


class GlauberStateprepArithmetic(Gate):
    r"""
    Exact state-preparation version of the generalized Glauber / MH acceptance
    amplitude.

    Register convention:

        A | B | coin

    where A and B each have n qubits, and coin has c qubits.

    For finite a >= 1,

        g(x, y) = (1 / (1 + exp(beta * DeltaE(x, y)))) ** (1 / (2 * a)),

    where

        DeltaE(x, y) = E(y) - E(x)

    and

        E(z) = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.

    For a = np.inf, this switches to the Metropolis-Hastings rule

        g(x, y) = sqrt(min(1, exp(-beta * DeltaE(x, y)))).

    Intended clean-coin action:

        |x>|y>|0...0> -> |x>|y>(
            g(x, y)|0...0> + sqrt(1 - g(x, y)^2)|1...1>
        ).

    No other computational-basis component appears in the coin register.
    """

    def __init__(
        self,
        n: int,
        c: int,
        h: np.ndarray,
        J: np.ndarray,
        beta: float,
        a: float = 1.0,
        label=None,
        tol: float = 1e-12,
    ) -> None:
        if n < 1:
            raise ValueError("n must be positive.")
        if c < 1:
            raise ValueError("c must be positive.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if not (np.isinf(a) or a >= 1.0):
            raise ValueError("a must satisfy a >= 1 or a = np.inf.")

        assert h.shape == (n,)
        assert J.shape == (n, n)

        self.n = int(n)
        self.c = int(c)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.beta = float(beta)
        self.a = float(a)
        self.tol = float(tol)

        name = "GlauberStateprepArithmetic" if not np.isinf(self.a) else "MHStateprepArithmetic"

        super().__init__(
            name=name,
            num_qubits=2 * self.n + self.c,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        A = list(range(0, self.n))
        B = list(range(self.n, 2 * self.n))
        coin = list(range(2 * self.n, 2 * self.n + self.c))

        return {
            "A": A,
            "B": B,
            "coin": coin,
            "control": coin,
            "system": A + B,
            "aux": [],
        }

    @staticmethod
    def _bits(index: int, n: int) -> np.ndarray:
        """
        Big-endian bits matching ket(index, n).

        For n=2:
            0 -> |00> -> [0, 0]
            1 -> |01> -> [0, 1]
            2 -> |10> -> [1, 0]
            3 -> |11> -> [1, 1]
        """
        return np.array([(index >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)

    def _energy(self, bits: np.ndarray) -> float:
        z = 1 - 2 * bits
        value = float(np.dot(self.h, z))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                value += float(self.J[i, j]) * int(z[i]) * int(z[j])

        return value

    def _delta_energy(self, x: int, y: int) -> float:
        return self._energy(self._bits(y, self.n)) - self._energy(self._bits(x, self.n))

    def _amplitude(self, x: int, y: int) -> float:
        delta = self._delta_energy(x, y)
        log_r = -self.beta * delta

        if np.isinf(self.a):
            log_acceptance = min(0.0, log_r)
        else:
            log_acceptance = log_r - np.logaddexp(0.0, self.a * log_r) / self.a

        amp = np.exp(0.5 * log_acceptance)
        return float(np.clip(amp, 0.0, 1.0))

    def _stateprep(self, x: int, y: int) -> StatePreparation:
        amp0 = self._amplitude(x, y)
        amp1 = np.sqrt(max(0.0, 1.0 - amp0 * amp0))

        state = np.zeros(2**self.c, dtype=complex)
        state[0] = amp0
        state[-1] = amp1

        return StatePreparation(state)

    @staticmethod
    def _apply_controlled_stateprep_on_basis(
        qc: QuantumCircuit,
        controls: list[int],
        coin: list[int],
        basis_bits: list[int],
        prep: StatePreparation,
    ) -> None:
        for q, bit in zip(controls, basis_bits):
            if bit == 0:
                qc.x(q)

        qc.append(prep.control(num_ctrl_qubits=len(controls)), controls + coin)

        for q, bit in reversed(list(zip(controls, basis_bits))):
            if bit == 0:
                qc.x(q)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        A = self.layout["A"]
        B = self.layout["B"]
        coin = self.layout["coin"]
        controls = A + B

        for x in range(2**self.n):
            x_bits = self._bits(x, self.n).tolist()

            for y in range(2**self.n):
                y_bits = self._bits(y, self.n).tolist()
                basis_bits = x_bits + y_bits

                self._apply_controlled_stateprep_on_basis(
                    qc,
                    controls,
                    coin,
                    basis_bits,
                    self._stateprep(x, y),
                )

        return qc