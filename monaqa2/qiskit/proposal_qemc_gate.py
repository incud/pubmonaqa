import numpy as np
import scipy.linalg as la
from qiskit.circuit import QuantumCircuit, Gate

from monaqa2.mcmc.utils_hamiltonian import energies, get_cached_mixing_hamiltonian
from monaqa2.qiskit.hamiltonian_simulation_gqsp_gate import HamiltonianSimulationGQSP
from monaqa2.qiskit.trotterized_ising_tf_gate import TrotterizedOperatorIsingTF


class ProposalQemc(Gate):
    r"""
    Proposal gate for QEMC.

    Acts as

        |x>_A |0>_B |0>_aux -> |x>_A U(t)|x>_B |0>_aux

    by copying A into B, then applying one of:

        evolution="gqsp":    HamiltonianSimulationGQSP on B plus auxiliaries;
        evolution="trotter": TrotterizedOperatorIsingTF directly on B;
        evolution="exact":   dense expm evolution directly on B.

    For evolution="gqsp" and evolution="trotter", gamma must be an array of shape (n,)
    and the Hamiltonian is

        H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + sum_i gamma_i X_i.

    For evolution="exact", gamma must be a scalar and the Hamiltonian is

        H = (1 - gamma) diag(E) + gamma H_mixer,

    where E is computed from h, J and H_mixer = sum_i X_i.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        gamma: np.ndarray | float,
        t: float,
        eps: float | None = None,
        evolution: str = "gqsp",
        num_trotter_steps: int | None = None,
        mocked_reflection: bool = False,
        mocked_angles: bool = False,
        atol: float = 1e-8,
        label=None,
    ) -> None:
        if n < 1:
            raise ValueError("`n` must be a positive integer.")
        if evolution not in {"gqsp", "trotter", "exact"}:
            raise ValueError("evolution must be one of 'gqsp', 'trotter', or 'exact'.")
        if eps is not None and eps <= 0.0:
            raise ValueError("eps must be positive when provided.")
        if evolution == "gqsp" and eps is None:
            raise ValueError("eps must be provided for evolution='gqsp'.")
        if evolution == "trotter" and eps is None and num_trotter_steps is None:
            raise ValueError("Pass either eps or num_trotter_steps for evolution='trotter'.")

        h = np.asarray(h, dtype=float)
        J = np.asarray(J, dtype=float)

        if h.shape != (n,):
            raise ValueError(f"h must have shape ({n},), got {h.shape}.")
        if J.shape != (n, n):
            raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}.")

        self.n = int(n)
        self.h = h
        self.J = J
        self.t = float(t)
        self.eps = None if eps is None else float(eps)
        self.evolution = str(evolution)
        self.num_trotter_steps = num_trotter_steps
        self.mocked_reflection = bool(mocked_reflection)
        self.mocked_angles = bool(mocked_angles)
        self.atol = float(atol)

        

        if self.evolution == "exact":
            gamma_arr = np.asarray(gamma, dtype=float)
            if gamma_arr.ndim != 0:
                raise ValueError("For evolution='exact', gamma must be a scalar mixing parameter.")
            self.gamma = float(gamma_arr)
            self.hsim = None
            self.n_aux = 0
        else:
            if isinstance(gamma, float):
                gamma_arr = np.array([gamma] * n, dtype=float)
            else:
                gamma_arr = np.asarray(gamma, dtype=float)
            if gamma_arr.shape != (n,):
                raise ValueError(f"For evolution='{self.evolution}', gamma must have shape ({n},), got {gamma_arr.shape}.")
            self.gamma = gamma_arr
            self.hsim = self._build_evolution_gate()
            self.n_aux = self.hsim.num_qubits - self.n

        self.name = f"ProposalQemc(n={self.n},evolution={self.evolution})"

        super().__init__(self.name, 2 * self.n + self.n_aux, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {
            "A": list(range(self.n)),
            "B": list(range(self.n, 2 * self.n)),
            "aux": list(range(2 * self.n, self.num_qubits)),
        }

    def _build_evolution_gate(self) -> Gate:
        if self.evolution == "gqsp":
            return HamiltonianSimulationGQSP(self.n, self.h, self.J, self.gamma, self.t, self.eps, mocked_reflection=self.mocked_reflection, mocked_angles=self.mocked_angles)

        if self.evolution == "trotter":
            return TrotterizedOperatorIsingTF(self.n, self.h, self.J, self.gamma, time=self.t, eps=self.eps, num_trotter_steps=self.num_trotter_steps, atol=self.atol)

        raise RuntimeError("_build_evolution_gate should not be called for evolution='exact'.")

    def _exact_hamiltonian(self) -> np.ndarray:
        H_ising = np.diag(energies(self.h, self.J))
        H_mixer = get_cached_mixing_hamiltonian(self.n)
        return (1.0 - self.gamma) * H_ising + self.gamma * H_mixer

    def _hsim_qubits(self) -> list[int]:
        if self.evolution == "trotter":
            return self.layout["B"]

        if self.evolution != "gqsp":
            raise RuntimeError("_hsim_qubits is only used for evolution='gqsp' or evolution='trotter'.")

        hsim_qubits = [None] * self.hsim.num_qubits
        system = [1 + q for q in self.hsim.qubitization.layout["system"]]

        for local_q, global_q in zip(system, self.layout["B"]):
            hsim_qubits[local_q] = global_q

        aux = iter(self.layout["aux"])
        return [next(aux) if q is None else q for q in hsim_qubits]

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        for a, b in zip(self.layout["A"], self.layout["B"]):
            qc.cx(a, b)

        if self.evolution == "exact":
            # Big endianness: need to reverse the stuff
            qc.unitary(la.expm(-1j * self.t * self._exact_hamiltonian()), list(reversed(self.layout["B"])))
            # Wrong: qc.unitary(la.expm(-1j * self.t * self._exact_hamiltonian()), self.layout["B"])
        else:
            qc.append(self.hsim, self._hsim_qubits())

        return qc
    