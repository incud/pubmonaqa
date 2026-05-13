import numpy as np
from qiskit.circuit import Gate, QuantumCircuit

from monaqa2.qiskit.prepare_unary_gate import PrepareUnary


class LcuUnaryIsingTF(Gate):
    """
    Unary LCU block encoding for the transverse-field Ising Hamiltonian

        H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + sum_i gamma_i X_i.

    Only terms with abs(coefficient) > tol are instantiated.

    It implements

        <0_anc| U |0_anc> = H / alpha,

    where

        alpha = sum_l |a_l|

    over the active terms.
    """

    def __init__(
        self,
        n: int,
        h: np.ndarray,
        J: np.ndarray,
        gamma: np.ndarray,
        label=None,
        tol: float = 1e-8,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")

        assert h.shape == (n,)
        assert J.shape == (n, n)
        assert gamma.shape == (n,)

        self.n = int(n)
        self.h = np.asarray(h, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.tol = float(tol)

        self.terms = self._build_terms()
        self.coeffs = np.array([coeff for coeff, _ in self.terms], dtype=float)
        self.magnitudes = np.abs(self.coeffs)
        self.alpha = float(np.sum(self.magnitudes))
        self.n_terms = len(self.terms)
        self.n_tree = max(self.n_terms - 1, 0)

        if self.n_terms < 2:
            raise ValueError("LcuUnaryIsingTF requires at least two active term.")
        if self.alpha <= 0.0:
            raise ValueError("LCU normalization alpha is zero.")

        super().__init__(
            name="LcuUnary",
            num_qubits=self.n + self.n_tree + self.n_terms,
            params=[],
            label=label,
        )

        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {
            "system": list(range(self.n)),
            "tree": list(range(self.n, self.n + self.n_tree)),
            "prepare": list(range(self.n + self.n_tree, self.num_qubits)),
        }

    def _is_active(self, coeff: float) -> bool:
        return abs(float(coeff)) > self.tol

    def _build_terms(self) -> list[tuple[float, tuple]]:
        terms = []

        for i in range(self.n):
            if self._is_active(self.h[i]):
                terms.append((self.h[i], ("Z", i)))

        for i in range(self.n):
            if self._is_active(self.gamma[i]):
                terms.append((self.gamma[i], ("X", i)))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self._is_active(self.J[i, j]):
                    terms.append((self.J[i, j], ("ZZ", i, j)))

        return terms

    def _term_index(self) -> dict[tuple, int]:
        return {pauli: i for i, (_, pauli) in enumerate(self.terms)}

    @staticmethod
    def _matchings(n: int) -> list[list[tuple[int, int]]]:
        vertices = list(range(n))
        dummy = n if n % 2 else None

        if dummy is not None:
            vertices.append(dummy)

        rounds = []
        m = len(vertices)

        for _ in range(m - 1):
            matching = []

            for a, b in zip(vertices[: m // 2], reversed(vertices[m // 2 :])):
                if a != dummy and b != dummy:
                    matching.append((min(a, b), max(a, b)))

            rounds.append(matching)
            vertices = [vertices[0], vertices[-1], *vertices[1:-1]]

        return rounds

    def _apply_signs(self, qc: QuantumCircuit, prepare: list[int]) -> None:
        for q, coeff in zip(prepare, self.coeffs):
            if coeff < 0.0:
                qc.z(q)

    def _select_z_terms(
        self,
        qc: QuantumCircuit,
        system: list[int],
        prepare: list[int],
        term_index: dict[tuple, int],
    ) -> None:
        for i in range(self.n):
            pauli = ("Z", i)

            if pauli in term_index:
                qc.cz(prepare[term_index[pauli]], system[i])

    def _select_x_terms(
        self,
        qc: QuantumCircuit,
        system: list[int],
        prepare: list[int],
        term_index: dict[tuple, int],
    ) -> None:
        for i in range(self.n):
            pauli = ("X", i)

            if pauli in term_index:
                qc.cx(prepare[term_index[pauli]], system[i])

    def _select_zz_terms(
        self,
        qc: QuantumCircuit,
        system: list[int],
        prepare: list[int],
        term_index: dict[tuple, int],
    ) -> None:
        for matching in self._matchings(self.n):
            active_edges = [
                (i, j)
                for i, j in matching
                if ("ZZ", i, j) in term_index
            ]

            for i, j in active_edges:
                q = prepare[term_index[("ZZ", i, j)]]
                qc.cz(q, system[i])

            for i, j in active_edges:
                q = prepare[term_index[("ZZ", i, j)]]
                qc.cz(q, system[j])

    def _select(
        self,
        qc: QuantumCircuit,
        system: list[int],
        prepare: list[int],
    ) -> None:
        term_index = self._term_index()

        self._apply_signs(qc, prepare)
        self._select_z_terms(qc, system, prepare, term_index)
        self._select_x_terms(qc, system, prepare, term_index)
        self._select_zz_terms(qc, system, prepare, term_index)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        system = self.layout["system"]
        tree = self.layout["tree"]
        prepare = self.layout["prepare"]

        prepare_gate = PrepareUnary(self.magnitudes)

        qc.append(prepare_gate, tree + prepare)
        self._select(qc, system, prepare)
        qc.append(prepare_gate.inverse(), tree + prepare)

        return qc