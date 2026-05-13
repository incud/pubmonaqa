import numpy as np
from qiskit.circuit import Gate, QuantumCircuit


class PrepareUnary(Gate):
    """
    Prepare a one-hot unary amplitude state from the all-zero state. This gate acts on:

        H_tree ⊗ H_leaf

    with layout

        tree: internal routing qubits, length n_terms - 1
        leaf: one-hot output qubits, length n_terms

    and implements

        U_prepare |0...0>_tree |0...0>_leaf
            =
        |0...0>_tree
            ⊗
        sum_l sqrt(abs(coeffs[l]) / alpha) |l>_unary,

    where

        alpha = sum_l abs(coeffs[l]).

    The tree qubits are temporary routing qubits and are returned to |0...0>.
    """

    def __init__(self, coeffs: np.ndarray, label=None, tol: float = 1e-8) -> None:
    
        # Validate coeffs
        coeffs = np.asarray(coeffs).reshape(-1)
        if np.iscomplexobj(coeffs): 
            coeffs = np.real_if_close(coeffs)
        if np.iscomplexobj(coeffs): 
            raise ValueError("PrepareUnary coefficients must be real.")
        if np.any(coeffs < 0): 
            raise ValueError("PrepareUnary coefficients must be non-negative.")

        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        active = np.flatnonzero(np.abs(coeffs) > tol)

        if len(active) == 0:
            raise ValueError("PrepareUnary requires at least one nonzero coefficient.")

        self.original_coeffs = coeffs
        self.active_indices = active
        self.coeffs = coeffs[active]
        self.weights = self.coeffs
        self.alpha = float(np.sum(self.weights))
        self.n_terms = len(self.coeffs)
        self.n_tree = max(self.n_terms - 1, 0)

        self.root = self._tree(0, self.n_terms, [0])

        super().__init__("PrepareUnary", self.n_tree + self.n_terms, [], label=label)
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {
            "tree": list(range(self.n_tree)),
            "leaf": list(range(self.n_tree, self.n_tree + self.n_terms)),
        }

    def _leaf(self, i: int) -> int:
        return self.n_tree + i

    def _weight(self, lo: int, hi: int) -> float:
        return float(np.sum(self.weights[lo:hi]))

    def _tree(self, lo: int, hi: int, next_q: list[int]) -> dict:
        if hi - lo == 1:
            return {
                "leaf": True,
                "q": self._leaf(lo),
                "w": self._weight(lo, hi),
            }

        q = next_q[0]
        next_q[0] += 1
        mid = (lo + hi) // 2
        left = self._tree(lo, mid, next_q)
        right = self._tree(mid, hi, next_q)

        return {
            "leaf": False,
            "q": q,
            "left": left,
            "right": right,
            "w": left["w"] + right["w"],
        }

    def _levels(self) -> list[list[dict]]:
        levels = []

        def visit(node: dict, depth: int) -> None:
            if node["leaf"]:
                return

            if depth == len(levels):
                levels.append([])

            levels[depth].append(node)
            visit(node["left"], depth + 1)
            visit(node["right"], depth + 1)

        visit(self.root, 0)
        return levels

    @staticmethod
    def _theta(left_w: float, right_w: float) -> float:
        return 2.0 * np.arctan2(np.sqrt(right_w), np.sqrt(left_w))

    def _split(self, qc: QuantumCircuit, nodes: list[dict]) -> None:
        for node in nodes:
            p = node["q"]
            l = node["left"]["q"]
            r = node["right"]["q"]
            theta = self._theta(node["left"]["w"], node["right"]["w"])

            qc.cry(theta, p, r)
            qc.cx(p, l)
            qc.cx(r, l)
            qc.cx(l, p)
            qc.cx(r, p)

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        if self.n_terms == 1:
            qc.x(self.layout["leaf"][0])
            return qc

        qc.x(self.root["q"])

        for nodes in self._levels():
            self._split(qc, nodes)

        return qc