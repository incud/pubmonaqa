import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit.primitives import Cry, Ccry



class SCS(Gate):
    r"""
    Split-and-Cyclic-Shift gate ``SCS(n, k)``.

    This gate is the elementary local block used in the Bärtschi-Eidenbenz
    Dicke-state preparation construction. It is parameterized by ``n`` and ``k``,
    but it acts on only ``k + 1`` qubits.

    On the unary basis states
    ``|0^{k+1-\ell} 1^\ell>`` for ``\ell = 1, ..., k``, it acts as
    """

    def __init__(self, n: int, k: int, label=None) -> None:
        """
        Constructor.

        :param n: Global SCS size parameter.
        :param k: Hamming-weight parameter of the SCS block.
        :param label: Optional gate label.
        :return: None.
        """
        if n < 2:
            raise ValueError("SCS requires n >= 2.")
        if not (1 <= k < n):
            raise ValueError(f"SCS requires 1 <= k < n, got k={k}, n={n}.")

        self.n = int(n)
        self.k = int(k)

        super().__init__(
            name="SCS",
            num_qubits=self.k + 1,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @staticmethod
    def _gate_i(qc: QuantumCircuit, n: int, wires: list[int]) -> None:
        """
        Apply the type-I two-qubit SCS block.

        :param qc: Circuit to modify.
        :param n: Global SCS size parameter.
        :param wires: Two local qubits on which the block acts.
        :return: None.
        """
        qc.cx(wires[0], wires[1])
        theta = 2.0 * float(np.arccos(np.sqrt(1.0 / n)))
        qc.append(Cry(theta), [wires[1], wires[0]])
        qc.cx(wires[0], wires[1])

    @staticmethod
    def _gate_ii_l(qc: QuantumCircuit, l: int, n: int, wires: list[int]) -> None:
        """
        Apply the type-II three-qubit SCS block.

        :param qc: Circuit to modify.
        :param l: Intermediate SCS parameter.
        :param n: Global SCS size parameter.
        :param wires: Three local qubits [target, second control, first control].
        :return: None.
        """
        qc.cx(wires[0], wires[2])
        theta = 2.0 * float(np.arccos(np.sqrt(float(l) / n)))
        qc.append(Ccry(theta), [wires[2], wires[1], wires[0]])
        qc.cx(wires[0], wires[2])

    def _build_definition(self) -> QuantumCircuit:
        """
        Build the local SCS(n, k) block on `k + 1` qubits.

        Local qubit ordering is `[0, 1, ..., k]`.

        :return: Circuit implementing SCS(n, k).
        """
        qc = QuantumCircuit(self.num_qubits, name=f"SCS_{self.n}_{self.k}")
        wires = list(range(self.num_qubits))

        self._gate_i(qc, self.n, [wires[self.k - 1], wires[self.k]])

        for l in range(2, self.k + 1):
            self._gate_ii_l(
                qc,
                l,
                self.n,
                [wires[self.k - l], wires[self.k - l + 1], wires[self.k]],
            )

        return qc
    


class WDB(Gate):
    r"""
    Weight Distribution Block :math:`\mathrm{WDB}^{n,m}_k`.

    Let

    * ``A = q[0 : n-m]`` be the first register, and
    * ``B = q[n-m : n]`` be the second register.

    This implementation assumes :math:`k \le n-m`, so the input unary weight
    fits entirely inside ``A``. The second register may be truncated when
    :math:`m < k`, exactly as described in the paper.
    """

    def __init__(self, n: int, m: int, k: int, label=None) -> None:
        """
        Constructor.

        :param n: Total number of qubits in the block.
        :param m: Size of the right child block.
        :param k: Leaf-size / active-prefix parameter.
        :param label: Optional gate label.
        :return: None.
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        if not (0 < m < n):
            raise ValueError(f"m must satisfy 0 < m < n, got m={m}, n={n}.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if k > n - m:
            raise ValueError(
                "This WDB construction assumes k <= n - m so that the input "
                "unary weight fits inside the first register."
            )

        self.n = int(n)
        self.m = int(m)
        self.k = int(k)

        super().__init__(
            name="WDB",
            num_qubits=self.n,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        """
        Register layout for the WDB block.

        :return: A mapping with left and right child block indices.
        """
        left = list(range(0, self.n - self.m))
        right = list(range(self.n - self.m, self.n))
        return {"left": left, "right": right}

    @staticmethod
    def _compute_wdb_angles(n: int, m: int, ell: int) -> np.ndarray:
        """
        Compute the controlled-addition rotation angles.

        :param n: Total block size.
        :param m: Right-child size.
        :param ell: Weight parameter.
        :return: Array of rotation angles.
        """
        x = np.array([math.comb(m, i) * math.comb(n - m, ell - i) if (ell - i) >= 0 else 0 for i in range(ell + 1)], dtype=float)
        s = np.cumsum(x[::-1])[::-1]
        ratio = np.divide(x, s, out=np.ones_like(s), where=s > 1e-15)
        theta = 2.0 * np.arccos(np.sqrt(np.clip(ratio, 0.0, 1.0)))
        return np.where(s > 0.0, theta, 0.0)

    @staticmethod
    def _cnot_ladder(qc: QuantumCircuit, register: list[int], reverse: bool = False) -> None:
        """
        Convert unary to one-hot or invert the conversion by a CNOT ladder.

        :param qc: Circuit to modify.
        :param register: Register to encode/decode.
        :param reverse: If True, perform the inverse ladder.
        :return: None.
        """
        if reverse:
            for i in reversed(range(1, len(register))):
                qc.cx(register[i], register[i - 1])
        else:
            for i in range(len(register) - 1):
                qc.cx(register[i + 1], register[i])

    @classmethod
    def _controlled_addition(
        cls,
        qc: QuantumCircuit,
        reg_a: list[int],
        reg_b: list[int],
        n: int,
        m: int,
    ) -> None:
        """
        Perform the controlled weighted-addition step.

        :param qc: Circuit to modify.
        :param reg_a: Active left register.
        :param reg_b: Active right register.
        :param n: Total block size.
        :param m: Right-child size.
        :return: None.
        """
        for ell in range(len(reg_a) - 1, -1, -1):
            angles = cls._compute_wdb_angles(n, m, ell + 1)
            for j in range(min(len(reg_b), ell + 1)):
                ctrls = [reg_a[ell]]
                if j > 0:
                    ctrls.append(reg_b[j - 1])

                theta = float(angles[j])
                if abs(theta) < 1e-15:
                    continue

                if len(ctrls) == 1:
                    qc.append(Cry(theta), [ctrls[0], reg_b[j]])
                elif len(ctrls) == 2:
                    qc.append(Ccry(theta), [ctrls[0], ctrls[1], reg_b[j]])
                else:
                    raise AssertionError("Never")

    @staticmethod
    def _fredkin_stair(
        qc: QuantumCircuit,
        reg_a: list[int],
        reg_b: list[int],
    ) -> None:
        """
        Apply the Fredkin staircase subtraction step.

        :param qc: Circuit to modify.
        :param reg_a: Active left register.
        :param reg_b: Active right register.
        :return: None.
        """
        const = 1
        if len(reg_a) == len(reg_b):
            qc.cx(reg_b[-1], reg_a[-1])
            const = 2

        for i in range(len(reg_b) - const, -1, -1):
            for j in range(i, len(reg_a) - 1):
                # 3 x Toffoli
                qc.cswap(reg_b[i], reg_a[j], reg_a[j + 1])
            qc.cx(reg_b[i], reg_a[-1])

    def _build_definition(self) -> QuantumCircuit:
        """
        Build the WDB circuit.

        :return: Circuit implementing the WDB block.
        """
        qc = QuantumCircuit(self.num_qubits, name=f"WDB_{self.n}_{self.m}_{self.k}")

        reg_a = self.layout["left"][: self.k]
        reg_b = self.layout["right"][: self.k]

        # Free
        self._cnot_ladder(qc, reg_a)
        # CRY -> k times (2 x RZ)
        # CCRY -> (\sum_{i=0}^{k-1} i) times (2 x RZ + 2 x Toffoli)
        self._controlled_addition(qc, reg_a, reg_b, self.n, self.m)
        # Free
        self._cnot_ladder(qc, reg_a, reverse=True)
        # \sum_{i=0}^{k-1} \sum_{j=i}^k (3 x Toffoli)
        self._fredkin_stair(qc, reg_a, reg_b)

        return qc
    

class DickePreparation(Gate):
    r"""
    Log-depth all-to-all Dicke-state preparation using the WDB partition-tree
    algorithm with SCS preparation on the leaves.

    The construction follows the public WDB+SCS implementation pattern:
    initialize |1^k 0^{n-k}>, apply WDB blocks over a balanced partition tree,
    then apply the leaf unary-to-Dicke transform built from SCS blocks.
    """

    def __init__(self, n: int, k: int, label=None) -> None:
        """
        Constructor.

        :param n: Number of qubits.
        :param k: Hamming weight.
        :param label: Optional gate label.
        :return: None.
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        if not (0 <= k <= n):
            raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}.")

        self.n = int(n)
        self.k = int(k)

        super().__init__(
            name="DickeStatePreparationLogDepth",
            num_qubits=self.n,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @staticmethod
    def _is_leaf(node: tuple[list[int], object, object]) -> bool:
        return node[1] is None

    def _build_partition_tree(self, qubits: list[int]) -> tuple[list[int], object, object]:
        if len(qubits) <= self.k:
            return qubits, None, None

        chunks = [qubits[i : i + self.k] for i in range(0, len(qubits), self.k)]

        def from_chunks(chunk_list: list[list[int]]):
            if len(chunk_list) == 1:
                return chunk_list[0], None, None

            mid = len(chunk_list) // 2
            left = from_chunks(chunk_list[:mid])
            right = from_chunks(chunk_list[mid:])
            return left[0] + right[0], left, right

        return from_chunks(chunks)

    def _internal_nodes(self, node: tuple[list[int], object, object]):
        if self._is_leaf(node):
            return []
        result = [node]
        result.extend(self._internal_nodes(node[1]))
        result.extend(self._internal_nodes(node[2]))
        return result

    def _leaves(self, node: tuple[list[int], object, object]):
        if self._is_leaf(node):
            return [node]
        result = []
        result.extend(self._leaves(node[1]))
        result.extend(self._leaves(node[2]))
        return result

    @staticmethod
    def _reverse_register(qc: QuantumCircuit, register: list[int]) -> None:
        """
        Reverse qubit order by a SWAP ladder.

        :param qc: Circuit to modify.
        :param register: Register to reverse.
        :return: None.
        """
        for i in range(len(register) // 2):
            qc.swap(register[i], register[len(register) - 1 - i])

    def _build_definition(self) -> QuantumCircuit:
        """
        Build the Dicke-state preparation circuit.

        :return: Circuit implementing the all-to-all log-depth Dicke preparation.
        """
        qc = QuantumCircuit(self.num_qubits, name=f"DickeLog_{self.n}_{self.k}")

        if self.k == 0:
            return qc

        if self.k == self.n:
            for q in range(self.n):
                qc.x(q)
            return qc

        # Initial unary state |1^k 0^{n-k}>.
        for q in range(self.k):
            qc.x(q)

        tree = self._build_partition_tree(list(range(self.n)))

        for qubits, left, right in self._internal_nodes(tree):
            m = len(right[0])
            qc.append(WDB(n=len(qubits), m=m, k=self.k), qubits)

        for qubits, _, _ in self._leaves(tree):

            # The leaf unitary expects the unary state in the opposite ordering.
            self._reverse_register(qc, qubits)

            # Apply the universal unary-to-Dicke transform on the leaf:
            # prod_{i = |leaf|, ..., 2} SCS(i, i-1).
            for i in reversed(range(2, len(qubits) + 1)):
                qc.append(SCS(i, i - 1), qubits[:i])

        return qc


class ProposalLocal(Gate):
    r"""
    This gate acts effectively on the system

        H_A \\otimes H_B 

    as 
    
        P_local |x> |0> = |x> |x \oplus dicke(n, k)>)
    """

    def __init__(self, n: int, k: int, label=None) -> None:
        if n < 1:
            raise ValueError("`n` must be a positive integer.")
        if not (1 <= k <= n):
            raise ValueError("`k` must satisfy 1 <= k <= n.")

        self.n = int(n)
        self.k = int(k)
        self.name = f"ProposalLocal(n={self.n},k={self.k})"
        super().__init__(
            name=self.name,
            num_qubits=2 * self.n,
            params=[],
            label=label,
        )
        self.definition = self._build_definition()

    @property
    def layout(self) -> dict[str, list[int]]:
        return {
            "A": list(range(0, self.n)),
            "B": list(range(self.n, 2 * self.n)),
        }

    def _build_definition(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.append(DickePreparation(self.n, self.k), self.layout["B"])
        for a, b in zip(self.layout["A"], self.layout["B"]):
            qc.cx(a, b)
        return qc
