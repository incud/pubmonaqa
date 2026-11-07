from qiskit.circuit import Gate, QuantumCircuit
from monaqa2.qiskit2.ccx import Ccx
from monaqa2.qiskit2.reversible_fanout import ReversibleFanout


class ReversibleConditionalLoader(Gate):
    """Load q_l*(parity_l(x)-parity_l(y)): each word is 0, +q_l, or -q_l.

    Coefficients are integers: fields first, then the upper-triangular couplings.
    Field parity is x_i; pair parity is x_i XOR x_j (likewise for y).
    This compares arbitrary full states, without a single-spin restriction.
    Every selector has W copies, so word writes take constant depth after fanout.
    Inputs are preserved, output words are XOR-loaded, and all work is cleaned.
    """

    def __init__(self, n: int, coefficients, W: int, label=None):
        self.n, self.W = int(n), int(W)
        self.M = self.n * (self.n + 1) // 2
        self.coefficients = [int(value) for value in coefficients]
        if min(self.n, self.W) < 1 or len(self.coefficients) != self.M:
            raise ValueError("Require n,W > 0 and n*(n+1)//2 coefficients.")
        if any(abs(value) >= 2**(self.W - 1) for value in self.coefficients):
            raise ValueError("Both coefficient signs must fit the signed word.")
        self.scratch = 2 * self.n * (self.n - 1) + 4 * self.M + 2 * self.M * (self.W - 1)
        super().__init__("reversible_conditional_loader", 2 * self.n + self.M * self.W + self.scratch, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        prepare = QuantumCircuit(self.num_qubits)
        x, y, terms, work = (self.layout[key] for key in ("x", "y", "terms", "work"))
        n, M, W = self.n, self.M, self.W
        end_copies = 2 * n * (n - 1)
        copies = [[source] + work[i * (n - 1):(i + 1) * (n - 1)] for i, source in enumerate(x + y)]
        for wires in copies:
            prepare.append(ReversibleFanout(n), wires)
        px = work[end_copies:end_copies + M]
        py = work[end_copies + M:end_copies + 2 * M]
        positive = work[end_copies + 2 * M:end_copies + 3 * M]
        negative = work[end_copies + 3 * M:end_copies + 4 * M]
        selector_work = work[end_copies + 4 * M:]

        # Independent copies feed every field and pair parity in parallel.
        prepare.cx(x, px[:n])
        prepare.cx(y, py[:n])
        used, term = [1] * n, n
        for i in range(n):
            for j in range(i + 1, n):
                prepare.cx(copies[i][used[i]], px[term])
                prepare.cx(copies[j][used[j]], px[term])
                prepare.cx(copies[n + i][used[i]], py[term])
                prepare.cx(copies[n + j][used[j]], py[term])
                used[i] += 1
                used[j] += 1
                term += 1

        for term in range(M):
            # Positive = px AND NOT py; negative = py AND NOT px.
            prepare.cx(px[term], negative[term])
            prepare.cx(py[term], negative[term])
            prepare.append(Ccx(), [px[term], negative[term], positive[term]])
            prepare.cx(positive[term], negative[term])
        selectors = [[source] + selector_work[i * (W - 1):(i + 1) * (W - 1)]
                     for i, source in enumerate(positive + negative)]
        for wires in selectors:
            prepare.append(ReversibleFanout(W), wires)

        qc.compose(prepare, inplace=True)
        for term, value in enumerate(self.coefficients):
            for bit in range(W):
                target = terms[term * W + bit]
                if (value >> bit) & 1:
                    qc.cx(selectors[term][bit], target)
                if ((-value) >> bit) & 1:
                    qc.cx(selectors[M + term][bit], target)
        qc.compose(prepare.inverse(), inplace=True)
        return qc

    @property
    def layout(self):
        end_inputs, end_terms = 2 * self.n, 2 * self.n + self.M * self.W
        return {"x": list(range(self.n)), "y": list(range(self.n, end_inputs)),
                "terms": list(range(end_inputs, end_terms)), "work": list(range(end_terms, self.num_qubits))}
