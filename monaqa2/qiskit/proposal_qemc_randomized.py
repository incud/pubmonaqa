"""Coherent midpoint-grid approximation to the randomized QEMC proposal."""

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit


class ProposalQemcRandomized(Gate):
    r"""Coherently prepare the randomized Hamiltonian-simulation proposal.

    For 2**k midpoint values of each parameter, the gate acts as

        |x>_A |0>_B |0>_time |0>_gamma |0>_work
          -> (1/2**k) sum_{j,l} |x>_A U_r(t_j,gamma_l)|x>_B
                               |j>_time |l>_gamma |0>_work.

    To agree with monaqa2.mcmc.proposal.create_proposal_matrix_quantum_exact,
    H(gamma) = (1-gamma) H_Ising + gamma H_X. Here H_Ising is the diagonal
    matrix of normalized energies E(x) = -sum_i h_i x_i - sum_{i<j} J_ij x_i x_j,
    and H_X = sum_i X_i. U_r uses r steps of the X/2--Ising--X/2 formula.

    Each (j,l) occupies a different orthogonal register state. 
    """

    def __init__(self, n: int, h: np.ndarray, J: np.ndarray, k: int,
                 time_lims: tuple[float, float] = (2.0, 20.0),
                 gamma_lims: tuple[float, float] = (0.25, 0.60),
                 num_trotter_steps: int = 1, atol: float = 1e-8, label=None) -> None:
        """Build the uniform grid and the coherent proposal gate.

        :param n: Number of spins.
        :param h: Local Ising fields.
        :param J: Ising coupling matrix.
        :param k: Bits per parameter grid.
        :param time_lims: Time interval endpoints.
        :param gamma_lims: Mixing interval endpoints.
        :param num_trotter_steps: Product-formula step count.
        :param atol: Ising term cutoff.
        :param label: Qiskit display label.
        :return: None.
        """
        if n < 1 or k < 0 or num_trotter_steps < 1:
            raise ValueError("Require n > 0, k >= 0, and num_trotter_steps > 0.")
        self.n, self.k, self.num_trotter_steps = int(n), int(k), int(num_trotter_steps)
        h, J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
        self.atol = float(atol)
        if h.shape != (n,) or J.shape != (n, n):
            raise ValueError("h and J must have shapes (n,) and (n,n).")
        # mcmc.utils_hamiltonian.energies uses IsingModel's normalization and
        # the negative Ising sign. Use the same coefficients in our Z rotations.
        norm = np.dot(h, h) + np.sum(np.triu(J, 1) ** 2)
        if norm <= 0:
            raise ValueError("The Ising coefficients cannot all be zero.")
        self.h, self.J = h, J
        t0, t1 = time_lims
        g0, g1 = gamma_lims
        if t1 <= t0 or g1 <= g0:
            raise ValueError("Parameter ranges must have increasing endpoints.")
        # Grid cell widths; each parameter is placed at the center of its cell.
        dt, dg = (t1 - t0) / 2**k, (g1 - g0) / 2**k
        self.times = t0 + (np.arange(2**k) + 0.5) * dt
        self.gammas = g0 + (np.arange(2**k) + 0.5) * dg
        self.z_terms = [(i, h_i) for i, h_i in enumerate(self.h) if abs(h_i) >= atol]
        self.zz_terms = [(i, j, self.J[i, j]) for i in range(n)
                         for j in range(i + 1, n) if abs(self.J[i, j]) >= atol]

        # 2n system qubits + 2k parameter bits, each with n-1 work copies.
        super().__init__("ProposalQemcRandomized", 2 * n + 2 * k * n, [], label=label)
        self.definition = self._build_definition(dt, dg)

    @property
    def layout(self) -> dict[str, list[int]]:
        """Give the qubit indices of each register.

        :return: Indices for A, B, time, gamma, and clean work qubits.
        """
        n, k = self.n, self.k
        return {"A": list(range(n)), "B": list(range(n, 2*n)),
                "time": list(range(2*n, 2*n+k)),
                "gamma": list(range(2*n+k, 2*n+2*k)),
                "work": list(range(2*n+2*k, self.num_qubits))}

    @staticmethod
    def _matchings(n: int) -> list[list[tuple[int, int]]]:
        """Partition all Ising edges into rounds of disjoint pairs.

        :param n: Number of vertices (spins).
        :return: A list of matchings covering every pair once.
        """
        # A dummy vertex makes the usual round-robin schedule work for odd n.
        vertices = list(range(n)) + ([None] if n % 2 else [])
        rounds = []
        for _ in range(len(vertices) - 1):
            rounds.append([tuple(sorted((a, b))) for a, b in
                           zip(vertices[:len(vertices)//2], reversed(vertices[len(vertices)//2:]))
                           if a is not None and b is not None])
            vertices = [vertices[0], vertices[-1], *vertices[1:-1]]
        return rounds

    @staticmethod
    def _rotate(qc: QuantumCircuit, axis: str, angle: float, target: int,
                controls: tuple[int, ...]) -> None:
        """Apply an X or Z rotation controlled by at most two qubits.

        :param qc: Circuit receiving the rotation.
        :param axis: Either "x" or "z".
        :param angle: Rotation angle in radians.
        :param target: Target qubit index.
        :param controls: Zero, one, or two positive control qubit indices.
        :return: None.
        """
        # The expansion of t or t*gamma produces at most two controls per term.
        if axis == "z":
            if not controls:
                qc.rz(angle, target)
            elif len(controls) == 1:
                qc.crz(angle, controls[0], target)
            else:
                qc.mcrz(angle, list(controls), target)
        elif axis == "x":
            if not controls:
                qc.rx(angle, target)
            elif len(controls) == 1:
                qc.crx(angle, controls[0], target)
            else:
                qc.mcrx(angle, list(controls), target)
        else:
            raise ValueError("axis must be 'x' or 'z'.")

    def _build_definition(self, dt: float, dg: float) -> QuantumCircuit:
        """Prepare grid bits, run the controlled simulation, and clear work.

        :param dt: Distance between neighboring time midpoints.
        :param dg: Distance between neighboring gamma midpoints.
        :return: Circuit implementing the full proposal gate.
        """
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        a, b = self.layout["A"], self.layout["B"]
        parameter_bits = self.layout["time"] + self.layout["gamma"]
        # copies[bit][slot] chooses a distinct control line for each concurrent
        # single-spin rotation or ZZ edge in a matching. Index 0 is the bit itself.
        copies = {bit: [bit] + self.layout["work"][p*(self.n-1):(p+1)*(self.n-1)]
                  for p, bit in enumerate(parameter_bits)}

        # First create |x>_A|x>_B. Hadamards give every (j,l) amplitude 1/2**k.
        for source, target in zip(a, b):
            qc.cx(source, target)
        for bit in parameter_bits:
            qc.h(bit)
            # This CNOT fanout makes entangled controls; it does not clone a
            # free-standing unknown state. It is undone after all rotations.
            for copy in copies[bit][1:]:
                qc.cx(bit, copy)

        # t0 and g0 denote the FIRST MIDPOINTS, not the interval endpoints.
        # On |j> and |l>, let j_p,l_q be their binary digits. Then
        # t_j=t0+sum_p dt*2**p*j_p and gamma_l=g0+sum_q dg*2**q*l_q.
        # Each (weight, bits) below contributes weight only when all bits are 1.
        t0, g0 = self.times[0], self.gammas[0]
        # First expand time once. All Z, ZZ and X angles reuse these weights.
        time = [(t0, ())] + [(dt * 2**p, (bit,))
                             for p, bit in enumerate(self.layout["time"])]
        # X angles depend on t_j*gamma_l. Expanding the product gives:
        # t0*g0 + (time bits)*g0 + t0*(gamma bits) + (time bits)*(gamma bits).
        # The final terms need two controls, one from each parameter register.
        mixer = [(weight*g0, controls) for weight, controls in time]
        mixer += [(t0*dg*2**p, (bit,)) for p, bit in enumerate(self.layout["gamma"])]
        mixer += [(dt*dg*2**(p+q), (tbit, gbit))
                  for p, tbit in enumerate(self.layout["time"])
                  for q, gbit in enumerate(self.layout["gamma"])]
        # Ising uses t_j*(1-gamma_l): change gamma coefficients' signs.
        ising = [(weight*(1-g0), bits) for weight, bits in time]
        ising += [(-t0*dg*2**p, (bit,)) for p, bit in enumerate(self.layout["gamma"])]
        ising += [(-dt*dg*2**(p+q), (tbit, gbit))
                  for p, tbit in enumerate(self.layout["time"])
                  for q, gbit in enumerate(self.layout["gamma"])]
        coeff_by_edge = {(i, j): coeff for i, j, coeff in self.zz_terms}
        r = self.num_trotter_steps

        def rotate(axis, coeff, target, terms, slot, fraction=1.0):
            """Append rotations for a polynomial in the grid bits.

            :param axis: Rotation axis, "x" or "z".
            :param coeff: Pauli coefficient, including the Ising sign.
            :param target: System qubit index.
            :param terms: (angle weight, activating grid bits) pairs.
            :param slot: Which fanout copy controls this system rotation.
            :param fraction: Half-step factor for the mixer layers.
            :return: None.
            """
            for weight, bits in terms:
                # Select the fanout lines assigned to this spin or matching edge.
                controls = tuple(copies[bit][slot] for bit in bits)
                self._rotate(qc, axis, 2*coeff*weight*fraction/r, target, controls)

        def x_layer(fraction):
            """Apply a fractional mixer step on every branch.

            :param fraction: 0.5 at the ends, 1.0 between Ising steps.
            :return: None.
            """
            for i, target in enumerate(b):
                rotate("x", 1.0, target, mixer, i, fraction)

        def z_layer():
            """Apply the Ising step, scheduling ZZ terms by matching.

            :return: None.
            """
            # Energy index x treats spin 0 as its most significant bit;
            # Qiskit treats B[0] as its least significant qubit.
            spin = list(reversed(b))
            for i, coeff in self.z_terms:
                rotate("z", coeff, spin[i], ising, i)
            for matching in self._matchings(self.n):
                for slot, (i, j) in enumerate(matching):
                    coeff = coeff_by_edge.get((i, j))
                    if coeff is not None:
                        # CX computes the Z_i Z_j parity on b[j]. Rotating
                        # that parity and undoing CX implements each RZZ term.
                        qc.cx(spin[i], spin[j])
                        rotate("z", coeff, spin[j], ising, slot)
                        qc.cx(spin[i], spin[j])

        # Adjacent half mixer steps have identical controls and no Ising layer
        # between them, so they combine exactly into a single full mixer step.
        x_layer(0.5)
        for step in range(r):
            z_layer()
            x_layer(0.5 if step == r - 1 else 1.0)

        # Unfanout leaves the parameter bits entangled with B as required,
        # while every work qubit returns to |0> for later uncomputation.
        for bit in reversed(parameter_bits):
            for copy in reversed(copies[bit][1:]):
                qc.cx(bit, copy)
        return qc


def _test_proposal_qemc_randomized() -> None:
    """Compare the circuit proposal with the exact averaged proposal matrix.

    :return: None; raises AssertionError if the circuit or cleanup is wrong.
    """
    from qiskit.quantum_info import Statevector
    from monaqa2.mcmc.proposal import create_proposal_matrix_quantum_exact

    n, k = 2, 1
    h = np.array([0.7, -0.3])  # Unequal fields reveal reversed spin ordering.
    J = np.array([[0.0, 0.4], [0.4, 0.0]])
    gate = ProposalQemcRandomized(n, h, J, k, time_lims=(0.4, 1.0),
                                  gamma_lims=(0.2, 0.6), num_trotter_steps=40)
    dimension = 2**n
    observed = np.zeros((dimension, dimension))
    work_start = 2*n + 2*k

    for x in range(dimension):
        # A begins in |x>; B, parameter registers, and work begin in |0>.
        state = Statevector.from_int(x, 2**gate.num_qubits).evolve(gate.definition).data
        assert np.allclose(state[2**work_start:], 0, atol=1e-10)
        for y in range(dimension):
            # Qiskit stores the registers as A + 2**n B + 2**(2n) time + ...
            observed[y, x] = sum(
                abs(state[x + (y << n) + (j << (2*n)) + (l << (2*n+k))])**2
                for j in range(2**k) for l in range(2**k)
            )

    expected = sum(
        create_proposal_matrix_quantum_exact(h, J, gamma=float(g), t=float(t))
        for t in gate.times for g in gate.gammas
    ) / 2**(2*k)
    # This compares the Trotter circuit against exact matrix exponentials.
    assert np.allclose(observed.sum(axis=0), 1.0, atol=1e-10)
    assert np.allclose(observed, expected, atol=1e-4), np.max(abs(observed - expected))
    print(f"Randomized proposal OK; max error = {np.max(abs(observed - expected)):.2e}; work is clean.")


if __name__ == "__main__":
    _test_proposal_qemc_randomized()
