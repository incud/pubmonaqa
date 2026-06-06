import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.walk_uniform_gate import WalkUniform
from monaqa2.qiskit.walk_uniform_symbolic import (
    walk_uniform_coins,
    walk_uniform_number_qubits,
    walk_uniform_nc_depth,
    walk_uniform_t_count,
    walk_uniform_rz_count,
    walk_uniform_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import get_nc_depth, get_t_count, get_rz_count, get_toffoli_count


def _make_cases():
    return [
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "mh", 1.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "glauber", 1.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "glauber", 5.0),
        (3, np.array([0.25, 0.0, -0.5]), np.array([[0.0, 0.2, 0.0], [0.2, 0.0, -0.3], [0.0, -0.3, 0.0]]), 0.20, 2e-2, "glauber", 10.0),
    ]


def _make_structure_cases():
    return [
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "mh", 1.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "glauber", 5.0),
    ]


def _make_spectral_cases():
    return [
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "glauber", 1.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "glauber", 5.0),
        (2, np.array([0.5, -0.25]), np.array([[0.0, 0.2], [0.2, 0.0]]), 0.30, 1e-2, "glauber", 10.0),
    ]


def _sum_abs(h: np.ndarray, J: np.ndarray) -> float:
    return float(np.sum(np.abs(h))) + float(np.sum(np.abs(np.triu(J, k=1))))


def _active_counts(h: np.ndarray, J: np.ndarray, tol: float = 1e-7) -> tuple[int, int]:
    return int(np.count_nonzero(np.abs(h) > tol)), int(np.count_nonzero(np.abs(np.triu(J, k=1)) > tol))


def _as_int(expr) -> int:
    value = complex(sp.N(expr))
    assert abs(value.imag) <= 1e-9
    real = float(value.real)
    assert np.isfinite(real)
    return int(round(real))


def _symbolic_kwargs(n: int, h: np.ndarray, J: np.ndarray, beta: float, eps: float, coin: str, a: float) -> dict:
    n_terms_z, n_terms_zz = _active_counts(h, J)
    return {"n": sp.Integer(n), "sum_abs": sp.Float(_sum_abs(h, J)), "beta": sp.Float(beta), "eps": sp.Float(eps), "coin": coin, "n_terms_z": sp.Integer(n_terms_z), "n_terms_zz": sp.Integer(n_terms_zz), "a": sp.Float(a)}


def _bits(index: int, n: int) -> np.ndarray:
    return np.array([(index >> i) & 1 for i in range(n)], dtype=int)


def _spin_energy(bits: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    z = 1 - 2 * bits
    value = float(np.dot(h, z))

    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            value += float(J[i, j]) * int(z[i]) * int(z[j])

    return value


def _glauber_acceptance(x: int, y: int, n: int, h: np.ndarray, J: np.ndarray, beta: float, a: float) -> float:
    delta = _spin_energy(_bits(y, n), h, J) - _spin_energy(_bits(x, n), h, J)
    return float((1.0 / (1.0 + np.exp(beta * delta))) ** (1.0 / a))


def _transition_matrix(n: int, h: np.ndarray, J: np.ndarray, beta: float, coin: str, a: float) -> np.ndarray:
    dim = 2**n
    proposal = 1.0 / dim
    P = np.zeros((dim, dim), dtype=float)

    for x in range(dim):
        accepted_out = 0.0

        for y in range(dim):
            if x == y:
                continue

            if coin == "mh":
                assert False
            else:
                acc = _glauber_acceptance(x, y, n, h, J, beta, a)
            P[x, y] = proposal * acc
            accepted_out += proposal * acc

        P[x, x] = 1.0 - accepted_out

    return P


def _stationary_weights(n: int, h: np.ndarray, J: np.ndarray, beta: float, coin:str, a: float) -> np.ndarray:
    energies = np.array([_spin_energy(_bits(x, n), h, J) for x in range(2**n)], dtype=float)
    weights = np.exp(-beta * energies / a)
    return weights / np.sum(weights)


def _discriminant_from_transition(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    sqrt_pi = np.sqrt(pi)
    return (sqrt_pi[:, None] * P) / sqrt_pi[None, :]


def _basis_state(index: int, num_qubits: int) -> np.ndarray:
    return Statevector.from_int(index, 2**num_qubits).data


def _a_register_basis_index(walk: WalkUniform, x: int) -> int:
    bits = _bits(x, walk.n)
    return sum(int(bit) << int(q) for bit, q in zip(bits, walk.layout["A"]))


def _walk_prefix_without_reflection(walk: WalkUniform) -> QuantumCircuit:
    """
    Build S = V† B† F B V from the actual WalkUniform definition.

    WalkUniform.definition is expected to contain

        V, B, F, B†, V†, R0.

    This function drops only the final reflection.
    """
    qc = QuantumCircuit(walk.num_qubits)

    for inst in walk.definition.data[:-1]:
        qargs = [walk.definition.find_bit(q).index for q in inst.qubits]
        qc.append(inst.operation, qargs)

    return qc


def _evolve_vector(vec: np.ndarray, qc: QuantumCircuit) -> np.ndarray:
    return Statevector(vec).evolve(qc).data


def _actual_projected_discriminant(walk: WalkUniform) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Compute D_actual = Π S Π from the actual WalkUniform circuit.

    Here Π projects B, coins, accept work, and reflection work to zero while
    leaving the A register free.
    """
    dim_a = 2**walk.n
    qc_s = _walk_prefix_without_reflection(walk)

    phi = [_basis_state(_a_register_basis_index(walk, x), walk.num_qubits) for x in range(dim_a)]
    s_phi = [_evolve_vector(vec, qc_s) for vec in phi]

    D = np.zeros((dim_a, dim_a), dtype=complex)

    for y in range(dim_a):
        for x in range(dim_a):
            D[x, y] = np.vdot(phi[x], s_phi[y])

    return D, phi, s_phi


def _orthonormal_basis(columns: list[np.ndarray], tol: float = 1e-10) -> np.ndarray:
    basis = []

    for col in columns:
        vec = np.array(col, dtype=complex, copy=True)

        for q in basis:
            vec -= q * np.vdot(q, vec)

        norm = np.linalg.norm(vec)

        if norm > tol:
            basis.append(vec / norm)

    return np.column_stack(basis)



def _expected_walk_eigs_from_discriminant(D: np.ndarray, target_dim: int, tol: float) -> np.ndarray:
    """
    Expected eigenvalues of the actual walk on the numerically generated
    subspace span{Π, SΠ}.

    For lambda strictly inside (-1, 1), the walk contributes both

        exp(+i arccos(lambda)) and exp(-i arccos(lambda)).

    If lambda is numerically close to +1 or -1, the two branches are degenerate
    in exact arithmetic. Depending on the numerical rank of span{Π, SΠ}, the
    generated subspace may contain either one or two copies of that degenerate
    eigenvalue. `target_dim` is the actual dimension of the restricted walk
    matrix and fixes the multiplicity.
    """
    lambdas = np.linalg.eigvalsh((D + D.conj().T) / 2)
    lambdas = np.clip(lambdas, -1.0, 1.0)

    eigs = []
    collapsible = []

    for lam in lambdas:
        if abs(lam - 1.0) <= tol:
            collapsible.append((1.0 + 0.0j, 1.0 + 0.0j))
        elif abs(lam + 1.0) <= tol:
            collapsible.append((-1.0 + 0.0j, -1.0 + 0.0j))
        else:
            theta = np.arccos(lam)
            eigs.append(np.exp(1j * theta))
            eigs.append(np.exp(-1j * theta))

    remaining = target_dim - len(eigs)

    if remaining < 0 or remaining > 2 * len(collapsible):
        raise AssertionError(
            f"Cannot build expected spectrum of length {target_dim}: "
            f"{len(eigs)} non-degenerate eigenvalues and {len(collapsible)} "
            "near-degenerate discriminant modes."
        )

    for value, duplicate in collapsible:
        if remaining >= 2:
            eigs.append(value)
            eigs.append(duplicate)
            remaining -= 2
        elif remaining == 1:
            eigs.append(value)
            remaining -= 1

    assert len(eigs) == target_dim
    return np.array(eigs, dtype=complex)


def _assert_spectra_close(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-7) -> None:
    assert len(actual) == len(expected)

    unused = list(actual)

    for target in expected:
        distances = [abs(value - target) for value in unused]
        idx = int(np.argmin(distances))
        assert distances[idx] <= atol
        unused.pop(idx)


@pytest.mark.parametrize("n,h,J,beta,eps,coin,a", _make_structure_cases())
def test_walk_uniform_layout_and_top_level_structure(n: int, h: np.ndarray, J: np.ndarray, beta: float, eps: float, coin: str, a: float) -> None:
    """
    Check the top-level WalkUniform composition and register layout.

    This verifies that WalkUniform allocates disjoint A, B, coin, accept-work,
    and reflection-work regions, and that the top-level circuit has the expected
    operator order

        V, B, F, B†, V†, R0.
    """
    walk = WalkUniform(n, h, J, beta, eps, coin=coin, a=a, mocked_circuit=True, mocked_angles=True)
    layout = walk.layout

    assert layout["A"] == list(range(0, n))
    assert layout["B"] == list(range(n, 2 * n))
    assert len(layout["coins"]) == walk.coins
    assert len(layout["accept_work"]) == walk.n_accept_work
    assert len(layout["reflection_work"]) == walk.n_reflection_work

    all_regions = layout["A"] + layout["B"] + layout["coins"] + layout["accept_work"] + layout["reflection_work"]

    assert sorted(all_regions) == list(range(walk.num_qubits))
    assert len(set(all_regions)) == walk.num_qubits

    names = [inst.operation.name for inst in walk.definition.data]

    assert names[0] == "ProposalUniform"
    assert "AcceptPath" in names
    assert names[-1] == "Reflection"

    accept_idx = names.index("AcceptPath")

    assert 0 < accept_idx < len(names) - 1


def _actual_walk_subspace_matrix(
    walk: WalkUniform,
    phi: list[np.ndarray],
    s_phi: list[np.ndarray],
    atol: float,
) -> np.ndarray:
    """
    Restrict the actual WalkUniform unitary to span{Π, SΠ}.

    In exact arithmetic, this subspace is invariant under

        W = R0 S,

    where S = V† B† F B V. With an approximate arithmetic coin and dense
    Statevector simulation, there can be small numerical/approximation leakage,
    so the residual tolerance is passed by the test and tied to eps.
    """
    qc_w = QuantumCircuit(walk.num_qubits)
    qc_w.append(walk, list(range(walk.num_qubits)))

    Q = _orthonormal_basis(phi + s_phi)
    WQ = np.column_stack([_evolve_vector(Q[:, j], qc_w) for j in range(Q.shape[1])])

    projected = Q @ (Q.conj().T @ WQ)
    residual = np.linalg.norm(WQ - projected, ord=2)

    assert residual <= atol

    return Q.conj().T @ WQ


@pytest.mark.parametrize("n,h,J,beta,eps,coin,a", _make_spectral_cases())
def test_walk_uniform_actual_spectrum_matches_transition_eigenvalues(
    n: int,
    h: np.ndarray,
    J: np.ndarray,
    beta: float,
    eps: float,
    coin: str,
    a: float,
) -> None:
    """
    Check the actual WalkUniform spectral relation.

    This test instantiates WalkUniform and uses its actual circuit. It constructs

        S = V† B† F B V

    by taking the WalkUniform definition and dropping only the final reflection.
    It then computes the actual projected discriminant block

        D_actual = Π S Π,

    where Π projects B, coins, and work registers to zero. The actual full walk

        W = R0 S

    is then restricted to span{Π, SΠ}. Its eigenvalues must be

        exp(± i arccos(lambda_j)),

    with near-±1 branches collapsed or duplicated according to the actual
    numerical dimension of span{Π, SΠ}.

    Finally, D_actual is compared against the desired discriminant built from
    the uniform proposal and the requested generalized Glauber acceptance rule.
    """
    assert coin == "glauber"

    walk = WalkUniform(n, h, J, beta, eps, coin=coin, a=a, mocked_circuit=True, mocked_angles=False)
    subspace_atol = max(1e-6, 1e-3 * eps)

    D_actual, phi, s_phi = _actual_projected_discriminant(walk)
    W_subspace = _actual_walk_subspace_matrix(walk, phi, s_phi, atol=subspace_atol)

    np.testing.assert_allclose(D_actual, D_actual.conj().T, atol=subspace_atol)

    actual_walk_eigs = np.linalg.eigvals(W_subspace)
    expected_walk_eigs = _expected_walk_eigs_from_discriminant(
        D_actual,
        target_dim=W_subspace.shape[0],
        tol=subspace_atol,
    )

    _assert_spectra_close(actual_walk_eigs, expected_walk_eigs, atol=1e-5)

    P_target = _transition_matrix(n, h, J, beta, coin, a)
    pi_target = _stationary_weights(n, h, J, beta, coin, a)
    D_target = _discriminant_from_transition(P_target, pi_target)

    np.testing.assert_allclose(P_target.sum(axis=1), np.ones(P_target.shape[0]), atol=1e-12)
    np.testing.assert_allclose(np.diag(pi_target) @ P_target, (np.diag(pi_target) @ P_target).T, atol=1e-10)
    np.testing.assert_allclose(D_actual, D_target, atol=max(5e-2, 20 * eps))


@pytest.mark.parametrize("n,h,J,beta,eps,coin,a", _make_cases())
def test_walk_uniform_symbolic_qubit_bound(n: int, h: np.ndarray, J: np.ndarray, beta: float, eps: float, coin: str, a: float) -> None:
    """
    Check the symbolic qubit bound against an actual WalkUniform instance.
    """
    walk = WalkUniform(n, h, J, beta, eps, coin=coin, a=a, mocked_circuit=True, mocked_angles=True)
    kwargs = _symbolic_kwargs(n, h, J, beta, eps, coin, a)

    symbolic_coins = _as_int(walk_uniform_coins(**kwargs))
    symbolic_num_qubits = _as_int(walk_uniform_number_qubits(**kwargs))

    assert walk.coins <= symbolic_coins
    assert walk.num_qubits <= symbolic_num_qubits


@pytest.mark.parametrize("n,h,J,beta,eps,coin,a", _make_cases())
def test_walk_uniform_symbolic_resource_bounds(n: int, h: np.ndarray, J: np.ndarray, beta: float, eps: float, coin: str, a: float) -> None:
    """
    Check symbolic resource upper bounds against static counters on WalkUniform.

    The test appends the actual WalkUniform gate and uses the existing recursive
    static counters. It does not execute the circuit and does not extract a dense
    unitary.
    """
    walk = WalkUniform(n, h, J, beta, eps, coin=coin, a=a, mocked_circuit=False, mocked_angles=True)
    qc = QuantumCircuit(walk.num_qubits)
    qc.append(walk, list(range(walk.num_qubits)))

    kwargs = _symbolic_kwargs(n, h, J, beta, eps, coin, a)

    symbolic_nc_depth = _as_int(walk_uniform_nc_depth(**kwargs))
    symbolic_t_count = _as_int(walk_uniform_t_count(**kwargs))
    symbolic_rz_count = _as_int(walk_uniform_rz_count(**kwargs))
    symbolic_toffoli_count = _as_int(walk_uniform_toffoli_count(**kwargs))

    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)

    assert actual_nc_depth <= symbolic_nc_depth
    assert actual_t_count <= symbolic_t_count
    assert actual_rz_count <= symbolic_rz_count
    assert actual_toffoli_count <= symbolic_toffoli_count


def test_walk_uniform_rejects_invalid_coin() -> None:
    """
    Check that WalkUniform accepts only documented coin names.
    """
    n = 2
    h = np.array([0.5, -0.25])
    J = np.array([[0.0, 0.2], [0.2, 0.0]])

    with pytest.raises(ValueError, match="coin must be either 'mh' or 'glauber'"):
        WalkUniform(n, h, J, beta=0.3, eps=1e-2, coin="bad")