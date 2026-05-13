import math
import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from monaqa2.qiskit.walk_qemc_gate import WalkQemc
from monaqa2.qiskit.walk_qemc_symbolic import (
    walk_qemc_coins,
    walk_qemc_number_qubits,
    walk_qemc_nc_depth,
    walk_qemc_t_count,
    walk_qemc_rz_count,
    walk_qemc_toffoli_count,
)
from monaqa2.qiskit.utils_qiskit import get_nc_depth, get_t_count, get_rz_count, get_toffoli_count


def _make_cases():
    return [
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.30, 0.08, 2e-3, "mh", 1.0),
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.30, 0.08, 2e-3, "glauber", 1.0),
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.30, 0.08, 2e-3, "glauber", 5.0),
    ]


def _make_structure_cases():
    return [
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.30, 0.08, 2e-3, "mh", 1.0),
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.30, 0.08, 2e-3, "glauber", 5.0),
    ]


def _make_spectral_cases():
    return [
        (2, np.array([0.20, -0.10]), np.array([[0.0, 0.05], [0.05, 0.0]]), np.array([0.10, 0.15]), 0.30, 0.08, 2e-3, "glauber", 1.0),
    ]


def _sum_abs(h: np.ndarray, J: np.ndarray) -> float:
    return float(np.sum(np.abs(h))) + float(np.sum(np.abs(np.triu(J, k=1))))


def _alpha_qemc(h: np.ndarray, J: np.ndarray, gamma: np.ndarray) -> float:
    return _sum_abs(h, J) + float(np.sum(np.abs(gamma)))


def _active_counts(h: np.ndarray, J: np.ndarray, tol: float = 1e-7) -> tuple[int, int]:
    return int(np.count_nonzero(np.abs(h) > tol)), int(np.count_nonzero(np.abs(np.triu(J, k=1)) > tol))


def _active_qemc_terms(h: np.ndarray, J: np.ndarray, gamma: np.ndarray, tol: float = 1e-7) -> int:
    return int(np.count_nonzero(np.abs(h) > tol) + np.count_nonzero(np.abs(np.triu(J, k=1)) > tol) + np.count_nonzero(np.abs(gamma) > tol))


def _as_upper_int(expr) -> int:
    value = complex(sp.N(expr))
    assert abs(value.imag) <= 1e-9
    real = float(value.real)
    assert np.isfinite(real)
    return int(math.ceil(real - 1e-12))


def _symbolic_kwargs(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, beta: float, t: float, eps: float, coin: str, a: float) -> dict:
    n_terms_z, n_terms_zz = _active_counts(h, J)
    return {
        "n": sp.Integer(n),
        "n_terms_qemc": sp.Integer(_active_qemc_terms(h, J, gamma)),
        "alpha_qemc": sp.Float(_alpha_qemc(h, J, gamma)),
        "t": sp.Float(t),
        "sum_abs": sp.Float(_sum_abs(h, J)),
        "beta": sp.Float(beta),
        "eps": sp.Float(eps),
        "coin": coin,
        "n_terms_z": sp.Integer(n_terms_z),
        "n_terms_zz": sp.Integer(n_terms_zz),
        "a": sp.Float(a),
    }


def _bits(index: int, n: int) -> np.ndarray:
    return np.array([(index >> i) & 1 for i in range(n)], dtype=int)


def _basis_state(index: int, num_qubits: int) -> np.ndarray:
    return Statevector.from_int(index, 2**num_qubits).data


def _a_register_basis_index(walk: WalkQemc, x: int) -> int:
    bits = _bits(x, walk.n)
    return sum(int(bit) << int(q) for bit, q in zip(bits, walk.layout["A"]))


def _walk_prefix_without_reflection(walk: WalkQemc) -> QuantumCircuit:
    qc = QuantumCircuit(walk.num_qubits)

    for inst in walk.definition.data[:-1]:
        qargs = [walk.definition.find_bit(q).index for q in inst.qubits]
        qc.append(inst.operation, qargs)

    return qc


def _evolve_vector(vec: np.ndarray, qc: QuantumCircuit) -> np.ndarray:
    return Statevector(vec).evolve(qc).data


def _actual_projected_discriminant(walk: WalkQemc) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
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


def _actual_walk_subspace_matrix(walk: WalkQemc, phi: list[np.ndarray], s_phi: list[np.ndarray], atol: float) -> np.ndarray:
    qc_w = QuantumCircuit(walk.num_qubits)
    qc_w.append(walk, list(range(walk.num_qubits)))

    Q = _orthonormal_basis(phi + s_phi, tol=atol / 10)
    WQ = np.column_stack([_evolve_vector(Q[:, j], qc_w) for j in range(Q.shape[1])])
    projected = Q @ (Q.conj().T @ WQ)
    residual = np.linalg.norm(WQ - projected, ord=2)

    assert residual <= atol

    return Q.conj().T @ WQ


def _expected_walk_eigs_from_discriminant(D: np.ndarray, target_dim: int, tol: float) -> np.ndarray:
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
        raise AssertionError(f"Cannot build expected spectrum of length {target_dim} from {len(eigs)} non-degenerate and {len(collapsible)} near-degenerate modes.")

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


@pytest.mark.parametrize("n,h,J,gamma,beta,t,eps,coin,a", _make_structure_cases())
def test_walk_qemc_layout_and_top_level_structure(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, beta: float, t: float, eps: float, coin: str, a: float) -> None:
    """
    Check the top-level WalkQemc composition and register layout.

    The expected operator order is V, B, F, B†, V†, R0.
    """
    walk = WalkQemc(n, h, J, gamma, beta, t, eps, coin=coin, a=a, mocked_reflection=True, mocked_circuit=True, mocked_angles=True)
    layout = walk.layout

    assert layout["A"] == list(range(0, n))
    assert layout["B"] == list(range(n, 2 * n))
    assert len(layout["proposal_aux"]) == walk.n_proposal_aux
    assert len(layout["coins"]) == walk.coins
    assert len(layout["accept_work"]) == walk.n_accept_work
    assert len(layout["reflection_work"]) == walk.n_reflection_work

    all_regions = layout["A"] + layout["B"] + layout["proposal_aux"] + layout["coins"] + layout["accept_work"] + layout["reflection_work"]

    assert sorted(all_regions) == list(range(walk.num_qubits))
    assert len(set(all_regions)) == walk.num_qubits

    names = [inst.operation.name for inst in walk.definition.data]

    assert names[0].startswith("ProposalQemc")
    assert "AcceptPath" in names
    assert names[-1] == "Reflection"

    accept_idx = names.index("AcceptPath")

    assert 0 < accept_idx < len(names) - 1


@pytest.mark.parametrize("n,h,J,gamma,beta,t,eps,coin,a", _make_spectral_cases())
def test_walk_qemc_actual_spectrum_matches_discriminant_eigenvalues(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, beta: float, t: float, eps: float, coin: str, a: float) -> None:
    """
    Check the actual WalkQemc spectral relation.

    This instantiates WalkQemc and uses its actual circuit. It computes
    D_actual = Π S Π with S = V† B† F B V extracted from the actual definition,
    then restricts the full actual W = R0 S to span{Π, SΠ}. The restricted
    eigenvalues must be exp(±i arccos(lambda_j)), where lambda_j are eigenvalues
    of D_actual.
    """
    assert coin == "glauber"

    walk = WalkQemc(n, h, J, gamma, beta, t, eps, coin=coin, a=a, mocked_reflection=True, mocked_circuit=True, mocked_angles=False)

    # if walk.num_qubits > 16:
    #     pytest.skip(f"Dense spectral test skipped for {walk.num_qubits} qubits.")

    subspace_atol = max(1e-6, 1e-3 * eps)

    D_actual, phi, s_phi = _actual_projected_discriminant(walk)
    W_subspace = _actual_walk_subspace_matrix(walk, phi, s_phi, atol=subspace_atol)

    np.testing.assert_allclose(D_actual, D_actual.conj().T, atol=subspace_atol)

    actual_walk_eigs = np.linalg.eigvals(W_subspace)
    expected_walk_eigs = _expected_walk_eigs_from_discriminant(D_actual, target_dim=W_subspace.shape[0], tol=subspace_atol)

    _assert_spectra_close(actual_walk_eigs, expected_walk_eigs, atol=1e-5)


@pytest.mark.parametrize("n,h,J,gamma,beta,t,eps,coin,a", _make_cases())
def test_walk_qemc_symbolic_qubit_bound(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, beta: float, t: float, eps: float, coin: str, a: float) -> None:
    """
    Check the symbolic qubit bound against an actual WalkQemc instance.
    """
    walk = WalkQemc(n, h, J, gamma, beta, t, eps, coin=coin, a=a, mocked_reflection=False, mocked_circuit=False, mocked_angles=True)
    kwargs = _symbolic_kwargs(n, h, J, gamma, beta, t, eps, coin, a)

    symbolic_coins = _as_upper_int(walk_qemc_coins(**kwargs))
    symbolic_num_qubits = _as_upper_int(walk_qemc_number_qubits(**kwargs))

    assert walk.coins <= symbolic_coins
    assert walk.num_qubits <= symbolic_num_qubits


@pytest.mark.parametrize("n,h,J,gamma,beta,t,eps,coin,a", _make_cases())
def test_walk_qemc_symbolic_resource_bounds(n: int, h: np.ndarray, J: np.ndarray, gamma: np.ndarray, beta: float, t: float, eps: float, coin: str, a: float) -> None:
    """
    Check symbolic resource upper bounds against static counters on WalkQemc.

    The QEMC proposal is non-mocked. The acceptance coin is mocked at circuit
    level for tractability, as in the other walk tests, so the symbolic coin
    formula remains an upper bound.
    """
    walk = WalkQemc(n, h, J, gamma, beta, t, eps, coin=coin, a=a, mocked_reflection=False, mocked_circuit=False, mocked_angles=True)
    qc = QuantumCircuit(walk.num_qubits)
    qc.append(walk, list(range(walk.num_qubits)))

    kwargs = _symbolic_kwargs(n, h, J, gamma, beta, t, eps, coin, a)

    symbolic_nc_depth = _as_upper_int(walk_qemc_nc_depth(**kwargs))
    symbolic_t_count = _as_upper_int(walk_qemc_t_count(**kwargs))
    symbolic_rz_count = _as_upper_int(walk_qemc_rz_count(**kwargs))
    symbolic_toffoli_count = _as_upper_int(walk_qemc_toffoli_count(**kwargs))

    actual_nc_depth = get_nc_depth(qc)
    actual_t_count = get_t_count(qc)
    actual_rz_count = get_rz_count(qc)
    actual_toffoli_count = get_toffoli_count(qc)

    assert actual_nc_depth <= symbolic_nc_depth
    assert actual_t_count <= symbolic_t_count
    assert actual_rz_count <= symbolic_rz_count
    assert actual_toffoli_count <= symbolic_toffoli_count

