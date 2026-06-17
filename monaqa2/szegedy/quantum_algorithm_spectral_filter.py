import numpy as np
import pennylane as qml
from numpy.polynomial import Polynomial, Chebyshev
import matplotlib.pyplot as plt


def discriminant_matrix(P: np.ndarray) -> np.ndarray:
    """
    Build the symmetric discriminant matrix associated with a reversible transition matrix.

    :param P: Column-stochastic transition matrix.
    :return: Symmetric discriminant matrix.
    """
    X = np.sqrt(P * P.T)
    return 0.5 * (X + X.T)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of the transition matrix.

    :param P: Column-stochastic transition matrix.
    :return: Normalized stationary distribution.
    """
    vals, vecs = np.linalg.eig(P)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = -v if v.sum() < 0 else v
    v = np.maximum(v, 0.0)
    return v / v.sum()


def chebyshev_gap_filter(degree: int, gap: float) -> Polynomial:
    """
    Build the even Chebyshev filter polynomial that preserves lambda=1 and suppresses |lambda| <= 1-gap.

    :param degree: Polynomial degree.
    :param gap: Spectral gap around the stationary eigenvalue.
    :return: Filter polynomial in the monomial basis.
    """
    degree = degree if degree % 2 == 0 else degree + 1
    m = degree // 2
    a = 1.0 - float(gap)
    y = Polynomial([-1.0, 0.0, 2.0 / a**2])
    return Chebyshev.basis(m).convert(kind=Polynomial)(y) / Chebyshev.basis(m)(2.0 / a**2 - 1.0)


def szegedy_v(P: np.ndarray, wires: list[int]) -> None:
    """
    Apply the Szegedy preparation isometry V.

    :param P: Column-stochastic transition matrix.
    :param wires: Wires for the two-register system A|B.
    :return: None.
    """
    P = np.asarray(P, dtype=float)
    N = P.shape[0]
    n = len(wires) // 2
    dim = 1 << n
    A, B = wires[:n], wires[n:]

    for x in range(N):
        amps = np.zeros(dim)
        amps[:N] = np.sqrt(P[:, x])
        amps /= np.linalg.norm(amps)

        # PennyLane control values follow the wire order in A, so use big-endian bits.
        bits = [(x >> (n - 1 - j)) & 1 for j in range(n)]
        qml.ctrl(qml.StatePrep, control=A, control_values=bits)(amps, wires=B)


def szegedy_w(P: np.ndarray, wires: list[int]) -> None:
    """
    Apply the V-rotated one-reflection Szegedy walk V^\\dagger S V R_0.

    :param P: Column-stochastic transition matrix.
    :param wires: Wires for the two-register system A|B.
    :return: None.
    """
    n = len(wires) // 2
    A, B = wires[:n], wires[n:]

    # Reflection about B=0, i.e. R_0 = 2|0...0><0...0| - I on the B register.
    qml.Reflection(qml.Identity(wires=B), alpha=np.pi)

    szegedy_v(P, wires)

    for j in range(n):
        qml.SWAP(wires=[A[j], B[j]])

    qml.adjoint(szegedy_v)(P, wires)


def apply_spectral_filter_to_uniform_distribution(
    P: np.ndarray,
    degree: int,
    gap: float | None = None,
    scale: float = 1.0 - 1e-8,
    angle_solver: str = "root-finding",
) -> dict[str, object]:
    """
    Apply QSVT to the Szegedy walk and return the decoded B=0 slice of the statevector.

    :param P: Column-stochastic transition matrix.
    :param degree: Polynomial degree.
    :param gap: Optional spectral gap. If omitted, it is inferred from the discriminant matrix.
    :param scale: Polynomial rescaling factor used before angle synthesis.
    :param angle_solver: PennyLane angle solver passed to qml.poly_to_angles.
    :return: Dictionary with filtered state, distribution, success probability, fidelities, angles, and statevector.
    """
    P = np.asarray(P, dtype=float)
    N = P.shape[0]
    n = int(np.ceil(np.log2(N))) if N > 1 else 1
    dim = 1 << n

    if N != dim:
        raise ValueError("Hadamard initialization assumes N is a power of two.")

    wires = list(range(2 * n))
    A, B = wires[:n], wires[n:]

    X = discriminant_matrix(P)
    lam, vecs = np.linalg.eigh(X)

    if gap is None:
        rest = np.delete(lam, np.argmax(lam))
        gap = 1.0 - float(np.max(np.abs(rest)))

    if gap <= 0.0:
        raise ValueError("The inferred spectral gap is non-positive.")

    degree = degree if degree % 2 == 0 else degree + 1
    poly = scale * chebyshev_gap_filter(degree, gap)

    coeffs = np.pad(np.asarray(poly.coef, dtype=float), (0, degree + 1 - len(poly.coef)))
    coeffs[np.abs(coeffs) < 1e-14] = 0.0
    coeffs[1::2] = 0.0

    angles = qml.poly_to_angles(coeffs, "QSVT", angle_solver=angle_solver)
    dev = qml.device("default.qubit", wires=2 * n)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        for q in A:
            qml.Hadamard(wires=q)

        # PCPhase is PennyLane's QSVT projector phase exp(i phi (2Pi - I)).
        qml.PCPhase(angles[0], dim=1, wires=B)

        for k in range(1, len(angles)):
            szegedy_w(P, wires) if k % 2 else qml.adjoint(szegedy_w)(P, wires)
            qml.PCPhase(angles[k], dim=1, wires=B)

        return qml.state()

    state = circuit()

    # The B=0 amplitudes occupy indices a * dim.
    filtered = np.asarray([state[a * dim] for a in range(N)], dtype=complex)
    success = float(np.vdot(filtered, filtered).real)
    filtered_state = filtered / np.sqrt(success) if success > 0 else filtered
    filtered_distribution = np.abs(filtered_state) ** 2

    pi = stationary_distribution(P)
    fidelity = float(np.dot(np.sqrt(filtered_distribution), np.sqrt(pi)) ** 2)

    uniform = np.ones(N, dtype=complex) / np.sqrt(N)
    expected_filtered = vecs @ (poly(lam) * (vecs.T @ uniform))
    expected_success = float(np.vdot(expected_filtered, expected_filtered).real)
    expected_state = expected_filtered / np.sqrt(expected_success) if expected_success > 0 else expected_filtered
    expected_distribution = np.abs(expected_state) ** 2
    expected_fidelity = float(np.dot(np.sqrt(expected_distribution), np.sqrt(pi)) ** 2)

    return {
        "filtered_state": filtered_state,
        "filtered_distribution": filtered_distribution,
        "success": success,
        "fidelity": fidelity,
        "expected_fidelity": expected_fidelity,
        "expected_success": expected_success,
        "angles": np.asarray(angles),
        "gap": gap,
        "degree": degree,
        "statevector": state,
    }


def plot_spectral_filter_infidelity_vs_degree(
    P: np.ndarray,
    degrees: list[int],
    gap: float | None = None,
    scale: float = 1.0 - 1e-8,
    angle_solver: str = "root-finding",
    include_qsvt: bool = True,
    include_expected: bool = True,
    plot_infidelity: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, np.ndarray]]:
    """
    Plot spectral-filter fidelity, or infidelity, as a function of the QSVT polynomial degree, using a logarithmic y axis.

    :param P: Column-stochastic transition matrix.
    :param degrees: Polynomial degrees.
    :param gap: Optional spectral gap. If omitted, it is inferred from the discriminant matrix.
    :param scale: Polynomial rescaling factor.
    :param angle_solver: PennyLane angle solver passed to qml.poly_to_angles.
    :param include_qsvt: Whether to run the PennyLane QSVT simulation.
    :param include_expected: Whether to plot the classical Chebyshev-filter prediction.
    :param plot_infidelity: Whether to plot 1 - fidelity instead of fidelity.
    :param ax: Optional Matplotlib axis.
    :return: Figure, axis, and collected numerical data.
    """
    if not include_qsvt and not include_expected:
        raise ValueError("At least one of include_qsvt or include_expected must be True.")

    P = np.asarray(P, dtype=float)
    degrees = np.asarray(degrees, dtype=int)

    X = discriminant_matrix(P)
    lam, vecs = np.linalg.eigh(X)

    if gap is None:
        rest = np.delete(lam, np.argmax(lam))
        gap = 1.0 - float(np.max(np.abs(rest)))

    if gap <= 0.0:
        raise ValueError("The inferred spectral gap is non-positive.")

    pi = stationary_distribution(P)
    uniform = np.ones(P.shape[0], dtype=complex) / np.sqrt(P.shape[0])

    fidelities = []
    expected_fidelities = []
    successes = []

    for d in degrees:
        d = int(d)

        if include_expected or not include_qsvt:
            d_even = d if d % 2 == 0 else d + 1
            m = d_even // 2
            a = 1.0 - float(gap)

            # Stable Chebyshev evaluation, avoiding the high-degree monomial coefficients.
            z = 2.0 * lam**2 / a**2 - 1.0
            den = Chebyshev.basis(m)(2.0 / a**2 - 1.0)
            values = scale * Chebyshev.basis(m)(z) / den

            expected_filtered = vecs @ (values * (vecs.T @ uniform))
            expected_success = float(np.vdot(expected_filtered, expected_filtered).real)
            expected_state = expected_filtered / np.sqrt(expected_success) if expected_success > 0 else expected_filtered
            expected_distribution = np.abs(expected_state) ** 2
            expected_fidelity = float(np.dot(np.sqrt(expected_distribution), np.sqrt(pi)) ** 2)
        else:
            expected_fidelity = np.nan

        if include_qsvt:
            out = apply_spectral_filter_to_uniform_distribution(P, degree=d, gap=gap, scale=scale, angle_solver=angle_solver)
            fidelities.append(out["fidelity"])
            successes.append(out["success"])
            if not include_expected:
                expected_fidelity = out["expected_fidelity"]
        else:
            fidelities.append(np.nan)
            successes.append(np.nan)

        expected_fidelities.append(expected_fidelity)

    data = {
        "degree": degrees,
        "fidelity": np.asarray(fidelities, dtype=float),
        "expected_fidelity": np.asarray(expected_fidelities, dtype=float),
        "success": np.asarray(successes, dtype=float),
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
    else:
        fig = ax.figure

    eps = np.finfo(float).tiny

    if include_qsvt:
        y = 1.0 - data["fidelity"] if plot_infidelity else data["fidelity"]
        ax.plot(data["degree"], np.maximum(y, eps), "o-", label="QSVT simulation")

    if include_expected:
        y_expected = 1.0 - data["expected_fidelity"] if plot_infidelity else data["expected_fidelity"]
        ax.plot(data["degree"], np.maximum(y_expected, eps), "--", label="Polynomial prediction")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Infidelity" if plot_infidelity else "Fidelity")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()

    return fig, ax, data