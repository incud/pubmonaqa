import numpy as np
import mpmath as mp


def _gqsp_u3_gate(theta: mp.mpf, phi: mp.mpf, lambd: mp.mpf):
    """
    Build the 2x2 GQSP U3 gate matrix in arbitrary precision.

    :param theta: Rotation angle theta.
    :param phi: Rotation angle phi.
    :param lambd: Rotation angle lambda.
    :return: Nested tuple representing the 2x2 gate.
    """
    return (
        (mp.e ** (1j * (lambd + phi)) * mp.cos(theta), mp.e ** (1j * phi) * mp.sin(theta)),
        (mp.e ** (1j * lambd) * mp.sin(theta), -mp.cos(theta)),
    )


def _to_mpc_list(coeffs) -> list:
    """
    Convert coefficients to a list of mpmath complex numbers.

    :param coeffs: Iterable of coefficients.
    :return: List of mp.mpc coefficients.
    """
    return [mp.mpc(complex(c)) for c in coeffs]


def _trim_trailing_small(coeffs, tol: mp.mpf) -> list:
    """
    Remove trailing coefficients whose magnitude is below tolerance.

    :param coeffs: Iterable of coefficients.
    :param tol: Trimming tolerance.
    :return: Trimmed list of coefficients.
    """
    coeffs = list(coeffs)
    while len(coeffs) > 1 and abs(coeffs[-1]) < tol:
        coeffs.pop()
    return coeffs


def _poly_eval(coeffs, z: mp.mpc) -> mp.mpc:
    """
    Evaluate a polynomial in ascending-power coefficient convention.

    :param coeffs: Polynomial coefficients.
    :param z: Evaluation point.
    :return: Polynomial value at z.
    """
    out, zk = mp.mpc(0), mp.mpc(1)
    for c in coeffs:
        out += c * zk
        zk *= z
    return out


def _poly_mul(a, b) -> list:
    """
    Multiply two polynomials in ascending-power coefficient convention.

    :param a: First polynomial coefficients.
    :param b: Second polynomial coefficients.
    :return: Product polynomial coefficients.
    """
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _poly_from_roots(roots) -> list:
    """
    Build a monic polynomial from its roots.

    :param roots: Iterable of roots.
    :return: Polynomial coefficients in ascending-power convention.
    """
    out = [mp.mpc(1)]
    for r in roots:
        out = _poly_mul(out, [-r, mp.mpc(1)])
    return out


def _complementary_poly(poly_coeffs, dps: int = 120, trim_tol=None, root_tol=None) -> list:
    """
    Compute the complementary polynomial Q for a polynomial P using arbitrary precision.

    If some roots are too close to the unit circle, a warning is printed and
    the ambiguous roots are assigned heuristically so the factorization can proceed.

    :param poly_coeffs: Coefficients of the polynomial P.
    :param dps: Decimal precision used by mpmath.
    :param trim_tol: Tolerance for trimming trailing coefficients.
    :param root_tol: Tolerance for deciding whether a root is too close to the unit circle.
    :return: Coefficients of the complementary polynomial Q.
    """
    mp.mp.dps = dps
    trim_tol = mp.mpf(10) ** (-(dps // 2)) if trim_tol is None else trim_tol
    root_tol = mp.mpf(10) ** (-(dps // 3)) if root_tol is None else root_tol

    p = _trim_trailing_small(_to_mpc_list(poly_coeffs), trim_tol)
    deg = len(p) - 1

    # Build the polynomial R(z) = z^degree * (1 - conj(P(1/z)) * P(z)).
    # In coefficient form:
    #   R(z) = z^degree - P(z) * conj(P(z)[::-1]).
    prod = _poly_mul(p, [mp.conj(c) for c in reversed(p)])
    R = [mp.mpc(0)] * (2 * deg + 1)
    R[deg] = mp.mpc(1)
    for i, val in enumerate(prod):
        R[i] -= val

    # mpmath expects coefficients in descending order.
    roots = mp.polyroots(list(reversed(R)), maxsteps=200, extraprec=100, error=False)

    inside_circle, outside_circle, near_circle = [], [], []
    for r in roots:
        mod = abs(r)
        delta = abs(mod - 1)
        if delta <= root_tol:
            near_circle.append(r)
        elif mod < 1:
            inside_circle.append(r)
        else:
            outside_circle.append(r)

    if near_circle:
        debug_info = "\n".join(
            f" - | |r| - 1 | = {abs(abs(r) - 1)} ; |r| = {abs(r)}" for r in near_circle
        )
        print(
            f"WARNING: {len(near_circle)} roots lie too close to the unit circle.\n"
            f"Root tolerance: {root_tol}\n"
            f"Ambiguous roots:\n{debug_info}\n"
            "Proceeding with a heuristic split of ambiguous roots."
        )

        need_inside = deg - len(inside_circle)
        need_outside = deg - len(outside_circle)

        if need_inside < 0 or need_outside < 0 or need_inside + need_outside != len(near_circle):
            raise RuntimeError(
                f"Heuristic root split failed before assignment: "
                f"inside={len(inside_circle)}, outside={len(outside_circle)}, "
                f"ambiguous={len(near_circle)}, expected degree={deg}."
            )

        # Sort by modulus: the slightly smaller ones go inside, the slightly larger ones go outside.
        near_circle.sort(key=abs)
        inside_circle.extend(near_circle[:need_inside])
        outside_circle.extend(near_circle[need_inside:])

    if len(inside_circle) != deg or len(outside_circle) != deg:
        raise RuntimeError(
            f"Root split failed: inside={len(inside_circle)}, "
            f"outside={len(outside_circle)}, expected {deg} each."
        )

    outside_prod = mp.mpc(1)
    for r in outside_circle:
        outside_prod *= r

    scale_factor = mp.sqrt(abs(R[-1] * outside_prod))
    return [scale_factor * c for c in _poly_from_roots(inside_circle)]


def poly_to_gqsp_angles(
    poly, dps: int = 200, trim_tol=None, check_tol=None, zero_tol=None, root_tol=None, safety: float = 1e-14
):
    """
    Compute the Generalized Quantum Signal Processing angles for a polynomial using arbitrary precision.

    :param poly: Polynomial coefficients in ascending-power convention.
    :param dps: Decimal precision used by mpmath.
    :param trim_tol: Tolerance for trimming trailing coefficients.
    :param check_tol: Tolerance for boundedness checks at -1, 0, and 1.
    :param zero_tol: Tolerance for deciding whether a coefficient is numerically zero.
    :param root_tol: Tolerance for deciding whether a root is too close to the unit circle.
    :param safety: Multiplicative safety margin applied to the polynomial.
    :return: Tuple of NumPy arrays (angles_theta, angles_phi, angles_lambda).
    """
    poly = np.trim_zeros(np.asarray(poly, dtype=np.complex128), trim="b") * (1 - safety)
    if len(poly) == 1:
        raise AssertionError("The polynomial must have at least degree 1.")

    mp.mp.dps = dps
    trim_tol = mp.mpf(10) ** (-(dps // 2)) if trim_tol is None else trim_tol
    check_tol = mp.mpf(10) ** (-(dps // 3)) if check_tol is None else check_tol
    zero_tol = mp.mpf(10) ** (-(dps // 3)) if zero_tol is None else zero_tol

    p = _to_mpc_list(poly)
    for x in (-1, 0, 1):
        # Check that |P(x)| ≤ 1. Only points -1, 0, 1 are checked,
        # matching PennyLane's public wrapper.
        if abs(_poly_eval(p, mp.mpf(x))) > 1 + check_tol:
            raise AssertionError("The polynomial must satisfy that |P(x)| ≤ 1 for all x in [-1, 1]")

    p = _trim_trailing_small(p, trim_tol)
    q = _complementary_poly(p, dps=dps, trim_tol=trim_tol, root_tol=root_tol)

    # Algorithm 1 in [arXiv:2308.01501]
    input_row0, input_row1 = list(p), list(q)
    n = len(input_row0)
    angles_theta = [mp.mpf(0)] * n
    angles_phi = [mp.mpf(0)] * n
    angles_lambda = [mp.mpf(0)] * n

    for idx in range(n - 1, -1, -1):
        a, b = input_row0[idx], input_row1[idx]
        angles_theta[idx] = mp.atan2(abs(b), abs(a))
        angles_phi[idx] = mp.mpf(0) if abs(b) <= zero_tol else mp.arg(a * mp.conj(b))

        if idx == 0:
            angles_lambda[0] = mp.mpf(0) if abs(b) <= zero_tol else mp.arg(b)
            continue

        (g00, g01), (g10, g11) = _gqsp_u3_gate(angles_theta[idx], angles_phi[idx], mp.mpf(0))
        gd00, gd01, gd10, gd11 = mp.conj(g00), mp.conj(g10), mp.conj(g01), mp.conj(g11)

        updated0 = [gd00 * input_row0[j] + gd01 * input_row1[j] for j in range(idx + 1)]
        updated1 = [gd10 * input_row0[j] + gd11 * input_row1[j] for j in range(idx + 1)]
        input_row0, input_row1 = updated0[1 : idx + 1], updated1[:idx]

    return tuple(
        np.array([float(x) for x in angle_list], dtype=float)
        for angle_list in (angles_theta, angles_phi, angles_lambda)
    )