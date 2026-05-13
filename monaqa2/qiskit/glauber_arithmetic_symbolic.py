import sympy as sp

from monaqa2.qiskit.qubitized_ising_tf_symbolic import (
    qubitized_ising_tf_nc_depth,
    qubitized_ising_tf_t_count,
    qubitized_ising_tf_rz_count,
    qubitized_ising_tf_toffoli_count,
    controlled_qubitized_ising_tf_number_qubits,
    controlled_qubitized_ising_tf_nc_depth,
    controlled_qubitized_ising_tf_t_count,
    controlled_qubitized_ising_tf_rz_count,
    controlled_qubitized_ising_tf_toffoli_count,
)


def _delta_terms(
    n_terms_z: sp.Symbol,
    n_terms_zz: sp.Symbol,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return (
        sp.simplify(2 * sp.sympify(n_terms_z)),
        sp.simplify(2 * sp.sympify(n_terms_zz)),
        sp.Integer(0),
    )


def glauber_arithmetic_degree(
    alpha: sp.Symbol,
    beta: sp.Symbol,
    eps: sp.Symbol,
    a: sp.Symbol = 1,
) -> sp.Expr:
    alpha = sp.sympify(alpha)
    beta = sp.sympify(beta)
    eps = sp.sympify(eps)
    a = sp.sympify(a)

    scale = sp.simplify(beta * alpha)
    tau = sp.pi / scale
    rho = tau + sp.sqrt(1 + tau**2)

    branch_prefactor = 1 + sp.Rational(1, 2) / a

    return sp.simplify(
        sp.Piecewise(
            (1, scale <= 0),
            (
                sp.Max(
                    1,
                    sp.ceiling(sp.log(8 * branch_prefactor / eps) / sp.log(rho)) + 2,
                ),
                True,
            ),
        )
    )


def glauber_arithmetic_number_qubits(
    n: sp.Symbol,
    alpha: sp.Symbol,
    beta: sp.Symbol,
    eps: sp.Symbol,
    n_terms_z: sp.Symbol,
    n_terms_zz: sp.Symbol,
    a: sp.Symbol = 1,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    delta_z, delta_zz, delta_x = _delta_terms(n_terms_z, n_terms_zz)

    return sp.simplify(
        controlled_qubitized_ising_tf_number_qubits(
            n_system,
            n_terms_z=delta_z,
            n_terms_zz=delta_zz,
            n_terms_x=delta_x,
        )
    )


def glauber_arithmetic_nc_depth(
    n: sp.Symbol,
    alpha: sp.Symbol,
    beta: sp.Symbol,
    eps: sp.Symbol,
    n_terms_z: sp.Symbol,
    n_terms_zz: sp.Symbol,
    a: sp.Symbol = 1,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    degree = glauber_arithmetic_degree(alpha, beta, eps, a)
    delta_z, delta_zz, delta_x = _delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = sp.ceiling(
        controlled_qubitized_ising_tf_nc_depth(
            n_system,
            n_terms_z=delta_z,
            n_terms_zz=delta_zz,
            n_terms_x=delta_x,
        )
    )
    inverse_walk = sp.ceiling(
        qubitized_ising_tf_nc_depth(
            n_system,
            n_terms_z=delta_z,
            n_terms_zz=delta_zz,
            n_terms_x=delta_x,
        )
    )

    phase_blocks = 2 * degree + 1

    return sp.simplify(
        sp.ceiling(
            2 * degree * controlled_walk
            + degree * inverse_walk
            + 3 * phase_blocks
        )
    )


def glauber_arithmetic_t_count(
    n: sp.Symbol,
    alpha: sp.Symbol,
    beta: sp.Symbol,
    eps: sp.Symbol,
    n_terms_z: sp.Symbol,
    n_terms_zz: sp.Symbol,
    a: sp.Symbol = 1,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    degree = glauber_arithmetic_degree(alpha, beta, eps, a)
    delta_z, delta_zz, delta_x = _delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = controlled_qubitized_ising_tf_t_count(
        n_system,
        n_terms_z=delta_z,
        n_terms_zz=delta_zz,
        n_terms_x=delta_x,
    )
    inverse_walk = qubitized_ising_tf_t_count(
        n_system,
        n_terms_z=delta_z,
        n_terms_zz=delta_zz,
        n_terms_x=delta_x,
    )

    return sp.simplify(2 * degree * controlled_walk + degree * inverse_walk)


def glauber_arithmetic_rz_count(
    n: sp.Symbol,
    alpha: sp.Symbol,
    beta: sp.Symbol,
    eps: sp.Symbol,
    n_terms_z: sp.Symbol,
    n_terms_zz: sp.Symbol,
    a: sp.Symbol = 1,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    degree = glauber_arithmetic_degree(alpha, beta, eps, a)
    delta_z, delta_zz, delta_x = _delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = controlled_qubitized_ising_tf_rz_count(
        n_system,
        n_terms_z=delta_z,
        n_terms_zz=delta_zz,
        n_terms_x=delta_x,
    )
    inverse_walk = qubitized_ising_tf_rz_count(
        n_system,
        n_terms_z=delta_z,
        n_terms_zz=delta_zz,
        n_terms_x=delta_x,
    )

    phase_blocks = 2 * degree + 1

    return sp.simplify(
        2 * degree * controlled_walk
        + degree * inverse_walk
        + 3 * phase_blocks
    )


def glauber_arithmetic_toffoli_count(
    n: sp.Symbol,
    alpha: sp.Symbol,
    beta: sp.Symbol,
    eps: sp.Symbol,
    n_terms_z: sp.Symbol,
    n_terms_zz: sp.Symbol,
    a: sp.Symbol = 1,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    degree = glauber_arithmetic_degree(alpha, beta, eps, a)
    delta_z, delta_zz, delta_x = _delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = controlled_qubitized_ising_tf_toffoli_count(
        n_system,
        n_terms_z=delta_z,
        n_terms_zz=delta_zz,
        n_terms_x=delta_x,
    )
    inverse_walk = qubitized_ising_tf_toffoli_count(
        n_system,
        n_terms_z=delta_z,
        n_terms_zz=delta_zz,
        n_terms_x=delta_x,
    )

    return sp.simplify(2 * degree * controlled_walk + degree * inverse_walk)