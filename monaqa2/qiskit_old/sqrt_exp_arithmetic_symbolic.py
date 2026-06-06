import math
import sympy as sp

from monaqa2.qiskit.qubitized_ising_tf_symbolic import (
    qubitized_ising_tf_number_qubits,
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
from monaqa2.qiskit.utils_symbolic import fast_simplify_logs



def _is_numeric_leq_one(x: sp.Expr) -> bool:
    x = sp.sympify(x)
    return bool(x.is_number and float(sp.N(x)) <= 1.0)


def _sqrt_exp_uncontrolled_walk_nc_depth(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(0)

    return sp.ceiling(
        qubitized_ising_tf_nc_depth(
            b,
            n_terms_z=n_terms_z,
            n_terms_zz=0,
            n_terms_x=0,
        )
    )


def _sqrt_exp_uncontrolled_walk_t_count(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(0)

    return qubitized_ising_tf_t_count(
        b,
        n_terms_z=n_terms_z,
        n_terms_zz=0,
        n_terms_x=0,
    )


def _sqrt_exp_uncontrolled_walk_rz_count(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(0)

    return qubitized_ising_tf_rz_count(
        b,
        n_terms_z=n_terms_z,
        n_terms_zz=0,
        n_terms_x=0,
    )


def _sqrt_exp_uncontrolled_walk_toffoli_count(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(0)

    return qubitized_ising_tf_toffoli_count(
        b,
        n_terms_z=n_terms_z,
        n_terms_zz=0,
        n_terms_x=0,
    )


def _sqrt_exp_controlled_walk_nc_depth(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        # One active Z term:
        #   no controlled PREPARE,
        #   controlled SELECT is one CCZ layer,
        #   controlled reflection is Clifford.
        return sp.Integer(1)

    return sp.ceiling(
        controlled_qubitized_ising_tf_nc_depth(
            b,
            n_terms_z=n_terms_z,
            n_terms_zz=0,
            n_terms_x=0,
        )
    )


def _sqrt_exp_controlled_walk_t_count(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(0)

    return controlled_qubitized_ising_tf_t_count(
        b,
        n_terms_z=n_terms_z,
        n_terms_zz=0,
        n_terms_x=0,
    )


def _sqrt_exp_controlled_walk_rz_count(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(0)

    return controlled_qubitized_ising_tf_rz_count(
        b,
        n_terms_z=n_terms_z,
        n_terms_zz=0,
        n_terms_x=0,
    )


def _sqrt_exp_controlled_walk_toffoli_count(b: sp.Symbol, n_terms_z: sp.Expr) -> sp.Expr:
    if _is_numeric_leq_one(n_terms_z):
        return sp.Integer(1)

    return controlled_qubitized_ising_tf_toffoli_count(
        b,
        n_terms_z=n_terms_z,
        n_terms_zz=0,
        n_terms_x=0,
    )



def sqrt_exp_arithmetic_alpha_signal(b: sp.Symbol) -> sp.Expr:
    b = sp.sympify(b)
    return sp.simplify(1 - 2 ** (-b))


def sqrt_exp_arithmetic_lambda(beta: sp.Symbol, normalization: sp.Symbol) -> sp.Expr:
    return sp.simplify(sp.sympify(beta) * sp.sympify(normalization) / 2)


def sqrt_exp_arithmetic_mu(b: sp.Symbol, beta: sp.Symbol, normalization: sp.Symbol) -> sp.Expr:
    return sqrt_exp_arithmetic_lambda(beta, normalization)


def sqrt_exp_arithmetic_degree(b: sp.Symbol, beta: sp.Symbol, normalization: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    mu = sqrt_exp_arithmetic_mu(b, beta, normalization)
    return sp.simplify(sp.E * mu + sp.log(4 / sp.sympify(eps)) + 1)


def sqrt_exp_arithmetic_alpha_signal(b: sp.Symbol) -> sp.Expr:
    b = sp.sympify(b)
    return sp.simplify(1 - 2 ** (-b))


def sqrt_exp_arithmetic_n_terms_z(b: sp.Symbol, tol: float = 1e-8) -> sp.Expr:
    """
    Active Z terms in SqrtExpArithmetic after LcuUnaryIsingTF pruning.

    h[-1] = 1/2 is always active.
    For j = 0, ..., b-2, the coefficient is 2^{j-b};
    it is active iff 2^{j-b} >= tol.
    """
    return b + 1
    # if isinstance(b, int):
    #     j_min = max(0, math.ceil(b + math.log2(tol)))
    #     fractional_terms = max(0, (b - 1) - j_min)
    #     return sp.Integer(1 + fractional_terms)
    # 
    # b = sp.sympify(b)
    # j_min = sp.Max(0, sp.ceiling(b + sp.log(sp.Float(tol)) / sp.log(2)))
    # fractional_terms = sp.Max(0, b - 1 - j_min)
    # return sp.simplify(1 + fractional_terms)


def sqrt_exp_arithmetic_number_qubits(
    b: sp.Symbol,
    beta: sp.Symbol,
    normalization: sp.Symbol,
    eps: sp.Symbol,
) -> sp.Expr:
    b = sp.sympify(b)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)

    if _is_numeric_leq_one(n_terms_z):
        return sp.simplify(1 + b)

    return sp.simplify(
        controlled_qubitized_ising_tf_number_qubits(
            b,
            n_terms_z=n_terms_z,
            n_terms_zz=0,
            n_terms_x=0,
        )
    )


def sqrt_exp_arithmetic_nc_depth(
    b: sp.Symbol,
    beta: sp.Symbol,
    normalization: sp.Symbol,
    eps: sp.Symbol,
) -> sp.Expr:
    b = sp.sympify(b)
    d = sp.symbols('d', integer=True, positive=True)
    degree = d # sqrt_exp_arithmetic_degree(b, beta, normalization, eps)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)

    controlled_walk = _sqrt_exp_controlled_walk_nc_depth(b, n_terms_z)
    inverse_walk = _sqrt_exp_uncontrolled_walk_nc_depth(b, n_terms_z)

    expr = sp.simplify(
            2 * degree * controlled_walk
            + degree * inverse_walk
            + 3 * (2 * degree + 1)
    )
    expr = fast_simplify_logs(expr)
    f = sp.symbols('f', integer=True, positive=True)
    return expr.subs(b, f+1).xreplace({
        sp.log(f+1): sp.log(f),
        sp.log(f+2): sp.log(f),
        sp.log(2*f+1): 2*sp.log(f),
        sp.log(2): 1
    })


def sqrt_exp_arithmetic_t_count(
    b: sp.Symbol,
    beta: sp.Symbol,
    normalization: sp.Symbol,
    eps: sp.Symbol,
) -> sp.Expr:
    b = sp.sympify(b)
    d = sp.symbols('d', integer=True, positive=True)
    degree = d # sqrt_exp_arithmetic_degree(b, beta, normalization, eps)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)

    controlled_walk = _sqrt_exp_controlled_walk_t_count(b, n_terms_z)
    inverse_walk = _sqrt_exp_uncontrolled_walk_t_count(b, n_terms_z)

    return sp.simplify(2 * degree * controlled_walk + degree * inverse_walk)


def sqrt_exp_arithmetic_rz_count(
    b: sp.Symbol,
    beta: sp.Symbol,
    normalization: sp.Symbol,
    eps: sp.Symbol,
) -> sp.Expr:
    b = sp.sympify(b)
    d = sp.symbols('d', integer=True, positive=True)
    degree = d # sqrt_exp_arithmetic_degree(b, beta, normalization, eps)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)

    controlled_walk = _sqrt_exp_controlled_walk_rz_count(b, n_terms_z)
    inverse_walk = _sqrt_exp_uncontrolled_walk_rz_count(b, n_terms_z)

    return sp.simplify(
        2 * degree * controlled_walk
        + degree * inverse_walk
        + 3 * (2 * degree + 1)
    )


def sqrt_exp_arithmetic_toffoli_count(
    b: sp.Symbol,
    beta: sp.Symbol,
    normalization: sp.Symbol,
    eps: sp.Symbol,
) -> sp.Expr:
    b = sp.sympify(b)
    d = sp.symbols('d', integer=True, positive=True)
    degree = d # sqrt_exp_arithmetic_degree(b, beta, normalization, eps)
    n_terms_z = sqrt_exp_arithmetic_n_terms_z(b)

    controlled_walk = _sqrt_exp_controlled_walk_toffoli_count(b, n_terms_z)
    inverse_walk = _sqrt_exp_uncontrolled_walk_toffoli_count(b, n_terms_z)

    return sp.simplify(2 * degree * controlled_walk + degree * inverse_walk)
