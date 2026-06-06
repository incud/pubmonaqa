import sympy as sp

from monaqa2.qiskit.kogge_stone_in_place_adder_symbolic import (
    kogge_stone_in_place_adder_number_qubits,
    kogge_stone_in_place_adder_nc_depth,
    kogge_stone_in_place_adder_toffoli_count,
)
from monaqa2.qiskit.utils_symbolic import fast_simplify_logs


def metropolis_hastings_energy_upper_bound_energy_diff(
    sum_abs: sp.Expr,
) -> sp.Expr:
    """
    Energy-difference upper bound.

    sum_abs = sum_i |h_i| + sum_{i<j} |J_ij|.

    The difference E(y) - E(x) is bounded by 2 * sum_abs.
    """
    return sp.simplify(2 * sum_abs)


def metropolis_hastings_energy_fractional_bits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    eps = sp.sympify(eps)

    upper = metropolis_hastings_energy_upper_bound_energy_diff(sum_abs)
    scaled_eps = eps / upper
    # return sp.ceiling(sp.log(n * (n + 1) / (scaled_eps**2), 2))
    return 3 * sp.log((n / scaled_eps), 2) # 2 * (1 + 1/2) * log(...)


def metropolis_hastings_energy_acc_word_bits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    f = sp.symbols('f', integer=True, positive=True)
    return 3 + f #3 + metropolis_hastings_energy_fractional_bits(n, sum_abs, eps)


def metropolis_hastings_energy_signal_bits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    f = sp.symbols('f', integer=True, positive=True)
    return 1 + f #1 + metropolis_hastings_energy_fractional_bits(n, sum_abs, eps)


def metropolis_hastings_energy_number_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    """
    Dense worst-case qubit upper bound.

    Uses:
        T = n * (n + 1)

    because the circuit may contain two signed copies of all one-body and pair
    terms:
        2 * (n + n(n-1)/2) = n(n+1).
    """
    n = sp.sympify(n)
    word_bits = metropolis_hastings_energy_acc_word_bits(n, sum_abs, eps)
    signal_bits = metropolis_hastings_energy_signal_bits(n, sum_abs, eps)

    n_terms = n * (n + 1)
    copies_per_side = n * (n - 1)
    pair_flags = n * (n - 1)
    bit_copies = n_terms * (word_bits - 1)
    sign_copies = word_bits - 2

    adder_qubits = kogge_stone_in_place_adder_number_qubits(
        word_bits,
        with_carry_out=False,
    )
    adder_ancillas = adder_qubits - 2 * word_bits
    max_parallel_adders = n_terms / 2 + 1

    expr = sp.simplify(
        2 * n
        + 2 * copies_per_side
        + n_terms * word_bits
        + pair_flags
        + bit_copies
        + max_parallel_adders * adder_ancillas
        + sign_copies
        + word_bits
        + word_bits
        + signal_bits
    )

    f = next(s for s in expr.free_symbols if s.name == "f")
    expr = sp.collect(fast_simplify_logs(sp.simplify(expr)), [n**2, n]).xreplace({
        sp.log(2): 1,
        sp.log(f+3): 2*sp.log(f),
    })
    term_10 = 2 * f * sp.log(f) + sp.Rational(5, 2) * f + 6 * sp.log(f) + 10
    term_6 = 2 * f * sp.log(f) + sp.Rational(5, 2) * f + 6 * sp.log(f) + 6
    expr_approx = expr.xreplace({
        term_10: 5 * f * sp.log(f),
        term_6: 5 * f * sp.log(f),
        2 * term_10: 10 * f * sp.log(f),
        2 * term_6: 10 * f * sp.log(f),
    }).subs({
        2 * term_6: 10 * f * sp.log(f),
    })
    return expr_approx


def metropolis_hastings_energy_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    return sp.Integer(0)


def metropolis_hastings_energy_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    """
    Dense worst-case Toffoli-count upper bound.
    """
    n = sp.sympify(n)
    word_bits = metropolis_hastings_energy_acc_word_bits(n, sum_abs, eps)

    n_terms = n * (n + 1)
    pair_terms = n * (n - 1)

    adder_toffoli = kogge_stone_in_place_adder_toffoli_count(
        word_bits,
        with_carry_out=False,
    )

    pair_toffoli = 4 * pair_terms
    clip_toffoli = word_bits - 1
    tree_toffoli = 2 * (n_terms - 1) * adder_toffoli
    constant_add_toffoli = adder_toffoli

    expr = sp.simplify(
        pair_toffoli
        + clip_toffoli
        + tree_toffoli
        + constant_add_toffoli
    )
    f = next(s for s in expr.free_symbols if s.name == "f")
    # f >= 10
    term_m4 = 32 * f * sp.log(f) - 8 * f + 96 * sp.log(f) - 4
    term_m12 = 32 * f * sp.log(f) - 8 * f + 96 * sp.log(f) - 12
    term_m16 = 16 * f * sp.log(f) - 5 * f + 48 * sp.log(f) - 6
    expr = fast_simplify_logs(expr).xreplace({
        sp.log(2): 1,
        sp.log(f+3): 2*sp.log(f),
        sp.log(n+1): sp.log(n),
    }).collect([n**2, n])
    return expr.xreplace({
        term_m4: 42 * f * sp.log(f),
        term_m12: 42 * f * sp.log(f),
        2 * term_m4: 84 * f * sp.log(f),
        2 * term_m12: 84 * f * sp.log(f),
    }).subs({
        term_m16: 21 * f * sp.log(f),
    })


def metropolis_hastings_energy_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    return sp.Integer(0)


def metropolis_hastings_energy_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    """
    Dense worst-case non-Clifford-depth upper bound in CCX-layer units.

    The addition reduction is a logarithmic tree of Kogge-Stone adders.  The
    extra constant in `adder_level_depth` covers the concrete DAG-layer skew
    introduced by reversible term loading/unloading and by reusing clean adder
    work blocks between tree levels.  It does not change the asymptotic depth:

        O(log(T) * log(w)),

    where T = n(n + 1) is the dense two-sided term count and w is the accumulator
    word size.
    """
    n = sp.sympify(n)
    word_bits = metropolis_hastings_energy_acc_word_bits(n, sum_abs, eps)

    n_terms = n * (n + 1)
    tree_height = sp.log(n_terms, 2) + 1 # ceiling out

    adder_depth = kogge_stone_in_place_adder_nc_depth(
        word_bits,
        with_carry_out=False,
    )

    adder_level_depth = adder_depth + 4

    non_adder_envelope_depth = 8

    expr = sp.simplify(
        non_adder_envelope_depth
        + (2 * tree_height + 1) * adder_level_depth
    )
    f = next(s for s in expr.free_symbols if s.name == "f")
    return fast_simplify_logs(expr).xreplace({
        sp.log(2): 1,
        sp.log(f+3): 2*sp.log(f),
        sp.log(n+1): sp.log(n),
    })


def metropolis_hastings_energy_t_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    return sp.Integer(0)