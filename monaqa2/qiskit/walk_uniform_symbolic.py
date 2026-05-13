import sympy as sp

from monaqa2.qiskit.utils_symbolic import get_symbol_name, advanced_initial_simplify, replace_shifted_logs, leading_terms_upper_bound

from monaqa2.qiskit.multi_controlled_not_symbolic import (
    mcx_nc_depth,
    mcx_t_count,
    mcx_rz_count,
    mcx_toffoli_count,
)
from monaqa2.qiskit.metropolis_hastings_energy_symbolic import (
    metropolis_hastings_energy_upper_bound_energy_diff,
    metropolis_hastings_energy_fractional_bits,
)
from monaqa2.qiskit.kogge_stone_in_place_adder_symbolic import (
    kogge_stone_in_place_adder_number_qubits,
    kogge_stone_in_place_adder_nc_depth,
    kogge_stone_in_place_adder_toffoli_count,
)
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
from monaqa2.qiskit.glauber_arithmetic_symbolic import (
    glauber_arithmetic_number_qubits,
    glauber_arithmetic_nc_depth,
    glauber_arithmetic_t_count,
    glauber_arithmetic_rz_count,
    glauber_arithmetic_toffoli_count,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _check_coin(coin: str) -> None:
    if coin not in {"mh", "glauber"}:
        raise ValueError("coin must be either 'mh' or 'glauber'.")


def _require_glauber_terms(
    n_terms_z: sp.Expr | None,
    n_terms_zz: sp.Expr | None,
) -> tuple[sp.Expr, sp.Expr]:
    if n_terms_z is None or n_terms_zz is None:
        raise ValueError("coin='glauber' requires n_terms_z and n_terms_zz.")

    return sp.sympify(n_terms_z), sp.sympify(n_terms_zz)


def _require_eps(eps: sp.Expr | None, name: str = "eps") -> sp.Expr:
    if eps is None:
        raise ValueError(f"{name} is required for this resource estimate.")

    return sp.sympify(eps)


def _require_normalization(normalization: sp.Expr | None) -> sp.Expr:
    if normalization is None:
        raise ValueError("normalization is required when d_px is not supplied explicitly.")

    return sp.sympify(normalization)


def _mh_energy_term_count(n: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    return sp.simplify(n * (n + 1))


def _mh_signal_bits_from_f(f: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.sympify(f) + 1)


def _mh_acc_word_bits_from_f(f: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.sympify(f) + 3)


def walk_uniform_mh_fixed_point_eps(eps: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.sympify(eps) / 2)


def walk_uniform_mh_sqrt_exp_eps(eps: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.sympify(eps) / 2)


def walk_uniform_mh_normalization_from_f(sum_abs: sp.Expr, f: sp.Expr) -> sp.Expr:
    upper = metropolis_hastings_energy_upper_bound_energy_diff(sum_abs)
    return sp.simplify(upper / (2 - 2 ** (-sp.sympify(f))))


def walk_uniform_mh_normalization(n: sp.Expr, sum_abs: sp.Expr, eps: sp.Expr) -> sp.Expr:
    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    f = metropolis_hastings_energy_fractional_bits(n, sum_abs, eps_fixed_point)
    return walk_uniform_mh_normalization_from_f(sum_abs, f)


def walk_uniform_mh_fractional_bits_from_eps(
    n: sp.Expr,
    beta: sp.Expr,
    normalization: sp.Expr,
    eps_fx: sp.Expr,
) -> sp.Expr:
    """
    Explicit precision rule for the MH fixed-point block.

    It solves

        T u exp(2 beta D u) <= eps_fx^2,   u = 2^{-f},

    where T = n(n+1) and D is the fixed-point normalization.
    """
    n = sp.sympify(n)
    beta = sp.sympify(beta)
    normalization = sp.sympify(normalization)
    eps_fx = sp.sympify(eps_fx)
    T = _mh_energy_term_count(n)

    return sp.simplify(
        sp.ceiling(
            sp.log(
                2 * beta * normalization / sp.LambertW(2 * beta * normalization * eps_fx**2 / T),
                2,
            )
        )
    )


def walk_uniform_mh_sqrt_exp_degree_from_eps(
    beta: sp.Expr,
    normalization: sp.Expr,
    eps_px: sp.Expr,
) -> sp.Expr:
    """
    Explicit degree rule for the sqrt-exp GQSP block.

    The degree bound uses the b-independent upper bound alpha_signal <= 1 and
    the square-root scale beta * D / 2.
    """
    beta = sp.sympify(beta)
    normalization = sp.sympify(normalization)
    eps_px = sp.sympify(eps_px)

    return sp.simplify(
        sp.ceiling(sp.E * beta * normalization / 2 + sp.log(4 / eps_px) + 1)
    )


def _resolve_mh_parameters(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Resolve the parameters used by the Metropolis-Hastings coin.

    Preferred explicit mode:
        pass f and d_px.

    Mixed mode:
        pass f and eps_px plus normalization, and d_px is derived.

    Backward-compatible mode:
        pass eps only. Then eps is split into eps/2 and eps/2. The old
        fixed-point precision rule is used for f, while d_px is derived from the
        corrected sqrt-exp degree formula.
    """
    n = sp.sympify(n)
    beta = sp.sympify(beta)

    if eps_fx is None and eps is not None:
        eps_fx = walk_uniform_mh_fixed_point_eps(eps)
    if eps_px is None and eps is not None:
        eps_px = walk_uniform_mh_sqrt_exp_eps(eps)

    if f is None:
        if eps_fx is None:
            raise ValueError("Either f or eps_fx/eps must be supplied for the MH coin.")

        if normalization is None:
            # Backward-compatible explicit formula used by the fixed-point block.
            f = metropolis_hastings_energy_fractional_bits(n, sum_abs, eps_fx)
        else:
            f = walk_uniform_mh_fractional_bits_from_eps(n, beta, normalization, eps_fx)

    f = sp.sympify(f)

    if normalization is None:
        normalization = walk_uniform_mh_normalization_from_f(sum_abs, f)
    else:
        normalization = sp.sympify(normalization)

    if d_px is None:
        if eps_px is None:
            raise ValueError("Either d_px or eps_px/eps must be supplied for the MH coin.")
        d_px = walk_uniform_mh_sqrt_exp_degree_from_eps(beta, normalization, eps_px)

    return sp.simplify(f), sp.simplify(d_px), sp.simplify(normalization)


# -----------------------------------------------------------------------------
# Accept path and reflection resources
# -----------------------------------------------------------------------------


def walk_uniform_accept_path_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(n + coins - 2)


def walk_uniform_reflection_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(n + coins - 3)


def walk_uniform_accept_path_number_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * n + coins + walk_uniform_accept_path_work_qubits(n, coins))


def walk_uniform_reflection_number_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * n + coins + walk_uniform_reflection_work_qubits(n, coins))


def walk_uniform_accept_path_toffoli_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_toffoli_count(coins) + 3 * n)


def walk_uniform_accept_path_t_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_t_count(coins) + 21 * n)


def walk_uniform_accept_path_rz_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_rz_count(coins))


def walk_uniform_accept_path_nc_depth(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_nc_depth(coins) + 3)


def walk_uniform_reflection_toffoli_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1
    return sp.simplify(mcx_toffoli_count(controls))


def walk_uniform_reflection_t_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1
    return sp.simplify(mcx_t_count(controls))


def walk_uniform_reflection_rz_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1
    return sp.simplify(mcx_rz_count(controls))


def walk_uniform_reflection_nc_depth(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1
    return sp.simplify(mcx_nc_depth(controls))


# -----------------------------------------------------------------------------
# Explicit Metropolis-Hastings coin resources in terms of f and d_px
# -----------------------------------------------------------------------------


def walk_uniform_mh_energy_number_qubits_from_f(n: sp.Expr, f: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    f = sp.sympify(f)
    T = _mh_energy_term_count(n)
    word_bits = _mh_acc_word_bits_from_f(f)
    signal_bits = _mh_signal_bits_from_f(f)

    copies_per_side = n * (n - 1)
    pair_flags = n * (n - 1)
    bit_copies = T * (word_bits - 1)
    sign_copies = word_bits - 2

    adder_qubits = kogge_stone_in_place_adder_number_qubits(word_bits, with_carry_out=False)
    adder_ancillas = adder_qubits - 2 * word_bits
    max_parallel_adders = sp.floor(T / 2) + 1

    return sp.simplify(
        2 * n
        + 2 * copies_per_side
        + T * word_bits
        + pair_flags
        + bit_copies
        + max_parallel_adders * adder_ancillas
        + sign_copies
        + word_bits
        + word_bits
        + signal_bits
    )


def walk_uniform_mh_energy_toffoli_count_from_f(n: sp.Expr, f: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    f = sp.sympify(f)
    T = _mh_energy_term_count(n)
    word_bits = _mh_acc_word_bits_from_f(f)

    pair_terms = n * (n - 1)
    adder_toffoli = kogge_stone_in_place_adder_toffoli_count(word_bits, with_carry_out=False)

    pair_toffoli = 4 * pair_terms
    clip_toffoli = word_bits - 1
    tree_toffoli = 2 * (T - 1) * adder_toffoli
    constant_add_toffoli = adder_toffoli

    return sp.simplify(pair_toffoli + clip_toffoli + tree_toffoli + constant_add_toffoli)


def walk_uniform_mh_energy_t_count_from_f(n: sp.Expr, f: sp.Expr) -> sp.Expr:
    return sp.simplify(7 * walk_uniform_mh_energy_toffoli_count_from_f(n, f))


def walk_uniform_mh_energy_rz_count_from_f(n: sp.Expr, f: sp.Expr) -> sp.Expr:
    return sp.Integer(0)


def walk_uniform_mh_energy_nc_depth_from_f(n: sp.Expr, f: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    f = sp.sympify(f)
    T = _mh_energy_term_count(n)
    word_bits = _mh_acc_word_bits_from_f(f)

    tree_height = sp.ceiling(sp.log(T, 2))
    adder_depth = kogge_stone_in_place_adder_nc_depth(word_bits, with_carry_out=False)
    return sp.simplify(8 + (2 * tree_height + 1) * (adder_depth + 4))


def walk_uniform_mh_sqrt_exp_number_qubits_from_f_d(f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    b = _mh_signal_bits_from_f(f)
    return sp.simplify(
        controlled_qubitized_ising_tf_number_qubits(
            b,
            n_terms_z=b,
            n_terms_zz=0,
            n_terms_x=0,
        )
    )


def walk_uniform_mh_sqrt_exp_toffoli_count_from_f_d(f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    b = _mh_signal_bits_from_f(f)
    d_px = sp.sympify(d_px)

    controlled_walk = controlled_qubitized_ising_tf_toffoli_count(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    inverse_walk = qubitized_ising_tf_toffoli_count(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    return sp.simplify(2 * d_px * controlled_walk + d_px * inverse_walk)


def walk_uniform_mh_sqrt_exp_t_count_from_f_d(f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    b = _mh_signal_bits_from_f(f)
    d_px = sp.sympify(d_px)

    controlled_walk = controlled_qubitized_ising_tf_t_count(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    inverse_walk = qubitized_ising_tf_t_count(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    return sp.simplify(2 * d_px * controlled_walk + d_px * inverse_walk)


def walk_uniform_mh_sqrt_exp_rz_count_from_f_d(f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    b = _mh_signal_bits_from_f(f)
    d_px = sp.sympify(d_px)

    controlled_walk = controlled_qubitized_ising_tf_rz_count(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    inverse_walk = qubitized_ising_tf_rz_count(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    return sp.simplify(2 * d_px * controlled_walk + d_px * inverse_walk + 3 * (2 * d_px + 1))


def walk_uniform_mh_sqrt_exp_nc_depth_from_f_d(f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    b = _mh_signal_bits_from_f(f)
    d_px = sp.sympify(d_px)

    controlled_walk = controlled_qubitized_ising_tf_nc_depth(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    inverse_walk = qubitized_ising_tf_nc_depth(b, n_terms_z=b, n_terms_zz=0, n_terms_x=0)
    return sp.simplify(2 * d_px * controlled_walk + d_px * inverse_walk + 3 * (2 * d_px + 1))


def walk_uniform_mh_coin_core_qubits_from_f_d(n: sp.Expr, f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    b = _mh_signal_bits_from_f(f)

    energy_qubits = walk_uniform_mh_energy_number_qubits_from_f(n, f)
    energy_work = energy_qubits - 2 * n

    sqrt_qubits = walk_uniform_mh_sqrt_exp_number_qubits_from_f_d(f, d_px)
    sqrt_extra = sqrt_qubits - b

    return sp.simplify(energy_work + sqrt_extra)


def walk_uniform_mh_coin_toffoli_count_from_f_d(n: sp.Expr, f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    energy = walk_uniform_mh_energy_toffoli_count_from_f(n, f)
    sqrt_exp = walk_uniform_mh_sqrt_exp_toffoli_count_from_f_d(f, d_px)
    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_mh_coin_t_count_from_f_d(n: sp.Expr, f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    energy = walk_uniform_mh_energy_t_count_from_f(n, f)
    sqrt_exp = walk_uniform_mh_sqrt_exp_t_count_from_f_d(f, d_px)
    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_mh_coin_rz_count_from_f_d(n: sp.Expr, f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    energy = walk_uniform_mh_energy_rz_count_from_f(n, f)
    sqrt_exp = walk_uniform_mh_sqrt_exp_rz_count_from_f_d(f, d_px)
    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_mh_coin_nc_depth_from_f_d(n: sp.Expr, f: sp.Expr, d_px: sp.Expr) -> sp.Expr:
    energy = walk_uniform_mh_energy_nc_depth_from_f(n, f)
    sqrt_exp = walk_uniform_mh_sqrt_exp_nc_depth_from_f_d(f, d_px)
    return sp.simplify(4 * energy + 2 * sqrt_exp)


# -----------------------------------------------------------------------------
# Backward-compatible MH wrappers with optional explicit f/d_px parameters
# -----------------------------------------------------------------------------


def walk_uniform_mh_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    f, d_px, _ = _resolve_mh_parameters(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    return walk_uniform_mh_coin_core_qubits_from_f_d(n, f, d_px)


def walk_uniform_mh_coin_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    f, d_px, _ = _resolve_mh_parameters(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    return walk_uniform_mh_coin_toffoli_count_from_f_d(n, f, d_px)


def walk_uniform_mh_coin_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    f, d_px, _ = _resolve_mh_parameters(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    return walk_uniform_mh_coin_t_count_from_f_d(n, f, d_px)


def walk_uniform_mh_coin_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    f, d_px, _ = _resolve_mh_parameters(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    return walk_uniform_mh_coin_rz_count_from_f_d(n, f, d_px)


def walk_uniform_mh_coin_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    f, d_px, _ = _resolve_mh_parameters(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    return walk_uniform_mh_coin_nc_depth_from_f_d(n, f, d_px)


# -----------------------------------------------------------------------------
# Glauber coin. This path supports either eps-based degree selection or an
# explicit d_px supplied by the caller.
# -----------------------------------------------------------------------------


def _glauber_delta_terms(n_terms_z: sp.Expr, n_terms_zz: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return (
        sp.simplify(2 * sp.sympify(n_terms_z)),
        sp.simplify(2 * sp.sympify(n_terms_zz)),
        sp.Integer(0),
    )


def walk_uniform_glauber_coin_core_qubits_from_d(
    n: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    d_px: sp.Expr,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    delta_z, delta_zz, delta_x = _glauber_delta_terms(n_terms_z, n_terms_zz)

    return sp.simplify(
        controlled_qubitized_ising_tf_number_qubits(
            n_system,
            n_terms_z=delta_z,
            n_terms_zz=delta_zz,
            n_terms_x=delta_x,
        )
        - n_system
    )


def walk_uniform_glauber_coin_toffoli_count_from_d(
    n: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    d_px: sp.Expr,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    d_px = sp.sympify(d_px)
    delta_z, delta_zz, delta_x = _glauber_delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = controlled_qubitized_ising_tf_toffoli_count(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x)
    inverse_walk = qubitized_ising_tf_toffoli_count(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x)

    return sp.simplify(2 * (2 * d_px * controlled_walk + d_px * inverse_walk))


def walk_uniform_glauber_coin_t_count_from_d(
    n: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    d_px: sp.Expr,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    d_px = sp.sympify(d_px)
    delta_z, delta_zz, delta_x = _glauber_delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = controlled_qubitized_ising_tf_t_count(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x)
    inverse_walk = qubitized_ising_tf_t_count(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x)

    return sp.simplify(2 * (2 * d_px * controlled_walk + d_px * inverse_walk))


def walk_uniform_glauber_coin_rz_count_from_d(
    n: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    d_px: sp.Expr,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    d_px = sp.sympify(d_px)
    delta_z, delta_zz, delta_x = _glauber_delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = controlled_qubitized_ising_tf_rz_count(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x)
    inverse_walk = qubitized_ising_tf_rz_count(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x)
    phase_blocks = 2 * d_px + 1

    return sp.simplify(2 * (2 * d_px * controlled_walk + d_px * inverse_walk + 3 * phase_blocks))


def walk_uniform_glauber_coin_nc_depth_from_d(
    n: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    d_px: sp.Expr,
) -> sp.Expr:
    n_system = 2 * sp.sympify(n)
    d_px = sp.sympify(d_px)
    delta_z, delta_zz, delta_x = _glauber_delta_terms(n_terms_z, n_terms_zz)

    controlled_walk = sp.ceiling(controlled_qubitized_ising_tf_nc_depth(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x))
    inverse_walk = sp.ceiling(qubitized_ising_tf_nc_depth(n_system, n_terms_z=delta_z, n_terms_zz=delta_zz, n_terms_x=delta_x))
    phase_blocks = 2 * d_px + 1

    return sp.simplify(2 * sp.ceiling(2 * d_px * controlled_walk + d_px * inverse_walk + 3 * phase_blocks))


def walk_uniform_glauber_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
    *,
    d_px: sp.Expr | None = None,
) -> sp.Expr:
    if d_px is not None:
        return walk_uniform_glauber_coin_core_qubits_from_d(n, n_terms_z, n_terms_zz, d_px)

    alpha = sp.simplify(2 * sp.sympify(sum_abs))
    glauber_qubits = glauber_arithmetic_number_qubits(n, alpha, beta, _require_eps(eps), n_terms_z, n_terms_zz, a)
    return sp.simplify(glauber_qubits - 2 * sp.sympify(n))


def walk_uniform_glauber_coin_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
    *,
    d_px: sp.Expr | None = None,
) -> sp.Expr:
    if d_px is not None:
        return walk_uniform_glauber_coin_toffoli_count_from_d(n, n_terms_z, n_terms_zz, d_px)

    alpha = sp.simplify(2 * sp.sympify(sum_abs))
    return sp.simplify(2 * glauber_arithmetic_toffoli_count(n, alpha, beta, _require_eps(eps), n_terms_z, n_terms_zz, a))


def walk_uniform_glauber_coin_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
    *,
    d_px: sp.Expr | None = None,
) -> sp.Expr:
    if d_px is not None:
        return walk_uniform_glauber_coin_t_count_from_d(n, n_terms_z, n_terms_zz, d_px)

    alpha = sp.simplify(2 * sp.sympify(sum_abs))
    return sp.simplify(2 * glauber_arithmetic_t_count(n, alpha, beta, _require_eps(eps), n_terms_z, n_terms_zz, a))


def walk_uniform_glauber_coin_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
    *,
    d_px: sp.Expr | None = None,
) -> sp.Expr:
    if d_px is not None:
        return walk_uniform_glauber_coin_rz_count_from_d(n, n_terms_z, n_terms_zz, d_px)

    alpha = sp.simplify(2 * sp.sympify(sum_abs))
    return sp.simplify(2 * glauber_arithmetic_rz_count(n, alpha, beta, _require_eps(eps), n_terms_z, n_terms_zz, a))


def walk_uniform_glauber_coin_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
    *,
    d_px: sp.Expr | None = None,
) -> sp.Expr:
    if d_px is not None:
        return walk_uniform_glauber_coin_nc_depth_from_d(n, n_terms_z, n_terms_zz, d_px)

    alpha = sp.simplify(2 * sp.sympify(sum_abs))
    return sp.simplify(2 * glauber_arithmetic_nc_depth(n, alpha, beta, _require_eps(eps), n_terms_z, n_terms_zz, a))


# -----------------------------------------------------------------------------
# Coin dispatchers
# -----------------------------------------------------------------------------


def walk_uniform_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_core_qubits(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_core_qubits(n, sum_abs, beta, eps, z_terms, zz_terms, a, d_px=d_px)


def walk_uniform_coins(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    expr = sp.simplify(
        walk_uniform_coin_core_qubits(
            n,
            sum_abs,
            beta,
            eps,
            coin=coin,
            n_terms_z=n_terms_z,
            n_terms_zz=n_terms_zz,
            a=a,
            f=f,
            d_px=d_px,
            normalization=normalization,
            eps_fx=eps_fx,
            eps_px=eps_px,
        )
    )
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return expr


def walk_uniform_coin_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_toffoli_count(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_toffoli_count(n, sum_abs, beta, eps, z_terms, zz_terms, a, d_px=d_px)


def walk_uniform_coin_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_t_count(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_t_count(n, sum_abs, beta, eps, z_terms, zz_terms, a, d_px=d_px)


def walk_uniform_coin_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_rz_count(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_rz_count(n, sum_abs, beta, eps, z_terms, zz_terms, a, d_px=d_px)


def walk_uniform_coin_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_nc_depth(n, sum_abs, beta, eps, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_nc_depth(n, sum_abs, beta, eps, z_terms, zz_terms, a, d_px=d_px)


# -----------------------------------------------------------------------------
# Full uniform-walk resources
# -----------------------------------------------------------------------------


def walk_uniform_number_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = walk_uniform_coins(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    expr = sp.simplify(2 * n + coins + walk_uniform_accept_path_work_qubits(n, coins) + walk_uniform_reflection_work_qubits(n, coins))
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return expr


def walk_uniform_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    coins = walk_uniform_coins(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    expr = sp.simplify(
        walk_uniform_coin_toffoli_count(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
        + walk_uniform_accept_path_toffoli_count(n, coins)
        + walk_uniform_reflection_toffoli_count(n, coins)
    )
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return expr


def walk_uniform_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    coins = walk_uniform_coins(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    expr = sp.simplify(
        walk_uniform_coin_t_count(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
        + walk_uniform_accept_path_t_count(n, coins)
        + walk_uniform_reflection_t_count(n, coins)
    )
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return expr


def walk_uniform_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    coins = walk_uniform_coins(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    expr = sp.simplify(
        walk_uniform_coin_rz_count(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
        + walk_uniform_accept_path_rz_count(n, coins)
        + walk_uniform_reflection_rz_count(n, coins)
    )
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return expr


def walk_uniform_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    eps_fx: sp.Expr | None = None,
    eps_px: sp.Expr | None = None,
) -> sp.Expr:
    coins = walk_uniform_coins(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
    expr = sp.simplify(
        walk_uniform_coin_nc_depth(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a, f=f, d_px=d_px, normalization=normalization, eps_fx=eps_fx, eps_px=eps_px)
        + walk_uniform_accept_path_nc_depth(n, coins)
        + walk_uniform_reflection_nc_depth(n, coins)
    )
    expr = advanced_initial_simplify(expr)
    expr = replace_shifted_logs(expr, variables=[d_px, f, n])
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    expr = advanced_initial_simplify(expr).xreplace({
        256: 260,
        259: 260, 
        6*sp.log(12): 0})
    return expr
