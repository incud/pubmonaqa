import sympy as sp

from monaqa2.qiskit.multi_controlled_not_symbolic import (
    mcx_nc_depth,
    mcx_t_count,
    mcx_rz_count,
    mcx_toffoli_count,
)
from monaqa2.qiskit.metropolis_hastings_energy_symbolic import (
    metropolis_hastings_energy_upper_bound_energy_diff,
    metropolis_hastings_energy_fractional_bits,
    metropolis_hastings_energy_signal_bits,
    metropolis_hastings_energy_number_qubits,
    metropolis_hastings_energy_nc_depth,
    metropolis_hastings_energy_t_count,
    metropolis_hastings_energy_rz_count,
    metropolis_hastings_energy_toffoli_count,
)
from monaqa2.qiskit.sqrt_exp_arithmetic_symbolic import (
    sqrt_exp_arithmetic_number_qubits,
    sqrt_exp_arithmetic_nc_depth,
    sqrt_exp_arithmetic_t_count,
    sqrt_exp_arithmetic_rz_count,
    sqrt_exp_arithmetic_toffoli_count,
)
from monaqa2.qiskit.glauber_arithmetic_symbolic import (
    glauber_arithmetic_number_qubits,
    glauber_arithmetic_nc_depth,
    glauber_arithmetic_t_count,
    glauber_arithmetic_rz_count,
    glauber_arithmetic_toffoli_count,
)


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


def walk_uniform_accept_path_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(1 + sp.Max(n - 1, coins - 2))


def walk_uniform_reflection_work_qubits(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)

    return sp.simplify(n + coins - 3)


def walk_uniform_accept_path_number_qubits(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)

    return sp.simplify(
        2 * n
        + coins
        + walk_uniform_accept_path_work_qubits(n, coins)
    )


def walk_uniform_reflection_number_qubits(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)

    return sp.simplify(
        2 * n
        + coins
        + walk_uniform_reflection_work_qubits(n, coins)
    )


def walk_uniform_accept_path_toffoli_count(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)

    return sp.simplify(
        2 * mcx_toffoli_count(coins)
        + 3 * n
    )


def walk_uniform_accept_path_t_count(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)

    return sp.simplify(
        2 * mcx_t_count(coins)
        + 21 * n
    )


def walk_uniform_accept_path_rz_count(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    coins = sp.sympify(coins)

    return sp.simplify(2 * mcx_rz_count(coins))


def walk_uniform_accept_path_nc_depth(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    coins = sp.sympify(coins)

    return sp.simplify(
        2 * mcx_nc_depth(coins)
        + 3
    )


def walk_uniform_reflection_toffoli_count(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1

    return sp.simplify(mcx_toffoli_count(controls))


def walk_uniform_reflection_t_count(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1

    return sp.simplify(mcx_t_count(controls))


def walk_uniform_reflection_rz_count(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1

    return sp.simplify(mcx_rz_count(controls))


def walk_uniform_reflection_nc_depth(
    n: sp.Expr,
    coins: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    controls = n + coins - 1

    return sp.simplify(mcx_nc_depth(controls))


def walk_uniform_mh_fixed_point_eps(
    eps: sp.Expr,
) -> sp.Expr:
    return sp.simplify(sp.sympify(eps) / 2)


def walk_uniform_mh_sqrt_exp_eps(
    eps: sp.Expr,
) -> sp.Expr:
    return sp.simplify(sp.sympify(eps) / 2)


def walk_uniform_mh_normalization(
    n: sp.Expr,
    sum_abs: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    upper = metropolis_hastings_energy_upper_bound_energy_diff(sum_abs)
    fractional_bits = metropolis_hastings_energy_fractional_bits(
        n,
        sum_abs,
        eps_fixed_point,
    )

    return sp.simplify(
        upper / (2 - 2 ** (-fractional_bits))
    )


def walk_uniform_mh_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    n = sp.sympify(n)
    beta = sp.sympify(beta)

    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    eps_sqrt_exp = walk_uniform_mh_sqrt_exp_eps(eps)

    energy_qubits = metropolis_hastings_energy_number_qubits(
        n,
        sum_abs,
        eps_fixed_point,
    )
    energy_work = sp.simplify(energy_qubits - 2 * n)

    signal_bits = metropolis_hastings_energy_signal_bits(
        n,
        sum_abs,
        eps_fixed_point,
    )
    normalization = walk_uniform_mh_normalization(
        n,
        sum_abs,
        eps,
    )

    sqrt_qubits = sqrt_exp_arithmetic_number_qubits(
        signal_bits,
        beta,
        normalization,
        eps_sqrt_exp,
    )
    sqrt_extra = sp.simplify(sqrt_qubits - signal_bits)

    return sp.simplify(energy_work + sqrt_extra)


def walk_uniform_glauber_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
) -> sp.Expr:
    n = sp.sympify(n)
    alpha = sp.simplify(2 * sp.sympify(sum_abs))

    glauber_qubits = glauber_arithmetic_number_qubits(
        n,
        alpha,
        beta,
        eps,
        n_terms_z,
        n_terms_zz,
        a,
    )

    return sp.simplify(glauber_qubits - 2 * n)


def walk_uniform_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_core_qubits(
            n,
            sum_abs,
            beta,
            eps,
        )

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)

    return walk_uniform_glauber_coin_core_qubits(
        n,
        sum_abs,
        beta,
        eps,
        z_terms,
        zz_terms,
        a,
    )


def walk_uniform_coins(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    core = walk_uniform_coin_core_qubits(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
    )

    return sp.Max(3, core)


def walk_uniform_number_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    n = sp.sympify(n)
    coins = walk_uniform_coins(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
    )

    return sp.simplify(
        2 * n
        + coins
        + walk_uniform_accept_path_work_qubits(n, coins)
        + walk_uniform_reflection_work_qubits(n, coins)
    )


def walk_uniform_mh_coin_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    eps_sqrt_exp = walk_uniform_mh_sqrt_exp_eps(eps)

    signal_bits = metropolis_hastings_energy_signal_bits(
        n,
        sum_abs,
        eps_fixed_point,
    )
    normalization = walk_uniform_mh_normalization(
        n,
        sum_abs,
        eps,
    )

    energy = metropolis_hastings_energy_toffoli_count(
        n,
        sum_abs,
        eps_fixed_point,
    )
    sqrt_exp = sqrt_exp_arithmetic_toffoli_count(
        signal_bits,
        beta,
        normalization,
        eps_sqrt_exp,
    )

    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_mh_coin_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    eps_sqrt_exp = walk_uniform_mh_sqrt_exp_eps(eps)

    signal_bits = metropolis_hastings_energy_signal_bits(
        n,
        sum_abs,
        eps_fixed_point,
    )
    normalization = walk_uniform_mh_normalization(
        n,
        sum_abs,
        eps,
    )

    energy = metropolis_hastings_energy_t_count(
        n,
        sum_abs,
        eps_fixed_point,
    )
    sqrt_exp = sqrt_exp_arithmetic_t_count(
        signal_bits,
        beta,
        normalization,
        eps_sqrt_exp,
    )

    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_mh_coin_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    eps_sqrt_exp = walk_uniform_mh_sqrt_exp_eps(eps)

    signal_bits = metropolis_hastings_energy_signal_bits(
        n,
        sum_abs,
        eps_fixed_point,
    )
    normalization = walk_uniform_mh_normalization(
        n,
        sum_abs,
        eps,
    )

    energy = metropolis_hastings_energy_rz_count(
        n,
        sum_abs,
        eps_fixed_point,
    )
    sqrt_exp = sqrt_exp_arithmetic_rz_count(
        signal_bits,
        beta,
        normalization,
        eps_sqrt_exp,
    )

    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_mh_coin_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
) -> sp.Expr:
    eps_fixed_point = walk_uniform_mh_fixed_point_eps(eps)
    eps_sqrt_exp = walk_uniform_mh_sqrt_exp_eps(eps)

    signal_bits = metropolis_hastings_energy_signal_bits(
        n,
        sum_abs,
        eps_fixed_point,
    )
    normalization = walk_uniform_mh_normalization(
        n,
        sum_abs,
        eps,
    )

    energy = metropolis_hastings_energy_nc_depth(
        n,
        sum_abs,
        eps_fixed_point,
    )
    sqrt_exp = sqrt_exp_arithmetic_nc_depth(
        signal_bits,
        beta,
        normalization,
        eps_sqrt_exp,
    )

    return sp.simplify(4 * energy + 2 * sqrt_exp)


def walk_uniform_glauber_coin_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
) -> sp.Expr:
    alpha = sp.simplify(2 * sp.sympify(sum_abs))

    return sp.simplify(
        2 * glauber_arithmetic_toffoli_count(
            n,
            alpha,
            beta,
            eps,
            n_terms_z,
            n_terms_zz,
            a,
        )
    )


def walk_uniform_glauber_coin_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
) -> sp.Expr:
    alpha = sp.simplify(2 * sp.sympify(sum_abs))

    return sp.simplify(
        2 * glauber_arithmetic_t_count(
            n,
            alpha,
            beta,
            eps,
            n_terms_z,
            n_terms_zz,
            a,
        )
    )


def walk_uniform_glauber_coin_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
) -> sp.Expr:
    alpha = sp.simplify(2 * sp.sympify(sum_abs))

    return sp.simplify(
        2 * glauber_arithmetic_rz_count(
            n,
            alpha,
            beta,
            eps,
            n_terms_z,
            n_terms_zz,
            a,
        )
    )


def walk_uniform_glauber_coin_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    n_terms_z: sp.Expr,
    n_terms_zz: sp.Expr,
    a: sp.Expr = 1,
) -> sp.Expr:
    alpha = sp.simplify(2 * sp.sympify(sum_abs))

    return sp.simplify(
        2 * glauber_arithmetic_nc_depth(
            n,
            alpha,
            beta,
            eps,
            n_terms_z,
            n_terms_zz,
            a,
        )
    )


def walk_uniform_coin_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_toffoli_count(n, sum_abs, beta, eps)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)

    return walk_uniform_glauber_coin_toffoli_count(
        n,
        sum_abs,
        beta,
        eps,
        z_terms,
        zz_terms,
        a,
    )


def walk_uniform_coin_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_t_count(n, sum_abs, beta, eps)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)

    return walk_uniform_glauber_coin_t_count(
        n,
        sum_abs,
        beta,
        eps,
        z_terms,
        zz_terms,
        a,
    )


def walk_uniform_coin_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_rz_count(n, sum_abs, beta, eps)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)

    return walk_uniform_glauber_coin_rz_count(
        n,
        sum_abs,
        beta,
        eps,
        z_terms,
        zz_terms,
        a,
    )


def walk_uniform_coin_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    _check_coin(coin)

    if coin == "mh":
        return walk_uniform_mh_coin_nc_depth(n, sum_abs, beta, eps)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)

    return walk_uniform_glauber_coin_nc_depth(
        n,
        sum_abs,
        beta,
        eps,
        z_terms,
        zz_terms,
        a,
    )


def walk_uniform_toffoli_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_uniform_coins(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
    )

    return sp.simplify(
        walk_uniform_coin_toffoli_count(
            n,
            sum_abs,
            beta,
            eps,
            coin=coin,
            n_terms_z=n_terms_z,
            n_terms_zz=n_terms_zz,
            a=a,
        )
        + walk_uniform_accept_path_toffoli_count(n, coins)
        + walk_uniform_reflection_toffoli_count(n, coins)
    )


def walk_uniform_t_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_uniform_coins(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
    )

    return sp.simplify(
        walk_uniform_coin_t_count(
            n,
            sum_abs,
            beta,
            eps,
            coin=coin,
            n_terms_z=n_terms_z,
            n_terms_zz=n_terms_zz,
            a=a,
        )
        + walk_uniform_accept_path_t_count(n, coins)
        + walk_uniform_reflection_t_count(n, coins)
    )


def walk_uniform_rz_count(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_uniform_coins(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
    )

    return sp.simplify(
        walk_uniform_coin_rz_count(
            n,
            sum_abs,
            beta,
            eps,
            coin=coin,
            n_terms_z=n_terms_z,
            n_terms_zz=n_terms_zz,
            a=a,
        )
        + walk_uniform_accept_path_rz_count(n, coins)
        + walk_uniform_reflection_rz_count(n, coins)
    )


def walk_uniform_nc_depth(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_uniform_coins(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
    )

    return walk_uniform_coin_nc_depth(
            n,
            sum_abs,
            beta,
            eps,
            coin=coin,
            n_terms_z=n_terms_z,
            n_terms_zz=n_terms_zz,
            a=a,
        ) + walk_uniform_accept_path_nc_depth(n, coins) + walk_uniform_reflection_nc_depth(n, coins)
