import sympy as sp

from monaqa2.qiskit.multi_controlled_not_symbolic import mcx_nc_depth, mcx_t_count, mcx_rz_count, mcx_toffoli_count
from monaqa2.qiskit.proposal_local_symbolic import (
    proposal_local_number_qubits,
    proposal_local_nc_depth,
    proposal_local_t_count,
    proposal_local_rz_count,
    proposal_local_toffoli_count,
)
from monaqa2.qiskit.walk_uniform_symbolic import (
    walk_uniform_mh_coin_core_qubits,
    walk_uniform_glauber_coin_core_qubits,
    walk_uniform_coin_nc_depth,
    walk_uniform_coin_t_count,
    walk_uniform_coin_rz_count,
    walk_uniform_coin_toffoli_count,
)


def _check_coin(coin: str) -> None:
    if coin not in {"mh", "glauber"}:
        raise ValueError("coin must be either 'mh' or 'glauber'.")


def _require_glauber_terms(n_terms_z: sp.Expr | None, n_terms_zz: sp.Expr | None) -> tuple[sp.Expr, sp.Expr]:
    if n_terms_z is None or n_terms_zz is None:
        raise ValueError("coin='glauber' requires n_terms_z and n_terms_zz.")

    return sp.sympify(n_terms_z), sp.sympify(n_terms_zz)


def walk_local_accept_path_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(1 + sp.Max(n - 1, coins - 2))


def walk_local_reflection_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(n + coins - 3)


def walk_local_accept_path_nc_depth(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.ceiling(2 * mcx_nc_depth(coins) + 3)


def walk_local_accept_path_t_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_t_count(coins) + 21 * n)


def walk_local_accept_path_rz_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_rz_count(coins))


def walk_local_accept_path_toffoli_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_toffoli_count(coins) + 3 * n)


def walk_local_reflection_nc_depth(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.ceiling(mcx_nc_depth(n + coins - 1))


def walk_local_reflection_t_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(mcx_t_count(n + coins - 1))


def walk_local_reflection_rz_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(mcx_rz_count(n + coins - 1))


def walk_local_reflection_toffoli_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(mcx_toffoli_count(n + coins - 1))


def walk_local_coin_core_qubits(
    n: sp.Expr,
    k: sp.Expr,
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
        return walk_uniform_mh_coin_core_qubits(n, sum_abs, beta, eps)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_core_qubits(n, sum_abs, beta, eps, z_terms, zz_terms, a)


def walk_local_coins(
    n: sp.Expr,
    k: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    core = walk_local_coin_core_qubits(n, k, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    return sp.Max(3, core)


def walk_local_number_qubits(
    n: sp.Expr,
    k: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_local_coins(n, k, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    accept_work = walk_local_accept_path_work_qubits(n, coins)
    reflection_work = walk_local_reflection_work_qubits(n, coins)
    return sp.simplify(proposal_local_number_qubits(n, k) + coins + accept_work + reflection_work)


def walk_local_nc_depth(
    n: sp.Expr,
    k: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_local_coins(n, k, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    total = (
        2 * proposal_local_nc_depth(n, k)
        + walk_uniform_coin_nc_depth(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_local_accept_path_nc_depth(n, coins)
        + walk_local_reflection_nc_depth(n, coins)
    )
    return sp.ceiling(total)


def walk_local_t_count(
    n: sp.Expr,
    k: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_local_coins(n, k, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    return sp.simplify(
        2 * proposal_local_t_count(n, k)
        + walk_uniform_coin_t_count(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_local_accept_path_t_count(n, coins)
        + walk_local_reflection_t_count(n, coins)
    )


def walk_local_rz_count(
    n: sp.Expr,
    k: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_local_coins(n, k, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    return sp.simplify(
        2 * proposal_local_rz_count(n, k)
        + walk_uniform_coin_rz_count(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_local_accept_path_rz_count(n, coins)
        + walk_local_reflection_rz_count(n, coins)
    )


def walk_local_toffoli_count(
    n: sp.Expr,
    k: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    coins = walk_local_coins(n, k, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    return sp.simplify(
        2 * proposal_local_toffoli_count(n, k)
        + walk_uniform_coin_toffoli_count(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_local_accept_path_toffoli_count(n, coins)
        + walk_local_reflection_toffoli_count(n, coins)
    )
