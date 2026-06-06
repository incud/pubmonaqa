import sympy as sp

from monaqa2.qiskit.multi_controlled_not_symbolic import mcx_nc_depth, mcx_t_count, mcx_rz_count, mcx_toffoli_count
from monaqa2.qiskit.proposal_qemc_symbolic import (
    proposal_qemc_number_qubits,
    proposal_qemc_nc_depth,
    proposal_qemc_t_count,
    proposal_qemc_rz_count,
    proposal_qemc_toffoli_count,
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


def walk_qemc_proposal_aux_qubits(n: sp.Expr, n_terms_qemc: sp.Expr, alpha_qemc: sp.Expr, t: sp.Expr, eps: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    eps_proposal = sp.sympify(eps) / 2
    return sp.simplify(proposal_qemc_number_qubits(n, n_terms_qemc, alpha_qemc, t, eps_proposal) - 2 * n)


def walk_qemc_accept_path_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(1 + sp.Max(n - 1, coins - 2))


def walk_qemc_reflection_work_qubits(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    return sp.simplify(n + reflected_aux - 3)


def walk_qemc_accept_path_nc_depth(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.ceiling(2 * mcx_nc_depth(coins) + 3)


def walk_qemc_accept_path_t_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_t_count(coins) + 21 * n)


def walk_qemc_accept_path_rz_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_rz_count(coins))


def walk_qemc_accept_path_toffoli_count(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(2 * mcx_toffoli_count(coins) + 3 * n)


def walk_qemc_reflection_nc_depth(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    controls = n + reflected_aux - 1
    return sp.ceiling(mcx_nc_depth(controls))


def walk_qemc_reflection_t_count(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    controls = n + reflected_aux - 1
    return sp.simplify(mcx_t_count(controls))


def walk_qemc_reflection_rz_count(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    controls = n + reflected_aux - 1
    return sp.simplify(mcx_rz_count(controls))


def walk_qemc_reflection_toffoli_count(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    controls = n + reflected_aux - 1
    return sp.simplify(mcx_toffoli_count(controls))


def walk_qemc_coin_core_qubits(
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
    eps_coin = sp.sympify(eps) / 2

    if coin == "mh":
        return walk_uniform_mh_coin_core_qubits(n, sum_abs, beta, eps_coin)

    z_terms, zz_terms = _require_glauber_terms(n_terms_z, n_terms_zz)
    return walk_uniform_glauber_coin_core_qubits(n, sum_abs, beta, eps_coin, z_terms, zz_terms, a)


def walk_qemc_coins(
    n: sp.Expr,
    n_terms_qemc: sp.Expr,
    alpha_qemc: sp.Expr,
    t: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    return sp.Max(3, walk_qemc_coin_core_qubits(n, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a))


def walk_qemc_number_qubits(
    n: sp.Expr,
    n_terms_qemc: sp.Expr,
    alpha_qemc: sp.Expr,
    t: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    n = sp.sympify(n)
    proposal_aux = walk_qemc_proposal_aux_qubits(n, n_terms_qemc, alpha_qemc, t, eps)
    coins = walk_qemc_coins(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    accept_work = walk_qemc_accept_path_work_qubits(n, coins)
    reflected_aux = proposal_aux + coins
    reflection_work = walk_qemc_reflection_work_qubits(n, reflected_aux)

    return sp.simplify(2 * n + proposal_aux + coins + accept_work + reflection_work)


def walk_qemc_toffoli_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr,
    alpha_qemc: sp.Expr,
    t: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    eps_proposal = sp.sympify(eps) / 2
    coins = walk_qemc_coins(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    proposal_aux = walk_qemc_proposal_aux_qubits(n, n_terms_qemc, alpha_qemc, t, eps)
    reflected_aux = proposal_aux + coins

    return sp.simplify(
        2 * proposal_qemc_toffoli_count(n, n_terms_qemc, alpha_qemc, t, eps_proposal)
        + walk_uniform_coin_toffoli_count(n, sum_abs, beta, sp.sympify(eps) / 2, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_qemc_accept_path_toffoli_count(n, coins)
        + walk_qemc_reflection_toffoli_count(n, reflected_aux)
    )


def walk_qemc_t_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr,
    alpha_qemc: sp.Expr,
    t: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    eps_proposal = sp.sympify(eps) / 2
    coins = walk_qemc_coins(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    proposal_aux = walk_qemc_proposal_aux_qubits(n, n_terms_qemc, alpha_qemc, t, eps)
    reflected_aux = proposal_aux + coins

    return sp.simplify(
        2 * proposal_qemc_t_count(n, n_terms_qemc, alpha_qemc, t, eps_proposal)
        + walk_uniform_coin_t_count(n, sum_abs, beta, sp.sympify(eps) / 2, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_qemc_accept_path_t_count(n, coins)
        + walk_qemc_reflection_t_count(n, reflected_aux)
    )


def walk_qemc_rz_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr,
    alpha_qemc: sp.Expr,
    t: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    eps_proposal = sp.sympify(eps) / 2
    coins = walk_qemc_coins(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    proposal_aux = walk_qemc_proposal_aux_qubits(n, n_terms_qemc, alpha_qemc, t, eps)
    reflected_aux = proposal_aux + coins

    return sp.simplify(
        2 * proposal_qemc_rz_count(n, n_terms_qemc, alpha_qemc, t, eps_proposal)
        + walk_uniform_coin_rz_count(n, sum_abs, beta, sp.sympify(eps) / 2, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_qemc_accept_path_rz_count(n, coins)
        + walk_qemc_reflection_rz_count(n, reflected_aux)
    )


def walk_qemc_nc_depth(
    n: sp.Expr,
    n_terms_qemc: sp.Expr,
    alpha_qemc: sp.Expr,
    t: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> sp.Expr:
    eps_proposal = sp.sympify(eps) / 2
    coins = walk_qemc_coins(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
    proposal_aux = walk_qemc_proposal_aux_qubits(n, n_terms_qemc, alpha_qemc, t, eps)
    reflected_aux = proposal_aux + coins

    total = (
        2 * proposal_qemc_nc_depth(n, n_terms_qemc, alpha_qemc, t, eps_proposal)
        + walk_uniform_coin_nc_depth(n, sum_abs, beta, sp.sympify(eps) / 2, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a)
        + walk_qemc_accept_path_nc_depth(n, coins)
        + walk_qemc_reflection_nc_depth(n, reflected_aux)
    )

    return sp.ceiling(total)
