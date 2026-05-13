import sympy as sp

from monaqa2.qiskit.multi_controlled_not_symbolic import mcx_nc_depth, mcx_t_count, mcx_rz_count, mcx_toffoli_count
from monaqa2.qiskit.proposal_qemc_symbolic import (
    proposal_qemc_number_qubits,
    proposal_qemc_nc_depth,
    proposal_qemc_t_count,
    proposal_qemc_rz_count,
    proposal_qemc_toffoli_count,
)
from monaqa2.qiskit.gqsp_symbolic import (
    gqsp_number_qubits,
    gqsp_nc_depth,
    gqsp_t_count,
    gqsp_rz_count,
    gqsp_toffoli_count,
)
from monaqa2.qiskit.hamiltonian_simulation_gqsp_symbolic import hamiltonian_simulation_gqsp_degree
from monaqa2.qiskit.trotterized_ising_tf_symbolic import (
    trotterized_ising_tf_number_qubits,
    trotterized_ising_tf_nc_depth,
    trotterized_ising_tf_t_count,
    trotterized_ising_tf_rz_count,
    trotterized_ising_tf_toffoli_count,
)
from monaqa2.qiskit.utils_symbolic import advanced_initial_simplify, leading_terms_upper_bound, replace_shifted_logs
from monaqa2.qiskit.walk_uniform_symbolic import (
    walk_uniform_mh_fractional_bits_from_eps,
    walk_uniform_mh_sqrt_exp_degree_from_eps,
    walk_uniform_coin_core_qubits,
    walk_uniform_coin_nc_depth,
    walk_uniform_coin_t_count,
    walk_uniform_coin_rz_count,
    walk_uniform_coin_toffoli_count,
)


def _check_coin(coin: str) -> None:
    if coin not in {"mh", "glauber"}:
        raise ValueError("coin must be either 'mh' or 'glauber'.")


def _check_evolution(evolution: str) -> None:
    if evolution not in {"gqsp", "trotter", "exact"}:
        raise ValueError("evolution must be one of 'gqsp', 'trotter', or 'exact'.")


def _coin_kwargs(
    *,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
) -> dict:
    out = {
        "coin": coin,
        "n_terms_z": n_terms_z,
        "n_terms_zz": n_terms_zz,
        "a": a,
    }

    if eps_coin is not None:
        out["eps"] = eps_coin
    if f is not None:
        out["f"] = f
    if d_px is not None:
        out["d_px"] = d_px
    if normalization is not None:
        out["normalization"] = normalization

    return out


def _resolve_eps_proposal(eps: sp.Expr | None, eps_proposal: sp.Expr | None) -> sp.Expr | None:
    if eps_proposal is not None:
        return sp.sympify(eps_proposal)
    if eps is None:
        return None
    return sp.simplify(sp.sympify(eps) / 2)


def _resolve_eps_coin(eps: sp.Expr | None, eps_coin: sp.Expr | None) -> sp.Expr | None:
    if eps_coin is not None:
        return sp.sympify(eps_coin)
    if eps is None:
        return None
    return sp.simplify(sp.sympify(eps) / 2)


def walk_qemc_hs_degree_from_eps(alpha_qemc: sp.Expr, t: sp.Expr, eps_hs: sp.Expr) -> sp.Expr:
    return hamiltonian_simulation_gqsp_degree(alpha_qemc, t, eps_hs)


def walk_qemc_mh_fractional_bits_from_eps(n: sp.Expr, beta: sp.Expr, normalization: sp.Expr, eps_fx: sp.Expr) -> sp.Expr:
    return walk_uniform_mh_fractional_bits_from_eps(n, beta, normalization, eps_fx)


def walk_qemc_mh_sqrt_exp_degree_from_eps(beta: sp.Expr, normalization: sp.Expr, eps_px: sp.Expr) -> sp.Expr:
    return walk_uniform_mh_sqrt_exp_degree_from_eps(beta, normalization, eps_px)


def walk_qemc_proposal_aux_qubits(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None = None,
    alpha_qemc: sp.Expr | None = None,
    t: sp.Expr | None = None,
    eps: sp.Expr | None = None,
    *,
    eps_proposal: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    _check_evolution(evolution)
    n = sp.sympify(n)

    if evolution == "exact":
        return sp.Integer(0)

    if evolution == "trotter":
        hsim_qubits = trotterized_ising_tf_number_qubits(
            n,
            num_trotter_steps=num_trotter_steps,
            n_terms_z=qemc_n_terms_z,
            n_terms_zz=qemc_n_terms_zz,
            n_terms_x=qemc_n_terms_x,
            eps=eps_proposal,
            time=sp.Integer(1) if t is None else t,
        )
        return sp.simplify(hsim_qubits - n)

    if n_terms_qemc is None:
        raise ValueError("evolution='gqsp' requires n_terms_qemc.")

    if d_hs is not None:
        return sp.simplify(gqsp_number_qubits(n, n_terms_qemc) - n)

    eps_prop = _resolve_eps_proposal(eps, eps_proposal)
    if eps_prop is None or alpha_qemc is None or t is None:
        raise ValueError("evolution='gqsp' requires either d_hs or alpha_qemc, t, and an eps_proposal/eps.")

    return sp.simplify(proposal_qemc_number_qubits(n, n_terms_qemc, alpha_qemc, t, eps_prop) - 2 * n)


def walk_qemc_proposal_nc_depth(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None = None,
    alpha_qemc: sp.Expr | None = None,
    t: sp.Expr | None = None,
    eps: sp.Expr | None = None,
    *,
    eps_proposal: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    _check_evolution(evolution)

    if evolution == "exact":
        return sp.Integer(0)

    if evolution == "trotter":
        return trotterized_ising_tf_nc_depth(
            n,
            num_trotter_steps=num_trotter_steps,
            n_terms_z=qemc_n_terms_z,
            n_terms_zz=qemc_n_terms_zz,
            n_terms_x=qemc_n_terms_x,
            eps=eps_proposal,
            time=sp.Integer(1) if t is None else t,
        )

    if n_terms_qemc is None:
        raise ValueError("evolution='gqsp' requires n_terms_qemc.")

    if d_hs is not None:
        return sp.simplify(gqsp_nc_depth(n, n_terms_qemc, 2 * d_hs + 1, d_hs))

    eps_prop = _resolve_eps_proposal(eps, eps_proposal)
    if eps_prop is None or alpha_qemc is None or t is None:
        raise ValueError("evolution='gqsp' requires either d_hs or alpha_qemc, t, and an eps_proposal/eps.")

    return proposal_qemc_nc_depth(n, n_terms_qemc, alpha_qemc, t, eps_prop)


def walk_qemc_proposal_t_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None = None,
    alpha_qemc: sp.Expr | None = None,
    t: sp.Expr | None = None,
    eps: sp.Expr | None = None,
    *,
    eps_proposal: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    _check_evolution(evolution)

    if evolution == "exact":
        return sp.Integer(0)

    if evolution == "trotter":
        return trotterized_ising_tf_t_count(
            n,
            num_trotter_steps=num_trotter_steps,
            n_terms_z=qemc_n_terms_z,
            n_terms_zz=qemc_n_terms_zz,
            n_terms_x=qemc_n_terms_x,
            eps=eps_proposal,
            time=sp.Integer(1) if t is None else t,
        )

    if n_terms_qemc is None:
        raise ValueError("evolution='gqsp' requires n_terms_qemc.")

    if d_hs is not None:
        return sp.simplify(gqsp_t_count(n, n_terms_qemc, 2 * d_hs + 1, d_hs))

    eps_prop = _resolve_eps_proposal(eps, eps_proposal)
    if eps_prop is None or alpha_qemc is None or t is None:
        raise ValueError("evolution='gqsp' requires either d_hs or alpha_qemc, t, and an eps_proposal/eps.")

    return proposal_qemc_t_count(n, n_terms_qemc, alpha_qemc, t, eps_prop)


def walk_qemc_proposal_rz_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None = None,
    alpha_qemc: sp.Expr | None = None,
    t: sp.Expr | None = None,
    eps: sp.Expr | None = None,
    *,
    eps_proposal: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    _check_evolution(evolution)

    if evolution == "exact":
        return sp.Integer(0)

    if evolution == "trotter":
        return trotterized_ising_tf_rz_count(
            n,
            num_trotter_steps=num_trotter_steps,
            n_terms_z=qemc_n_terms_z,
            n_terms_zz=qemc_n_terms_zz,
            n_terms_x=qemc_n_terms_x,
            eps=eps_proposal,
            time=sp.Integer(1) if t is None else t,
        )

    if n_terms_qemc is None:
        raise ValueError("evolution='gqsp' requires n_terms_qemc.")

    if d_hs is not None:
        return sp.simplify(gqsp_rz_count(n, n_terms_qemc, 2 * d_hs + 1, d_hs))

    eps_prop = _resolve_eps_proposal(eps, eps_proposal)
    if eps_prop is None or alpha_qemc is None or t is None:
        raise ValueError("evolution='gqsp' requires either d_hs or alpha_qemc, t, and an eps_proposal/eps.")

    return proposal_qemc_rz_count(n, n_terms_qemc, alpha_qemc, t, eps_prop)


def walk_qemc_proposal_toffoli_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None = None,
    alpha_qemc: sp.Expr | None = None,
    t: sp.Expr | None = None,
    eps: sp.Expr | None = None,
    *,
    eps_proposal: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    _check_evolution(evolution)

    if evolution == "exact":
        return sp.Integer(0)

    if evolution == "trotter":
        return trotterized_ising_tf_toffoli_count(
            n,
            num_trotter_steps=num_trotter_steps,
            n_terms_z=qemc_n_terms_z,
            n_terms_zz=qemc_n_terms_zz,
            n_terms_x=qemc_n_terms_x,
            eps=eps_proposal,
            time=sp.Integer(1) if t is None else t,
        )

    if n_terms_qemc is None:
        raise ValueError("evolution='gqsp' requires n_terms_qemc.")

    if d_hs is not None:
        return sp.simplify(gqsp_toffoli_count(n, n_terms_qemc, 2 * d_hs + 1, d_hs))

    eps_prop = _resolve_eps_proposal(eps, eps_proposal)
    if eps_prop is None or alpha_qemc is None or t is None:
        raise ValueError("evolution='gqsp' requires either d_hs or alpha_qemc, t, and an eps_proposal/eps.")

    return proposal_qemc_toffoli_count(n, n_terms_qemc, alpha_qemc, t, eps_prop)


def walk_qemc_accept_path_work_qubits(n: sp.Expr, coins: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    coins = sp.sympify(coins)
    return sp.simplify(n + coins - 2)


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
    return sp.ceiling(mcx_nc_depth(n + reflected_aux - 1))


def walk_qemc_reflection_t_count(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    return sp.simplify(mcx_t_count(n + reflected_aux - 1))


def walk_qemc_reflection_rz_count(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    return sp.simplify(mcx_rz_count(n + reflected_aux - 1))


def walk_qemc_reflection_toffoli_count(n: sp.Expr, reflected_aux: sp.Expr) -> sp.Expr:
    n = sp.sympify(n)
    reflected_aux = sp.sympify(reflected_aux)
    return sp.simplify(mcx_toffoli_count(n + reflected_aux - 1))


def walk_qemc_coin_core_qubits(
    n: sp.Expr,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
) -> sp.Expr:
    _check_coin(coin)
    eps_c = _resolve_eps_coin(eps, eps_coin)

    return sp.simplify(
        walk_uniform_coin_core_qubits(
            n,
            sum_abs,
            beta,
            **_coin_kwargs(
                eps_coin=eps_c,
                f=f,
                d_px=d_px,
                normalization=normalization,
                coin=coin,
                n_terms_z=n_terms_z,
                n_terms_zz=n_terms_zz,
                a=a,
            ),
        )
    )


def walk_qemc_coins(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
) -> sp.Expr:
    expr = walk_qemc_coin_core_qubits(
        n,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
        eps_coin=eps_coin,
        f=f,
        d_px=d_px,
        normalization=normalization,
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
    return advanced_initial_simplify(expr)


def _common_terms(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None,
    coin: str,
    n_terms_z: sp.Expr | None,
    n_terms_zz: sp.Expr | None,
    a: sp.Expr,
    eps_proposal: sp.Expr | None,
    eps_coin: sp.Expr | None,
    f: sp.Expr | None,
    d_px: sp.Expr | None,
    d_hs: sp.Expr | None,
    normalization: sp.Expr | None,
    evolution: str,
    num_trotter_steps: sp.Expr | None,
    qemc_n_terms_z: sp.Expr | None,
    qemc_n_terms_zz: sp.Expr | None,
    qemc_n_terms_x: sp.Expr | None,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    n = sp.sympify(n)
    proposal_aux = walk_qemc_proposal_aux_qubits(
        n,
        n_terms_qemc,
        alpha_qemc,
        t,
        eps,
        eps_proposal=eps_proposal,
        d_hs=d_hs,
        evolution=evolution,
        num_trotter_steps=num_trotter_steps,
        qemc_n_terms_z=qemc_n_terms_z,
        qemc_n_terms_zz=qemc_n_terms_zz,
        qemc_n_terms_x=qemc_n_terms_x,
    )
    coins = walk_qemc_coins(
        n,
        n_terms_qemc,
        alpha_qemc,
        t,
        sum_abs,
        beta,
        eps,
        coin=coin,
        n_terms_z=n_terms_z,
        n_terms_zz=n_terms_zz,
        a=a,
        eps_coin=eps_coin,
        f=f,
        d_px=d_px,
        normalization=normalization,
    )
    reflected_aux = sp.simplify(proposal_aux + coins)
    return proposal_aux, coins, reflected_aux


def walk_qemc_number_qubits(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_proposal: sp.Expr | None = None,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    proposal_aux, coins, reflected_aux = _common_terms(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin, n_terms_z, n_terms_zz, a, eps_proposal, eps_coin, f, d_px, d_hs, normalization, evolution, num_trotter_steps, qemc_n_terms_z, qemc_n_terms_zz, qemc_n_terms_x)
    accept_work = walk_qemc_accept_path_work_qubits(n, coins)
    reflection_work = walk_qemc_reflection_work_qubits(n, reflected_aux)
    expr = sp.simplify(2 * sp.sympify(n) + proposal_aux + coins + accept_work + reflection_work)
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return advanced_initial_simplify(expr)


def walk_qemc_nc_depth(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_proposal: sp.Expr | None = None,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    proposal_aux, coins, reflected_aux = _common_terms(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin, n_terms_z, n_terms_zz, a, eps_proposal, eps_coin, f, d_px, d_hs, normalization, evolution, num_trotter_steps, qemc_n_terms_z, qemc_n_terms_zz, qemc_n_terms_x)
    proposal = walk_qemc_proposal_nc_depth(n, n_terms_qemc, alpha_qemc, t, eps, eps_proposal=eps_proposal, d_hs=d_hs, evolution=evolution, num_trotter_steps=num_trotter_steps, qemc_n_terms_z=qemc_n_terms_z, qemc_n_terms_zz=qemc_n_terms_zz, qemc_n_terms_x=qemc_n_terms_x)
    coin_depth = walk_uniform_coin_nc_depth(n, sum_abs, beta, **_coin_kwargs(eps_coin=_resolve_eps_coin(eps, eps_coin), f=f, d_px=d_px, normalization=normalization, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a))
    total = 2 * proposal + coin_depth + walk_qemc_accept_path_nc_depth(n, coins) + walk_qemc_reflection_nc_depth(n, reflected_aux)
    expr = total
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    return advanced_initial_simplify(expr).subs(sp.log(2*n**2-3), 4*sp.log(n))\
        .subs(sp.log(2*n**2-2), 4*sp.log(n))\
        .subs(sp.log(7*f*n**2*sp.log(f)-1), sp.log(8*f*n**2*sp.log(f)))\
        .subs(sp.log(7*f*n**2*sp.log(f)+5*n**2+2*n-3), sp.log(8*f*n**2*sp.log(f)))\
        .subs(sp.log(8*f*n**2*sp.log(f)), 12*sp.log(f)+12*sp.log(n)+18)\
        .collect([n, sp.log(n), sp.log(f)])



def walk_qemc_t_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_proposal: sp.Expr | None = None,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    proposal_aux, coins, reflected_aux = _common_terms(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin, n_terms_z, n_terms_zz, a, eps_proposal, eps_coin, f, d_px, d_hs, normalization, evolution, num_trotter_steps, qemc_n_terms_z, qemc_n_terms_zz, qemc_n_terms_x)
    proposal = walk_qemc_proposal_t_count(n, n_terms_qemc, alpha_qemc, t, eps, eps_proposal=eps_proposal, d_hs=d_hs, evolution=evolution, num_trotter_steps=num_trotter_steps, qemc_n_terms_z=qemc_n_terms_z, qemc_n_terms_zz=qemc_n_terms_zz, qemc_n_terms_x=qemc_n_terms_x)
    coin_count = walk_uniform_coin_t_count(n, sum_abs, beta, **_coin_kwargs(eps_coin=_resolve_eps_coin(eps, eps_coin), f=f, d_px=d_px, normalization=normalization, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a))
    expr = sp.simplify(2 * proposal + coin_count + walk_qemc_accept_path_t_count(n, coins) + walk_qemc_reflection_t_count(n, reflected_aux))
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return advanced_initial_simplify(expr)


def walk_qemc_rz_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_proposal: sp.Expr | None = None,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    proposal_aux, coins, reflected_aux = _common_terms(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin, n_terms_z, n_terms_zz, a, eps_proposal, eps_coin, f, d_px, d_hs, normalization, evolution, num_trotter_steps, qemc_n_terms_z, qemc_n_terms_zz, qemc_n_terms_x)
    proposal = walk_qemc_proposal_rz_count(n, n_terms_qemc, alpha_qemc, t, eps, eps_proposal=eps_proposal, d_hs=d_hs, evolution=evolution, num_trotter_steps=num_trotter_steps, qemc_n_terms_z=qemc_n_terms_z, qemc_n_terms_zz=qemc_n_terms_zz, qemc_n_terms_x=qemc_n_terms_x)
    coin_count = walk_uniform_coin_rz_count(n, sum_abs, beta, **_coin_kwargs(eps_coin=_resolve_eps_coin(eps, eps_coin), f=f, d_px=d_px, normalization=normalization, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a))
    expr = sp.simplify(2 * proposal + coin_count + walk_qemc_accept_path_rz_count(n, coins) + walk_qemc_reflection_rz_count(n, reflected_aux))
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return advanced_initial_simplify(expr)


def walk_qemc_toffoli_count(
    n: sp.Expr,
    n_terms_qemc: sp.Expr | None,
    alpha_qemc: sp.Expr | None,
    t: sp.Expr | None,
    sum_abs: sp.Expr,
    beta: sp.Expr,
    eps: sp.Expr | None = None,
    coin: str = "mh",
    n_terms_z: sp.Expr | None = None,
    n_terms_zz: sp.Expr | None = None,
    a: sp.Expr = 1,
    *,
    eps_proposal: sp.Expr | None = None,
    eps_coin: sp.Expr | None = None,
    f: sp.Expr | None = None,
    d_px: sp.Expr | None = None,
    d_hs: sp.Expr | None = None,
    normalization: sp.Expr | None = None,
    evolution: str = "gqsp",
    num_trotter_steps: sp.Expr | None = None,
    qemc_n_terms_z: sp.Expr | None = None,
    qemc_n_terms_zz: sp.Expr | None = None,
    qemc_n_terms_x: sp.Expr | None = None,
) -> sp.Expr:
    proposal_aux, coins, reflected_aux = _common_terms(n, n_terms_qemc, alpha_qemc, t, sum_abs, beta, eps, coin, n_terms_z, n_terms_zz, a, eps_proposal, eps_coin, f, d_px, d_hs, normalization, evolution, num_trotter_steps, qemc_n_terms_z, qemc_n_terms_zz, qemc_n_terms_x)
    proposal = walk_qemc_proposal_toffoli_count(n, n_terms_qemc, alpha_qemc, t, eps, eps_proposal=eps_proposal, d_hs=d_hs, evolution=evolution, num_trotter_steps=num_trotter_steps, qemc_n_terms_z=qemc_n_terms_z, qemc_n_terms_zz=qemc_n_terms_zz, qemc_n_terms_x=qemc_n_terms_x)
    coin_count = walk_uniform_coin_toffoli_count(n, sum_abs, beta, **_coin_kwargs(eps_coin=_resolve_eps_coin(eps, eps_coin), f=f, d_px=d_px, normalization=normalization, coin=coin, n_terms_z=n_terms_z, n_terms_zz=n_terms_zz, a=a))
    expr = sp.simplify(2 * proposal + coin_count + walk_qemc_accept_path_toffoli_count(n, coins) + walk_qemc_reflection_toffoli_count(n, reflected_aux))
    expr = replace_shifted_logs(
        advanced_initial_simplify(expr),
        variables=[d_px, f, n]
    )
    expr = leading_terms_upper_bound(
        expr,
        variables=[d_px, f, n],
        assumptions=[d_px >= 2, f >= 10, n >= 2]
    )
    return advanced_initial_simplify(expr)
