import sympy as sp


def proposal_uniform_number_qubits(n: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * n)


def proposal_uniform_nc_depth(n: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def proposal_uniform_t_count(n: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def proposal_uniform_rz_count(n: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def proposal_uniform_toffoli_count(n: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)