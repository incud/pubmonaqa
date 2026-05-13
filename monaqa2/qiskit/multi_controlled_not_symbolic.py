import sympy as sp


def mcx_number_qubits(controls: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * controls - 1)


def mcx_nc_depth(controls: sp.Symbol) -> sp.Expr:
    return 2 * (sp.log(controls - 1, 2) + 1) + 1


def mcx_t_count(controls: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def mcx_rz_count(controls: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def mcx_toffoli_count(controls: sp.Symbol) -> sp.Expr:
    return 2 * controls - 3
