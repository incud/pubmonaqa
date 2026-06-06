import sympy as sp
from monaqa2.qiskit.multi_controlled_not_symbolic import (
    mcx_nc_depth,
    mcx_t_count,
    mcx_rz_count,
    mcx_toffoli_count,
)


def reflection_number_qubits(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    return sp.sympify(2 * n + coins + n + coins - 3)


def reflection_nc_depth(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    controls = n + coins - 1
    return mcx_nc_depth(controls)


def reflection_t_count(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    controls = n + coins - 1
    return mcx_t_count(controls)


def reflection_rz_count(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    controls = n + coins - 1
    return mcx_rz_count(controls)


def reflection_toffoli_count(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    controls = n + coins - 1
    return mcx_toffoli_count(controls)