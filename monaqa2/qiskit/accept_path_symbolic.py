import sympy as sp
from monaqa2.qiskit.multi_controlled_not_symbolic import (
    mcx_nc_depth,
    mcx_t_count,
    mcx_rz_count,
    mcx_toffoli_count,
)


def accept_path_number_qubits(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    return sp.simplify(3 * n + 2 * coins - 2)


def accept_path_nc_depth(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    return 2 * mcx_nc_depth(coins) + 3


def accept_path_t_count(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    return 2 * mcx_t_count(coins)


def accept_path_rz_count(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    return 2 * mcx_rz_count(coins)


def accept_path_toffoli_count(n: sp.Symbol, coins: sp.Symbol) -> sp.Expr:
    return 2 * mcx_toffoli_count(coins) + 3 * n
