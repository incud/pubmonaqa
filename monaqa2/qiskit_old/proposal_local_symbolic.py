import sympy as sp
from monaqa2.qiskit.dicke_preparation_symbolic import (
    dicke_preparation_nc_depth,
    dicke_preparation_t_count,
    dicke_preparation_rz_count,
    dicke_preparation_toffoli_count,
)


def proposal_local_number_qubits(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * n)


def proposal_local_nc_depth(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return dicke_preparation_nc_depth(n, k)


def proposal_local_t_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return dicke_preparation_t_count(n, k)


def proposal_local_rz_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return dicke_preparation_rz_count(n, k)


def proposal_local_toffoli_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return dicke_preparation_toffoli_count(n, k)
