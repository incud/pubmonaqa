import sympy as sp
from monaqa2.qiskit.prepare_unary_symbolic import (
    prepare_unary_number_qubits,
    prepare_unary_nc_depth,
    prepare_unary_t_count,
    prepare_unary_rz_count,
    prepare_unary_toffoli_count,
)


def lcu_unary_number_qubits(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(n + prepare_unary_number_qubits(n_terms))


def lcu_unary_nc_depth(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * prepare_unary_nc_depth(n_terms))


def lcu_unary_t_count(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * prepare_unary_t_count(n_terms))


def lcu_unary_rz_count(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * prepare_unary_rz_count(n_terms))


def lcu_unary_toffoli_count(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * prepare_unary_toffoli_count(n_terms))