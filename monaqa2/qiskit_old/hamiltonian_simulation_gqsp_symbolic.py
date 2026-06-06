import sympy as sp

from monaqa2.qiskit.gqsp_symbolic import (
    gqsp_number_qubits,
    gqsp_nc_depth,
    gqsp_t_count,
    gqsp_rz_count,
    gqsp_toffoli_count,
)
from monaqa2.qiskit.utils_symbolic import fast_simplify_logs


def hamiltonian_simulation_gqsp_degree(alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    tau = sp.Abs(alpha * t)
    time_deg = sp.E * tau
    degree = sp.ceiling(time_deg + sp.log(4 / eps) + 1)
    return sp.simplify(degree)


def hamiltonian_simulation_gqsp_number_qubits(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    return sp.simplify(gqsp_number_qubits(n, n_terms))


def hamiltonian_simulation_gqsp_nc_depth(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    d = sp.symbols('d', integer=True, positive=True)
    degree = d # hamiltonian_simulation_gqsp_degree(alpha, t, eps)
    expr = sp.simplify(gqsp_nc_depth(n, n_terms, 2 * degree + 1, degree))
    return fast_simplify_logs(
        expr
    ).xreplace({
        sp.log(2*n_terms-2): sp.log(2*n_terms),
        sp.log(2*n_terms-3): sp.log(2*n_terms),
    }).xreplace({
        sp.log(2*n_terms): 2 * sp.log(n_terms)
    })


def hamiltonian_simulation_gqsp_t_count(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    degree = hamiltonian_simulation_gqsp_degree(alpha, t, eps)
    return sp.simplify(gqsp_t_count(n, n_terms, 2 * degree + 1, degree))


def hamiltonian_simulation_gqsp_rz_count(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    degree = hamiltonian_simulation_gqsp_degree(alpha, t, eps)
    return sp.simplify(gqsp_rz_count(n, n_terms, 2 * degree + 1, degree))


def hamiltonian_simulation_gqsp_toffoli_count(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    degree = hamiltonian_simulation_gqsp_degree(alpha, t, eps)
    return sp.simplify(gqsp_toffoli_count(n, n_terms, 2 * degree + 1, degree))
