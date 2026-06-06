import sympy as sp

from monaqa2.qiskit.hamiltonian_simulation_gqsp_symbolic import (
    hamiltonian_simulation_gqsp_number_qubits,
    hamiltonian_simulation_gqsp_nc_depth,
    hamiltonian_simulation_gqsp_t_count,
    hamiltonian_simulation_gqsp_rz_count,
    hamiltonian_simulation_gqsp_toffoli_count,
)
from monaqa2.qiskit.utils_symbolic import fast_simplify_logs


def proposal_qemc_number_qubits(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    # Proposal has:
    #   A register: n qubits
    #   HamiltonianSimulationGQSP acting on B + aux
    #
    # Since HamiltonianSimulationGQSP already includes its own n-qubit system register,
    # total = n + hsim_qubits.
    return sp.simplify(n + hamiltonian_simulation_gqsp_number_qubits(n, n_terms, alpha, t, eps))


def proposal_qemc_nc_depth(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    # The A -> B copy layer is CNOT-only, so it does not contribute to non-Clifford depth.
    return fast_simplify_logs(
        sp.simplify(hamiltonian_simulation_gqsp_nc_depth(n, n_terms, alpha, t, eps))
    )


def proposal_qemc_t_count(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    return sp.simplify(hamiltonian_simulation_gqsp_t_count(n, n_terms, alpha, t, eps))


def proposal_qemc_rz_count(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    return sp.simplify(hamiltonian_simulation_gqsp_rz_count(n, n_terms, alpha, t, eps))


def proposal_qemc_toffoli_count(n: sp.Symbol, n_terms: sp.Symbol, alpha: sp.Symbol, t: sp.Symbol, eps: sp.Symbol) -> sp.Expr:
    return sp.simplify(hamiltonian_simulation_gqsp_toffoli_count(n, n_terms, alpha, t, eps))