import sympy as sp

from monaqa2.qiskit.qubitized_ising_tf_symbolic import (
    qubitized_ising_tf_toffoli_count,
    qubitized_ising_tf_rz_count,
    qubitized_ising_tf_t_count,
    qubitized_ising_tf_nc_depth,
    qubitized_ising_tf_number_qubits,
    controlled_qubitized_ising_tf_toffoli_count,
    controlled_qubitized_ising_tf_rz_count,
    controlled_qubitized_ising_tf_t_count,
    controlled_qubitized_ising_tf_nc_depth,
    controlled_qubitized_ising_tf_number_qubits,
)


def gqsp_number_qubits(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(controlled_qubitized_ising_tf_number_qubits(n, n_terms))


def gqsp_nc_depth(
    n: sp.Symbol,
    n_terms: sp.Symbol,
    degree: sp.Symbol,
    laurent_negative_power: sp.Symbol = sp.Integer(0),
) -> sp.Expr:
    return sp.simplify(
        degree * controlled_qubitized_ising_tf_nc_depth(n, n_terms)
        + laurent_negative_power * qubitized_ising_tf_nc_depth(n, n_terms)
        + 3 * (degree + 1)
    )


def gqsp_t_count(
    n: sp.Symbol,
    n_terms: sp.Symbol,
    degree: sp.Symbol,
    laurent_negative_power: sp.Symbol = sp.Integer(0),
) -> sp.Expr:
    return sp.simplify(
        degree * controlled_qubitized_ising_tf_t_count(n, n_terms)
        + laurent_negative_power * qubitized_ising_tf_t_count(n, n_terms)
    )


def gqsp_rz_count(
    n: sp.Symbol,
    n_terms: sp.Symbol,
    degree: sp.Symbol,
    laurent_negative_power: sp.Symbol = sp.Integer(0),
) -> sp.Expr:
    return sp.simplify(
        degree * controlled_qubitized_ising_tf_rz_count(n, n_terms)
        + laurent_negative_power * qubitized_ising_tf_rz_count(n, n_terms)
        + 3 * (degree + 1)
    )


def gqsp_toffoli_count(
    n: sp.Symbol,
    n_terms: sp.Symbol,
    degree: sp.Symbol,
    laurent_negative_power: sp.Symbol = sp.Integer(0),
) -> sp.Expr:
    return sp.simplify(
        degree * controlled_qubitized_ising_tf_toffoli_count(n, n_terms)
        + laurent_negative_power * qubitized_ising_tf_toffoli_count(n, n_terms)
    )