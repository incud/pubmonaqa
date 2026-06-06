import sympy as sp
from monaqa2.qiskit.lcu_unary_ising_tf_symbolic import (
    lcu_unary_number_qubits,
    lcu_unary_nc_depth,
    lcu_unary_t_count,
    lcu_unary_rz_count,
    lcu_unary_toffoli_count,
)
from monaqa2.qiskit.multi_controlled_not_symbolic import (
    mcx_nc_depth,
    mcx_t_count,
    mcx_rz_count,
    mcx_toffoli_count,
)
from monaqa2.qiskit.utils_symbolic import fast_simplify_logs


def qubitized_ising_tf_number_qubits(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controls = sp.simplify(2 * n_terms - 2)
    # controls_overhead = sp.Max(controls - 2, 0)
    controls_overhead = controls
    return fast_simplify_logs(sp.simplify(lcu_unary_number_qubits(n, n_terms) + controls_overhead))


def qubitized_ising_tf_nc_depth(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controls = sp.simplify(2 * n_terms - 2)

    return fast_simplify_logs(sp.simplify(
        lcu_unary_nc_depth(n, n_terms)
        + mcx_nc_depth(controls)
    ))


def qubitized_ising_tf_t_count(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controls = sp.simplify(2 * n_terms - 2)

    return fast_simplify_logs(sp.simplify(
        lcu_unary_t_count(n, n_terms) + mcx_t_count(controls)
    ))


def qubitized_ising_tf_rz_count(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controls = sp.simplify(2 * n_terms - 2)

    return fast_simplify_logs(sp.simplify(
        lcu_unary_rz_count(n, n_terms) + mcx_rz_count(controls)
    ))


def qubitized_ising_tf_toffoli_count(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controls = sp.simplify(2 * n_terms - 2)

    return fast_simplify_logs(sp.simplify(
        lcu_unary_toffoli_count(n, n_terms) + mcx_toffoli_count(controls)
    ))




def _resolve_terms(
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> tuple[sp.Expr, sp.Expr | None, sp.Expr | None, sp.Expr | None]:
    has_total = n_terms is not None
    has_split = n_terms_z is not None or n_terms_zz is not None or n_terms_x is not None

    if has_total and has_split:
        raise ValueError("Pass either n_terms or n_terms_z/n_terms_zz/n_terms_x, not both.")

    if not has_total and not has_split:
        raise ValueError("Pass either n_terms or all of n_terms_z/n_terms_zz/n_terms_x.")

    if has_split and not all(v is not None for v in [n_terms_z, n_terms_zz, n_terms_x]):
        raise ValueError("If using split term counts, pass all of n_terms_z, n_terms_zz, and n_terms_x.")

    if has_total:
        return sp.sympify(n_terms), None, None, None

    n_terms_z = sp.sympify(n_terms_z)
    n_terms_zz = sp.sympify(n_terms_zz)
    n_terms_x = sp.sympify(n_terms_x)
    n_terms = sp.simplify(n_terms_z + n_terms_zz + n_terms_x)

    return n_terms, n_terms_z, n_terms_zz, n_terms_x


def _positive_indicator(x: sp.Expr) -> sp.Expr:
    x = sp.sympify(x)

    if x.is_zero:
        return sp.Integer(0)

    if x.is_positive:
        return sp.Integer(1)

    return sp.Piecewise((0, x <= 0), (1, True))


def _complete_graph_matching_rounds(n: sp.Symbol) -> sp.Expr:
    return sp.sympify(n)


def _prepare_control_work(n: sp.Symbol, n_terms: sp.Symbol) -> sp.Expr:
    n = sp.sympify(n)
    n_terms = sp.sympify(n_terms)

    if isinstance(n, int) and isinstance(n_terms, int):
        if n_terms <= 1:
            prepare_width = 0
        elif n_terms == 2:
            prepare_width = 1
        else:
            levels = []

            def visit(lo: int, hi: int, depth: int) -> None:
                if hi - lo == 1:
                    return

                if depth == len(levels):
                    levels.append(0)

                levels[depth] += 1
                mid = (lo + hi) // 2
                visit(lo, mid, depth + 1)
                visit(mid, hi, depth + 1)

            visit(0, n_terms, 0)
            prepare_width = max(levels)

        return sp.Integer(max(n, prepare_width))

    prepare_width = n_terms / 2

    return n + n_terms # sp.Max(n, prepare_width)


def _controlled_select_toffoli_count(
    n_terms: sp.Symbol,
    n_terms_z: sp.Expr | None,
    n_terms_zz: sp.Expr | None,
    n_terms_x: sp.Expr | None,
) -> sp.Expr:
    if n_terms_z is None:
        return sp.simplify(2 * n_terms)

    return sp.simplify(n_terms_z + 2 * n_terms_zz + n_terms_x)


def _controlled_select_nc_depth(
    n: sp.Symbol,
    n_terms: sp.Symbol,
    n_terms_z: sp.Expr | None,
    n_terms_zz: sp.Expr | None,
    n_terms_x: sp.Expr | None,
) -> sp.Expr:
    if n_terms_z is None:
        return sp.simplify(2 + 2 * _complete_graph_matching_rounds(n))

    n_terms_z = sp.sympify(n_terms_z)
    n_terms_zz = sp.Integer(0) if n_terms_zz is None else sp.sympify(n_terms_zz)
    n_terms_x = sp.Integer(0) if n_terms_x is None else sp.sympify(n_terms_x)

    z_depth = _positive_indicator(n_terms_z)
    x_depth = _positive_indicator(n_terms_x)
    zz_depth = _positive_indicator(n_terms_zz) * 2 * sp.Min(_complete_graph_matching_rounds(n), n_terms_zz)

    return fast_simplify_logs(sp.simplify(z_depth + x_depth + zz_depth))


def controlled_qubitized_ising_tf_number_qubits(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)

    reflection_ancillas = 2 * n_terms - 1
    prepare_work = _prepare_control_work(n, n_terms)
    reflection_work = reflection_ancillas

    return fast_simplify_logs(
        sp.simplify(1 + lcu_unary_number_qubits(n, n_terms) + prepare_work + reflection_work))


def controlled_qubitized_ising_tf_t_count(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    return sp.Integer(0)


def controlled_qubitized_ising_tf_rz_count(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    return sp.simplify(4 * (n_terms - 1))


def controlled_qubitized_ising_tf_toffoli_count(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)

    reflection_ancillas = 2 * n_terms - 1
    controlled_prepare = 12 * (n_terms - 1)
    controlled_select = _controlled_select_toffoli_count(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controlled_reflection = mcx_toffoli_count(reflection_ancillas)

    return fast_simplify_logs(
        sp.simplify(controlled_prepare + controlled_select + controlled_reflection))


def controlled_qubitized_ising_tf_nc_depth(
    n: sp.Symbol,
    n_terms: sp.Symbol | None = None,
    n_terms_z: sp.Symbol | None = None,
    n_terms_zz: sp.Symbol | None = None,
    n_terms_x: sp.Symbol | None = None,
) -> sp.Expr:
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)

    reflection_ancillas = 2 * n_terms - 1
    controlled_prepare_depth = 16 * sp.ceiling(sp.log(n_terms, 2))
    controlled_select_depth = _controlled_select_nc_depth(n, n_terms, n_terms_z, n_terms_zz, n_terms_x)
    controlled_reflection_depth = mcx_nc_depth(reflection_ancillas) # log(reflection_ancillas)

    return fast_simplify_logs(
        sp.simplify(controlled_prepare_depth + controlled_select_depth + controlled_reflection_depth))

