import sympy as sp


def trotterized_ising_tf_number_qubits(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    return sp.sympify(n)


def trotterized_ising_tf_nc_depth(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    r = _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    if n_terms_z is None:
        return sp.simplify(r * (n + 2) + 1)
    return sp.simplify(r * (_positive_indicator(n_terms_z) + _positive_indicator(n_terms_zz) * _complete_graph_matching_rounds(n) + _positive_indicator(n_terms_x)) + _positive_indicator(n_terms_x))


def trotterized_ising_tf_t_count(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    return sp.Integer(0)


def trotterized_ising_tf_rz_count(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, U: sp.Symbol | None = None, commutator_bound: sp.Symbol | None = None) -> sp.Expr:
    r = _resolve_steps(num_trotter_steps, eps=eps, time=time, U=U, commutator_bound=commutator_bound)
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    if n_terms_z is None:
        return sp.simplify(r * n_terms)
    return sp.simplify(r * (n_terms_z + n_terms_zz) + (r + 1) * n_terms_x)


def trotterized_ising_tf_toffoli_count(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    return sp.Integer(0)


def controlled_trotterized_ising_tf_number_qubits(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    return sp.simplify(2 * sp.sympify(n) + 1)


def controlled_trotterized_ising_tf_nc_depth(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    n = sp.sympify(n)
    r = _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)

    if n_terms_z is None:
        return sp.simplify(2 * n + 6 * r * n_terms)

    return sp.simplify(2 * n + r * (4 * n_terms_z + 6 * n_terms_zz + 6 * n_terms_x))


def controlled_trotterized_ising_tf_t_count(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)
    _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    return sp.Integer(0)


def controlled_trotterized_ising_tf_rz_count(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    r = _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    n_terms, n_terms_z, n_terms_zz, n_terms_x = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)

    if n_terms_z is None:
        return sp.simplify(2 * r * n_terms)

    return sp.simplify(2 * r * (n_terms_z + n_terms_zz + n_terms_x))


def controlled_trotterized_ising_tf_toffoli_count(n: sp.Symbol, num_trotter_steps: sp.Symbol | None = None, n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    r = _resolve_steps(num_trotter_steps, eps=eps, time=time, commutator_bound=commutator_bound, U=U)
    _, n_terms_z, n_terms_zz, _ = _resolve_terms(n_terms, n_terms_z, n_terms_zz, n_terms_x)

    if n_terms_z is None:
        return sp.simplify(2 * r * sp.sympify(n_terms))

    return sp.simplify(2 * r * n_terms_zz)


def _resolve_terms(n_terms: sp.Symbol | None = None, n_terms_z: sp.Symbol | None = None, n_terms_zz: sp.Symbol | None = None, n_terms_x: sp.Symbol | None = None) -> tuple[sp.Expr, sp.Expr | None, sp.Expr | None, sp.Expr | None]:
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


def _resolve_steps(num_trotter_steps: sp.Symbol | None, eps: sp.Symbol | None = None, time: sp.Symbol = 1, commutator_bound: sp.Symbol | None = None, U: sp.Symbol | None = None) -> sp.Expr:
    has_steps = num_trotter_steps is not None
    bound = commutator_bound if commutator_bound is not None else U
    has_eps_bound = eps is not None and bound is not None

    if has_steps and has_eps_bound:
        raise ValueError("Pass either num_trotter_steps or eps with commutator_bound/U, not both.")

    if has_steps:
        return sp.sympify(num_trotter_steps)

    if not has_eps_bound:
        raise ValueError("Pass either num_trotter_steps or both eps and commutator_bound/U.")

    return sp.Max(1, sp.ceiling(sp.Abs(sp.sympify(time)) ** 2 * sp.sympify(bound) / (2 * sp.sympify(eps))))


def _positive_indicator(x: sp.Expr) -> sp.Expr:
    x = sp.sympify(x)

    if x.is_zero:
        return sp.Integer(0)

    if x.is_positive:
        return sp.Integer(1)

    return sp.Piecewise((0, x <= 0), (1, True))


def _complete_graph_matching_rounds(n: sp.Symbol) -> sp.Expr:
    return sp.sympify(n)
