import sympy as sp


def prepare_unary_number_qubits(n_terms: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * n_terms - 1)


def prepare_unary_nc_depth(n_terms: sp.Symbol) -> sp.Expr:
    # Balanced tree has ceil(log2(n_terms)) split levels.
    # Each split uses one CRY, decomposed as 2 serial RY rotations.
    return sp.simplify(4 * sp.ceiling(sp.log(n_terms, 2)))


def prepare_unary_t_count(n_terms: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def prepare_unary_rz_count(n_terms: sp.Symbol) -> sp.Expr:
    # There is one CRY per internal tree node.
    # Number of internal nodes = n_terms - 1.
    # CRY = 2 RY, and every RY = 2 RZ.
    return sp.simplify(4 * (n_terms - 1))


def prepare_unary_toffoli_count(n_terms: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)
