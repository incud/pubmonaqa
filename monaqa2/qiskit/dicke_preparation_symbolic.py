import sympy as sp


def scs_number_qubits(k: sp.Symbol) -> sp.Expr:
    return sp.simplify(k + 1)


def scs_nc_depth(k: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * k)


def scs_t_count(k: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def scs_rz_count(k: sp.Symbol) -> sp.Expr:
    return sp.simplify(4 * k)


def scs_toffoli_count(k: sp.Symbol) -> sp.Expr:
    return sp.simplify(2 * (k - 1))



def wdb_number_qubits(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return sp.simplify(n)


def wdb_nc_depth(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return wdb_rz_count(n, k) + wdb_toffoli_count(n, k)


def wdb_t_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return sp.Integer(0)


def wdb_rz_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    num_ry_addition = k + k * (k+1) / 2
    return 2 * (num_ry_addition)


def wdb_toffoli_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    num_toffoli_addition = k * (k+1)
    num_toffoli_fredkin = 3 * k * (k + 3) / 2
    return num_toffoli_addition + num_toffoli_fredkin



def dicke_preparation_number_qubits(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    return sp.simplify(n)


def dicke_preparation_nc_depth(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    i = sp.symbols("i", integer=True, positive=True)
    # Number of leaves and height of the balanced partition tree.
    L = sp.ceiling(n / k)
    h = sp.log(L, 2) + 1
    # Along any root-to-leaf path:
    # - there are at most h internal WDB rounds;
    # - then one leaf unary-to-Dicke chain of size at most k.
    #
    # Gates on disjoint subtrees / leaves can be parallelized, so depth scales
    # with tree height, not with the total number of nodes.
    full_leaf_depth = sp.summation(scs_nc_depth(i), (i, 2, k))
    full_internal_depth = wdb_nc_depth(2 * k, k)
    return h * full_internal_depth + full_leaf_depth


def dicke_preparation_t_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    i = sp.symbols("i", integer=True, positive=True)
    L = sp.ceiling(n / k)                                               # Number of leaves in the balanced partition tree.
    full_leaf_cost = sp.summation(scs_t_count(i), (i, 2, k))     # Every leaf has size at most k, so each leaf unary-to-Dicke chain is upper-bounded by the full size-k chain.
    full_internal_cost = wdb_t_count(2 * k, k)                       # Every internal node has active prefixes of size at most k on both sides, so each internal WDB is upper-bounded by WDB(2k, k, k).
    return L * full_leaf_cost + (L - 1) * full_internal_cost            # A binary tree with L leaves has exactly L - 1 internal nodes.


def dicke_preparation_rz_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    i = sp.symbols("i", integer=True, positive=True)
    L = sp.ceiling(n / k)                                               # Number of leaves in the balanced partition tree.
    full_leaf_cost = sp.summation(scs_rz_count(i), (i, 2, k))    # Every leaf has size at most k, so each leaf unary-to-Dicke chain is upper-bounded by the full size-k chain.
    full_internal_cost = wdb_rz_count(2 * k, k)                      # Every internal node has active prefixes of size at most k on both sides, so each internal WDB is upper-bounded by WDB(2k, k, k).
    return L * full_leaf_cost + (L - 1) * full_internal_cost            # A binary tree with L leaves has exactly L - 1 internal nodes.


def dicke_preparation_toffoli_count(n: sp.Symbol, k: sp.Symbol) -> sp.Expr:
    i = sp.symbols("i", integer=True, positive=True)
    L = sp.ceiling(n / k)                                               # Number of leaves in the balanced partition tree.
    full_leaf_cost = sp.summation(scs_toffoli_count(i), (i, 2, k)) # Every leaf has size at most k, so each leaf unary-to-Dicke chain is upper-bounded by the full size-k chain.
    full_internal_cost = wdb_toffoli_count(2 * k, k)                 # Every internal node has active prefixes of size at most k on both sides, so each internal WDB is upper-bounded by WDB(2k, k, k).
    return L * full_leaf_cost + (L - 1) * full_internal_cost            # A binary tree with L leaves has exactly L - 1 internal nodes.

