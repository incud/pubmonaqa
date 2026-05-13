import sympy as sp


def t_per_toffoli(k): 
    return 7 * k


def simplify_expression(n, expr):
    # Define wildcards to catch coefficients and constants
    a = sp.Wild('a')
    b = sp.Wild('b')
    
    expr_1 = expr.replace(sp.log(2), 1).replace(sp.ceiling, lambda x: x + 1)
    # expr_1 = expr.replace(sp.ceiling, lambda x: x + 1)
    expr_2 = sp.expand_log(expr_1, force=True).replace(sp.log(2), 1)
    expr_3 = sp.expand(expr_2)
    expr_4 = sp.collect(expr_3, [n**2 * sp.log(n), n**2, n * sp.log(n), n, sp.log(n)])
    return expr_4

def fast_simplify_logs(expr):
    expr_1 = expr.replace(sp.ceiling, lambda x: x + 1)
    # expr_1 = expr.replace(sp.ceiling, lambda x: x + 1)
    expr_2 = sp.expand_log(expr_1, force=True)
    expr_3 = sp.expand(expr_2)
    return expr_3


def get_symbol_name(expr, name):
    matches = sorted([s for s in sp.sympify(expr).free_symbols if s.name == name], key=str)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one symbol named {name!r}, found {matches}.")
    return matches[0]


def substitute_by_symbol_name(expr, substitutions):
    out = sp.sympify(expr)
    for name, value in substitutions.items():
        matches = [s for s in out.free_symbols if s.name == name]
        if matches:
            out = out.subs({s: value for s in matches})
    return out


def advanced_initial_simplify(expr):
    a, b = sp.Wild('a'), sp.Wild('b')
    expr = expr.replace(sp.log(2), 1)
    expr = expr.replace(sp.ceiling(a), a+1)
    expr = expr.replace(sp.floor(a), a)
    expr = sp.expand_log(expr, force=True)
    expr = sp.expand(expr)
    return expr



def leading_terms_upper_bound(expr, variables, assumptions=(), *, default_lower=1):
    import sympy as sp
    from sympy.core.relational import Relational

    variables = tuple(map(sp.sympify, variables))

    def clean(e):
        e = sp.sympify(e).subs(sp.log(2), 1)

        def repl(x):
            arg = x.args[0]
            if arg in variables or not arg.has(*variables):
                return x

            rec = leading_terms_upper_bound(
                arg,
                variables,
                assumptions,
                default_lower=default_lower,
            )
            return sp.log(rec)

        return sp.expand(e.replace(lambda x: x.func is sp.log, repl).subs(sp.log(2), 1))

    expr = clean(expr)

    lbs = {v: sp.sympify(default_lower) for v in variables} if default_lower is not None else {}

    for a in assumptions:
        if not isinstance(a, Relational):
            raise TypeError(f"Invalid assumption: {a}")

        l, r = sp.sympify(a.lhs), sp.sympify(a.rhs)

        if a.rel_op in (">=", ">") and l in variables and not r.has(*variables):
            lbs[l] = sp.Max(lbs.get(l, r), r)
        elif a.rel_op in ("<=", "<") and r in variables and not l.has(*variables):
            lbs[r] = sp.Max(lbs.get(r, l), l)

    logs = sorted(
        [x for x in expr.atoms(sp.Function) if x.func is sp.log],
        key=str,
    )

    gens = tuple(dict.fromkeys((*variables, *logs)))

    for g in gens:
        if g in lbs:
            continue

        if g.func is sp.log and g.args[0] in lbs:
            lbs[g] = sp.simplify(sp.log(lbs[g.args[0]]).subs(sp.log(2), 1))
        elif default_lower is not None:
            lbs[g] = sp.sympify(default_lower)
        else:
            raise ValueError(f"Missing lower bound for {g}")

        if lbs[g].is_positive is False or lbs[g] == 0:
            raise ValueError(f"Generator {g} needs a positive lower bound")

    coeffs = {}

    for term in sp.Add.make_args(expr):
        powers = term.as_powers_dict()
        exp = tuple(sp.sympify(powers.get(g, 0)) for g in gens)
        monomial = sp.prod(g**e for g, e in zip(gens, exp))
        coeff = sp.simplify(term / monomial)

        if coeff.has(*variables):
            raise ValueError(f"Could not decompose term: {term}")

        coeffs[exp] = sp.simplify(coeffs.get(exp, 0) + coeff)

    coeffs = {e: c for e, c in coeffs.items() if c != 0}

    def dominates(a, b):
        return all(sp.simplify(x - y).is_nonnegative is True for x, y in zip(a, b))

    leading = [
        e for e in coeffs
        if not any(f != e and dominates(f, e) for f in coeffs)
    ]

    inflated = {e: sp.Integer(0) for e in leading}

    for e, c in coeffs.items():
        c = sp.Abs(c)

        if e in leading:
            inflated[e] += c
            continue

        choices = [f for f in leading if dominates(f, e)]

        def penalty(f):
            return sp.prod(
                lbs[g] ** sp.simplify(ei - fi)
                for g, ei, fi in zip(gens, e, f)
            )

        try:
            chosen = min(choices, key=lambda f: float(sp.N(penalty(f))))
        except Exception:
            chosen = choices[0]

        inflated[chosen] += c * penalty(chosen)

    bound = sum(
        sp.ceiling(sp.simplify(c).subs(sp.log(2), 1))
        * sp.prod(g**e for g, e in zip(gens, exp))
        for exp, c in inflated.items()
    )

    return sp.factor_terms(sp.simplify(bound.subs(sp.log(2), 1)))


def replace_shifted_logs(expr, variables):
    import sympy as sp

    variables = tuple(map(sp.sympify, variables))

    def safe_log_const(c):
        c = sp.simplify(c)

        if c == 1:
            return 0

        if c.is_positive is True:
            return sp.log(c)

        # Avoid nan / complex logs from negative or unknown constants.
        return 0

    def repl(x):
        arg = sp.expand(x.args[0])

        for v in variables:
            a = sp.simplify(arg.coeff(v))
            b = sp.simplify(arg - a*v)

            if a != 0 and not a.has(*variables) and not b.has(*variables):
                return safe_log_const(a) + sp.log(v)

        return x

    return sp.sympify(expr).replace(
        lambda x: x.func is sp.log,
        repl,
    ).subs(sp.log(2), 1)


def remove_lambertw_upper_bound(expr):
    import sympy as sp

    expr = sp.sympify(expr)

    def bound_log_arg(arg):
        out = arg

        while True:
            hits = [
                p for p in out.atoms(sp.Pow)
                if p.base.func is sp.LambertW and p.exp.is_negative is True
            ]

            if not hits:
                return out

            p = hits[0]
            z = p.base.args[0]
            k = sp.simplify(-p.exp)
            rest = sp.factor_terms(out / p)

            if k == 1:
                out = sp.Add(
                    rest,
                    sp.factor_terms(rest / z),
                    evaluate=False,
                )
            elif k.is_integer is True and k.is_positive is True:
                out = sp.expand(rest * ((1 + z) / z) ** k)
            else:
                out = rest * ((1 + z) / z) ** k

    def repl_log(x):
        old_arg = x.args[0]
        new_arg = bound_log_arg(old_arg)

        if new_arg == old_arg:
            return x

        return sp.log(new_arg, evaluate=False)

    return expr.replace(lambda x: x.func is sp.log, repl_log)