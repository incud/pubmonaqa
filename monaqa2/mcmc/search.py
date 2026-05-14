def search_iterative(step: "callable[any, any]", compare: "callable[any, float]", start_elem: any, max_iter: int, info: str="") -> int:
    """Return the smallest t in [0, max_iter] with compare(step^t(start_elem)) <= 0,
    assuming compare(step^t(start_elem)) is monotone non-increasing in t. Uses linear search; raises ValueError if none exists."""
    elem = start_elem
    for t in range(max_iter + 1):
        cost = compare(elem)
        if cost <= 0:
            return t
        elem = step(elem)
    raise ValueError(f"not found in [0, {max_iter}]: residual is {cost} | {info}")


def search_monotone(fun: "callable[int, any]", compare: "callable[any, float]", start_iter: int, max_iter: int, info: str="", permit_continued_search: bool = False) -> int:
    """Return the smallest t in [start_iter, max_iter] with compare(fun(t)) <= 0,
    assuming compare(fun(t)) is monotone non-increasing in t. Uses doubling to
    bracket a solution, then binary search; raises ValueError if none exists."""
    # find hi by doubling
    lo = start_iter
    hi = max(lo, 1)
    cost = compare(fun(lo))

    while hi <= max_iter:
        if lo == hi == max_iter:
            break
        cost = compare(fun(hi))
        if cost <= 0:
            break
        lo, hi = hi, min(2 * hi, max_iter)

    if cost > 0:
        raise ValueError(f"Not found in [{start_iter=}, {max_iter=}]: residual is {compare(fun(hi))} [{hi=} cost={cost}] {info}.")
        # search_monotone_debug(fun, compare, start_iter, max_iter)

    # binary search in (lo, hi]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if compare(fun(mid)) <= 0:
            hi = mid
        else:
            lo = mid
    return hi


def search_monotone_debug(fun: "callable[int, any]", compare: "callable[any, float]", start_iter: int, max_iter: int, info: str="", permit_continued_search: bool = False) -> int:
    """Return the smallest t in [start_iter, max_iter] with compare(fun(t)) <= 0,
    assuming compare(fun(t)) is monotone non-increasing in t. Uses doubling to
    bracket a solution, then binary search; raises ValueError if none exists."""
    # find hi by doubling
    lo = start_iter
    hi = max(lo, 1)
    cost = compare(fun(lo))
    debug = ""

    while hi <= max_iter:
        if lo == hi == max_iter:
            break
        cost = compare(fun(hi))
        debug += f"  search_monotone: compare(fun({hi}))={cost} > 0, doubling hi to {2*hi}\n"
        if cost <= 0:
            debug += f"  search_monotone: found hi={hi} with compare(fun({hi}))={cost} <= 0\n"
            break
        lo, hi = hi, min(2 * hi, max_iter)

    debug += f"  search_monotone: final lo={lo}, compare(fun({lo}))={compare(fun(lo))}; hi={hi}, compare(fun({hi}))={compare(fun(hi))}\n"
    if cost > 0:
        raise ValueError(f"Not found in [{start_iter=}, {max_iter=}]: residual is {compare(fun(hi))} [{hi=} cost={cost}] {info}. Debug msg:\n{debug}")

    # binary search in (lo, hi]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if compare(fun(mid)) <= 0:
            hi = mid
        else:
            lo = mid
    return hi
