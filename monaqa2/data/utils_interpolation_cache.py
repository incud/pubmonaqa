import json
import hashlib
from pathlib import Path
from functools import wraps
from itertools import product

import numpy as np


def _key(func):
    """Return the default cache key for a function."""
    return f"{func.__module__}.{func.__qualname__}"


def _prefix(key):
    """Return a short deterministic prefix used to name arrays inside the .npz file."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _load(filename):
    """
    Load metadata and arrays from a shared interpolation-cache .npz file.

    The file stores one JSON metadata object under "__meta__" and all numerical
    arrays separately. If the file does not exist, an empty cache is returned.
    """
    path = Path(filename)
    if not path.exists():
        return {"functions": {}}, {}

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["__meta__"].item())
        arrays = {name: data[name] for name in data.files if name != "__meta__"}

    meta.setdefault("functions", {})
    return meta, arrays


def _save(filename, meta, arrays):
    """Save metadata and numerical arrays to a compressed shared .npz cache file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(arrays)
    payload["__meta__"] = np.array(json.dumps(meta))

    np.savez_compressed(path, **payload)


def _interp_grid(x, axes, values):
    """
    Multilinear interpolation on a regular tensor-product grid.

    Parameters
    ----------
    x:
        Point where the interpolant is evaluated, already transformed according
        to the stored coordinate system, e.g. log(x) for logarithmic axes.

    axes:
        Stored grid axes. These may be linear axes or logarithmically transformed axes.

    values:
        Tensor of precomputed function values on the grid.
    """
    js = []
    ts = []

    # For each coordinate, locate the enclosing grid interval and interpolation weight.
    for xi, axis in zip(x, axes):
        j = np.searchsorted(axis, xi, side="right") - 1

        if j < 0 or j >= len(axis):
            raise ValueError("point outside interpolation grid")

        # If xi is exactly on the last grid point, use the last interval with t = 1.
        if j == len(axis) - 1:
            j = len(axis) - 2
            t = 1.0
        else:
            t = (xi - axis[j]) / (axis[j + 1] - axis[j])

        js.append(j)
        ts.append(t)

    y = 0.0

    # Sum over all corners of the enclosing hypercube.
    for corner in product((0, 1), repeat=len(axes)):
        w = 1.0
        idx = []

        for d, bit in enumerate(corner):
            w *= ts[d] if bit else 1.0 - ts[d]
            idx.append(js[d] + bit)

        y += w * values[tuple(idx)]

    return float(y)


def build_interpolation_cache(filename, func, axes, log_axes=(), log_output=False, key=None, dtype=np.float64, overwrite=True):
    """
    Precompute a scalar function on a regular tensor-product grid and store it in a shared .npz file.

    Parameters
    ----------
    filename:
        Path to the shared cache file.

    func:
        Scalar function to precompute. It must accept only positional int/float
        arguments and return a scalar int/float value.

    axes:
        Iterable of one-dimensional grids, one per function argument. For example,
        for f(x, y), pass axes=(x_grid, y_grid).

    log_axes:
        Indices of input axes to interpolate in logarithmic coordinates. For example,
        log_axes=(0,) means that the first argument is interpolated as log(x).

    log_output:
        If True, store log(func(...)) and exponentiate after interpolation. This is
        useful for positive quantities with power-law or exponential scaling.

    key:
        Optional explicit cache key. By default, the key is module + qualname.

    dtype:
        Numerical dtype used to store the cached values.

    overwrite:
        If False, raise an error when this function is already present in the cache.
    """
    cache_key = key or _key(func)
    pref = _prefix(cache_key)
    log_axes = set(log_axes)

    # Keep the original axes for evaluating func, and transformed axes for interpolation.
    raw_axes = [np.asarray(axis, dtype=float) for axis in axes]

    if not raw_axes:
        raise ValueError("at least one axis is required")

    stored_axes = []

    for d, axis in enumerate(raw_axes):
        if axis.ndim != 1 or len(axis) < 2:
            raise ValueError("each axis must be one-dimensional and contain at least two points")

        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must contain only finite values")

        # Logarithmic interpolation is implemented by storing log(axis).
        if d in log_axes:
            if np.any(axis <= 0):
                raise ValueError("log-interpolated axes must be strictly positive")
            axis = np.log(axis)

        if np.any(np.diff(axis) <= 0):
            raise ValueError("axes must be strictly increasing")

        stored_axes.append(axis)

    values = np.empty(tuple(len(axis) for axis in raw_axes), dtype=dtype)

    # Evaluate the original function at every grid point.
    for idx in np.ndindex(values.shape):
        args = tuple(float(raw_axes[d][idx[d]]) for d in range(len(raw_axes)))
        y = float(func(*args))

        if not np.isfinite(y):
            raise ValueError(f"function returned non-finite value {y} at args={args}")

        # Store log-output when interpolation in output-log-space is requested.
        if log_output:
            if y <= 0:
                raise ValueError("log_output=True requires strictly positive function values")
            y = np.log(y)

        values[idx] = y

    meta, arrays = _load(filename)

    if cache_key in meta["functions"] and not overwrite:
        raise ValueError(f"cache already contains function {cache_key!r}")

    # Remove stale arrays belonging to the same cache key before overwriting.
    for name in list(arrays):
        if name.startswith(pref + "__"):
            del arrays[name]

    # Store each axis separately, then the tensor of values.
    for d, axis in enumerate(stored_axes):
        arrays[f"{pref}__axis_{d}"] = axis

    arrays[f"{pref}__values"] = values

    # Metadata tells the decorator how to reconstruct this function's cache.
    meta["functions"][cache_key] = {"prefix": pref, "ndim": len(raw_axes), "log_axes": sorted(log_axes), "log_output": bool(log_output)}

    _save(filename, meta, arrays)


def interpolation_cache(filename, key=None, fallback=True):
    """
    Decorator for replacing a scalar function by interpolation from a precomputed cache.

    Example
    -------
    @interpolation_cache("interpolation_cache.npz")
    def f(x):
        return expensive_exact_f(x)

    f.build_cache(axes=(np.logspace(-300, 0, 4096),), log_axes=(0,), log_output=True)

    y = f(1e-16)

    Notes
    -----
    The cache is loaded lazily on the first function call. If the cache is missing
    and fallback=True, the original function is evaluated instead.
    """
    def decorator(func):
        cache_key = key or _key(func)
        state = {"loaded": False, "available": False}

        def load():
            """Load this function's interpolation data from disk, once per process."""
            if state["loaded"]:
                return

            meta, arrays = _load(filename)
            info = meta["functions"].get(cache_key)

            if info is None:
                state.update({"loaded": True, "available": False})
                return

            pref = info["prefix"]
            ndim = int(info["ndim"])

            state["axes"] = [arrays[f"{pref}__axis_{d}"] for d in range(ndim)]
            state["values"] = arrays[f"{pref}__values"]
            state["log_axes"] = set(info.get("log_axes", []))
            state["log_output"] = bool(info.get("log_output", False))
            state.update({"loaded": True, "available": True})

        @wraps(func)
        def wrapped(*args):
            """Evaluate the cached interpolant, falling back to the original function if needed."""
            load()

            if not state["available"]:
                if fallback:
                    return func(*args)
                raise ValueError(f"no interpolation cache found for {cache_key!r}")

            axes = state["axes"]

            if len(args) != len(axes):
                raise ValueError("number of arguments does not match interpolation cache dimension")

            x = np.asarray(args, dtype=float)

            if not np.all(np.isfinite(x)):
                if fallback:
                    return func(*args)
                raise ValueError("arguments must be finite")

            # Apply the same coordinate transform used when the cache was built.
            for d in state["log_axes"]:
                if x[d] <= 0:
                    if fallback:
                        return func(*args)
                    raise ValueError("log-interpolated arguments must be strictly positive")
                x[d] = np.log(x[d])

            # Outside-grid calls are not extrapolated; fallback keeps behavior safe.
            for xi, axis in zip(x, axes):
                if xi < axis[0] or xi > axis[-1]:
                    if fallback:
                        return func(*args)
                    raise ValueError("point outside interpolation grid")

            y = _interp_grid(x, axes, state["values"])

            # If log-output was stored, interpolation happened in log-space.
            return float(np.exp(y) if state["log_output"] else y)

        def build_cache(axes, log_axes=(), log_output=False, dtype=np.float64, overwrite=True):
            """Convenience method attached to the decorated function."""
            build_interpolation_cache(filename, func, axes, log_axes=log_axes, log_output=log_output, key=cache_key, dtype=dtype, overwrite=overwrite)

            # Force reload after rebuilding the cache.
            state["loaded"] = False

        wrapped.build_cache = build_cache
        wrapped.cache_key = cache_key
        return wrapped

    return decorator
