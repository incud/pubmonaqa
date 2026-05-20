from monaqa2.data.filename import TIMING_CPU_FOLDER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


def get_average_cpu_time_per_n(move: str) -> dict[int, float]:
    """
    Return the average measured CPU time per move for each system size.

    :param move: Move type to select. Must be either ``"local1"`` or ``"uniform"``.
    :return: Dictionary mapping ``n`` to the average value of ``seconds_per_move``.
    """
    assert move in ["local1", "uniform"]
    out = {}

    for file in TIMING_CPU_FOLDER.glob("*.csv"):
        df = pd.read_csv(file)
        df = df[df["move"] == move]

        if df.empty:
            continue

        n = df["n"].astype(int).unique()

        if len(n) != 1:
            raise ValueError(f"{file} has multiple n values: {n}")

        out[int(n[0])] = float(df["seconds_per_move"].mean())

    return dict(sorted(out.items()))


def _positive_lstsq(X: np.ndarray, y: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Solve a non-negative least-squares problem.

    :param X: Design matrix.
    :param y: Target vector.
    :param eps: Lower bound imposed on every coefficient.
    :return: Least-squares coefficients minimizing ``||X c - y||_2`` with ``c >= eps``.
    """
    res = lsq_linear(X, y, bounds=(eps, np.inf))

    if not res.success:
        raise RuntimeError(res.message)

    return res.x


def calculate_polynomial_cpu_time_local_move() -> tuple[float, float]:
    """
    Fit the local-move CPU time model.

    :return: Coefficients ``(a, b)`` for ``a + b n``, with ``a,b >= 0``.
    """
    points = get_average_cpu_time_per_n("local1")

    n = np.array(list(points.keys()), dtype=float)
    y = np.array(list(points.values()), dtype=float)

    X = np.column_stack([np.ones_like(n), n])
    a, b = _positive_lstsq(X, y)

    return float(a), float(b)


def calculate_polynomial_cpu_time_uniform_move() -> tuple[float, float, float]:
    """
    Fit the uniform-move CPU time model.

    :return: Coefficients ``(a, b, c)`` for ``a + b n + c n^2``, with ``a,b,c >= 0``.
    """
    points = get_average_cpu_time_per_n("uniform")

    n = np.array(list(points.keys()), dtype=float)
    y = np.array(list(points.values()), dtype=float)

    X = np.column_stack([np.ones_like(n), n, n**2])
    a, b, c = _positive_lstsq(X, y)

    return float(a), float(b), float(c)


def cpu_time_per_local_move(n: int) -> float:
    """
    Return the fitted CPU time for one local move on a system of n spins.

    :param n: Number of spins.
    :return: Estimated seconds per local move on a system of n spins.
    """
    # calculate_polynomial_cpu_time_local_move() -> a + b*n
    # a = 4.200080736659944e-08
    # b = 4.621352693480281e-10
    return 4.621352693480281e-10 * n


def cpu_time_per_uniform_move(n: int) -> float:
    """
    Return the fitted CPU time for one uniform move on a system of n spins.

    :param n: Number of spins.
    :return: Estimated seconds per uniform move on a system of n spins.
    """
    # calculate_polynomial_cpu_time_uniform_move() -> a + b*n + c*n^2
    # a = 2.6081189365793093e-09
    # b = 2.004293998940997e-09
    # c = 1.7274948135461728e-09
    return 2.004293998940997e-09 * n + 1.7274948135461728e-09 * n * n


def plot_average_cpu_time_per_n(move: str, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot all measured CPU times and their average per n, then fit the move-specific timing model with zero offset.

    :param move: Move type to select. Must be either ``"local1"`` or ``"uniform"``.
    :param ax: Optional matplotlib axis.
    :return: Matplotlib figure and axis.
    """
    assert move in ["local1", "uniform"]

    raw_n = []
    raw_y = []

    for file in TIMING_CPU_FOLDER.glob("*.csv"):
        df = pd.read_csv(file)
        df = df[df["move"] == move]

        if df.empty:
            continue

        raw_n.extend(df["n"].astype(float).to_numpy())
        raw_y.extend(pd.to_numeric(df["seconds_per_move"], errors="coerce").to_numpy())

    raw_n = np.asarray(raw_n, dtype=float)
    raw_y = np.asarray(raw_y, dtype=float)
    mask = np.isfinite(raw_n) & np.isfinite(raw_y) & (raw_y > 0.0)
    raw_n, raw_y = raw_n[mask], raw_y[mask]

    points = get_average_cpu_time_per_n(move)

    if not points:
        raise ValueError(f"No CPU timing data found for move={move}")

    n = np.array(list(points.keys()), dtype=float)
    y = np.array(list(points.values()), dtype=float)
    n_grid = np.linspace(float(np.min(n)), float(np.max(n)), 300)

    if move == "local1":
        X = n[:, None]
        (b,) = _positive_lstsq(X, y)
        y_fit = b * n_grid
        fit_label = f"fit: {b:.3e} n"
    else:
        X = np.column_stack([n, n * n])
        b, c = _positive_lstsq(X, y)
        y_fit = b * n_grid + c * n_grid * n_grid
        fit_label = f"fit: {b:.3e} n + {c:.3e} n^2"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
    else:
        fig = ax.figure

    ax.scatter(raw_n, raw_y, s=18, color="grey", alpha=0.10, edgecolors="none", label="raw measurements")
    ax.scatter(n, y, s=42, color="blue", edgecolors="none", label=f"{move} average")
    ax.plot(n_grid, y_fit, color="black", linewidth=2.0, label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Seconds per move")
    ax.set_title(f"CPU time per {move} move")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")
    ax.legend(frameon=False)

    return fig, ax