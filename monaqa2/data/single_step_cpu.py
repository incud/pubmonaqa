from monaqa2.data.filename import TIMING_CPU_FOLDER
import pandas as pd
import numpy as np
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
    # a = 3.850208997728436e-09
    # b = 2.7169295429837303e-09
    return 2.7169295429837303e-09 * n


def cpu_time_per_uniform_move(n: int) -> float:
    """
    Return the fitted CPU time for one uniform move on a system of n spins.

    :param n: Number of spins.
    :return: Estimated seconds per uniform move on a system of n spins.
    """
    # calculate_polynomial_cpu_time_uniform_move() -> a + b*n + c*n^2
    # a = 7.579962345745788e-09
    # b = 7.577190184025888e-09
    # c = 7.56288226735683e-09
    return 7.577190184025888e-09 * n + 7.56288226735683e-09 * n * n