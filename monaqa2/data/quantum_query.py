"""Quantum query costs: calculate each instance first, then summarize and fit."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from monaqa2.data.filename import SPECTRAL_GAP_FILE
from monaqa2.data.runtime import get_annealing_queries_quantum_walks
from monaqa2.data.utils_queries import annealing_query_rows, annealing_query_stats, fit_query_statistic_by_n, query_temperature_window
from monaqa2.data.utils_stats import fit_exponential_from_stats


def load_quantum_queries(
    proposal: str,
    a: int | float,
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    in_file: Path = SPECTRAL_GAP_FILE,
    zeno_overlap_probability: float = 1.0 / np.e,
    *,
    n_min: int | None = None,
    n_max: int | None = None,
) -> pd.DataFrame:
    """Return complete annealing costs, with columns n, idx, beta, queries.

    Input is the existing spectral-gap pickle. For each (n, idx), interpolate
    that instance's gaps log-linearly in beta, then apply the existing complete
    annealing-query formula. The output beta is the target temperature.
    Nonpositive/unresolved gaps needed by the schedule raise ValueError;
    instances are never silently removed or replaced by a mean gap.
    """
    overlap = float(zeno_overlap_probability)
    if not np.isfinite(overlap) or not 0.0 < overlap <= 1.0:
        raise ValueError("zeno_overlap_probability must satisfy 0 < p <= 1.")
    table = pd.read_pickle(in_file)
    table = table.loc[(table["proposal"] == proposal) & (table["a"] == a), ["n", "idx", "beta", "delta"]]
    if n_min is not None:
        table = table[table["n"] >= n_min]
    if n_max is not None:
        table = table[table["n"] <= n_max]
    return annealing_query_rows(
        table, "delta", beta, epsilon, annealing_schedule_generator,
        lambda n, eps, gaps: get_annealing_queries_quantum_walks(n, eps, gaps, overlap),
        is_gap=True,
    )


def get_quantum_query_stats(
    proposal: str,
    a: int | float,
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    zeno_overlap_probability: float = 1.0 / np.e,
    *,
    n_min: int | None = None,
    n_max: int | None = None,
) -> pd.DataFrame:
    """Summarize full per-instance costs by n at one target beta.

    Return n, beta, center, spread, count, statistic, q25, q75, q0, q100. Existing
    columns follow classical_query and spectral_gap. By default center is
    the arithmetic mean and spread is the sample standard deviation (ddof=1).
    q25/q75 are linear sample quantiles of the complete instance costs.
    q0/q100 are the minimum/maximum of those same complete instance costs.
    The spread describes disorder-instance variability, not confidence in a fit.
    """
    rows = load_quantum_queries(proposal, a, beta, epsilon, annealing_schedule_generator, in_file, zeno_overlap_probability, n_min=n_min, n_max=n_max)
    return annealing_query_stats(rows, statistic)


def get_quantum_query_mean_fit_by_n(
    stats: pd.DataFrame,
    n_min: int | None = None,
    n_max: int | None = None,
    *,
    fixed_beta: float | None = None,
) -> tuple[float, float]:
    """Fit mean query cost to A*exp(b*n) from a mean+std statistics table.

    Pass the table returned by get_quantum_query_stats so that mean and std
    fits reuse the same instance calculation. Return (A, b). With fixed_beta,
    fit the per-temperature table from get_quantum_query_step_stats instead;
    absent temperatures use the existing interpolation of log(A) and b.
    """
    if not stats["statistic"].eq("mean+std").all():
        raise ValueError("The mean fit requires statistic='mean+std'.")
    if fixed_beta is not None:
        return fit_exponential_from_stats(stats, beta=fixed_beta, n_min=n_min, n_max=n_max)
    return fit_query_statistic_by_n(stats, "center", n_min, n_max)


def get_quantum_query_std_fit_by_n(
    stats: pd.DataFrame,
    n_min: int | None = None,
    n_max: int | None = None,
) -> tuple[float, float]:
    """Fit sample standard deviation to A*exp(b*n); return (A, b).

    Identical instance costs at every fitted size give (0, 0). Mixed zero
    and positive spreads are fitted using the positive points only.
    """
    if not stats["statistic"].eq("mean+std").all():
        raise ValueError("The std fit requires statistic='mean+std'.")
    return fit_query_statistic_by_n(stats, "spread", n_min, n_max)


def get_quantum_query_step_stats(
    proposal: str,
    a: int | float,
    epsilon: float,
    beta_min: float,
    beta_max: float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    zeno_overlap_probability: float = 1.0 / np.e,
    *,
    n_min: int | None = None,
    n_max: int | None = None,
) -> pd.DataFrame:
    """Summarize per-instance one-filter reference costs at each temperature.

    Each reference cost uses get_annealing_queries_quantum_walks with L=1,
    before averaging. For a complete L-filter schedule the plot rescales the
    sum by the ratio of the L-filter and one-filter precision prefactors.
    This keeps the actual schedule and the existing filter error allocation.
    """
    overlap = float(zeno_overlap_probability)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must satisfy 0 < epsilon < 1.")
    if not np.isfinite(overlap) or not 0.0 < overlap <= 1.0:
        raise ValueError("zeno_overlap_probability must satisfy 0 < p <= 1.")
    table = pd.read_pickle(in_file)
    table = table.loc[(table["proposal"] == proposal) & (table["a"] == a), ["n", "idx", "beta", "delta"]]
    if n_min is not None:
        table = table[table["n"] >= n_min]
    if n_max is not None:
        table = table[table["n"] <= n_max]
    rows = query_temperature_window(table, "delta", beta_min, beta_max, is_gap=True)
    rows["queries"] = [get_annealing_queries_quantum_walks(n, epsilon, [gap], overlap) for n, gap in zip(rows["n"], rows["delta"])]
    return annealing_query_stats(rows, statistic)
