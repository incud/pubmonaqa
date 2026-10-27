"""Shared helpers for statistics of complete per-instance annealing costs."""

from typing import Callable

import numpy as np
import pandas as pd

from monaqa2.data.utils_stats import grouped_statistics


def query_temperature_window(
    table: pd.DataFrame,
    value_col: str,
    beta_min: float,
    beta_max: float,
    is_gap: bool = False,
) -> pd.DataFrame:
    """Select measured temperatures bracketing the requested schedule range."""
    if not np.isfinite(beta_min) or not np.isfinite(beta_max) or beta_min < 0.0 or beta_max < beta_min:
        raise ValueError("Invalid requested temperature range.")
    if table.empty:
        raise ValueError("No instance data match the requested query calculation.")
    temperatures = np.sort(table["beta"].astype(float).unique())
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("Nonfinite temperatures in query data.")
    if beta_min < temperatures[0] and not np.isclose(beta_min, temperatures[0]):
        raise ValueError("Schedule below measured temperature range.")
    if beta_max > temperatures[-1] and not np.isclose(beta_max, temperatures[-1]):
        raise ValueError("Schedule above measured temperature range.")
    left = max(0, int(np.searchsorted(temperatures, beta_min, side="right")) - 1)
    right = min(len(temperatures) - 1, int(np.searchsorted(temperatures, beta_max)))
    out = table.loc[table["beta"].between(temperatures[left], temperatures[right])].copy()
    values = out[value_col].to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0.0)
    if is_gap:
        invalid |= values > 1.0
    if np.any(invalid):
        row = out.loc[invalid].iloc[0]
        raise ValueError(f"Invalid {value_col} for n={row['n']}, idx={row['idx']}, beta={row['beta']}; cannot estimate a finite instance cost.")
    if out.duplicated(["n", "idx", "beta"]).any():
        out = out.drop_duplicates()
        if out.duplicated(["n", "idx", "beta"]).any():
            raise ValueError("Conflicting duplicate instance/temperature rows.")
    return out


def annealing_query_rows(
    table: pd.DataFrame,
    value_col: str,
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    cost: Callable[[int, float, list[float]], float],
    is_gap: bool = False,
) -> pd.DataFrame:
    """Interpolate each instance in temperature, then calculate its total cost.

    Interpolation is linear in log(value). Values outside the instance's
    measured temperature range are rejected rather than silently extrapolated.
    Invalid endpoints are errors, including unresolved/nonpositive gaps and
    failed classical-query sentinels. Only endpoints used by the schedule are
    checked, so failures at unrelated temperatures do not affect a calculation.
    """
    beta, epsilon = float(beta), float(epsilon)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative.")
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must satisfy 0 < epsilon < 1.")
    if table.empty:
        raise ValueError("No instance data match the requested query calculation.")

    schedules = {}
    rows = []
    for (n, idx), group in table.groupby(["n", "idx"], sort=True):
        n = int(n)
        if n not in schedules:
            schedule = np.asarray(annealing_schedule_generator(n, beta), dtype=float)
            if schedule.ndim != 1 or not np.all(np.isfinite(schedule)):
                raise ValueError(f"Invalid annealing schedule for n={n}.")
            if np.any(schedule < 0.0) or np.any(np.diff(schedule) <= 0.0):
                raise ValueError(f"The schedule for n={n} must be non-negative and strictly increasing.")
            # A leading beta=0 is the directly prepared initial state.
            schedule = schedule[schedule > 0.0]
            if (beta > 0.0 and (not schedule.size or not np.isclose(schedule[-1], beta))) or (beta == 0.0 and schedule.size):
                raise ValueError(f"The schedule for n={n} must end at beta={beta}.")
            schedules[n] = schedule
        schedule = schedules[n]
        if not schedule.size:
            rows.append({"n": n, "idx": idx, "beta": beta, "queries": 0.0})
            continue

        data = group[["beta", value_col]].copy()
        data["beta"] = pd.to_numeric(data["beta"], errors="coerce")
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
        if not np.all(np.isfinite(data["beta"])):
            raise ValueError(f"Nonfinite temperature for n={n}, idx={idx}.")
        data = data.drop_duplicates().sort_values("beta")
        if data["beta"].duplicated().any():
            raise ValueError(f"Conflicting duplicate temperatures for n={n}, idx={idx}.")
        temperatures = data["beta"].to_numpy(dtype=float)
        values = data[value_col].to_numpy(dtype=float)
        if schedule[0] < temperatures[0] and not np.isclose(schedule[0], temperatures[0]):
            raise ValueError(f"Schedule below measured temperature range for n={n}, idx={idx}.")
        if schedule[-1] > temperatures[-1] and not np.isclose(schedule[-1], temperatures[-1]):
            raise ValueError(f"Schedule above measured temperature range for n={n}, idx={idx}.")

        interpolated = []
        for beta_t in schedule:
            exact = np.flatnonzero(np.isclose(temperatures, beta_t))
            if exact.size:
                left = right = int(exact[0])
            else:
                right = int(np.searchsorted(temperatures, beta_t))
                left = right - 1
            endpoints = values[[left, right]]
            if not np.all(np.isfinite(endpoints)) or np.any(endpoints <= 0.0) or (is_gap and np.any(endpoints > 1.0)):
                raise ValueError(f"Invalid {value_col} at beta={beta_t:g} for n={n}, idx={idx}; cannot estimate a finite instance cost.")
            if left == right:
                value = values[left]
            else:
                weight = (beta_t - temperatures[left]) / (temperatures[right] - temperatures[left])
                value = np.exp((1.0 - weight) * np.log(values[left]) + weight * np.log(values[right]))
            interpolated.append(float(value))
        total = float(cost(n, epsilon, interpolated))
        if not np.isfinite(total) or total < 0.0:
            raise ValueError(f"Invalid total query cost for n={n}, idx={idx}.")
        rows.append({"n": n, "idx": idx, "beta": beta, "queries": total})
    return pd.DataFrame(rows, columns=["n", "idx", "beta", "queries"])


def annealing_query_stats(rows: pd.DataFrame, statistic: str = "mean+std") -> pd.DataFrame:
    """Summarize instance totals, including quartiles and sample extrema.

    q25/q75 use linear sample quantiles; q0/q100 are the minimum/maximum.
    All four columns use the untrimmed instance costs.
    Existing center/spread columns retain their meaning, including ddof=1.
    Other supported statistics retain the package's trimming/MAD conventions.
    A sample standard deviation is undefined for fewer than two instances.
    """
    stats = grouped_statistics(rows, value_col="queries", statistic=statistic)
    if statistic in ("mean+std", "mean+std-tail"):
        count = stats["count"].to_numpy(dtype=float)
        correction = np.full(count.shape, np.nan)
        np.divide(count, count - 1.0, out=correction, where=count > 1.0)
        stats["spread"] = stats["spread"] * np.sqrt(correction)
    if stats.empty:
        return stats.assign(q25=pd.Series(dtype=float), q75=pd.Series(dtype=float), q0=pd.Series(dtype=float), q100=pd.Series(dtype=float))
    quantiles = rows.groupby(["n", "beta"])["queries"].quantile([0.0, 0.25, 0.75, 1.0], interpolation="linear").unstack()
    quantiles.columns = ["q0", "q25", "q75", "q100"]
    return stats.merge(quantiles[["q25", "q75", "q0", "q100"]], on=["n", "beta"], validate="one_to_one")


def query_percentile_columns(percentiles: tuple[int, int]) -> tuple[str, str]:
    """Validate the supported percentile bands and return their table columns."""
    message = "band_percentiles must be (25, 75) or (0, 100)."
    try:
        values = np.asarray(percentiles, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if values.shape != (2,) or not (np.array_equal(values, (25, 75)) or np.array_equal(values, (0, 100))):
        raise ValueError(message)
    return f"q{int(values[0])}", f"q{int(values[1])}"


def fit_query_quantile_ratios_by_n(
    stats: pd.DataFrame,
    n_min: int | None = None,
    n_max: int | None = None,
    *,
    percentiles: tuple[int, int] = (25, 75),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fit lower/mean and (upper-lower)/mean separately to A*exp(b*n).

    percentiles selects (25, 75) quartiles or (0, 100) sample extrema.
    Applied to the full-path mean, these fits give positive, ordered percentile
    bounds while retaining changes in the actual annealing schedule. The mean
    need not lie between fitted bounds. An identically zero range stays zero;
    otherwise only positive widths enter the logarithmic width fit.
    """
    lower_col, upper_col = query_percentile_columns(percentiles)
    ratios = stats.copy()
    if n_min is not None:
        ratios = ratios[ratios["n"] >= n_min]
    if n_max is not None:
        ratios = ratios[ratios["n"] <= n_max]
    if not ratios["statistic"].eq("mean+std").all():
        raise ValueError("Percentile ratios require arithmetic means: statistic='mean+std'.")
    values = ratios[["center", lower_col, upper_col]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:, :2] <= 0.0) or np.any(values[:, 2] < values[:, 1]):
        raise ValueError("Percentile fits need finite positive means and 0 < lower <= upper.")
    ratios["spread"] = (ratios[upper_col] - ratios[lower_col]) / ratios["center"]
    ratios["center"] = ratios[lower_col] / ratios["center"]
    return fit_query_statistic_by_n(ratios, "center"), fit_query_statistic_by_n(ratios, "spread")


def fit_query_statistic_by_n(
    stats: pd.DataFrame,
    column: str,
    n_min: int | None = None,
    n_max: int | None = None,
) -> tuple[float, float]:
    """Fit center or spread to A*exp(b*n) for a single target temperature.

    All-zero statistics return (0, 0). Otherwise at least two positive,
    finite values at distinct sizes are required for the logarithmic fit.
    """
    if column not in ("center", "spread"):
        raise ValueError("column must be 'center' or 'spread'.")
    data = stats.copy()
    if n_min is not None:
        data = data[data["n"] >= n_min]
    if n_max is not None:
        data = data[data["n"] <= n_max]
    if data.empty:
        raise ValueError("No query statistics in the requested fitting range.")
    if data["beta"].nunique() != 1:
        raise ValueError("Fit query statistics for one target beta at a time.")
    if data["n"].duplicated().any():
        raise ValueError("Expected one query statistic per system size.")
    n = data["n"].to_numpy(dtype=float)
    values = data[column].to_numpy(dtype=float)
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"Cannot fit nonfinite or negative {column}; standard deviations need at least two instances per size.")
    if np.all(values == 0.0):
        return 0.0, 0.0
    positive = values > 0.0
    if np.count_nonzero(positive) < 2:
        raise ValueError(f"Need at least two positive {column} values to fit.")
    slope, log_A = np.polyfit(n[positive], np.log(values[positive]), deg=1)
    return float(np.exp(log_A)), float(slope)
