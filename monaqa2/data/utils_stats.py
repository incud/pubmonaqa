import numpy as np
import pandas as pd


STATISTICS = ("mean+std", "mean+std-tail", "median+mad")


def summarize_values(values, statistic: str = "mean+std", positive_only: bool = False) -> tuple[float, float, int]:
    """
    Summarize a one-dimensional sample by one of the common statistics used in the data tables.

    :param values: Input sample.
    :param statistic: One of "mean+std", "mean+std-tail", or "median+mad".
    :param positive_only: If True, discard non-positive values before summarizing.
    :return: Tuple (center, spread, count).
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {list(STATISTICS)}")

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if positive_only:
        x = x[x > 0.0]

    if x.size == 0:
        return np.nan, np.nan, 0

    if statistic == "mean+std":
        return float(np.mean(x)), float(np.std(x)), int(x.size)

    if statistic == "mean+std-tail":
        q1, q3 = np.percentile(x, [25, 75])
        x = x[(x >= q1) & (x <= q3)]
        return (float(np.mean(x)), float(np.std(x)), int(x.size)) if x.size else (np.nan, np.nan, 0)

    if statistic == "median+mad":
        center = float(np.median(x))
        return center, float(np.median(np.abs(x - center))), int(x.size)

    raise RuntimeError("unreachable statistic branch")


def grouped_statistics(
    df: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] = ("n", "beta"),
    statistic: str = "mean+std",
    positive_only: bool = False,
    extra_cols: dict | None = None,
    sort_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Calculate grouped statistics for tables indexed by columns such as (n, beta).

    :param df: Input table.
    :param value_col: Column containing the sample values.
    :param group_cols: Columns to group by before summarizing.
    :param statistic: One of "mean+std", "mean+std-tail", or "median+mad".
    :param positive_only: If True, discard non-positive values before summarizing.
    :param extra_cols: Constant columns appended to every output row.
    :param sort_cols: Columns used for sorting the output; defaults to group_cols.
    :return: DataFrame with group_cols + ["center", "spread", "count", "statistic"] + extra_cols.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {list(STATISTICS)}")

    rows = []
    extra_cols = {} if extra_cols is None else dict(extra_cols)
    sort_cols = tuple(group_cols) if sort_cols is None else tuple(sort_cols)

    if df.empty:
        columns = list(group_cols) + ["center", "spread", "count", "statistic"] + list(extra_cols.keys())
        return pd.DataFrame(columns=columns)

    for key, group in df.groupby(list(group_cols), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        center, spread, count = summarize_values(group[value_col].to_numpy(dtype=float), statistic=statistic, positive_only=positive_only)
        row = {col: val for col, val in zip(group_cols, key)}
        row.update({"center": float(center), "spread": float(spread), "count": int(count), "statistic": statistic})
        row.update(extra_cols)
        rows.append(row)

    columns = list(group_cols) + ["center", "spread", "count", "statistic"] + list(extra_cols.keys())
    return pd.DataFrame(rows, columns=columns).sort_values(list(sort_cols)).reset_index(drop=True)


def filter_stats_table(
    table: pd.DataFrame,
    beta: float | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    center_positive: bool = True,
) -> pd.DataFrame:
    """
    Filter a statistics table before exponential fitting.

    :param table: Statistics table containing at least n and center columns.
    :param beta: Optional beta value to select.
    :param n_min: Optional lower n cutoff.
    :param n_max: Optional upper n cutoff.
    :param center_positive: If True, keep only positive centers.
    :return: Filtered table.
    """
    out = table.copy()

    if beta is not None:
        out = out[np.isclose(out["beta"].astype(float), float(beta))]

    if n_min is not None:
        out = out[out["n"].astype(int) >= int(n_min)]

    if n_max is not None:
        out = out[out["n"].astype(int) <= int(n_max)]

    mask = np.isfinite(out["n"].astype(float)) & np.isfinite(out["center"].astype(float))
    if center_positive:
        mask = mask & (out["center"].astype(float) > 0.0)

    return out.loc[mask].copy()


def _fit_exponential_from_filtered_stats(table: pd.DataFrame, sign: int) -> tuple[float, float]:
    n = table["n"].to_numpy(dtype=float)
    y = table["center"].to_numpy(dtype=float)

    if y.size < 2:
        raise ValueError("Need at least two positive points to fit.")

    slope, log_A = np.polyfit(n, np.log(y), deg=1)
    return float(np.exp(log_A)), float(sign * slope)


def interpolate_exponential_from_stats(
    table: pd.DataFrame,
    beta: float,
    n_min: int | None = None,
    n_max: int | None = None,
    sign: int = 1,
) -> tuple[float, float]:
    """
    Interpolate the exponential fit y(n,beta)=A(beta)*exp(sign*b(beta)*n) when the requested beta is missing.

    The method fits only the two closest bracketing beta values and linearly interpolates log(A) and b.
    If beta lies outside the available range, it extrapolates from the two nearest endpoint beta values.
    """
    available = filter_stats_table(table, beta=None, n_min=n_min, n_max=n_max, center_positive=True)
    beta_values = np.sort(available["beta"].astype(float).dropna().unique())

    if beta_values.size < 2:
        raise ValueError(f"Cannot interpolate fit for beta={beta}; fewer than two beta values are available.")

    j = int(np.searchsorted(beta_values, float(beta)))

    if j == 0:
        beta_0, beta_1 = beta_values[0], beta_values[1]
    elif j == beta_values.size:
        beta_0, beta_1 = beta_values[-2], beta_values[-1]
    else:
        beta_0, beta_1 = beta_values[j - 1], beta_values[j]

    table_0 = filter_stats_table(available, beta=beta_0, n_min=None, n_max=None, center_positive=True)
    table_1 = filter_stats_table(available, beta=beta_1, n_min=None, n_max=None, center_positive=True)

    A_0, b_0 = _fit_exponential_from_filtered_stats(table_0, sign)
    A_1, b_1 = _fit_exponential_from_filtered_stats(table_1, sign)

    weight = (float(beta) - beta_0) / (beta_1 - beta_0)
    log_A = (1.0 - weight) * np.log(A_0) + weight * np.log(A_1)
    b = (1.0 - weight) * b_0 + weight * b_1

    return float(np.exp(log_A)), float(b)


def fit_exponential_from_stats(
    table: pd.DataFrame,
    beta: float,
    n_min: int | None = None,
    n_max: int | None = None,
    sign: int = 1,
) -> tuple[float, float]:
    """
    Fit y(n) = A * exp(sign * b * n) from a statistics table.

    Use sign=1 for growing quantities such as query counts. Use sign=-1 for decaying quantities such as spectral gaps. The returned b is always the coefficient in the requested model y(n)=A*exp(sign*b*n). If the requested beta is missing, the method interpolates log(A) and b from the two closest available beta values.

    :param table: Statistics table with columns n and center.
    :param beta: Optional beta value to select before fitting.
    :param n_min: Optional lower n cutoff.
    :param n_max: Optional upper n cutoff.
    :param sign: Either 1 or -1.
    :return: Tuple (A, b).
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be either 1 or -1")

    filtered = filter_stats_table(table, beta=beta, n_min=n_min, n_max=n_max, center_positive=True)

    if filtered.empty:
        return interpolate_exponential_from_stats(table, beta=beta, n_min=n_min, n_max=n_max, sign=sign)

    return _fit_exponential_from_filtered_stats(filtered, sign)
