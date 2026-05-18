from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE


# PLOT_ORDER = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
# LEGEND_ORDER = ["local1", "local2", "local3", "uniform", "qemc", "layden"]

PLOT_ORDER = ["local1", "uniform", "qemc", "layden"]
LEGEND_ORDER = ["local1", "uniform", "qemc", "layden"]

PROPOSAL_LABELS = {
    "uniform": "Uniform",
    "local1": "Local spin-flip (single)",
    "local2": "Local spin-flip (double)",
    "local3": "Local spin-flip (triple)",
    "qemc": "Quantum enhanced (best hyperparameters)",
    "layden": "Quantum enhanced (randomized)",
}

PROPOSAL_COLORS = {
    "uniform": "#7A7A7A",
    "local1": "#56B4E9",
    "local2": "#0072B2",
    "local3": "#009E73",
    "qemc": "#E69F00",
    "layden": "#D55E00",
}


def _a_mask(series: pd.Series, a: int | float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.isinf(values) if np.isinf(a) else np.isclose(values, float(a))


def _epsilon_column(epsilon: float) -> str:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    k = int(round(-np.log10(epsilon)))
    if not np.isclose(epsilon, 10.0 ** (-k)):
        raise ValueError("epsilon must be one of 1e-2, ..., 1e-8.")
    return f"queries_eps_1e-{k}"


def _summarize(x: np.ndarray, statistic: str) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]

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
    raise ValueError("statistic must be one of ['mean+std', 'mean+std-tail', 'median+mad']")


def _fit_exp_from_stats(table: pd.DataFrame, beta: float, n_min: int | None, n_max: int | None) -> tuple[float, float]:
    table = table[np.isclose(table["beta"].astype(float), float(beta))].copy()
    if n_min is not None:
        table = table[table["n"].astype(int) >= int(n_min)]
    if n_max is not None:
        table = table[table["n"].astype(int) <= int(n_max)]

    n = table["n"].to_numpy(dtype=float)
    y = table["center"].to_numpy(dtype=float)
    mask = np.isfinite(n) & np.isfinite(y) & (y > 0.0)
    n, y = n[mask], y[mask]

    if y.size < 2:
        raise ValueError("Need at least two positive points to fit.")

    b, log_A = np.polyfit(n, np.log(y), deg=1)
    return float(np.exp(log_A)), float(b)


def _get_classical_query_stats(
    proposal: str,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    in_file: Path,
    statistic: str,
    only_ok: bool = True,
) -> pd.DataFrame:
    df = pd.read_pickle(in_file)
    eps_col = _epsilon_column(epsilon)
    mask = (df["proposal"] == proposal) & _a_mask(df["a"], a) & (df["q0_mode"] == q0_mode)
    if only_ok and "ok" in df.columns:
        mask = mask & df["ok"].astype(bool)
    df = df.loc[mask, ["n", "beta", eps_col]].copy()
    df[eps_col] = pd.to_numeric(df[eps_col], errors="coerce")

    rows = []
    for (n, beta), group in df.groupby(["n", "beta"], sort=True):
        center, spread, count = _summarize(group[eps_col].to_numpy(dtype=float), statistic)
        rows.append({"n": int(n), "beta": float(beta), "center": center, "spread": spread, "count": count, "statistic": statistic, "q0_mode": q0_mode, "kind": "classical"})

    return pd.DataFrame(rows).sort_values(["n", "beta"]).reset_index(drop=True)


def _get_quantum_query_stats(
    proposal: str,
    a: int | float,
    q0_mode: str,
    in_file: Path,
    statistic: str,
    only_ok: bool = True,
) -> pd.DataFrame:
    df = pd.read_pickle(in_file)
    mask = (df["proposal"] == proposal) & _a_mask(df["a"], a) & (df["q0_mode"] == q0_mode)
    if only_ok and "ok" in df.columns:
        mask = mask & df["ok"].astype(bool)
    df = df.loc[mask, ["n", "beta", "queries"]].copy()
    df["queries"] = pd.to_numeric(df["queries"], errors="coerce")

    rows = []
    for (n, beta), group in df.groupby(["n", "beta"], sort=True):
        center, spread, count = _summarize(group["queries"].to_numpy(dtype=float), statistic)
        rows.append({"n": int(n), "beta": float(beta), "center": center, "spread": spread, "count": count, "statistic": statistic, "q0_mode": q0_mode, "kind": "quantum"})

    return pd.DataFrame(rows).sort_values(["n", "beta"]).reset_index(drop=True)


def plot_classical_and_quantum_queries(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 8,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    only_ok: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot classical and quantum query counts versus n for fixed beta, acceptance a, and initialization mode.

    Both use the same y-axis (# queries, log scale) and the same proposal colors.
    Classical traces are solid lines with circle markers.
    Quantum traces are dashed lines with square markers.
    Fits use Q(n) = A * exp(b n).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    stats = {"classical": {}, "quantum": {}}
    all_n = []

    for proposal in PLOT_ORDER:
        classical_table = _get_classical_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=classical_query_file, statistic=statistic, only_ok=only_ok)
        classical_table = classical_table[np.isclose(classical_table["beta"].astype(float), float(beta)) & (classical_table["count"].astype(int) >= min_count) & np.isfinite(classical_table["n"].astype(float)) & np.isfinite(classical_table["center"].astype(float)) & (classical_table["center"].astype(float) > 0.0)].copy()
        if not classical_table.empty:
            classical_table["n"] = classical_table["n"].astype(float)
            classical_table["center"] = classical_table["center"].astype(float)
            classical_table["spread"] = classical_table["spread"].fillna(0.0).astype(float)
            classical_table = classical_table.sort_values("n")
            stats["classical"][proposal] = classical_table
            all_n.extend(classical_table["n"].tolist())

        quantum_table = _get_quantum_query_stats(proposal=proposal, a=a, q0_mode=q0_mode, in_file=quantum_query_file, statistic=statistic, only_ok=only_ok)
        quantum_table = quantum_table[np.isclose(quantum_table["beta"].astype(float), float(beta)) & (quantum_table["count"].astype(int) >= min_count) & np.isfinite(quantum_table["n"].astype(float)) & np.isfinite(quantum_table["center"].astype(float)) & (quantum_table["center"].astype(float) > 0.0)].copy()
        if not quantum_table.empty:
            quantum_table["n"] = quantum_table["n"].astype(float)
            quantum_table["center"] = quantum_table["center"].astype(float)
            quantum_table["spread"] = quantum_table["spread"].fillna(0.0).astype(float)
            quantum_table = quantum_table.sort_values("n")
            stats["quantum"][proposal] = quantum_table
            all_n.extend(quantum_table["n"].tolist())

    if not all_n:
        raise ValueError(f"No valid data found for beta={beta}, a={a}, q0_mode={q0_mode}, epsilon={epsilon}, statistic={statistic}.")

    n_min = float(np.min(all_n)) if n_plot_min is None else float(n_plot_min)
    n_max = float(np.max(all_n)) if n_plot_max is None else float(n_plot_max)
    n_grid = np.linspace(n_min, n_max, 300)

    legend_handles = {}
    legend_labels = {}

    def _positive_band(center: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = center - spread
        upper = center + spread
        positive = np.concatenate([center[center > 0.0], upper[upper > 0.0]])
        floor = np.min(positive) * 1e-2 if positive.size else np.finfo(float).tiny
        return np.maximum(lower, floor), np.maximum(upper, floor)

    styles = {
        "classical": {"linestyle": "-", "marker": "o", "alpha_fill": 0.18, "markersize": 34, "linewidth": 2.2, "name": "classical"},
        "quantum": {"linestyle": "--", "marker": "s", "alpha_fill": 0.10, "markersize": 30, "linewidth": 2.0, "name": "quantum"},
    }

    for proposal in PLOT_ORDER:
        color = PROPOSAL_COLORS[proposal]
        for kind in ("classical", "quantum"):
            if proposal not in stats[kind]:
                continue

            table = stats[kind][proposal]
            n_vals = table["n"].to_numpy(dtype=float)
            center = table["center"].to_numpy(dtype=float)
            spread = table["spread"].to_numpy(dtype=float)
            style = styles[kind]

            if show_spread:
                lower, upper = _positive_band(center, spread)
                ax.fill_between(n_vals, lower, upper, color=color, alpha=style["alpha_fill"], linewidth=0.0, zorder=1)

            ax.scatter(n_vals, center, s=style["markersize"], marker=style["marker"], color=color, edgecolors="none", alpha=0.95, zorder=4)

            try:
                A, b = _fit_exp_from_stats(table, beta=beta, n_min=n_fit_min, n_max=n_fit_max)
                y_fit = A * np.exp(b * n_grid)
                (line,) = ax.plot(n_grid, y_fit, color=color, linewidth=style["linewidth"], linestyle=style["linestyle"], alpha=0.95, zorder=3)
                legend_handles[(proposal, kind)] = line
                legend_labels[(proposal, kind)] = rf"{PROPOSAL_LABELS[proposal]} ({style['name']}): fit ${A:.3f} \times \exp({b:.3f} n)$"
            except ValueError:
                pass

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Number of queries")
    ax.set_title(rf"Classical and quantum queries, $\beta={beta}$, $a={a}$, q0={q0_mode}, $\epsilon={epsilon:g}$")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    handles = []
    labels = []
    for proposal in LEGEND_ORDER:
        if (proposal, "classical") in legend_handles:
            handles.append(legend_handles[(proposal, "classical")])
            labels.append(legend_labels[(proposal, "classical")])
        if (proposal, "quantum") in legend_handles:
            handles.append(legend_handles[(proposal, "quantum")])
            labels.append(legend_labels[(proposal, "quantum")])

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.6)

    return fig, ax
