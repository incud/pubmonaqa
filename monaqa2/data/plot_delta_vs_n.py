import matplotlib.pyplot as plt
from monaqa2.data.filename import SPECTRAL_GAP_FILE
from monaqa2.data.spectral_gap import get_spectral_gap_stats, get_spectral_gap_fit_by_n
import numpy as np
import pandas as pd
from pathlib import Path


def plot_spectral_gap_vs_n(
    beta: float,
    a: int | float,
    in_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    show_spread: bool = True,
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 8,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot spectral gap versus n for fixed beta and acceptance parameter a.

    Scatter points show the statistic center over instances.
    Transparent bands show center +/- spread.
    Solid lines show fits delta(n) = A * exp(-b n).
    """
    # plot_order = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
    # legend_order = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
    plot_order = ["local1", "uniform", "layden"]
    legend_order = ["local1", "uniform", "layden"]

    proposal_labels = {
        "layden": "Quantum enhanced",
        "local1": "Local spin-flip",
        "local2": "Local spin-flip (double)",
        "local3": "Local spin-flip (triple)",
        "uniform": "Uniform",
        "qemc": "Quantum enhanced (best hyperparameters)",
    }

    proposal_colors = {
        "uniform": "#7A7A7A",
        "local1": "#56B4E9",
        "local2": "#0072B2",
        "local3": "#009E73",
        "qemc": "#E69F00",
        "layden": "#D55E00",
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    stats_by_proposal = {}
    all_n = []

    for proposal in plot_order:
        table = get_spectral_gap_stats(proposal=proposal, a=a, in_file=in_file, statistic=statistic)

        table = table[
            np.isclose(table["beta"].astype(float), float(beta))
            & (table["count"].astype(int) >= min_count)
            & np.isfinite(table["n"].astype(float))
            & np.isfinite(table["center"].astype(float))
            & (table["center"].astype(float) > 0.0)
        ].copy()

        if table.empty:
            continue

        table["n"] = table["n"].astype(float)
        table["center"] = table["center"].astype(float)
        table["spread"] = table["spread"].fillna(0.0).astype(float)

        table = table.sort_values("n")
        stats_by_proposal[proposal] = table
        all_n.extend(table["n"].tolist())

    if not all_n:
        raise ValueError(f"No valid data found for beta={beta}, a={a}, statistic={statistic}.")

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

    for proposal in plot_order:
        if proposal not in stats_by_proposal:
            continue

        color = proposal_colors[proposal]
        table = stats_by_proposal[proposal]

        n_vals = table["n"].to_numpy(dtype=float)
        center = table["center"].to_numpy(dtype=float)
        spread = table["spread"].to_numpy(dtype=float)

        if show_spread:
            lower, upper = _positive_band(center, spread)
            ax.fill_between(n_vals, lower, upper, color=color, alpha=0.25, linewidth=0.0, zorder=1)

        ax.scatter(n_vals, center, s=36, color=color, edgecolors="none", alpha=0.95, zorder=3)

        try:
            A, b = get_spectral_gap_fit_by_n(proposal=proposal, a=a, fixed_beta=beta, in_file=in_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
        except ValueError:
            continue

        if b < 0.0:
            b = 0.0

        delta_fit = A * np.exp(-b * n_grid)
        (fit_line,) = ax.plot(n_grid, delta_fit, color=color, linewidth=2.0, linestyle="-", alpha=0.90, zorder=2)

        legend_handles[proposal] = fit_line
        legend_labels[proposal] = rf"{proposal_labels[proposal]}: fit ${A:.3f} \times \exp(-{b:.3f} n)$"

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Spectral gap $\delta$")
    ax.set_title(rf"Spectral gap over instances, $\beta={beta}$, $a={a}$, statistic={statistic}")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(False, which="minor")

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    handles = [legend_handles[p] for p in legend_order if p in legend_handles]
    labels = [legend_labels[p] for p in legend_order if p in legend_labels]

    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, borderaxespad=0.0, handlelength=2.4)

    return fig, ax
