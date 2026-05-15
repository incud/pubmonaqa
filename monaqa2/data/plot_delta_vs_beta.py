from pathlib import Path

from monaqa2.data.filename import SPECTRAL_GAP_FILE
from monaqa2.data.spectral_gap import get_spectral_gap_stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_spectral_gap_vs_beta(
    n: int,
    a: int | float,
    statistic: str = "mean+std",
    in_file: Path = SPECTRAL_GAP_FILE,
    show_spread: bool = True,
    min_count: int = 1,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot spectral-gap statistics versus beta for a fixed n and acceptance parameter a.

    :param n: System size.
    :param a: Acceptance parameter, one of np.inf, 1, 10.
    :param statistic: One of "mean+std", "mean+std-tail", "median+mad".
    :param in_file: Pickle file containing spectral-gap data.
    :param show_spread: If True, shade center +/- spread.
    :param min_count: Minimum number of instances required to plot a point.
    :param ax: Optional matplotlib axis.
    :return: The matplotlib figure and axis.
    """
    plot_order = ["local1", "local2", "local3", "uniform", "qemc", "layden"]
    legend_order = ["local1", "local2", "local3", "uniform", "qemc", "layden"]

    proposal_labels = {
        "layden": "Quantum enhanced (randomized)",
        "local1": "Local spin-flip (single)",
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

    stats = []

    for proposal in plot_order:
        proposal_stats = get_spectral_gap_stats(
            proposal=proposal,
            a=a,
            in_file=in_file,
            statistic=statistic,
        )

        proposal_stats = proposal_stats.copy()
        proposal_stats["proposal"] = proposal
        stats.append(proposal_stats)

    stats = pd.concat(stats, ignore_index=True)

    stats = stats.loc[
        (stats["n"] == n)
        & (stats["count"] >= min_count)
        & np.isfinite(stats["beta"].astype(float))
        & np.isfinite(stats["center"].astype(float))
        & (stats["beta"].astype(float) > 0.0)
        & (stats["center"].astype(float) > 0.0)
    ].copy()

    stats["beta"] = stats["beta"].astype(float)
    stats["center"] = stats["center"].astype(float)
    stats["spread"] = stats["spread"].astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
    else:
        fig = ax.figure

    line_handles = {}

    def _positive_band(center: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = center - spread
        upper = center + spread

        positive = np.concatenate([center[center > 0.0], upper[upper > 0.0]])
        floor = np.min(positive) * 1e-2 if positive.size else np.finfo(float).tiny

        return np.maximum(lower, floor), np.maximum(upper, floor)

    for proposal in plot_order:
        sub = stats.loc[stats["proposal"] == proposal].sort_values("beta")

        if sub.empty:
            continue

        beta = sub["beta"].to_numpy()
        center = sub["center"].to_numpy()
        spread = sub["spread"].fillna(0.0).to_numpy()
        color = proposal_colors[proposal]

        if show_spread:
            lower, upper = _positive_band(center, spread)

            ax.fill_between(
                beta,
                lower,
                upper,
                color=color,
                alpha=0.25,
                linewidth=0.0,
                zorder=1,
            )

        (line,) = ax.plot(
            beta,
            center,
            color=color,
            linewidth=2.5,
            label=proposal_labels[proposal],
            zorder=3,
        )

        line_handles[proposal] = line

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"Spectral gap $\delta$")
    ax.set_title(rf"Spectral gap over instances, $n={n}$, $a={a}$, statistic={statistic}")

    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    legend_handles = [line_handles[p] for p in legend_order if p in line_handles]
    legend_labels = [proposal_labels[p] for p in legend_order if p in line_handles]

    ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=1.8,
        handlelength=2.4,
    )

    offset = ScaledTranslation(
        0.5 / 2.54,
        1.5 / 2.54,
        fig.dpi_scale_trans,
    )

    axins = inset_axes(
        ax,
        width="50%",
        height="50%",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes + offset,
        borderpad=2.0,
    )

    for proposal in plot_order:
        sub = stats.loc[stats["proposal"] == proposal].sort_values("beta")

        if sub.empty:
            continue

        beta = sub["beta"].to_numpy()
        center = sub["center"].to_numpy()
        spread = sub["spread"].fillna(0.0).to_numpy()
        color = proposal_colors[proposal]

        if show_spread:
            lower, upper = _positive_band(center, spread)

            axins.fill_between(
                beta,
                lower,
                upper,
                color=color,
                alpha=0.25,
                linewidth=0.0,
                zorder=1,
            )

        axins.plot(
            beta,
            center,
            color=color,
            linewidth=1.8,
            zorder=3,
        )

    axins.set_xscale("log")
    axins.set_yscale("log")

    axins.set_xlim(0.1, 10.0)
    axins.set_ylim(1e-4, 9e-1)

    axins.tick_params(axis="both", which="major", labelsize=8)

    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.5",
        alpha=0.7,
    )

    ax.grid(True, which="major", alpha=0.35, zorder=0)
    ax.grid(False, which="minor")

    axins.grid(True, which="major", alpha=0.30, zorder=0)
    axins.grid(False, which="minor")

    return fig, ax
