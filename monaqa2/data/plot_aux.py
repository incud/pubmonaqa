from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from monaqa2.mcmc.model import IsingModel, RandomIsingModel


CM = 1.0 / 2.54
PT_TO_CM = 2.54 / 72.0
APS_FIGURE_WIDTH_CM = 8.6
APS_FIGURE_HEIGHT_CM = 5.6

AUX_COLORS = {
    "one_norm": "#4477AA",
    "op_norm": "#CC6677",
    "one_norm_expected": "#4477AA",
    "op_norm_expected": "#CC6677",
}

AUX_LABELS = {
    "one_norm": r"Empirical coefficient $1$-norm",
    "op_norm": "Empirical operator norm",
    "one_norm_expected": r"Expected coefficient $1$-norm: $n^{3/2}/\sqrt{\pi}$",
    "op_norm_expected": r"Expected operator norm: $n$",
}


def coefficient_one_norm(m: IsingModel) -> float:
    """Return the coefficient 1-norm of the rescaled Hamiltonian."""
    return float(np.abs(m.h_rescaled).sum() + np.abs(np.triu(m.J_rescaled, 1)).sum())


def operator_norm(m: IsingModel) -> float:
    """Return the operator norm of the rescaled diagonal Hamiltonian."""
    return float(np.abs(m.energies_rescaled).max())


one_norm_ub = coefficient_one_norm
op_norm_ub = operator_norm


def _n_grid(n_min: int, n_max: int) -> np.ndarray:
    if int(n_max) < int(n_min):
        raise ValueError("n_max must be greater than or equal to n_min.")
    return np.arange(int(n_min), int(n_max) + 1, dtype=int)


def _load_or_compute_bounds(
    ns: np.ndarray,
    n_models: int,
    one_norm_file: Path | None,
    op_norm_file: Path | None,
    seed_fn: Callable[[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    if one_norm_file is not None and Path(one_norm_file).exists():
        one_norm_values = np.load(one_norm_file)
    else:
        one_norm_values = None

    if op_norm_file is not None and Path(op_norm_file).exists():
        op_norm_values = np.load(op_norm_file)
    else:
        op_norm_values = None

    if one_norm_values is not None and op_norm_values is not None:
        return np.asarray(one_norm_values, dtype=float), np.asarray(op_norm_values, dtype=float)

    models = [[RandomIsingModel(n=int(n), seed=int(seed_fn(int(n), i))) for i in range(int(n_models))] for n in ns]
    if one_norm_values is None:
        one_norm_values = np.asarray([[coefficient_one_norm(m) for m in row] for row in models], dtype=float)
    if op_norm_values is None:
        op_norm_values = np.asarray([[operator_norm(m) for m in row] for row in models], dtype=float)

    return np.asarray(one_norm_values, dtype=float), np.asarray(op_norm_values, dtype=float)


def _finish_aux_axis(
    ax: plt.Axes,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    grid_color: str,
    grid_linewidth: float,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    ax.grid(False, which="major", axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_sk_upper_bounds(
    n_min: int = 5,
    n_max: int = 24,
    n_models: int = 30,
    one_norm_file: Path | str | None = Path("../data/sk_upper_bound_one_norm.npy"),
    op_norm_file: Path | str | None = Path("../data/sk_upper_bound_op_norm.npy"),
    output_file: Path | str | None = Path("plots/1_sk_norm_scalings.png"),
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    legend_placement: str = "right",
    legend_y_shift: float = -0.26,
    legend_x_shift: float = 1.04,
    xlabel_labelpad: float = 4.0,
    marker_size: float = 3.8,
    capsize: float = 2.4,
    line_width: float = 1.35,
    expected_line_width: float = 1.1,
    line_alpha: float = 0.94,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical and expected SK norm scalings versus system size."""
    if legend_placement not in {"top_left", "out", "right"}:
        raise ValueError("legend_placement must be either 'top_left', 'out', or 'right'.")

    ns = _n_grid(n_min, n_max)
    seed_fn = lambda n, i: int(n_models) * int(n) + int(i)
    one_norm_file = None if one_norm_file is None else Path(one_norm_file)
    op_norm_file = None if op_norm_file is None else Path(op_norm_file)
    U1, U2 = _load_or_compute_bounds(ns, n_models, one_norm_file, op_norm_file, seed_fn)

    if U1.shape[0] != len(ns) or U2.shape[0] != len(ns):
        raise ValueError("Loaded arrays must have one row per n value.")

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(APS_FIGURE_WIDTH_CM * CM, APS_FIGURE_HEIGHT_CM * CM))
        if legend_placement == "out":
            fig.subplots_adjust(left=0.18, right=0.985, top=0.985, bottom=0.36)
        elif legend_placement == "right":
            fig.subplots_adjust(left=0.18, right=0.64, top=0.985, bottom=0.17)
        else:
            fig.subplots_adjust(left=0.18, right=0.985, top=0.985, bottom=0.17)

    one_norm = ax.errorbar(ns, U1.mean(1), yerr=U1.std(1), fmt="o", markersize=marker_size, capsize=capsize, color=AUX_COLORS["one_norm"], linewidth=line_width, alpha=line_alpha, label=AUX_LABELS["one_norm"], zorder=3)
    op_norm = ax.errorbar(ns, U2.mean(1), yerr=U2.std(1), fmt="o", markersize=marker_size, capsize=capsize, color=AUX_COLORS["op_norm"], linewidth=line_width, alpha=line_alpha, label=AUX_LABELS["op_norm"], zorder=3)
    (one_expected,) = ax.plot(ns, ns**1.5 / np.sqrt(np.pi), linestyle="--", color=AUX_COLORS["one_norm_expected"], linewidth=expected_line_width, alpha=line_alpha, label=AUX_LABELS["one_norm_expected"], zorder=2)
    (op_expected,) = ax.plot(ns, ns, linestyle="--", color=AUX_COLORS["op_norm_expected"], linewidth=expected_line_width, alpha=line_alpha, label=AUX_LABELS["op_norm_expected"], zorder=2)

    _finish_aux_axis(ax, title, r"$n$", r"Norm", grid_color, grid_linewidth)
    ax.xaxis.labelpad = xlabel_labelpad

    if show_legend:
        handles = [one_norm, op_norm, one_expected, op_expected]
        labels = [AUX_LABELS["one_norm"], AUX_LABELS["op_norm"], AUX_LABELS["one_norm_expected"], AUX_LABELS["op_norm_expected"]]
        if legend_placement == "top_left":
            ax.legend(handles, labels, frameon=False, loc="upper left", ncol=1, borderaxespad=0.35, handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
        elif legend_placement == "right":
            ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(legend_x_shift, 1.0), ncol=1, borderaxespad=0.0, handlelength=2.2, columnspacing=1.0, labelspacing=0.35)
        else:
            ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.0, legend_y_shift), ncol=1, borderaxespad=0.0, handlelength=2.2, columnspacing=1.0, labelspacing=0.35)

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=int(dpi), bbox_inches="tight")

    return fig, ax
