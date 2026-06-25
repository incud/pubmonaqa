from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_n
from monaqa2.data.classical_query import get_classical_query_fit_by_n
from monaqa2.data.runtime import get_annealing_time_classical_walk_uniform, get_annealing_time_quantum_walk_uniform, get_annealing_time_quantum_walk_qemc


CM = 1.0 / 2.54
APS_COLUMN_WIDTH_CM = 8.5
APS_FIGURESTAR_WIDTH_CM = 17.8

RUNTIME_SERIES = ("uniform_classical", "uniform_quantum", "layden_quantum")

RUNTIME_LABELS = {
    "uniform_classical": "Best classical walk",
    "uniform_quantum": "Quantized classical walk",
    "layden_quantum": "This work",
}

RUNTIME_COLORS = {
    "uniform_classical": "#66CCEE",
    "uniform_quantum": "#4477AA",
    "layden_quantum": "#AA3377",
}


def _n_grid(n_plot_min: int | None, n_plot_max: int | None) -> np.ndarray:
    n_plot_min = 3 if n_plot_min is None else int(n_plot_min)
    n_plot_max = 120 if n_plot_max is None else int(n_plot_max)
    if n_plot_max < n_plot_min:
        raise ValueError("n_plot_max must be greater than or equal to n_plot_min.")
    return np.arange(n_plot_min, n_plot_max + 1, dtype=int)


def _time_constants() -> dict[str, float]:
    one_second = 1.0
    one_minute = 60.0
    one_hour = 60.0 * one_minute
    one_day = 24.0 * one_hour
    one_year = 365.25 * one_day
    return {
        "one_second": one_second,
        "one_minute": one_minute,
        "one_hour": one_hour,
        "one_day": one_day,
        "one_month": one_year / 12.0,
        "one_year": one_year,
        "ten_years": 10.0 * one_year,
        "hundred_years": 100.0 * one_year,
        "thousand_years": 1000.0 * one_year,
    }


def _time_ticks() -> tuple[list[float], list[str]]:
    t = _time_constants()
    # long description -> return [t["one_second"], t["one_minute"], t["one_hour"], t["one_day"], t["one_month"], t["one_year"], t["ten_years"], t["hundred_years"], t["thousand_years"]], ["one second", "one minute", "one hour", "one day", "one month", "one year", "10 years", "100 years", "1000 years"]
    # short description -> return [t["one_second"], t["one_minute"], t["one_hour"], t["one_day"], t["one_month"], t["one_year"], t["ten_years"], t["hundred_years"], t["thousand_years"]], ["1 s", "1 min", "1 h", "1 d", "1 mo", "1 yr", "10 yr", "100 yr", "1000 yr"]
    # return [t["one_second"], t["one_day"],  t["one_year"], t["thousand_years"]], ["one second", "one day", "one year", "1000 years"]
    return [t["one_second"], t["one_day"],  t["one_year"], t["thousand_years"]], ["1 s", "1 d", "1 yr", "1000 yr"]


def _classical_query_fit_value(proposal: str, n: int, beta_t: float, epsilon: float, classical_query_file: Path, statistic: str, n_fit_min: int | None, n_fit_max: int | None) -> float:
    A_q, b_q = get_classical_query_fit_by_n(proposal=proposal, a=np.inf, q0_mode="bhattacharyya", epsilon=epsilon, beta=beta_t, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
    return float(A_q * np.exp(b_q * int(n)))


def _spectral_gap_fit_value(proposal: str, n: int, beta_t: float, spectral_gap_file: Path, statistic: str, n_fit_min: int | None, n_fit_max: int | None) -> float:
    A_g, b_g = get_spectral_gap_fit_by_n(proposal=proposal, a=np.inf, fixed_beta=beta_t, in_file=spectral_gap_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max)
    b_g = max(float(b_g), 0.0)
    return float(A_g * np.exp(-b_g * int(n)))


def _positive_floor(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    floor = np.min(positive) * 1e-4 if positive.size else np.finfo(float).tiny
    return np.maximum(values, floor)


def _compute_runtime_curves(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    n_vals: np.ndarray,
    classical_device: str,
    physical_error_rate_min: float,
    physical_error_rate_max: float,
    physical_operation_time_min: float,
    physical_operation_time_max: float,
    physical_measurement_time_min: float,
    physical_measurement_time_max: float,
    num_trotter_steps: int,
    classical_query_file: Path,
    spectral_gap_file: Path,
    statistic: str,
    n_fit_min: int | None,
    n_fit_max: int | None,
    debug: bool,
) -> dict[str, np.ndarray]:
    curves = {key: [] for key in RUNTIME_SERIES}
    eps_SF = float(epsilon) / 4.0

    for n in n_vals:
        schedule = [float(beta_t) for beta_t in annealing_schedule_generator(int(n), float(beta))]
        uniform_queries = []
        uniform_gaps = []
        layden_gaps = []

        if debug:
            print(f"\n[debug] n={int(n)}, beta={float(beta):.12g}, epsilon={float(epsilon):.12g}, schedule={schedule}")

        for beta_t in schedule:
            uniform_query = _classical_query_fit_value("uniform", int(n), beta_t, epsilon, classical_query_file, statistic, n_fit_min, n_fit_max)
            uniform_gap = _spectral_gap_fit_value("uniform", int(n), beta_t, spectral_gap_file, statistic, n_fit_min, n_fit_max)
            layden_gap = _spectral_gap_fit_value("layden", int(n), beta_t, spectral_gap_file, statistic, n_fit_min, n_fit_max)
            uniform_queries.append(uniform_query)
            uniform_gaps.append(uniform_gap)
            layden_gaps.append(layden_gap)

            if debug:
                print(f"  beta_t={beta_t:.12g}, uniform_query={uniform_query:.12g}, uniform_gap={uniform_gap:.12g}, layden_gap={layden_gap:.12g}")

        classical_uniform = get_annealing_time_classical_walk_uniform(int(n), uniform_queries, device=classical_device)
        quantum_uniform_min = get_annealing_time_quantum_walk_uniform(int(n), epsilon, schedule, uniform_gaps, physical_operation_time_min, physical_measurement_time_min, physical_error_rate_min)
        quantum_uniform_max = get_annealing_time_quantum_walk_uniform(int(n), epsilon, schedule, uniform_gaps, physical_operation_time_max, physical_measurement_time_max, physical_error_rate_max)
        quantum_layden_min = get_annealing_time_quantum_walk_qemc(int(n), epsilon, schedule, layden_gaps, physical_operation_time_min, physical_measurement_time_min, physical_error_rate_min, num_trotter_steps=num_trotter_steps)
        quantum_layden_max = get_annealing_time_quantum_walk_qemc(int(n), epsilon, schedule, layden_gaps, physical_operation_time_max, physical_measurement_time_max, physical_error_rate_max, num_trotter_steps=num_trotter_steps)

        curves["uniform_classical"].append(float(classical_uniform))
        curves["uniform_quantum"].append((float(quantum_uniform_min), float(quantum_uniform_max)))
        curves["layden_quantum"].append((float(quantum_layden_min), float(quantum_layden_max)))

    return {
        "uniform_classical": _positive_floor(np.asarray(curves["uniform_classical"], dtype=float)),
        "uniform_quantum_min": _positive_floor(np.asarray([x[0] for x in curves["uniform_quantum"]], dtype=float)),
        "uniform_quantum_max": _positive_floor(np.asarray([x[1] for x in curves["uniform_quantum"]], dtype=float)),
        "layden_quantum_min": _positive_floor(np.asarray([x[0] for x in curves["layden_quantum"]], dtype=float)),
        "layden_quantum_max": _positive_floor(np.asarray([x[1] for x in curves["layden_quantum"]], dtype=float)),
    }


def _add_band(ax: plt.Axes, n_vals: np.ndarray, lower: np.ndarray, upper: np.ndarray, color: str, label: str, line_width: float, band_alpha: float, line_alpha: float, zorder: int):
    lower_raw = np.asarray(lower, dtype=float)
    upper_raw = np.asarray(upper, dtype=float)
    lower = _positive_floor(np.minimum(lower_raw, upper_raw))
    upper = _positive_floor(np.maximum(lower_raw, upper_raw))
    ax.fill_between(n_vals, lower, upper, color=color, alpha=band_alpha, edgecolor="none", linewidth=0.0, zorder=zorder)
    (optimistic_line,) = ax.plot(n_vals, lower, color=color, linewidth=0.55 * line_width, alpha=line_alpha, linestyle="-", label=label, zorder=zorder + 1)
    ax.plot(n_vals, upper, color=color, linewidth=0.55 * line_width, alpha=line_alpha, linestyle="-", label=None, zorder=zorder + 1)
    return optimistic_line


def _plot_runtime_curves(ax: plt.Axes, n_vals: np.ndarray, curves: dict[str, np.ndarray], line_width: float, line_alpha: float, band_alpha: float) -> tuple[list[plt.Line2D], list[str]]:
    handles = []
    labels = []
    (classical_line,) = ax.plot(n_vals, curves["uniform_classical"], color=RUNTIME_COLORS["uniform_classical"], linewidth=line_width, alpha=line_alpha, linestyle="-", label=RUNTIME_LABELS["uniform_classical"], zorder=4)
    handles.append(classical_line)
    labels.append(RUNTIME_LABELS["uniform_classical"])
    uniform_handle = _add_band(ax, n_vals, curves["uniform_quantum_min"], curves["uniform_quantum_max"], RUNTIME_COLORS["uniform_quantum"], RUNTIME_LABELS["uniform_quantum"], line_width, band_alpha, line_alpha, zorder=2)
    handles.append(uniform_handle)
    labels.append(RUNTIME_LABELS["uniform_quantum"])
    layden_handle = _add_band(ax, n_vals, curves["layden_quantum_min"], curves["layden_quantum_max"], RUNTIME_COLORS["layden_quantum"], RUNTIME_LABELS["layden_quantum"], line_width, band_alpha, line_alpha, zorder=3)
    handles.append(layden_handle)
    labels.append(RUNTIME_LABELS["layden_quantum"])
    return handles, labels


def _first_log_intercept_x(n_vals: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> float | None:
    x = np.asarray(n_vals, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    mask = np.isfinite(x) & np.isfinite(baseline) & np.isfinite(candidate) & (baseline > 0.0) & (candidate > 0.0)
    if np.count_nonzero(mask) < 2:
        return None
    x = x[mask]
    log_ratio = np.log(candidate[mask] / baseline[mask])
    if log_ratio[0] <= 0.0:
        return float(x[0])
    for i in range(1, len(x)):
        if log_ratio[i - 1] > 0.0 and log_ratio[i] <= 0.0:
            if np.isclose(log_ratio[i - 1], log_ratio[i]):
                return float(x[i])
            return float(x[i - 1] + (0.0 - log_ratio[i - 1]) * (x[i] - x[i - 1]) / (log_ratio[i] - log_ratio[i - 1]))
    return None


def _integer_intercept_ticks(n_vals: np.ndarray, curves: dict[str, np.ndarray], enabled: bool) -> list[int]:
    if not enabled:
        return []
    baseline = curves["uniform_classical"]
    ticks = []
    for key in ["layden_quantum_min", "layden_quantum_max"]:
        x = _first_log_intercept_x(n_vals, baseline, curves[key])
        if x is not None:
            ticks.append(int(round(x)))
    return sorted(set(ticks))


def _draw_intercept_lines(ax: plt.Axes, intercept_ticks: list[int]) -> None:
    for x in intercept_ticks:
        ax.axvline(float(x), color="0.48", linewidth=0.65, linestyle="-", alpha=0.90, zorder=0)


def _runtime_xticks(n_vals: np.ndarray, intercept_ticks: list[int]) -> list[int]:
    n_min = int(np.ceil(float(n_vals[0]) / 10.0) * 10)
    n_max = int(np.floor(float(n_vals[-1]) / 10.0) * 10)
    ticks = list(range(n_min, n_max + 1, 10))
    if intercept_ticks:
        ticks = [tick for tick in ticks if tick not in [30, 40]]
        ticks.extend(intercept_ticks)
    return sorted(set(ticks))




def _finish_runtime_axis(
    ax: plt.Axes,
    n_vals: np.ndarray,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    show_regime_separator: bool,
    regime_separator_n: float,
    show_one_year_line: bool,
    grid_color: str,
    grid_linewidth: float,
    x_right_padding: float,
    y_top_padding: float,
    intercept_ticks: list[int],
) -> None:
    t = _time_constants()
    ticks, labels = _time_ticks()
    xticks = _runtime_xticks(n_vals, intercept_ticks)

    ax.set_yscale("log")
    ax.set_xlim(float(n_vals[0]), float(n_vals[-1]) + max(0.0, float(x_right_padding)))
    ax.set_ylim(t["one_second"], t["thousand_years"] * (1.0 + max(0.0, float(y_top_padding))))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xticks(xticks)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    ax.grid(False, which="major", axis="x")

    if len(intercept_ticks) == 0:
        for x in xticks:
            if show_regime_separator and float(x) <= float(10.1):
                continue
            ax.axvline(float(x), color="0.90", linewidth=0.40, linestyle="-", alpha=1.0, zorder=0)

    if show_regime_separator and float(n_vals[0]) <= float(regime_separator_n) <= float(n_vals[-1]):
        ax.axvline(float(regime_separator_n), color="0.62", linewidth=0.55, linestyle=(0, (2.0, 2.0)), alpha=0.80, zorder=1)

    if show_one_year_line:
        ax.axhline(t["one_year"], color="black", linewidth=0.75, linestyle="-", alpha=1.0, zorder=5)

    _draw_intercept_lines(ax, intercept_ticks)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _add_runtime_legend(fig: plt.Figure, ax: plt.Axes, handles: list[plt.Line2D], labels: list[str], legend_placement: str, legend_y_shift: float) -> None:
    if legend_placement == "top_left":
        ax.legend(handles, labels, frameon=False, loc="upper left", ncol=1, borderaxespad=0.35, handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    elif legend_placement == "bottom_right":
        ax.legend(handles, labels, frameon=False, loc="lower right", ncol=1, borderaxespad=0.35, handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    elif legend_placement == "out":
        ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.0, legend_y_shift), ncol=1, borderaxespad=0.0, handlelength=2.2, columnspacing=1.0, labelspacing=0.35)
    else:
        raise ValueError("legend_placement must be either 'top_left', 'bottom_right', or 'out'.")


def plot_annealing_classical_and_quantum_runtime_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_device: str = "fpga",
    physical_error_rate_min: float = 1e-4,
    physical_error_rate_max: float = 1e-4,
    physical_operation_time_min: float = 200e-9,
    physical_operation_time_max: float = 20_000e-9,
    physical_measurement_time_min: float = 20e-9,
    physical_measurement_time_max: float = 2_000e-9,
    num_trotter_steps: int = 50,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    legend_placement: str = "top_left",
    legend_y_shift: float = -0.22,
    xlabel_labelpad: float = 4,
    line_width: float = 2.0,
    line_alpha: float = 0.94,
    band_alpha: float = 0.18,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    x_right_padding: float = 0.0,
    show_regime_separator: bool = True,
    regime_separator_n: float = 10.0,
    show_one_year_line: bool = False,
    y_top_padding: float = 0.0,
    optimistic_pessimistic_intercept: bool = False,
    debug: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot APS single-column annealing runtime estimates versus n."""
    n_vals = _n_grid(n_plot_min, n_plot_max)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(APS_COLUMN_WIDTH_CM * CM, 5.6 * CM))
        if legend_placement == "out":
            fig.subplots_adjust(left=0.20, right=0.985, top=0.985, bottom=0.43)
        else:
            fig.subplots_adjust(left=0.20, right=0.985, top=0.985, bottom=0.17)
    curves = _compute_runtime_curves(beta, epsilon, annealing_schedule_generator, n_vals, classical_device, physical_error_rate_min, physical_error_rate_max, physical_operation_time_min, physical_operation_time_max, physical_measurement_time_min, physical_measurement_time_max, num_trotter_steps, classical_query_file, spectral_gap_file, statistic, n_fit_min, n_fit_max, debug)
    handles, labels = _plot_runtime_curves(ax, n_vals, curves, line_width, line_alpha, band_alpha)
    intercept_ticks = _integer_intercept_ticks(n_vals, curves, optimistic_pessimistic_intercept)
    _finish_runtime_axis(ax, n_vals, title, r"$n$", r"Runtime", show_regime_separator, regime_separator_n, show_one_year_line, grid_color, grid_linewidth, x_right_padding, y_top_padding, intercept_ticks)
    ax.xaxis.labelpad = xlabel_labelpad
    if show_legend:
        _add_runtime_legend(fig, ax, handles, labels, legend_placement, legend_y_shift)
    return fig, ax


def _as_sequence(value: float | Sequence[float], length: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a float or a sequence of floats.")
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError:
        values = [float(value)] * length
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}.")
    return [float(v) for v in values]


def plot_annealing_classical_and_quantum_runtime_vs_n_three(
    betas: Sequence[float],
    epsilon: float | Sequence[float],
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_device: str = "fpga",
    physical_error_rate_min: float = 1e-4,
    physical_error_rate_max: float = 1e-4,
    physical_operation_time_min: float = 200e-9,
    physical_operation_time_max: float = 20_000e-9,
    physical_measurement_time_min: float = 20e-9,
    physical_measurement_time_max: float = 2_000e-9,
    num_trotter_steps: int = 50,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = SPECTRAL_GAP_FILE,
    statistic: str = "mean+std",
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_plot_min: int | None = None,
    n_plot_max: int | None = None,
    titles: Sequence[str | None] | None = None,
    show_legend: bool = True,
    legend_placement: str = "out",
    legend_y_shift: float = -0.22,
    xlabel_labelpad: float = 4,
    line_width: float = 2.0,
    line_alpha: float = 0.94,
    band_alpha: float = 0.18,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    x_right_padding: float = 0.0,
    show_regime_separator: bool = True,
    regime_separator_n: float = 10.0,
    show_one_year_line: bool = False,
    y_top_padding: float = 0.0,
    optimistic_pessimistic_intercept: bool = False,
    debug: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot three APS figure*-style annealing runtime panels with a shared y-axis."""
    betas = [float(beta) for beta in betas]
    if len(betas) != 3:
        raise ValueError("betas must contain exactly three values.")
    epsilons = _as_sequence(epsilon, len(betas), "epsilon")
    titles = [None] * len(betas) if titles is None else list(titles)
    if len(titles) != len(betas):
        raise ValueError("titles must have the same length as betas.")

    fig, axes = plt.subplots(1, 3, figsize=(APS_FIGURESTAR_WIDTH_CM * CM, 5.6 * CM), sharey=True)
    if legend_placement == "out":
        fig.subplots_adjust(left=0.13, right=0.995, top=0.985, bottom=0.34, wspace=0.08)
    else:
        fig.subplots_adjust(left=0.13, right=0.995, top=0.985, bottom=0.17, wspace=0.08)

    first_handles = None
    first_labels = None
    for i, (ax, beta, eps, title) in enumerate(zip(axes, betas, epsilons, titles)):
        _, _ = plot_annealing_classical_and_quantum_runtime_vs_n(beta=beta, epsilon=eps, annealing_schedule_generator=annealing_schedule_generator, classical_device=classical_device, physical_error_rate_min=physical_error_rate_min, physical_error_rate_max=physical_error_rate_max, physical_operation_time_min=physical_operation_time_min, physical_operation_time_max=physical_operation_time_max, physical_measurement_time_min=physical_measurement_time_min, physical_measurement_time_max=physical_measurement_time_max, num_trotter_steps=num_trotter_steps, classical_query_file=classical_query_file, spectral_gap_file=spectral_gap_file, statistic=statistic, n_fit_min=n_fit_min, n_fit_max=n_fit_max, n_plot_min=n_plot_min, n_plot_max=n_plot_max, fig=fig, ax=ax, title=title, show_legend=False, legend_placement="top_left", legend_y_shift=legend_y_shift, xlabel_labelpad=xlabel_labelpad, line_width=line_width, line_alpha=line_alpha, band_alpha=band_alpha, grid_color=grid_color, grid_linewidth=grid_linewidth, x_right_padding=x_right_padding, show_regime_separator=show_regime_separator, regime_separator_n=regime_separator_n, show_one_year_line=show_one_year_line, y_top_padding=y_top_padding, optimistic_pessimistic_intercept=optimistic_pessimistic_intercept, debug=debug)
        if i > 0:
            ax.set_ylabel(None)
        first_handles, first_labels = ax.get_legend_handles_labels() if first_handles is None else (first_handles, first_labels)

    if show_legend and first_handles is not None and first_labels is not None:
        if legend_placement == "out":
            fig.legend(first_handles, first_labels, frameon=False, loc="lower left", bbox_to_anchor=(0.13, 0.02), bbox_transform=fig.transFigure, ncol=1, borderaxespad=0.0, handlelength=2.2, labelspacing=0.35)
        elif legend_placement == "top_left":
            axes[0].legend(first_handles, first_labels, frameon=False, loc="upper left", ncol=1, borderaxespad=0.35, handlelength=2.0, labelspacing=0.35)
        elif legend_placement == "bottom_right":
            axes[-1].legend(first_handles, first_labels, frameon=False, loc="lower right", ncol=1, borderaxespad=0.35, handlelength=2.0, labelspacing=0.35)
        else:
            raise ValueError("legend_placement must be either 'top_left', 'bottom_right', or 'out'.")

    return fig, axes
