from typing import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np
from pathlib import Path

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_n
from monaqa2.data.classical_query import get_classical_query_fit_by_n
from monaqa2.data.runtime import get_annealing_time_classical_walk_uniform, get_annealing_time_quantum_walk_uniform, get_annealing_time_quantum_walk_qemc, get_time_direct_enumeration


CM = 1.0 / 2.54
APS_COLUMN_WIDTH_CM = 8.5
APS_FIGURESTAR_WIDTH_CM = 17.8

RUNTIME_SERIES = ("uniform_classical", "uniform_quantum", "layden_quantum")

RUNTIME_LABELS = {
    "uniform_classical": "Best classical walk",
    "uniform_quantum": "Quantized classical walk",
    "layden_quantum": "This work",
    "direct_enumeration": "Direct enumeration",
}


RUNTIME_COLORS = {
    # Neutral family: near-black and gray
    "local1_classical": "#222222",
    "local1_quantum": "#999999",
    # Cold family: navy blue and cyan
    "uniform_classical": "#004488",
    "uniform_quantum": "#66CCEE",
    # Warm family: magenta and orange
    "layden_classical": "#EE7733",
    "layden_quantum": "#AA3377",
    # Extra
    "direct_enumeration": "#D62728",
}

CLASSICAL_MOVE_COLORS = {
    "local": "#4477AA",
    "uniform": "#CC6677",
}

CLASSICAL_DEVICE_STYLES = {
    "fpga": {"linestyle": "-", "marker": None},
    "cpu": {"linestyle": "--", "marker": None},
    "gpu": {"linestyle": "None", "marker": "o"},
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


def _add_direct_enumeration_line(ax: plt.Axes, n_vals: np.ndarray, line_alpha: float) -> plt.Line2D:
    values = _positive_floor(np.asarray([get_time_direct_enumeration(int(n)) for n in n_vals], dtype=float))
    (line,) = ax.plot(n_vals, values, color=RUNTIME_COLORS["direct_enumeration"], linewidth=0.85, alpha=line_alpha, linestyle="-", label=RUNTIME_LABELS["direct_enumeration"], zorder=6)
    return line


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
    xticks = _runtime_xticks(n_vals, intercept_ticks)

    ax.set_yscale("log")
    ax.set_xlim(float(n_vals[0]), float(n_vals[-1]) + max(0.0, float(x_right_padding)))
    time_ticks, time_tick_labels = _time_ticks()
    ax.set_ylim(t["one_second"], t["thousand_years"] * (1.0 + max(0.0, float(y_top_padding))))
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_tick_labels)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
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
    show_direct_enumeration_line: bool = False,
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
    if show_direct_enumeration_line:
        direct_line = _add_direct_enumeration_line(ax, n_vals, line_alpha)
        handles.append(direct_line)
        labels.append(RUNTIME_LABELS["direct_enumeration"])
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
    show_direct_enumeration_line: bool = False,
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
        _, _ = plot_annealing_classical_and_quantum_runtime_vs_n(beta=beta, epsilon=eps, annealing_schedule_generator=annealing_schedule_generator, classical_device=classical_device, physical_error_rate_min=physical_error_rate_min, physical_error_rate_max=physical_error_rate_max, physical_operation_time_min=physical_operation_time_min, physical_operation_time_max=physical_operation_time_max, physical_measurement_time_min=physical_measurement_time_min, physical_measurement_time_max=physical_measurement_time_max, num_trotter_steps=num_trotter_steps, classical_query_file=classical_query_file, spectral_gap_file=spectral_gap_file, statistic=statistic, n_fit_min=n_fit_min, n_fit_max=n_fit_max, n_plot_min=n_plot_min, n_plot_max=n_plot_max, fig=fig, ax=ax, title=title, show_legend=False, legend_placement="top_left", legend_y_shift=legend_y_shift, xlabel_labelpad=xlabel_labelpad, line_width=line_width, line_alpha=line_alpha, band_alpha=band_alpha, grid_color=grid_color, grid_linewidth=grid_linewidth, x_right_padding=x_right_padding, show_regime_separator=show_regime_separator, regime_separator_n=regime_separator_n, show_one_year_line=show_one_year_line, show_direct_enumeration_line=show_direct_enumeration_line, y_top_padding=y_top_padding, optimistic_pessimistic_intercept=optimistic_pessimistic_intercept, debug=debug)
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



def _classical_transition_time(
    proposal: str,
    device: str,
    n_vals: np.ndarray,
) -> np.ndarray:
    """Return the fitted time per attempted Markov-chain transition in seconds."""
    n_vals = np.asarray(n_vals, dtype=float)
    if np.any(n_vals <= 0.0):
        raise ValueError("All system sizes must be positive.")

    if proposal == "local":
        if device == "cpu":
            values = 5.959e-9 + 1.429e-10 * n_vals
        elif device == "gpu":
            values = 7.837e-7 + 1.459e-9 * n_vals
        elif device == "fpga":
            values = (0.2679 + 0.0018 * np.log2(n_vals)) * 1e-6
        else:
            raise ValueError("device must be 'fpga', 'cpu', or 'gpu'.")
    elif proposal == "uniform":
        if device == "cpu":
            values = 1.173e-8 * n_vals + 6.964e-11 * n_vals**2
        elif device == "gpu":
            values = 2.215e-10 * n_vals**2
        elif device == "fpga":
            values = (0.2541 + 0.0042 * np.log2(n_vals)) * 1e-6
        else:
            raise ValueError("device must be 'fpga', 'cpu', or 'gpu'.")
    else:
        raise ValueError("proposal must be 'local' or 'uniform'.")

    return _positive_floor(np.asarray(values, dtype=float))


def plot_classical_transition_time_vs_n(
    n_plot_min: int = 3,
    n_plot_max: int = 120,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    legend_placement: str = "top_left",
    legend_y_shift: float = -0.30,
    line_width: float = 1.8,
    line_alpha: float = 0.94,
    marker_size: float = 3.2,
    gpu_markevery: int = 4,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    x_right_padding: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare local and uniform transition times on FPGA, CPU, and GPU.

    Colors encode the proposal rule, while styles encode the device:
    solid lines for FPGA, dashed lines for CPU, and point markers for GPU.
    """
    n_vals = _n_grid(n_plot_min, n_plot_max)
    if gpu_markevery < 1:
        raise ValueError("gpu_markevery must be at least 1.")

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(APS_COLUMN_WIDTH_CM * CM, 5.6 * CM))
        if legend_placement == "out":
            fig.subplots_adjust(left=0.22, right=0.985, top=0.985, bottom=0.42)
        else:
            fig.subplots_adjust(left=0.22, right=0.985, top=0.985, bottom=0.18)

    for proposal in ("local", "uniform"):
        color = CLASSICAL_MOVE_COLORS[proposal]
        for device in ("fpga", "cpu", "gpu"):
            style = CLASSICAL_DEVICE_STYLES[device]
            values = _classical_transition_time(proposal, device, n_vals)
            kwargs = {
                "color": color,
                "alpha": line_alpha,
                "zorder": 4 if device == "gpu" else 3,
            }
            if device == "gpu":
                kwargs.update(
                    linestyle="None",
                    marker=style["marker"],
                    markersize=marker_size,
                    markeredgewidth=0.0,
                    markevery=gpu_markevery,
                )
            else:
                kwargs.update(
                    linestyle=style["linestyle"],
                    linewidth=line_width,
                )
            ax.plot(n_vals, values, **kwargs)

    ax.set_yscale("log")
    ax.set_xlim(
        float(n_vals[0]),
        float(n_vals[-1]) + max(0.0, float(x_right_padding)),
    )
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Time per transition (s)")
    if title is not None:
        ax.set_title(title)

    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both",
        color=grid_color,
        linewidth=grid_linewidth,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        color=grid_color,
        linewidth=0.5 * grid_linewidth,
        alpha=0.55,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_legend:
        move_handles = [
            Line2D(
                [0],
                [0],
                color=CLASSICAL_MOVE_COLORS["local"],
                linewidth=line_width,
                label="local",
            ),
            Line2D(
                [0],
                [0],
                color=CLASSICAL_MOVE_COLORS["uniform"],
                linewidth=line_width,
                label="uniform",
            ),
        ]
        device_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=line_width,
                linestyle="-",
                label="FPGA",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=line_width,
                linestyle="--",
                label="CPU",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="None",
                marker="o",
                markersize=marker_size,
                markeredgewidth=0.0,
                label="GPU",
            ),
        ]
        handles = move_handles + device_handles

        if legend_placement == "top_left":
            ax.legend(
                handles=handles,
                frameon=False,
                loc="upper left",
                ncol=1,
                borderaxespad=0.35,
                handlelength=2.0,
                labelspacing=0.35,
            )
        elif legend_placement == "bottom_right":
            ax.legend(
                handles=handles,
                frameon=False,
                loc="lower right",
                ncol=1,
                borderaxespad=0.35,
                handlelength=2.0,
                labelspacing=0.35,
            )
        elif legend_placement == "out":
            ax.legend(
                handles=handles,
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(0.0, legend_y_shift),
                ncol=2,
                borderaxespad=0.0,
                handlelength=2.2,
                columnspacing=1.2,
                labelspacing=0.35,
            )
        else:
            raise ValueError(
                "legend_placement must be 'top_left', 'bottom_right', or 'out'."
            )

    return fig, ax



CLASSICAL_ANNEALING_RUNTIME_KEYS = (
    "cpu_local",
    "cpu_uniform",
    "gpu_local",
    "gpu_uniform",
    "fpga_local",
    "fpga_uniform",
)


def _compute_classical_annealing_runtime_curves(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    n_vals: np.ndarray,
    classical_query_file: Path,
    statistic: str,
    n_fit_min: int | None,
    n_fit_max: int | None,
    debug: bool,
) -> dict[str, np.ndarray]:
    """Compute local and uniform annealing runtimes for CPU, GPU, and FPGA."""
    curves: dict[str, list[float]] = {
        key: [] for key in CLASSICAL_ANNEALING_RUNTIME_KEYS
    }

    for n in n_vals:
        schedule = [
            float(beta_t)
            for beta_t in annealing_schedule_generator(int(n), float(beta))
        ]

        local_queries = [
            _classical_query_fit_value(
                proposal="local1",
                n=int(n),
                beta_t=beta_t,
                epsilon=epsilon,
                classical_query_file=classical_query_file,
                statistic=statistic,
                n_fit_min=n_fit_min,
                n_fit_max=n_fit_max,
            )
            for beta_t in schedule
        ]
        uniform_queries = [
            _classical_query_fit_value(
                proposal="uniform",
                n=int(n),
                beta_t=beta_t,
                epsilon=epsilon,
                classical_query_file=classical_query_file,
                statistic=statistic,
                n_fit_min=n_fit_min,
                n_fit_max=n_fit_max,
            )
            for beta_t in schedule
        ]

        total_queries = {
            "local": float(np.sum(local_queries)),
            "uniform": float(np.sum(uniform_queries)),
        }

        if debug:
            print(
                f"\n[debug] n={int(n)}, beta={float(beta):.12g}, "
                f"epsilon={float(epsilon):.12g}, schedule={schedule}"
            )
            print(f"  local_queries={local_queries}")
            print(f"  uniform_queries={uniform_queries}")

        for device in ("cpu", "gpu", "fpga"):
            for proposal in ("local", "uniform"):
                transition_time = float(
                    _classical_transition_time(
                        proposal=proposal,
                        device=device,
                        n_vals=np.asarray([n], dtype=float),
                    )[0]
                )
                runtime = total_queries[proposal] * transition_time
                curves[f"{device}_{proposal}"].append(runtime)

                if debug:
                    print(
                        f"  {device}_{proposal}: "
                        f"queries={total_queries[proposal]:.12g}, "
                        f"transition_time={transition_time:.12g}, "
                        f"runtime={runtime:.12g}"
                    )

    return {
        key: _positive_floor(np.asarray(values, dtype=float))
        for key, values in curves.items()
    }


def _plot_classical_annealing_runtime_curves(
    ax: plt.Axes,
    n_vals: np.ndarray,
    curves: dict[str, np.ndarray],
    line_width: float,
    line_alpha: float,
    marker_size: float,
    gpu_markevery: int,
) -> dict[str, plt.Line2D]:
    """Plot the six classical annealing-runtime curves."""
    handles: dict[str, plt.Line2D] = {}

    for device in ("cpu", "gpu", "fpga"):
        for proposal in ("local", "uniform"):
            key = f"{device}_{proposal}"
            color = CLASSICAL_MOVE_COLORS[proposal]

            if device == "gpu":
                (line,) = ax.plot(
                    n_vals,
                    curves[key],
                    color=color,
                    linestyle="None",
                    marker="o",
                    markersize=marker_size,
                    markeredgewidth=0.0,
                    markevery=gpu_markevery,
                    alpha=line_alpha,
                    zorder=5,
                )
            else:
                (line,) = ax.plot(
                    n_vals,
                    curves[key],
                    color=color,
                    linestyle=CLASSICAL_DEVICE_STYLES[device]["linestyle"],
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=4 if device == "fpga" else 3,
                )

            handles[key] = line

    return handles


def _add_classical_annealing_runtime_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    handles_by_key: dict[str, plt.Line2D],
    legend_placement: str,
    legend_y_shift: float,
) -> None:
    """Add a 2-column by 3-row legend: local/uniform by CPU/GPU/FPGA."""
    # Matplotlib fills multi-column legends column by column. This ordering gives
    # local in the first column, uniform in the second, and CPU/GPU/FPGA by row.
    ordered_keys = [
        "cpu_local",
        "gpu_local",
        "fpga_local",
        "cpu_uniform",
        "gpu_uniform",
        "fpga_uniform",
    ]
    labels = [
        "CPU local",
        "GPU local",
        "FPGA local",
        "CPU uniform",
        "GPU uniform",
        "FPGA uniform",
    ]
    handles = [handles_by_key[key] for key in ordered_keys]

    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        frameon=False,
        ncol=2,
        handlelength=2.2,
        columnspacing=1.15,
        labelspacing=0.35,
    )

    if legend_placement == "top_left":
        ax.legend(
            loc="upper left",
            borderaxespad=0.35,
            **legend_kwargs,
        )
    elif legend_placement == "bottom_right":
        ax.legend(
            loc="lower right",
            borderaxespad=0.35,
            **legend_kwargs,
        )
    elif legend_placement == "out":
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, legend_y_shift),
            borderaxespad=0.0,
            **legend_kwargs,
        )
    else:
        raise ValueError(
            "legend_placement must be 'top_left', 'bottom_right', or 'out'."
        )


def plot_annealing_classical_runtime_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
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
    legend_y_shift: float = -0.30,
    xlabel_labelpad: float = 4,
    line_width: float = 2.0,
    line_alpha: float = 0.94,
    marker_size: float = 3.2,
    gpu_markevery: int = 4,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    x_right_padding: float = 0.0,
    show_regime_separator: bool = True,
    regime_separator_n: float = 10.0,
    show_one_year_line: bool = False,
    y_top_padding: float = 0.0,
    debug: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot classical annealing runtimes for local/uniform moves on three devices.

    Colors encode the proposal rule: local is blue and uniform is red.
    Styles encode the device: CPU is dashed, GPU is shown by point markers,
    and FPGA is solid. The legend has two columns and three rows.
    """
    n_vals = _n_grid(n_plot_min, n_plot_max)
    if gpu_markevery < 1:
        raise ValueError("gpu_markevery must be at least 1.")

    if fig is None or ax is None:
        fig, ax = plt.subplots(
            figsize=(APS_COLUMN_WIDTH_CM * CM, 5.6 * CM)
        )
        if legend_placement == "out":
            fig.subplots_adjust(
                left=0.20,
                right=0.985,
                top=0.985,
                bottom=0.40,
            )
        else:
            fig.subplots_adjust(
                left=0.20,
                right=0.985,
                top=0.985,
                bottom=0.17,
            )

    curves = _compute_classical_annealing_runtime_curves(
        beta=beta,
        epsilon=epsilon,
        annealing_schedule_generator=annealing_schedule_generator,
        n_vals=n_vals,
        classical_query_file=classical_query_file,
        statistic=statistic,
        n_fit_min=n_fit_min,
        n_fit_max=n_fit_max,
        debug=debug,
    )
    handles_by_key = _plot_classical_annealing_runtime_curves(
        ax=ax,
        n_vals=n_vals,
        curves=curves,
        line_width=line_width,
        line_alpha=line_alpha,
        marker_size=marker_size,
        gpu_markevery=gpu_markevery,
    )

    _finish_runtime_axis(
        ax=ax,
        n_vals=n_vals,
        title=title,
        xlabel=r"$n$",
        ylabel=r"Runtime",
        show_regime_separator=show_regime_separator,
        regime_separator_n=regime_separator_n,
        show_one_year_line=show_one_year_line,
        grid_color=grid_color,
        grid_linewidth=grid_linewidth,
        x_right_padding=x_right_padding,
        y_top_padding=y_top_padding,
        intercept_ticks=[],
    )
    ax.xaxis.labelpad = xlabel_labelpad

    if show_legend:
        _add_classical_annealing_runtime_legend(
            fig=fig,
            ax=ax,
            handles_by_key=handles_by_key,
            legend_placement=legend_placement,
            legend_y_shift=legend_y_shift,
        )

    return fig, ax


def _normalize_physical_cost_multipliers(
    physical_cost_multipliers: Sequence[int],
) -> list[int]:
    """Return sorted, unique physical-cost multipliers including the implicit 1."""
    multipliers = {1}
    for value in physical_cost_multipliers:
        if isinstance(value, bool):
            raise TypeError("physical_cost_multipliers must contain positive integers.")
        multiplier = int(value)
        if float(value) != float(multiplier) or multiplier < 1:
            raise ValueError(
                "physical_cost_multipliers must contain integers greater than or equal to 1."
            )
        multipliers.add(multiplier)
    return sorted(multipliers)


def _compute_runtime_curves_shadowed(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    n_vals: np.ndarray,
    classical_device: str,
    physical_error_rate: float,
    physical_operation_time_min: float,
    physical_measurement_time_min: float,
    physical_cost_multipliers: Sequence[int],
    num_trotter_steps: int,
    classical_query_file: Path,
    spectral_gap_file: Path,
    statistic: str,
    n_fit_min: int | None,
    n_fit_max: int | None,
    debug: bool,
) -> tuple[
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[int],
]:
    """Compute runtime curves for a fixed error rate and several timing multipliers."""
    multipliers = _normalize_physical_cost_multipliers(
        physical_cost_multipliers
    )
    classical_values: list[float] = []
    uniform_quantum_values: dict[int, list[float]] = {
        multiplier: [] for multiplier in multipliers
    }
    layden_quantum_values: dict[int, list[float]] = {
        multiplier: [] for multiplier in multipliers
    }

    for n in n_vals:
        schedule = [
            float(beta_t)
            for beta_t in annealing_schedule_generator(int(n), float(beta))
        ]
        uniform_queries: list[float] = []
        uniform_gaps: list[float] = []
        layden_gaps: list[float] = []

        if debug:
            print(
                f"\n[debug] n={int(n)}, beta={float(beta):.12g}, "
                f"epsilon={float(epsilon):.12g}, schedule={schedule}, "
                f"multipliers={multipliers}"
            )

        for beta_t in schedule:
            uniform_query = _classical_query_fit_value(
                "uniform",
                int(n),
                beta_t,
                epsilon,
                classical_query_file,
                statistic,
                n_fit_min,
                n_fit_max,
            )
            uniform_gap = _spectral_gap_fit_value(
                "uniform",
                int(n),
                beta_t,
                spectral_gap_file,
                statistic,
                n_fit_min,
                n_fit_max,
            )
            layden_gap = _spectral_gap_fit_value(
                "layden",
                int(n),
                beta_t,
                spectral_gap_file,
                statistic,
                n_fit_min,
                n_fit_max,
            )
            uniform_queries.append(uniform_query)
            uniform_gaps.append(uniform_gap)
            layden_gaps.append(layden_gap)

            if debug:
                print(
                    f"  beta_t={beta_t:.12g}, "
                    f"uniform_query={uniform_query:.12g}, "
                    f"uniform_gap={uniform_gap:.12g}, "
                    f"layden_gap={layden_gap:.12g}"
                )

        classical_uniform = get_annealing_time_classical_walk_uniform(
            int(n),
            uniform_queries,
            device=classical_device,
        )
        classical_values.append(float(classical_uniform))

        for multiplier in multipliers:
            operation_time = float(physical_operation_time_min) * multiplier
            measurement_time = (
                float(physical_measurement_time_min) * multiplier
            )
            quantum_uniform = get_annealing_time_quantum_walk_uniform(
                int(n),
                epsilon,
                schedule,
                uniform_gaps,
                operation_time,
                measurement_time,
                physical_error_rate,
            )
            quantum_layden = get_annealing_time_quantum_walk_qemc(
                int(n),
                epsilon,
                schedule,
                layden_gaps,
                operation_time,
                measurement_time,
                physical_error_rate,
                num_trotter_steps=num_trotter_steps,
            )
            uniform_quantum_values[multiplier].append(float(quantum_uniform))
            layden_quantum_values[multiplier].append(float(quantum_layden))

            if debug:
                print(
                    f"  multiplier={multiplier}, "
                    f"operation_time={operation_time:.12g}, "
                    f"measurement_time={measurement_time:.12g}, "
                    f"uniform_quantum={float(quantum_uniform):.12g}, "
                    f"layden_quantum={float(quantum_layden):.12g}"
                )

    classical_curve = _positive_floor(
        np.asarray(classical_values, dtype=float)
    )
    uniform_quantum_curves = {
        multiplier: _positive_floor(np.asarray(values, dtype=float))
        for multiplier, values in uniform_quantum_values.items()
    }
    layden_quantum_curves = {
        multiplier: _positive_floor(np.asarray(values, dtype=float))
        for multiplier, values in layden_quantum_values.items()
    }
    return (
        classical_curve,
        uniform_quantum_curves,
        layden_quantum_curves,
        multipliers,
    )


def _add_shadowed_runtime_family(
    ax: plt.Axes,
    n_vals: np.ndarray,
    curves_by_multiplier: dict[int, np.ndarray],
    multipliers: Sequence[int],
    color: str,
    label: str,
    line_width: float,
    line_alpha: float,
    band_alpha: float,
    zorder: int,
) -> plt.Line2D:
    """Plot multiplier curves and progressively lighter fills between them."""
    multipliers = list(multipliers)
    interval_count = max(1, len(multipliers) - 1)

    for index, (lower_multiplier, upper_multiplier) in enumerate(
        zip(multipliers[:-1], multipliers[1:])
    ):
        lower_raw = np.asarray(
            curves_by_multiplier[lower_multiplier], dtype=float
        )
        upper_raw = np.asarray(
            curves_by_multiplier[upper_multiplier], dtype=float
        )
        lower = _positive_floor(np.minimum(lower_raw, upper_raw))
        upper = _positive_floor(np.maximum(lower_raw, upper_raw))
        fade = 1.0 - 0.70 * index / max(1, interval_count - 1)
        ax.fill_between(
            n_vals,
            lower,
            upper,
            color=color,
            alpha=band_alpha * fade,
            edgecolor="none",
            linewidth=0.0,
            zorder=zorder,
        )

    family_handle: plt.Line2D | None = None
    line_count = max(1, len(multipliers) - 1)
    for index, multiplier in enumerate(multipliers):
        fade = 1.0 - 0.35 * index / line_count
        (line,) = ax.plot(
            n_vals,
            curves_by_multiplier[multiplier],
            color=color,
            linewidth=0.55 * line_width,
            alpha=line_alpha * fade,
            linestyle="-",
            label=label if multiplier == 1 else None,
            zorder=zorder + 1,
        )
        if multiplier == 1:
            family_handle = line

    if family_handle is None:
        raise RuntimeError("The implicit multiplier 1 is missing.")
    return family_handle


def _add_multiplier_label(
    ax: plt.Axes,
    n_vals: np.ndarray,
    values: np.ndarray,
    multiplier: int,
    color: str,
    x_fraction: float,
    alpha: float,
) -> None:
    """Place a label above a curve, rotated along its local display-space slope."""
    x = np.asarray(n_vals, dtype=float)
    y = _positive_floor(np.asarray(values, dtype=float))
    if x.size < 2:
        return

    index = int(round(float(x_fraction) * (x.size - 1)))
    index = int(np.clip(index, 0, x.size - 1))
    left_index = max(0, index - 1)
    right_index = min(x.size - 1, index + 1)
    if left_index == right_index:
        return

    left_display = ax.transData.transform(
        (float(x[left_index]), float(y[left_index]))
    )
    right_display = ax.transData.transform(
        (float(x[right_index]), float(y[right_index]))
    )
    angle = float(
        np.degrees(
            np.arctan2(
                right_display[1] - left_display[1],
                right_display[0] - left_display[0],
            )
        )
    )

    if multiplier != 1:
        ax.annotate(
            rf"$\times {multiplier}$",
            xy=(float(x[index]), float(y[index])),
            xytext=(0.0, 2.0),
            textcoords="offset points",
            color=color,
            alpha=alpha,
            fontsize="small",
            ha="center",
            va="bottom",
            rotation=angle,
            rotation_mode="anchor",
            clip_on=True,
            zorder=10,
        )


def plot_annealing_classical_and_quantum_runtime_vs_n_shadowed(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    classical_device: str = "fpga",
    physical_error_rate: float = 1e-4,
    physical_operation_time_min: float = 200e-9,
    physical_measurement_time_min: float = 20e-9,
    physical_cost_multipliers: list[int] = [10, 100, 1000],
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
    show_direct_enumeration_line: bool = False,
    y_top_padding: float = 0.0,
    optimistic_pessimistic_intercept: bool = False,
    show_multiplier_label: bool = True,
    debug: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot annealing runtimes with timing-cost multiplier lines and shading.

    The optimistic quantum curves use the implicit multiplier 1. Each explicit
    multiplier scales both ``physical_operation_time_min`` and
    ``physical_measurement_time_min`` while leaving ``physical_error_rate``
    fixed. The regions between consecutive multiplier curves are filled with
    progressively lower opacity toward the larger multipliers.
    """
    n_vals = _n_grid(n_plot_min, n_plot_max)
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            figsize=(APS_COLUMN_WIDTH_CM * CM, 5.6 * CM)
        )
        if legend_placement == "out":
            fig.subplots_adjust(
                left=0.20,
                right=0.985,
                top=0.985,
                bottom=0.43,
            )
        else:
            fig.subplots_adjust(
                left=0.20,
                right=0.985,
                top=0.985,
                bottom=0.17,
            )

    (
        classical_curve,
        uniform_quantum_curves,
        layden_quantum_curves,
        multipliers,
    ) = _compute_runtime_curves_shadowed(
        beta=beta,
        epsilon=epsilon,
        annealing_schedule_generator=annealing_schedule_generator,
        n_vals=n_vals,
        classical_device=classical_device,
        physical_error_rate=physical_error_rate,
        physical_operation_time_min=physical_operation_time_min,
        physical_measurement_time_min=physical_measurement_time_min,
        physical_cost_multipliers=physical_cost_multipliers,
        num_trotter_steps=num_trotter_steps,
        classical_query_file=classical_query_file,
        spectral_gap_file=spectral_gap_file,
        statistic=statistic,
        n_fit_min=n_fit_min,
        n_fit_max=n_fit_max,
        debug=debug,
    )

    handles: list[plt.Line2D] = []
    labels: list[str] = []
    (classical_line,) = ax.plot(
        n_vals,
        classical_curve,
        color=RUNTIME_COLORS["uniform_classical"],
        linewidth=line_width,
        alpha=line_alpha,
        linestyle="-",
        label=RUNTIME_LABELS["uniform_classical"],
        zorder=4,
    )
    handles.append(classical_line)
    labels.append(RUNTIME_LABELS["uniform_classical"])

    uniform_handle = _add_shadowed_runtime_family(
        ax=ax,
        n_vals=n_vals,
        curves_by_multiplier=uniform_quantum_curves,
        multipliers=multipliers,
        color=RUNTIME_COLORS["uniform_quantum"],
        label=RUNTIME_LABELS["uniform_quantum"],
        line_width=line_width,
        line_alpha=line_alpha,
        band_alpha=band_alpha,
        zorder=2,
    )
    handles.append(uniform_handle)
    labels.append(RUNTIME_LABELS["uniform_quantum"])

    layden_handle = _add_shadowed_runtime_family(
        ax=ax,
        n_vals=n_vals,
        curves_by_multiplier=layden_quantum_curves,
        multipliers=multipliers,
        color=RUNTIME_COLORS["layden_quantum"],
        label=RUNTIME_LABELS["layden_quantum"],
        line_width=line_width,
        line_alpha=line_alpha,
        band_alpha=band_alpha,
        zorder=3,
    )
    handles.append(layden_handle)
    labels.append(RUNTIME_LABELS["layden_quantum"])

    if show_direct_enumeration_line:
        direct_line = _add_direct_enumeration_line(ax, n_vals, line_alpha)
        handles.append(direct_line)
        labels.append(RUNTIME_LABELS["direct_enumeration"])

    largest_multiplier = multipliers[-1]
    intercept_proxy = {
        "uniform_classical": classical_curve,
        "layden_quantum_min": layden_quantum_curves[1],
        "layden_quantum_max": layden_quantum_curves[largest_multiplier],
    }
    intercept_ticks = _integer_intercept_ticks(
        n_vals,
        intercept_proxy,
        optimistic_pessimistic_intercept,
    )

    _finish_runtime_axis(
        ax=ax,
        n_vals=n_vals,
        title=title,
        xlabel=r"$n$",
        ylabel=r"Runtime",
        show_regime_separator=show_regime_separator,
        regime_separator_n=regime_separator_n,
        show_one_year_line=show_one_year_line,
        grid_color=grid_color,
        grid_linewidth=grid_linewidth,
        x_right_padding=x_right_padding,
        y_top_padding=y_top_padding,
        intercept_ticks=intercept_ticks,
    )
    ax.xaxis.labelpad = xlabel_labelpad

    if show_multiplier_label:
        line_count = max(1, len(multipliers) - 1)
        for index, multiplier in enumerate(multipliers):
            fade = 1.0 - 0.35 * index / line_count
            _add_multiplier_label(
                ax=ax,
                n_vals=n_vals,
                values=uniform_quantum_curves[multiplier],
                multiplier=multiplier,
                color=RUNTIME_COLORS["uniform_quantum"],
                x_fraction=0.76,
                alpha=line_alpha * fade,
            )
            _add_multiplier_label(
                ax=ax,
                n_vals=n_vals,
                values=layden_quantum_curves[multiplier],
                multiplier=multiplier,
                color=RUNTIME_COLORS["layden_quantum"],
                x_fraction=0.58,
                alpha=line_alpha * fade,
            )

    if show_legend:
        _add_runtime_legend(
            fig,
            ax,
            handles,
            labels,
            legend_placement,
            legend_y_shift,
        )

    return fig, ax
