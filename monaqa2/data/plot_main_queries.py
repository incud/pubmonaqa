from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, SPECTRAL_GAP_FILE
from monaqa2.data.classical_query import get_annealing_classical_query_stats, load_classical_queries
from monaqa2.data.quantum_query import get_quantum_query_stats, get_quantum_query_mean_fit_by_n, get_quantum_query_step_stats
from monaqa2.data.runtime import get_annealing_queries_quantum_walks
from monaqa2.data.utils_queries import fit_query_quantile_ratios_by_n, annealing_query_stats, query_temperature_window, query_percentile_columns
from monaqa2.data.utils_stats import fit_exponential_from_stats


CM = 1.0 / 2.54
PT_TO_CM = 2.54 / 72.0
APS_FIGURESTAR_WIDTH_CM = 17.8
APS_FIGURESTAR_HEIGHT_CM = 5.8

PROPOSAL_COLORS = {
    # Neutral family: near-black and gray
    "local1_classical": "#222222",
    "local1_quantum": "#999999",

    # Cold family: navy blue and cyan
    "uniform_classical": "#004488",
    "uniform_quantum": "#66CCEE",

    # Warm family: magenta and orange
    "layden_classical": "#EE7733",
    "layden_quantum": "#AA3377",
}

QUERY_SERIES = ("uniform_classical", "uniform_quantum", "layden_quantum")

QUERY_LABELS = {
    "uniform_classical": "Best classical walk",
    "uniform_quantum": "Quantized classical walk",
    "layden_quantum": "This work",
}

QUERY_COLORS = {
    "uniform_classical": PROPOSAL_COLORS["uniform_classical"],
    "uniform_quantum": PROPOSAL_COLORS["uniform_quantum"],
    "layden_quantum": PROPOSAL_COLORS["layden_quantum"],
}


def _n_grid(n_plot_min: int | None, n_plot_max: int | None) -> np.ndarray:
    n_plot_min = 3 if n_plot_min is None else int(n_plot_min)
    n_plot_max = 120 if n_plot_max is None else int(n_plot_max)
    if n_plot_max < n_plot_min:
        raise ValueError("n_plot_max must be greater than or equal to n_plot_min.")
    return np.arange(n_plot_min, n_plot_max + 1, dtype=int)


def _compute_query_curves(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
    n_vals: np.ndarray,
    classical_query_file: Path,
    spectral_gap_file: Path,
    statistic: str,
    n_fit_min: int | None,
    n_fit_max: int | None,
    debug: bool = False,
    compute_bands: bool = True,
    band_percentiles: tuple[int, int] = (25, 75),
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    # The schedule must be evaluated at every plotted n, including extrapolation.
    schedules = {}

    def schedule(n, target_beta):
        if n not in schedules:
            values = np.asarray(annealing_schedule_generator(int(n), float(target_beta)), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(np.diff(values) <= 0.0):
                raise ValueError(f"Invalid annealing schedule for n={n}.")
            values = values[values > 0.0]
            if not values.size or not np.isclose(values[-1], target_beta):
                raise ValueError("A query plot needs a nonempty schedule ending at positive beta.")
            schedules[n] = values.tolist()
        return schedules[n]

    paths = [schedule(int(n), beta) for n in n_vals]
    beta_min = min(min(path) for path in paths)
    beta_max = max(max(path) for path in paths)
    classical_rows = load_classical_queries("uniform", np.inf, "bhattacharyya", epsilon, classical_query_file, only_ok=False)
    if n_fit_min is not None:
        classical_rows = classical_rows[classical_rows["n"] >= n_fit_min]
    if n_fit_max is not None:
        classical_rows = classical_rows[classical_rows["n"] <= n_fit_max]
    classical_steps = annealing_query_stats(query_temperature_window(classical_rows, "queries", beta_min, beta_max))
    steps = {
        "uniform_classical": classical_steps,
        "uniform_quantum": get_quantum_query_step_stats("uniform", np.inf, epsilon, beta_min, beta_max, spectral_gap_file, n_min=n_fit_min, n_max=n_fit_max),
        "layden_quantum": get_quantum_query_step_stats("layden", np.inf, epsilon, beta_min, beta_max, spectral_gap_file, n_min=n_fit_min, n_max=n_fit_max),
    }
    models = {}
    for key, table in steps.items():
        temperatures = np.sort(table["beta"].unique())
        if key == "uniform_classical":
            fits = [fit_exponential_from_stats(table, temperature, n_fit_min, n_fit_max) for temperature in temperatures]
        else:
            fits = [get_quantum_query_mean_fit_by_n(table, n_fit_min, n_fit_max, fixed_beta=temperature) for temperature in temperatures]
        models[key] = (temperatures, np.log([fit[0] for fit in fits]), np.array([fit[1] for fit in fits]))

    curves = {key: [] for key in QUERY_SERIES}
    for n, path in zip(n_vals, paths):
        length = len(path)
        # Reference step costs use L=1. Restore the actual L-dependent filter
        # precision using the existing query model, without averaging gaps.
        reference = get_annealing_queries_quantum_walks(int(n), epsilon, [1.0])
        precision_ratio = get_annealing_queries_quantum_walks(int(n), epsilon, [1.0] * length) / (length * reference)
        for key, (temperatures, log_A, rates) in models.items():
            per_step = np.exp(np.interp(path, temperatures, log_A) + np.interp(path, temperatures, rates) * n)
            total = float(np.sum(per_step))
            curves[key].append(total if key == "uniform_classical" else total * precision_ratio)
        if debug:
            print(f"\n[debug] n={int(n)}, schedule={path}, filter_count={length}")

    curves = {key: np.asarray(values) for key, values in curves.items()}
    bands = {}
    if compute_bands:
        # Quantiles are taken after summing each instance's complete path.
        # Fit the selected lower percentile and range relative to the mean.
        totals = {
            "uniform_classical": get_annealing_classical_query_stats("uniform", np.inf, "bhattacharyya", epsilon, beta, schedule, classical_query_file, n_min=n_fit_min, n_max=n_fit_max),
            "uniform_quantum": get_quantum_query_stats("uniform", np.inf, beta, epsilon, schedule, spectral_gap_file, n_min=n_fit_min, n_max=n_fit_max),
            "layden_quantum": get_quantum_query_stats("layden", np.inf, beta, epsilon, schedule, spectral_gap_file, n_min=n_fit_min, n_max=n_fit_max),
        }
        for key, table in totals.items():
            lower_fit, width_fit = fit_query_quantile_ratios_by_n(table, n_fit_min, n_fit_max, percentiles=band_percentiles)
            lower = curves[key] * lower_fit[0] * np.exp(lower_fit[1] * n_vals)
            width = curves[key] * width_fit[0] * np.exp(width_fit[1] * n_vals)
            bands[key] = (lower, lower + width)
            if debug:
                for label, bound in zip(query_percentile_columns(band_percentiles), bands[key]):
                    ratio = bound / curves[key]
                    print(f"[debug] {key}: {label}/mean first={ratio[0]:.6g}, last={ratio[-1]:.6g}")
    return curves, bands


def _plot_query_curves(ax: plt.Axes, n_vals: np.ndarray, curves: dict[str, np.ndarray], line_width: float, line_alpha: float, bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None, band_alpha: float = 0.18, band_label: str = "25–75% instance percentiles") -> tuple[list[plt.Line2D], list[str]]:
    handles = []
    labels = []
    for zorder, key in enumerate(QUERY_SERIES, start=2):
        if bands is not None:
            lower, upper = bands[key]
            ax.fill_between(n_vals, lower, upper, color=QUERY_COLORS[key], alpha=band_alpha, edgecolor="none", linewidth=0.0, label=f"{QUERY_LABELS[key]}: {band_label}", zorder=1)
        (line,) = ax.plot(n_vals, curves[key], color=QUERY_COLORS[key], linewidth=line_width, linestyle="-", alpha=line_alpha, label=QUERY_LABELS[key], zorder=zorder)
        handles.append(line)
        labels.append(QUERY_LABELS[key])
    return handles, labels


def _finish_query_axis(ax: plt.Axes, n_vals: np.ndarray, title: str | None, xlabel: str | None, ylabel: str | None, show_numerical_boundary: bool, numerical_boundary_n: float | None, grid_color: str, grid_linewidth: float, x_right_padding: float = 0.0) -> None:
    ax.set_yscale("log")
    ax.set_xlim(float(n_vals[0]), float(n_vals[-1]) + max(0.0, float(x_right_padding)))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if show_numerical_boundary and numerical_boundary_n is not None and float(n_vals[0]) <= float(numerical_boundary_n) <= float(n_vals[-1]):
        ax.axvline(float(numerical_boundary_n), color="0.62", linewidth=0.55, linestyle=(0, (2.0, 2.0)), alpha=0.80, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    ax.grid(False, which="major", axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _positive_log_ylim_from_curves(curve_groups: Sequence[dict[str, np.ndarray]]) -> tuple[float, float] | None:
    values = []
    for curves in curve_groups:
        for key in QUERY_SERIES:
            arr = np.asarray(curves[key], dtype=float)
            values.append(arr[np.isfinite(arr) & (arr > 0.0)])
    values = [arr for arr in values if arr.size]
    if not values:
        return None
    merged = np.concatenate(values)
    y_min = float(np.min(merged))
    y_max = float(np.max(merged))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min <= 0.0 or y_max <= 0.0:
        return None
    if np.isclose(y_min, y_max):
        return y_min / 2.0, y_max * 2.0
    return 10.0 ** (np.log10(y_min) - 0.06 * (np.log10(y_max) - np.log10(y_min))), 10.0 ** (np.log10(y_max) + 0.06 * (np.log10(y_max) - np.log10(y_min)))


def _validate_legend_placement(legend_placement: str) -> None:
    if legend_placement not in {"top_left", "out"}:
        raise ValueError("legend_placement must be either 'top_left' or 'out'.")


def _format_last_value(value: float) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    exponent_int = int(exponent)
    exponent_text = f"{exponent_int:02d}" if exponent_int >= 0 else f"-{abs(exponent_int):02d}"
    return rf"${mantissa}e{exponent_text}$"


def _draw_last_value_labels(ax: plt.Axes, n_vals: np.ndarray, curves: dict[str, np.ndarray], fontsize: float = 7.0) -> None:
    x_last = float(n_vals[-1])
    for key in QUERY_SERIES:
        values = np.asarray(curves[key], dtype=float)
        if values.size == 0 or not np.isfinite(values[-1]) or values[-1] <= 0.0:
            continue
        ax.annotate(_format_last_value(float(values[-1])), xy=(x_last, float(values[-1])), xytext=(-4.0, 0.0), textcoords="offset points", ha="right", va="center", fontsize=fontsize, color=QUERY_COLORS[key], clip_on=True, zorder=20)


def _add_single_axis_legend(fig: plt.Figure, ax: plt.Axes, handles: list[plt.Line2D], labels: list[str], legend_placement: str, legend_y_shift: float) -> None:
    if legend_placement == "top_left":
        ax.legend(handles, labels, frameon=False, loc="upper left", ncol=1, borderaxespad=0.35, handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    else:
        ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.0, legend_y_shift), ncol=1, borderaxespad=0.0, handlelength=2.2, columnspacing=1.0, labelspacing=0.35)


def _add_zoom_inset(
    ax: plt.Axes,
    n_vals: np.ndarray,
    curves: dict[str, np.ndarray],
    line_width: float,
    line_alpha: float,
    grid_color: str,
    grid_linewidth: float,
    zoom_xlim: tuple[float, float],
    zoom_ylim: tuple[float, float],
    zoom_bbox: tuple[float, float, float, float],
    zoom_tick_fontsize: float | None,
    show_zoom_in_border: bool,
    bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    band_alpha: float = 0.18,
    band_label: str = "25–75% instance percentiles",
) -> plt.Axes:
    inset_ax = ax.inset_axes(zoom_bbox)
    inset_ax.set_yscale("log")
    inset_ax.set_xlim(*zoom_xlim)
    inset_ax.set_ylim(*zoom_ylim)
    _plot_query_curves(inset_ax, n_vals, curves, line_width, line_alpha, bands=bands, band_alpha=band_alpha, band_label=band_label)
    inset_ax.set_xticks([int(zoom_xlim[0]), int(zoom_xlim[1])])
    inset_ax.set_yticks([1e3, 1e4])
    inset_ax.set_yticklabels([r"$10^3$", r"$10^4$"])
    # Matplotlib expands limits to include explicit ticks; preserve zoom_ylim.
    inset_ax.set_ylim(*zoom_ylim)
    tick_fontsize = max(1.0, float(plt.rcParams.get("font.size", 10.0)) - 1.0) if zoom_tick_fontsize is None else float(zoom_tick_fontsize)
    inset_ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, length=2.4, pad=1.4)
    inset_ax.set_axisbelow(True)
    inset_ax.grid(True, which="major", axis="y", color=grid_color, linewidth=grid_linewidth, zorder=0)
    inset_ax.grid(False, which="major", axis="x")
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    if show_zoom_in_border:
        for spine in ["top", "right"]:
            inset_ax.spines[spine].set_visible(True)
            inset_ax.spines[spine].set_color("black")
            inset_ax.spines[spine].set_linewidth(0.65)
    else:
        for spine in ["top", "right"]:
            inset_ax.spines[spine].set_visible(False)
    return inset_ax


def plot_annealing_classical_and_quantum_queries_vs_n(
    beta: float,
    epsilon: float,
    annealing_schedule_generator: Callable[[int, float], list[float]],
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
    debug: bool = False,
    show_numerical_boundary: bool = False,
    add_regime_separator: bool = False,
    numerical_boundary_n: float | None = None,
    legend_placement: str = "top_left",
    legend_y_shift: float = -0.22,
    show_last_value: bool = False,
    last_value_fontsize: float = 7.0,
    xlabel_labelpad: float = 4,
    line_width: float = 2.0,
    line_alpha: float = 0.94,
    grid_color: str = "0.92",
    grid_linewidth: float = 0.55,
    x_right_padding: float = 0.0,
    show_zoom_in: bool = True,
    zoom_xlim: tuple[float, float] = (3.0, 10.0),
    zoom_ylim: tuple[float, float] = (1e2, 1e4),
    zoom_bbox: tuple[float, float, float, float] = (0.10, 0.57, 0.42, 0.36),
    zoom_tick_fontsize: float | None = None,
    show_zoom_in_border: bool = False,
    *,
    show_bands: bool = True,
    band_scale: float = 1.0,
    band_alpha: float = 0.18,
    band_percentiles: tuple[int, int] = (25, 75),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot full annealing-path arithmetic means with selected percentile bands.

    Per-temperature query costs are calculated for each instance before
    averaging and fitting in n. Every plotted n gets its actual schedule,
    including the resulting filter count and precision allocation. Whole-path
    totals are not replaced by a single exponential. Percentiles are calculated
    from measured whole-instance totals, preserving temperature correlations.
    Fits of lower/mean and (upper-lower)/mean to A*exp(b*n) are multiplied by the
    full-path mean to estimate positive, ordered bounds at every plotted n.
    The (fig, ax) return value is unchanged; statistic must be 'mean+std'.

    At band_scale=1, (25, 75) estimates the middle 50% of instance costs and
    (0, 100) estimates the sample minimum–maximum range. Fitted bounds need
    not enclose every observed cost or the mean; they are not confidence or
    worst-case bounds. Extrapolation assumes the fitted means and relative
    percentiles continue beyond measured sizes. debug=True prints endpoint
    percentile/mean ratios.

    :param show_bands: Draw shading for all three curves and the zoom inset.
    :param band_scale: Scale log(bound/mean); 1 gives the selected range,
        0 the mean. Other values no longer represent the selected percentiles.
    :param band_alpha: Opacity of the shaded bands, between zero and one.
    :param band_percentiles: (25, 75) for the IQR (default), or (0, 100) for
        the fitted sample minimum–maximum range. Only these pairs are supported.
    """
    _validate_legend_placement(legend_placement)
    lower_col, upper_col = query_percentile_columns(band_percentiles)
    if statistic != "mean+std":
        raise ValueError("The full-path mean query plot requires statistic='mean+std'.")
    if not np.isfinite(band_scale) or band_scale < 0.0:
        raise ValueError("band_scale must be finite and non-negative.")
    if not np.isfinite(band_alpha) or not 0.0 <= band_alpha <= 1.0:
        raise ValueError("band_alpha must be between zero and one.")
    n_vals = _n_grid(n_plot_min, n_plot_max)
    show_numerical_boundary = show_numerical_boundary or add_regime_separator
    numerical_boundary_n = float(n_fit_max) if numerical_boundary_n is None and n_fit_max is not None else numerical_boundary_n
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8.5 * CM, 5.6 * CM))
        if legend_placement == "out":
            fig.subplots_adjust(left=0.18, right=0.985, top=0.985, bottom=0.43)
        else:
            fig.subplots_adjust(left=0.18, right=0.985, top=0.985, bottom=0.17)
    curves, percentiles = _compute_query_curves(beta, epsilon, annealing_schedule_generator, n_vals, classical_query_file, spectral_gap_file, statistic, n_fit_min, n_fit_max, debug, compute_bands=show_bands, band_percentiles=band_percentiles)
    bands = {
        key: tuple(curves[key] * (bound / curves[key]) ** band_scale for bound in bounds)
        for key, bounds in percentiles.items()
    } if show_bands else None
    curve_groups = [curves] if bands is None else [curves, {key: values[0] for key, values in bands.items()}, {key: values[1] for key, values in bands.items()}]
    ylim = _positive_log_ylim_from_curves(curve_groups)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _finish_query_axis(ax, n_vals, title, r"$n$", r"Queries", show_numerical_boundary, numerical_boundary_n, grid_color, grid_linewidth, x_right_padding=x_right_padding)
    band_label = f"{lower_col[1:]}–{upper_col[1:]}% instance percentiles" if lower_col != "q0" else "0–100% sample range"
    if band_scale != 1.0:
        band_label = "scaled percentile band"
    handles, labels = _plot_query_curves(ax, n_vals, curves, line_width, line_alpha, bands=bands, band_alpha=band_alpha, band_label=band_label)
    ax.xaxis.labelpad = xlabel_labelpad
    if show_last_value:
        _draw_last_value_labels(ax, n_vals, curves, fontsize=last_value_fontsize)
    if show_zoom_in:
        _add_zoom_inset(ax, n_vals, curves, line_width, line_alpha, grid_color, grid_linewidth, zoom_xlim, zoom_ylim, zoom_bbox, zoom_tick_fontsize, show_zoom_in_border, bands=bands, band_alpha=band_alpha, band_label=band_label)
    if show_legend:
        _add_single_axis_legend(fig, ax, handles, labels, legend_placement, legend_y_shift)
    return fig, ax
