"""Fit runtime centers and percentile bands from the same instance costs.

The full-schedule resource model supplies the size-dependent reference shape.
Instance statistics supply the center correction and relative percentile fits.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from monaqa2.data import plot_main_runtime as _base
from monaqa2.data.classical_query import load_classical_queries
from monaqa2.data.runtime import (
    get_annealing_time_classical_walk_uniform,
    get_annealing_time_quantum_walk_uniform,
    get_annealing_time_quantum_walk_qemc,
)
from monaqa2.data.utils_queries import (
    annealing_query_rows, annealing_query_stats,
    fit_query_statistic_by_n, query_percentile_columns,
)
from monaqa2.data.utils_stats import STATISTICS


def validate_band_options(statistic, band_percentiles, band_alpha):
    query_percentile_columns(band_percentiles)
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}.")
    if not np.isfinite(band_alpha) or not 0.0 <= band_alpha <= 1.0:
        raise ValueError("band_alpha must be between zero and one.")


def cached_schedule_generator(generator):
    cache = {}

    def schedule(n, beta):
        key = (int(n), float(beta))
        if key not in cache:
            values = np.asarray(generator(*key), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(np.diff(values) <= 0.0):
                raise ValueError(f"Invalid annealing schedule for n={n}.")
            values = values[values > 0.0]
            if not values.size or not np.isclose(values[-1], beta):
                raise ValueError("Runtime plots need a nonempty schedule ending at positive beta.")
            cache[key] = values.tolist()
        return list(cache[key])

    return schedule


def _instance_runtime_stats(table, proposal, beta, epsilon, schedule, classical_device, hardware, num_trotter_steps, *, k_randomized_grid=2, statistic="mean+std", arithmetic_type="FIXED_POINT"):
    """Evaluate the complete existing runtime model before taking statistics.

    Quantum precision and code distance are calculated for each instance.
    Interpolation and validation follow the query plot's instance convention.
    The temporary 'queries' field is the shared helper's name for a cost;
    center/spread/q* in the returned table are all in seconds.
    """
    if proposal == "uniform_classical":
        cost = lambda n, eps, values: get_annealing_time_classical_walk_uniform(n, values, device=classical_device)
        value_col, is_gap = "queries", False
    else:
        rate, operation_time, measurement_time = hardware
        runtime_fn = get_annealing_time_quantum_walk_uniform if proposal == "uniform_quantum" else get_annealing_time_quantum_walk_qemc
        resource_options = {"num_trotter_steps": num_trotter_steps, "arithmetic_type": arithmetic_type}
        if proposal == "layden_quantum":
            resource_options["k_randomized_grid"] = k_randomized_grid

        def cost(n, eps, gaps):
            return runtime_fn(n, eps, schedule(n, beta), gaps, operation_time, measurement_time, rate, **resource_options)

        value_col, is_gap = "delta", True
    rows = annealing_query_rows(table, value_col, beta, epsilon, schedule, cost, is_gap=is_gap)
    return annealing_query_stats(rows, statistic)


def _fit_values(stats, values, n_vals):
    relative = stats.copy()
    relative["center"] = values
    # A single positive offset cannot determine a slope; keep it constant.
    positive = np.asarray(values)[np.asarray(values) > 0.0]
    if positive.size == 1:
        return np.full(np.shape(n_vals), positive[0], dtype=float)
    amplitude, slope = fit_query_statistic_by_n(relative, "center")
    return amplitude * np.exp(slope * n_vals)


def _fit_relative_bounds(stats, n_vals, band_percentiles):
    """Fit percentile bounds relative to this same sample's runtime center.

    For sample extrema, positive offsets enforce lower <= center <= upper
    for every extrapolated size. IQR fits preserve lower <= upper without
    assuming an arithmetic mean must belong to the middle half of a sample.
    """
    lower_col, upper_col = query_percentile_columns(band_percentiles)
    center = stats["center"].to_numpy(dtype=float)
    lower = stats[lower_col].to_numpy(dtype=float)
    upper = stats[upper_col].to_numpy(dtype=float)
    if not np.all(np.isfinite([center, lower, upper])) or np.any(center <= 0.0) or np.any(lower <= 0.0) or np.any(upper < lower):
        raise ValueError("Runtime percentile fits need positive finite centers and 0 < lower <= upper.")
    if lower_col == "q0":
        offsets = [(center-lower)/lower, (upper-center)/center]
        fitted = []
        for offset in offsets:
            offset[np.abs(offset) <= 32*np.finfo(float).eps] = 0.0
            if np.any(offset < 0.0):
                raise ValueError("Runtime centers must lie within the sample minimum and maximum.")
            fitted.append(_fit_values(stats, offset, n_vals))
        return 1.0/(1.0+fitted[0]), 1.0+fitted[1]
    lower = _fit_values(stats, lower/center, n_vals)
    width = _fit_values(stats, (upper-stats[lower_col].to_numpy())/center, n_vals)
    return lower, lower + width


def apply_runtime_statistics(
    ax, previous_lines, previous_collections, *, beta, epsilon, schedule,
    classical_device, hardware_scenarios, multipliers, num_trotter_steps, k_randomized_grid, arithmetic_type,
    classical_query_file, spectral_gap_file, statistic, n_fit_min, n_fit_max,
    show_bands, band_percentiles, bands_on_multipliers, band_alpha, debug,
):
    """Replace the gap-based centers and hardware fills with instance fits.

    For ordinary plots the optimistic hardware setting is the main line;
    bands_on_multipliers also enables the pessimistic setting. For shadowed
    plots the main setting is multiplier 1; other timings scale both runtimes
    and their instance distributions exactly because the error rate is fixed.
    Every timing curve is updated even when its band is disabled. Existing
    artists on a reused axis are preserved; technique labels remain unchanged.
    """
    for collection in list(ax.collections):
        if collection not in previous_collections:
            collection.remove()
    lines = {
        key: [line for line in ax.lines if line not in previous_lines and to_rgba(line.get_color()) == to_rgba(_base.RUNTIME_COLORS[key])]
        for key in _base.RUNTIME_SERIES
    }
    n_vals = np.asarray(lines["uniform_classical"][0].get_xdata(), dtype=float)
    classical = load_classical_queries("uniform", np.inf, "bhattacharyya", epsilon, classical_query_file, only_ok=False)
    gaps = pd.read_pickle(spectral_gap_file)
    gaps = gaps[gaps["a"] == np.inf]
    tables = {
        "uniform_classical": classical,
        "uniform_quantum": gaps[gaps["proposal"] == "uniform"],
        "layden_quantum": gaps[gaps["proposal"] == "layden"],
    }
    for key, table in tables.items():
        if n_fit_min is not None:
            table = table[table["n"] >= n_fit_min]
        if n_fit_max is not None:
            table = table[table["n"] <= n_fit_max]
        tables[key] = table

    scenarios = hardware_scenarios if multipliers is None else hardware_scenarios[:1]
    for scenario_index, hardware in enumerate(scenarios):
        stats = {
            key: _instance_runtime_stats(table, key, beta, epsilon, schedule, classical_device, hardware, num_trotter_steps, k_randomized_grid=k_randomized_grid, arithmetic_type=arithmetic_type, statistic=statistic)
            for key, table in tables.items() if scenario_index == 0 or key != "uniform_classical"
        }
        measured_n = np.sort(np.unique(np.concatenate([table["n"].to_numpy() for table in stats.values()])))
        rate, operation_time, measurement_time = hardware
        classical_ref, uniform_ref, layden_ref, _ = _base._compute_runtime_curves_shadowed(
            beta, epsilon, schedule, measured_n, classical_device, rate,
            operation_time, measurement_time, [], num_trotter_steps,
            classical_query_file, spectral_gap_file, statistic, n_fit_min, n_fit_max, False,
            k_randomized_grid=k_randomized_grid, arithmetic_type=arithmetic_type,
        )
        references = {"uniform_classical": classical_ref, "uniform_quantum": uniform_ref[1], "layden_quantum": layden_ref[1]}
        for key, table in stats.items():
            reference = references[key][np.searchsorted(measured_n, table["n"].to_numpy())]
            center_ratio = _fit_values(table, table["center"].to_numpy()/reference, n_vals)
            shade = show_bands and (scenario_index == 0 or bands_on_multipliers)
            if shade:
                lower_ratio, upper_ratio = _fit_relative_bounds(table, n_vals, band_percentiles)
            if key == "uniform_classical":
                selected_lines = [(1, lines[key][0])]
            elif multipliers is None:
                selected_lines = [(scenario_index + 1, lines[key][scenario_index])]
            else:
                selected_lines = list(zip(multipliers, lines[key]))
            for setting, line in selected_lines:
                center = np.asarray(line.get_ydata(), dtype=float) * center_ratio
                line.set_ydata(center)
                if not shade or (multipliers is not None and setting != 1 and not bands_on_multipliers):
                    continue
                band = ax.fill_between(
                    n_vals, center * lower_ratio, center * upper_ratio,
                    color=line.get_color(), alpha=band_alpha * (line.get_alpha() if line.get_alpha() is not None else 1.0),
                    edgecolor="none", linewidth=0.0, zorder=1.5,
                )
                band.set_gid(f"runtime-percentile-{key}-{setting}")
                if debug:
                    print(f"[debug] {key}, setting={setting}: statistic={statistic}, center/reference={center_ratio[0]:.6g}..{center_ratio[-1]:.6g}, percentiles={band_percentiles}, lower/center={lower_ratio[0]:.6g}..{lower_ratio[-1]:.6g}, upper/center={upper_ratio[0]:.6g}..{upper_ratio[-1]:.6g}")


def refresh_runtime_annotations(
    ax, previous_lines, *, title, xlabel_labelpad, show_regime_separator,
    regime_separator_n, show_one_year_line, grid_color, grid_linewidth,
    x_right_padding, y_top_padding, optimistic_pessimistic_intercept,
    multipliers=None, show_multiplier_label=False,
):
    """Place crossings and multiplier labels on the corrected runtime curves."""
    series_colors = set(_base.RUNTIME_COLORS.values())
    for line in list(ax.lines):
        if line not in previous_lines and line.get_color() not in series_colors:
            line.remove()
    lines = {
        key: [line for line in ax.lines if line not in previous_lines and to_rgba(line.get_color()) == to_rgba(_base.RUNTIME_COLORS[key])]
        for key in _base.RUNTIME_SERIES
    }
    n_vals = np.asarray(lines["uniform_classical"][0].get_xdata())
    curves = {
        "uniform_classical": lines["uniform_classical"][0].get_ydata(),
        "layden_quantum_min": lines["layden_quantum"][0].get_ydata(),
        "layden_quantum_max": lines["layden_quantum"][-1].get_ydata(),
    }
    intercept_ticks = _base._integer_intercept_ticks(n_vals, curves, optimistic_pessimistic_intercept)
    _base._finish_runtime_axis(
        ax, n_vals, title, r"$n$", "Runtime", show_regime_separator,
        regime_separator_n, show_one_year_line, grid_color, grid_linewidth,
        x_right_padding, y_top_padding, intercept_ticks,
    )
    ax.xaxis.labelpad = xlabel_labelpad
    if multipliers is not None and show_multiplier_label:
        for key, fraction in (("uniform_quantum", 0.76), ("layden_quantum", 0.58)):
            for multiplier, line in zip(multipliers, lines[key]):
                _base._add_multiplier_label(ax, n_vals, line.get_ydata(), multiplier, line.get_color(), fraction, line.get_alpha())
