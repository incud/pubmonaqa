"""Drop-in extension of ``plot_main_runtime`` with optional equal runtime line widths.

All public names from ``plot_main_runtime`` are re-exported unchanged.  The only
behavioral change is in ``plot_annealing_classical_and_quantum_runtime_vs_n``:
when ``same_width_lines=True``, the two quantum runtime-series boundary lines
use the same ``line_width`` as the "Best classical walk" line.
"""

from typing import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from monaqa2.data import plot_main_runtime as _base


# Re-export every public symbol from the original module unchanged.
for _name in dir(_base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_base, _name)

__all__ = [name for name in dir(_base) if not name.startswith("_")]


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
    classical_query_file: Path = _base.CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = _base.SPECTRAL_GAP_FILE,
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
    same_width_lines: bool = False,
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
    """Plot APS single-column annealing runtime estimates versus n.

    ``same_width_lines=False`` preserves the original behavior: the quantum
    band boundary lines use ``0.55 * line_width`` while the best-classical line
    uses ``line_width``.

    If ``same_width_lines=True``, all three runtime series are drawn with the
    same width as the best-classical line, i.e. ``line_width``.  Both optimistic
    and pessimistic boundaries of each quantum band are updated consistently.
    """
    fig, ax = _base.plot_annealing_classical_and_quantum_runtime_vs_n(
        beta=beta,
        epsilon=epsilon,
        annealing_schedule_generator=annealing_schedule_generator,
        classical_device=classical_device,
        physical_error_rate_min=physical_error_rate_min,
        physical_error_rate_max=physical_error_rate_max,
        physical_operation_time_min=physical_operation_time_min,
        physical_operation_time_max=physical_operation_time_max,
        physical_measurement_time_min=physical_measurement_time_min,
        physical_measurement_time_max=physical_measurement_time_max,
        num_trotter_steps=num_trotter_steps,
        classical_query_file=classical_query_file,
        spectral_gap_file=spectral_gap_file,
        statistic=statistic,
        n_fit_min=n_fit_min,
        n_fit_max=n_fit_max,
        n_plot_min=n_plot_min,
        n_plot_max=n_plot_max,
        fig=fig,
        ax=ax,
        title=title,
        show_legend=show_legend,
        legend_placement=legend_placement,
        legend_y_shift=legend_y_shift,
        xlabel_labelpad=xlabel_labelpad,
        line_width=line_width,
        line_alpha=line_alpha,
        band_alpha=band_alpha,
        grid_color=grid_color,
        grid_linewidth=grid_linewidth,
        x_right_padding=x_right_padding,
        show_regime_separator=show_regime_separator,
        regime_separator_n=regime_separator_n,
        show_one_year_line=show_one_year_line,
        show_direct_enumeration_line=show_direct_enumeration_line,
        y_top_padding=y_top_padding,
        optimistic_pessimistic_intercept=optimistic_pessimistic_intercept,
        debug=debug,
    )

    if same_width_lines:
        quantum_colors = {
            _base.RUNTIME_COLORS["uniform_quantum"],
            _base.RUNTIME_COLORS["layden_quantum"],
        }
        for line in ax.lines:
            if line.get_color() in quantum_colors:
                line.set_linewidth(line_width)

        # Rebuild the legend so its quantum handles reflect the new widths too.
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            _base._add_runtime_legend(
                fig,
                ax,
                handles,
                labels,
                legend_placement,
                legend_y_shift,
            )

    return fig, ax


# Ensure the overridden public function is exported instead of the base one.
globals()["plot_annealing_classical_and_quantum_runtime_vs_n"] = (
    plot_annealing_classical_and_quantum_runtime_vs_n
)
if "plot_annealing_classical_and_quantum_runtime_vs_n" not in __all__:
    __all__.append("plot_annealing_classical_and_quantum_runtime_vs_n")


def __getattr__(name: str):
    """Forward any non-overridden attribute to the original module."""
    return getattr(_base, name)


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
    classical_query_file: Path = _base.CLASSICAL_QUERY_FILE,
    spectral_gap_file: Path = _base.SPECTRAL_GAP_FILE,
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
    same_width_lines: bool = False,
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
    """Plot shadowed annealing runtimes with optional equal line widths.

    ``same_width_lines=False`` preserves the original behavior: every quantum
    multiplier curve uses ``0.55 * line_width`` while the best-classical line
    uses ``line_width``.

    If ``same_width_lines=True``, every line in both quantum shadow families
    uses exactly ``line_width``, matching the best-classical line.  Shaded
    regions are unchanged.
    """
    fig, ax = _base.plot_annealing_classical_and_quantum_runtime_vs_n_shadowed(
        beta=beta,
        epsilon=epsilon,
        annealing_schedule_generator=annealing_schedule_generator,
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
        n_plot_min=n_plot_min,
        n_plot_max=n_plot_max,
        fig=fig,
        ax=ax,
        title=title,
        show_legend=show_legend,
        legend_placement=legend_placement,
        legend_y_shift=legend_y_shift,
        xlabel_labelpad=xlabel_labelpad,
        line_width=line_width,
        line_alpha=line_alpha,
        band_alpha=band_alpha,
        grid_color=grid_color,
        grid_linewidth=grid_linewidth,
        x_right_padding=x_right_padding,
        show_regime_separator=show_regime_separator,
        regime_separator_n=regime_separator_n,
        show_one_year_line=show_one_year_line,
        show_direct_enumeration_line=show_direct_enumeration_line,
        y_top_padding=y_top_padding,
        optimistic_pessimistic_intercept=optimistic_pessimistic_intercept,
        show_multiplier_label=show_multiplier_label,
        debug=debug,
    )

    if same_width_lines:
        quantum_colors = {
            _base.RUNTIME_COLORS["uniform_quantum"],
            _base.RUNTIME_COLORS["layden_quantum"],
        }
        for line in ax.lines:
            if line.get_color() in quantum_colors:
                line.set_linewidth(line_width)

        # Rebuild the legend so its two quantum family handles have the same
        # width as the plotted lines and the best-classical line.
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            _base._add_runtime_legend(
                fig,
                ax,
                handles,
                labels,
                legend_placement,
                legend_y_shift,
            )

    return fig, ax


# Ensure the overridden shadowed function is exported instead of the base one.
globals()["plot_annealing_classical_and_quantum_runtime_vs_n_shadowed"] = (
    plot_annealing_classical_and_quantum_runtime_vs_n_shadowed
)
if "plot_annealing_classical_and_quantum_runtime_vs_n_shadowed" not in __all__:
    __all__.append("plot_annealing_classical_and_quantum_runtime_vs_n_shadowed")
