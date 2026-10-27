"""Inspect the numerical inputs and resource model behind a runtime estimate."""

import hashlib
import inspect
from pathlib import Path

import numpy as np

from monaqa2.data import runtime
from monaqa2.data.utils_stats import grouped_statistics, fit_exponential_from_stats


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function_info(function):
    path = inspect.getsourcefile(function)
    return {"name": function.__name__, "signature": str(inspect.signature(function)),
            "file": str(Path(path).resolve()) if path else None,
            "file_sha256": file_sha256(path) if path else None}


def get_plot_runtime_provenance(plot_function):
    """Inspect the runtime functions actually bound to a notebook's plot API.

    Follow the extra-module wrappers to the base plotting module. This checks
    loaded function bindings, rather than assuming an edited file was reloaded.
    """
    namespace = plot_function.__globals__
    seen = set()
    while "_base" in namespace and id(namespace["_base"]) not in seen:
        module = namespace["_base"]
        seen.add(id(module))
        namespace = vars(module)
    result = {"plot": _function_info(plot_function)}
    for proposal in ("uniform", "qemc"):
        function = namespace[f"get_annealing_time_quantum_walk_{proposal}"]
        resource = function.__globals__[f"quantum_walk_{proposal}_circuit"]
        result[proposal] = {"annealing": _function_info(function), "resources": _function_info(resource)}
    return result


def fitted_gap_path(table, n, betas, *, proposal="uniform", statistic="mean+std", n_fit_min=5, n_fit_max=10):
    """Reproduce the base plot's gap fits directly from fresh raw table rows.

    Use the supplied package's statistics/fitting functions; no cached gap
    statistics or instance-cost correction is used. Return gaps and fit details.
    """
    selected = table[(table["proposal"] == proposal) & (table["a"] == np.inf)]
    stats = grouped_statistics(selected, value_col="delta", statistic=statistic, positive_only=False)
    gaps, fits = [], []
    for beta in betas:
        amplitude, slope = fit_exponential_from_stats(stats, beta, n_fit_min, n_fit_max, sign=-1)
        effective_slope = max(float(slope), 0.0)
        gap = float(amplitude*np.exp(-effective_slope*n))
        gaps.append(gap)
        fits.append({"beta": float(beta), "fit_amplitude": float(amplitude),
                     "fit_slope": float(slope), "effective_fit_slope": effective_slope})
    return gaps, fits


def get_quantum_runtime_diagnostic(
    n, epsilon, betas, spectral_gaps, physical_operation_time=20e-9,
    physical_measurement_time=200e-9, physical_error_rate=1e-3, *,
    proposal="uniform", arithmetic_type="FIXED_POINT", num_trotter_steps=50,
    k_randomized_grid=0, runtime_module=None,
):
    """Trace a complete gap-based runtime using actual public resource functions.

    Return summary, per-filter steps, and function provenance. The public
    annealing runtime is checked against an independently assembled spacetime
    volume and surface-code timing calculation. runtime_module permits an
    explicitly loaded original runtime for comparisons without monkeypatching.
    """
    rt = runtime if runtime_module is None else runtime_module
    if proposal not in ("uniform", "layden"):
        raise ValueError("proposal must be 'uniform' or 'layden'.")
    if len(betas) != len(spectral_gaps) or not len(betas):
        raise ValueError("Provide equally sized nonempty beta and gap paths.")
    betas = [float(value) for value in betas]
    if np.any(np.asarray(betas) <= 0) or np.any(np.diff(betas) <= 0):
        raise ValueError("Diagnostic beta paths must be positive and strictly increasing.")
    name = "uniform" if proposal == "uniform" else "qemc"
    resource = getattr(rt, f"quantum_walk_{name}_circuit")
    annealing = getattr(rt, f"get_annealing_time_quantum_walk_{name}")
    resource_options = {"arithmetic_type": arithmetic_type}
    options = {"arithmetic_type": arithmetic_type}
    if proposal == "layden":
        resource_options["num_trotter_steps"] = options["num_trotter_steps"] = num_trotter_steps
        if "k_randomized_grid" in inspect.signature(resource).parameters:
            resource_options["k_randomized_grid"] = options["k_randomized_grid"] = k_randomized_grid
        elif k_randomized_grid != 0:
            raise ValueError("This original runtime supports only a fixed proposal: use k_randomized_grid=0.")
    budget = rt.get_quantum_annealing_error_budget(epsilon, spectral_gaps)
    steps, logical_depth, logical_qubits = [], 0.0, 0.0
    for index, (beta, gap, queries) in enumerate(zip(betas, spectral_gaps, budget["queries_per_step"])):
        depth, qubits = resource(n, budget["eps_W"], beta, **resource_options)
        contribution = float(queries*depth)
        logical_depth += contribution
        logical_qubits = max(logical_qubits, float(qubits))
        steps.append({"step": index, "beta": beta, "gap": float(gap),
                      "inverse_phase_gap": float(rt.phase_gap_factor(gap)),
                      "filter_degree": float(budget["degree_filters"][index]),
                      "queries": float(queries), "walk_depth": float(depth),
                      "walk_qubits": float(qubits), "logical_depth_contribution": contribution})
    spacetime_volume = logical_depth*logical_qubits
    distance = rt.rotated_surface_code_distance(spacetime_volume, physical_error_rate, budget["eps_SF"])
    layer_time = distance*(4*physical_operation_time+physical_measurement_time)
    reconstructed = logical_depth*layer_time
    reported = float(annealing(n, epsilon, betas, spectral_gaps, physical_operation_time,
                              physical_measurement_time, physical_error_rate, **options))
    if not np.isclose(reported, reconstructed, rtol=2e-12, atol=0):
        raise ValueError("Public annealing runtime disagrees with the diagnostic's resource reconstruction.")
    W = rt._calculate_auxiliary_quantum_circuit_vars(n, budget["eps_W"], betas[-1])[2]
    summary = {"n": int(n), "proposal": proposal, "arithmetic_type": arithmetic_type,
               "k_randomized_grid": int(k_randomized_grid), "num_trotter_steps": int(num_trotter_steps),
               "filters": len(betas), "queries_total": float(budget["total_queries"]),
               "eps_W": float(budget["eps_W"]), "eps_SF": float(budget["eps_SF"]),
               "word_bits": int(W), "logical_depth": logical_depth, "logical_qubits": logical_qubits,
               "spacetime_volume": spacetime_volume, "code_distance": int(distance),
               "physical_operation_time": physical_operation_time,
               "physical_measurement_time": physical_measurement_time,
               "physical_error_rate": physical_error_rate, "layer_time": float(layer_time),
               "runtime_seconds": reported}
    return {"summary": summary, "steps": steps,
            "provenance": {"annealing": _function_info(annealing), "resources": _function_info(resource)}}
