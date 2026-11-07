"""Analytic runtime estimates using ||H||_1 = n**1.5 / sqrt(pi).

The fixed-point model uses the qiskit2 logical non-Clifford bounds from notebook 3.3.
Without individual coefficients, W is a conservative signed-word width.
Arbitrary Rz rotations count as one layer; routing, synthesis and factory
throughput are outside this logical-depth model.
"""

import numpy as np

ALLOWED_DEVICES = {"cpu", "gpu", "fpga"}
DEFAULT_TROTTER_STEPS = 20


def _validate_sizes(n):
    """Validate scalar or array sizes; continuous sizes support scaling curves."""
    values = np.asarray(n, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 1):
        raise ValueError("n must be finite and at least one.")
    return values


def _validate_probability(value, name, *, include_one=False):
    if not np.isfinite(value) or not 0 < value <= 1 or (value == 1 and not include_one):
        endpoint = "<=" if include_one else "<"
        raise ValueError(f"{name} must satisfy 0 < {name} {endpoint} 1.")


def _validate_trotter_steps(num_trotter_steps):
    if isinstance(num_trotter_steps, (bool, np.bool_)) or not isinstance(num_trotter_steps, (int, np.integer)) or num_trotter_steps < 1:
        raise ValueError("num_trotter_steps must be a positive integer.")
    return int(num_trotter_steps)


def _total_queries(vec_queries):
    values = np.asarray(vec_queries, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Query counts must be a finite, nonnegative vector.")
    total = float(np.sum(values))
    if not np.isfinite(total):
        raise ValueError("Total query count exceeds floating-point range.")
    return total


def _validate_randomized_grid(k_randomized_grid: int) -> int:
    """Return a non-negative integer grid level; zero selects a fixed proposal."""
    if isinstance(k_randomized_grid, (bool, np.bool_)) or not isinstance(k_randomized_grid, (int, np.integer)) or k_randomized_grid < 0:
        raise ValueError("k_randomized_grid must be a non-negative integer.")
    return int(k_randomized_grid)


def cpu_local_step(n: int | float) -> float:
    """Return the fitted CPU time for one local classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one local update.
    """
    # local/op: 5.959e-09 + 1.429e-10 n
    return 5.959e-09 + 1.429e-10 * _validate_sizes(n)


def cpu_uniform_step(n: int | float) -> float:
    """Return the fitted CPU time for one uniform classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one uniform update.
    """
    # uniform/op: 0.000e+00 + 1.173e-08 n + 6.964e-11 n^2
    n = _validate_sizes(n)
    return 1.173e-08 * n + 6.964e-11 * n**2


def gpu_local_step(n: int | float) -> float:
    """Return the fitted GPU time for one local classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one local update.
    """
    # local/op: 7.837e-07 + 1.459e-09 n
    return 7.837e-07 + 1.459e-09 * _validate_sizes(n)


def gpu_uniform_step(n: int | float) -> float:
    """Return the fitted GPU time for one uniform classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one uniform update.
    """
    # uniform/op: 0.000e+00 + 0.000e+00 n + 2.215e-10 n^2
    return 2.215e-10 * _validate_sizes(n)**2


def fpga_local_step(n: int | float) -> float:
    """Return the fitted FPGA time for one local classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one local update.
    """
    return 2.715000e-07 + 6.000000e-09 * np.log2(_validate_sizes(n))


def fpga_uniform_step(n: int | float) -> float:
    """Return the fitted FPGA time for one uniform classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one uniform update.
    """
    return 2.004000e-07 + 2.880000e-08 * np.log2(_validate_sizes(n))


def classical_step_time(n, proposal: str, device: str = "cpu"):
    """Return the shared CPU/GPU/FPGA timing fit in seconds, also for arrays."""
    steps = {
        "local": {"cpu": cpu_local_step, "gpu": gpu_local_step, "fpga": fpga_local_step},
        "uniform": {"cpu": cpu_uniform_step, "gpu": gpu_uniform_step, "fpga": fpga_uniform_step},
    }
    if proposal not in steps or device not in ALLOWED_DEVICES:
        raise ValueError("Use proposal='local' or 'uniform' and device='cpu', 'gpu' or 'fpga'.")
    return steps[proposal][device](n)


def split_quantum_error_budget(eps_TV: float) -> tuple[float, float, float, float]:
    """Split the target total-variation error into four resource budgets.

    :param eps_TV: Desired final total-variation distance error.
    :return: Tuple ``(eps_SF, eps_MS, eps_FLT, eps_W_budget)`` for surface-code failures, magic-state failures, spectral-filter approximation, and walk-implementation error.
    """
    _validate_probability(eps_TV, "eps_TV")
    eps_SF = eps_TV / 4.0
    eps_MS = eps_TV / 4.0
    eps_FLT = eps_TV / 4.0
    eps_W_budget = eps_TV / 4.0
    return eps_SF, eps_MS, eps_FLT, eps_W_budget


def rotated_surface_code_distance(spacetime_volume: int | float, physical_error_rate: float, eps_SF: float) -> int:
    """Return the smallest odd rotated-surface-code distance satisfying the surface-code budget.

    :param spacetime_volume: Logical spacetime volume, equal to logical time times logical space.
    :param physical_error_rate: Physical Clifford error rate.
    :param eps_SF: Error budget assigned to surface-code logical failures.
    :return: Smallest odd code distance ``d >= 3`` satisfying the error model.
    """
    if not np.all(np.isfinite([spacetime_volume, physical_error_rate])) or spacetime_volume < 0:
        raise ValueError("Require finite spacetime_volume >= 0 and physical_error_rate.")
    _validate_probability(eps_SF, "eps_SF")
    if not 0 <= physical_error_rate < .01:
        raise ValueError("The surface-code model requires 0 <= physical_error_rate < .01.")

    if spacetime_volume == 0 or physical_error_rate == 0:
        return 3
    # For odd d, t=(d+1)/2 is an integer. Solve V*0.03*(100*p)**t <= eps
    # in logarithms, avoiding underflow and an arbitrary search limit.
    log_ratio = np.log(np.longdouble(100) * np.longdouble(physical_error_rate))
    log_prefactor = np.log(np.longdouble(spacetime_volume)) + np.log(np.longdouble("0.03"))
    log_budget = np.log(np.longdouble(eps_SF))
    exponent = max(2, int(np.ceil((log_budget - log_prefactor) / log_ratio)))
    # Correct possible rounding at an integer boundary and check minimality.
    while exponent > 2 and log_prefactor + (exponent - 1) * log_ratio <= log_budget:
        exponent -= 1
    while log_prefactor + exponent * log_ratio > log_budget:
        exponent += 1
    return 2 * exponent - 1


def rotated_surface_code_time(logical_time: int | float, logical_space: int | float, physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, eps_SF: float) -> float:
    """Return the physical runtime of a logical computation under the rotated-surface-code model.

    :param logical_time: Logical depth in tile-time steps.
    :param logical_space: Logical space in surface-code tiles.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param eps_SF: Error budget assigned to surface-code logical failures.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    if not np.all(np.isfinite([logical_time, logical_space, physical_operation_time, physical_measurement_time])) or min(logical_time, logical_space, physical_operation_time, physical_measurement_time) < 0:
        raise ValueError("Logical resources and physical operation times must be finite and nonnegative.")
    spacetime_volume = logical_time * logical_space
    d = rotated_surface_code_distance(spacetime_volume, physical_error_rate, eps_SF)
    logical_cycle_time = d * (4 * physical_operation_time + physical_measurement_time)
    return logical_time * logical_cycle_time


def _ceil_log2(x: int | float) -> int:
    """Return ``ceil(log2(x))`` for a positive argument."""
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"Expected a finite positive value, got {x}.")
    return int(np.ceil(np.log2(x)))


def _calculate_auxiliary_quantum_circuit_vars(n: int | float, eps_W: float, beta: float) -> tuple[int, float, int, int, int, int]:
    """Return (F, M, W, ell_W, Wallace rounds, ell_n) for the rotation coin.

    Use ||H||_1 = n**1.5/sqrt(pi) and pass eps_W as epsilon_b in the
    coefficient-precision rule F=max(0,ceil(log2(beta*||H||_1*M/(2*eps_W)))).
    This convention estimates coefficient precision, not synthesis/Trotter error.

    Coefficients c are rounded as q=round(c*2**F/(2*||H||_1)). Therefore
    sum(abs(q)) <= 2**(F-1) + M/2. W conservatively fits both signs of that
    integer sum. For F=0 every q is zero (round-to-nearest-even), so W=2.
    Noninteger n is retained for analytic scaling curves.
    """
    n = float(n)
    beta = float(beta)
    eps_W = float(eps_W)

    if not np.isfinite(n) or n < 1.0:
        raise ValueError(f"Invalid n={n}. Expected n >= 1.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"Invalid beta={beta}. Expected beta >= 0.")
    if not np.isfinite(eps_W) or not (0.0 < eps_W < 1.0):
        raise ValueError(f"Invalid eps_W={eps_W}. Expected 0 < eps_W < 1.")

    M = n * (n + 1) / 2
    F = 0
    if beta:
        # Logarithms avoid overflowing the product beta*||H||_1*M/eps_W.
        log_norm_1 = 1.5 * np.log2(n) - 0.5 * np.log2(np.pi)
        F = max(0, int(np.ceil(np.log2(beta) + log_norm_1 + np.log2(M) - 1 - np.log2(eps_W))))
    rounded_bound = (1 << (F - 1)) + int(np.ceil(M)) // 2 if F else 0
    W = max(2, rounded_bound.bit_length() + 1)
    rounds = int(np.ceil(np.log(M) / np.log(1.5)))
    return F, M, W, _ceil_log2(W), rounds, _ceil_log2(n)


def fixed_point_arithmetic_circuit(n: int | float, eps_W: float, beta: float) -> tuple[float, float]:
    """Return upper bounds on BoltzmannCoin depth and logical qubits.

    Parallel controlled Ry rotations contribute at most two layers. Energy
    and positive-part arithmetic are computed and uncomputed. Loader and
    Wallace/prefix scratch are reused; W-1 coin qubits remain persistent.
    """
    _, M, W, ell_W, rounds, _ = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)
    boltz_depth = 36 * rounds + 24 * ell_W + 44
    loader_work = 2 * n * (n - 1) + 2 * M * (W + 1)
    prefix_work = (2 * ell_W + 3) * W
    adder_work = 0 if M == 1 else (M - 2) * (2 * W - 1) + 2 * W + prefix_work
    boltz_qubits = 2 * n + (M + 3) * W - 1 + max(loader_work, adder_work)
    return float(boltz_depth), float(boltz_qubits)


def _fixed_point_walk_blocks(n: int | float, eps_W: float, beta: float, extra: int | float = 0) -> tuple[float, float, float, float, float, float]:
    """Return fixed-point coin, reflection, and accept-path resources."""
    _, _, W, _, _, _ = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)
    boltz_depth, boltz_qubits = fixed_point_arithmetic_circuit(n, eps_W, beta)
    coins = W - 1
    reflection_depth = 12 * _ceil_log2(n + coins + extra) - 6
    reflection_qubits = 3 * n + 2 * coins + 2 * extra - 1
    accept_depth = 12 * _ceil_log2(coins) + 3
    accept_qubits = 2 * n + coins + 1 + max(coins - 2, n - 1)

    return boltz_depth, boltz_qubits, reflection_depth, reflection_qubits, accept_depth, accept_qubits


def _quantum_walk_circuit(n: int | float, eps_W: float, beta: float, proposal_depth: int | float, proposal_qubits: int | float) -> tuple[float, float]:
    """Combine the proposal with the fixed-point coin and walk reflections."""
    extra = proposal_qubits - 2 * n
    boltz_depth, boltz_qubits, reflection_depth, reflection_qubits, accept_depth, accept_qubits = _fixed_point_walk_blocks(n, eps_W, beta, extra)
    logical_depth = 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth
    # Proposal labels/copies coexist with the shared coin/accept/reflection work.
    logical_qubits = max(boltz_qubits + extra, reflection_qubits, accept_qubits + extra)
    return float(logical_depth), float(logical_qubits)


def quantum_walk_local_circuit(n: int | float, eps_W: float, beta: float) -> tuple[float, float]:
    """Return resources for one local-proposal Szegedy walk application.

    :param n: Number of spins.
    :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :return: Tuple ``(logical_depth, logical_qubits)`` for one walk application.
    """
    proposal_depth = 2 * _ceil_log2(n)
    proposal_qubits = 2 * n
    return _quantum_walk_circuit(n, eps_W, beta, proposal_depth, proposal_qubits)


def quantum_walk_uniform_circuit(n: int | float, eps_W: float, beta: float) -> tuple[float, float]:
    """Return resources for one uniform-proposal Szegedy walk application.

    :param n: Number of spins.
    :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :return: Tuple ``(logical_depth, logical_qubits)`` for one walk application.
    """
    proposal_depth = 0
    proposal_qubits = 2 * n
    return _quantum_walk_circuit(n, eps_W, beta, proposal_depth, proposal_qubits)


def quantum_walk_qemc_circuit(n: int | float, eps_W: float, beta: float, num_trotter_steps: int = DEFAULT_TROTTER_STEPS, k_randomized_grid: int = 2) -> tuple[float, float]:
    """Return resources for one Hamiltonian-simulation-proposal Szegedy walk.

    The randomized proposal implements H_tf = (1-gamma) H - gamma sum_i X_i.
    Both Ising and mixer angles contain products of the time and gamma bits.
    Their expansion has one ordinary, 2*k singly controlled, and k**2 doubly
    controlled rotations per Pauli term, at depths 1, 2, and 8 respectively.
    A single layer costs at most 1+4*k+8*k**2; merged X half-steps give
    (1+4*k+8*k**2)*(r*(n+2)+1). Default k=2 therefore gives factor 41.

    :param n: Number of spins.
    :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :param num_trotter_steps: Number of Trotter steps used by the proposal.
    :param k_randomized_grid: Bits per parameter, default 2 (a 4 by 4 grid); 0 uses fixed parameters.
    :return: Tuple ``(logical_depth, logical_qubits)`` for one walk application.
    """
    r = _validate_trotter_steps(num_trotter_steps)
    k = _validate_randomized_grid(k_randomized_grid)

    proposal_depth = (1 + 4 * k + 8 * k**2) * (r * (n + 2) + 1)
    proposal_qubits = 2 * n * (k + 1)

    return _quantum_walk_circuit(n, eps_W, beta, proposal_depth, proposal_qubits)


def get_time_direct_enumeration(n: int):
    """Return the fitted CPU direct-enumeration runtime in seconds."""
    n = _validate_sizes(n)
    return 6.323731e-10 * (n**2) * (2**n)


def get_annealing_time_classical_walk_local(n: int | float, vec_queries: list[float], device: str = "cpu") -> float:
    """Return the classical annealing runtime for local Metropolis updates.

    :param n: Number of spins.
    :param vec_queries: Number of classical update queries per annealing step.
    :param device: Classical device model, one of ``"cpu"``, ``"gpu"``, or ``"fpga"``.
    :return: Total runtime in seconds.
    """
    return _total_queries(vec_queries) * classical_step_time(n, "local", device)


def get_annealing_time_classical_walk_uniform(n: int | float, vec_queries: list[float], device: str = "cpu") -> float:
    """Return the classical annealing runtime for uniform Metropolis updates.

    :param n: Number of spins.
    :param vec_queries: Number of classical update queries per annealing step.
    :param device: Classical device model, one of ``"cpu"``, ``"gpu"``, or ``"fpga"``.
    :return: Total runtime in seconds.
    """
    return _total_queries(vec_queries) * classical_step_time(n, "uniform", device)


def _classical_qemc_resources(n, vec_queries, num_trotter_steps):
    """Classically sampled parameters need n qubits and no coherent controls."""
    n = float(_validate_sizes(n))
    r = _validate_trotter_steps(num_trotter_steps)
    return _total_queries(vec_queries) * (r * (n + 2) + 1), n


def get_annealing_time_classical_walk_qemc(n: int | float, vec_queries: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, eps_SF: float, num_trotter_steps: int = DEFAULT_TROTTER_STEPS) -> float:
    """Return the surface-code runtime for a QEMC proposal used inside a classical annealing schedule.

    :param n: Number of spins.
    :param vec_queries: Number of QEMC proposal queries per annealing step.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param eps_SF: Error budget assigned to surface-code logical failures.
    :param num_trotter_steps: Number of Trotter steps used by the QEMC proposal.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    logical_time, logical_space = _classical_qemc_resources(n, vec_queries, num_trotter_steps)
    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate, eps_SF)


def phase_gap_factor(spectral_gap: float | str | np.longdouble) -> np.longdouble:
    """Return ``1 / arccos(1 - delta)``, computed stably.

    :param spectral_gap: Classical spectral gap ``delta``.
    :return: Inverse Szegedy phase gap.
    """
    g = np.longdouble(spectral_gap)
    if not np.isfinite(g) or g <= np.longdouble(0) or g > np.longdouble(1):
        raise ValueError(f"Invalid spectral_gap={spectral_gap}. Expected 0 < spectral_gap <= 1.")
    return np.longdouble(1) / (
        np.longdouble(2) * np.arcsin(np.sqrt(g / np.longdouble(2)))
    )

def spectral_filter_polynomial_degree(spectral_gap: float, eps_filter: float) -> float:
    """Return the polynomial degree for one QSVT spectral filter.

    :param spectral_gap: Spectral gap of the Markov-chain discriminant matrix at the current annealing step.
    :param eps_filter: Leakage budget assigned to this single spectral filter.
    :return: Degree of the QSVT spectral-filter polynomial.
    """
    if not np.isfinite(eps_filter) or not 0 < eps_filter < 1:
        raise ValueError("eps_filter must satisfy 0 < eps_filter < 1.")
    return -2 * np.log2(eps_filter) * phase_gap_factor(spectral_gap)


def spectral_filter_polynomial_degree_list(spectral_gaps: list[float], eps_FLT: float, overlap: float = 1.0 / np.e) -> list[float]:
    """Return the QSVT spectral-filter degree for each annealing step.

    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param eps_FLT: Total error budget assigned to all spectral-filter polynomial approximations.
    :param overlap: Lower bound on the squared overlap between consecutive annealing states.
    :return: List of polynomial degrees, one for each spectral gap.
    """
    if not np.isfinite(overlap) or not 0 < overlap <= 1:
        raise ValueError("overlap must satisfy 0 < overlap <= 1.")
    _validate_probability(eps_FLT, "eps_FLT")
    num_filters = len(spectral_gaps)
    if num_filters == 0:
        return []
    num_zeno_filters = num_filters * (1.0 + 1.0 / overlap)
    eps_filter = eps_FLT / num_zeno_filters
    return [spectral_filter_polynomial_degree(spectral_gap, eps_filter) for spectral_gap in spectral_gaps]


def get_scheduled_filter_queries(degree_filters: list[float], overlap: float = 1.0 / np.e) -> list[float]:
    """Return the expected number of walk queries per annealing step after Zeno-rewind repetitions.

    :param degree_filters: Degree of each unique QSVT spectral filter.
    :param overlap: Lower bound on the squared overlap between consecutive annealing states.
    :return: Expected number of calls to ``W`` or ``W-dagger`` contributed by each annealing step.
    """
    if not np.isfinite(overlap) or not 0 < overlap <= 1:
        raise ValueError("overlap must satisfy 0 < overlap <= 1.")
    zeno_overhead = 1.0 + 1.0 / overlap
    _total_queries(degree_filters)
    return [zeno_overhead * degree for degree in degree_filters]


def get_quantum_annealing_error_budget(eps_TV: float, spectral_gaps: list[float], zeno_overlap_probability: float = 1.0 / np.e) -> dict[str, float | list[float]]:
    """Return the error budget and query counts for the quantum annealing schedule.

    :param eps_TV: Desired final total-variation distance error.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :return: Dictionary containing the four error budgets, QSVT degrees, expected queries per step, total expected queries, and per-walk error ``eps_W``.
    """
    eps_SF, eps_MS, eps_FLT, eps_W_budget = split_quantum_error_budget(eps_TV)
    degree_filters = spectral_filter_polynomial_degree_list(spectral_gaps, eps_FLT, zeno_overlap_probability)
    queries_per_step = get_scheduled_filter_queries(degree_filters, zeno_overlap_probability)
    total_queries = _total_queries(queries_per_step)
    eps_W = eps_W_budget / total_queries if total_queries else eps_W_budget
    if eps_W == 0:
        raise ValueError("Per-walk precision is below floating-point range.")
    return {
        "eps_TV": float(eps_TV),
        "eps_SF": eps_SF,
        "eps_MS": eps_MS,
        "eps_FLT": eps_FLT,
        "eps_W_budget": eps_W_budget,
        "eps_W": eps_W,
        "degree_filters": degree_filters,
        "queries_per_step": queries_per_step,
        "total_queries": total_queries,
    }


def _quantum_annealing_resources(n, eps_TV, betas, spectral_gaps, circuit_fn, zeno_overlap_probability=1.0 / np.e):
    """Return full-path depth, peak logical space and the surface-code budget.

    Runtime and code-distance plots use this same calculation. The volume is
    total depth times peak space, rather than a separately rounded per-step sum.
    """
    n = float(_validate_sizes(n))
    temperatures = np.asarray(betas, dtype=float)
    if temperatures.ndim != 1 or len(temperatures) != len(spectral_gaps):
        raise ValueError("betas and spectral_gaps must be equal-length vectors.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0) or np.any(np.diff(temperatures) <= 0):
        raise ValueError("betas must be finite, nonnegative and strictly increasing.")
    budget = get_quantum_annealing_error_budget(eps_TV, spectral_gaps, zeno_overlap_probability)
    logical_time, logical_space = 0.0, 0.0
    for beta, queries in zip(temperatures, budget["queries_per_step"]):
        depth, space = circuit_fn(n, budget["eps_W"], beta)
        if not np.all(np.isfinite([depth, space])) or min(depth, space) <= 0:
            raise ValueError("Walk depth and space must be finite and positive.")
        logical_time += queries * depth
        logical_space = max(logical_space, space)
    return logical_time, logical_space, budget["eps_SF"]


def _get_annealing_time_quantum_walk(n: int | float, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, circuit_fn, zeno_overlap_probability: float = 1.0 / np.e) -> float:
    """Return the surface-code runtime for a quantum annealing schedule.

    :param n: Number of spins.
    :param eps_TV: Desired final total-variation distance error.
    :param betas: Inverse temperatures along the annealing schedule.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param circuit_fn: Function returning one-walk logical resources for a selected proposal rule.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    logical_time, logical_space, eps_SF = _quantum_annealing_resources(n, eps_TV, betas, spectral_gaps, circuit_fn, zeno_overlap_probability)
    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate, eps_SF)


def get_annealing_time_quantum_walk_local(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, *, zeno_overlap_probability: float = 1.0 / np.e) -> float:
    """Return the surface-code runtime for quantum annealing with local-proposal Szegedy walks.

    :param n: Number of spins.
    :param eps_TV: Desired final total-variation distance error.
    :param betas: Inverse temperatures along the annealing schedule.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, quantum_walk_local_circuit, zeno_overlap_probability)


def get_annealing_time_quantum_walk_uniform(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, *, zeno_overlap_probability: float = 1.0 / np.e) -> float:
    """Return the surface-code runtime for quantum annealing with uniform-proposal Szegedy walks.

    :param n: Number of spins.
    :param eps_TV: Desired final total-variation distance error.
    :param betas: Inverse temperatures along the annealing schedule.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, quantum_walk_uniform_circuit, zeno_overlap_probability)


def get_annealing_time_quantum_walk_qemc(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, *, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = DEFAULT_TROTTER_STEPS, k_randomized_grid: int = 2) -> float:
    """Return the surface-code runtime for quantum annealing with QEMC-proposal Szegedy walks.

    :param n: Number of spins.
    :param eps_TV: Desired final total-variation distance error.
    :param betas: Inverse temperatures along the annealing schedule.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :param num_trotter_steps: Number of Trotter steps used by the QEMC proposal.
    :param k_randomized_grid: Non-negative grid level; 0 uses one fixed (t, gamma), positive k uses a 2**k by 2**k grid. Default 2.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    k_randomized_grid = _validate_randomized_grid(k_randomized_grid)
    num_trotter_steps = _validate_trotter_steps(num_trotter_steps)

    def circuit_fn(n_local: int | float, eps_W: float, beta: float) -> tuple[float, float]:
        """Wrap the QEMC circuit resource function with a fixed Trotter-step count.

        :param n_local: Number of spins.
        :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
        :param beta: Inverse temperature for the current annealing step.
        :return: Tuple ``(logical_depth, logical_qubits)`` for one QEMC walk application.
        """
        return quantum_walk_qemc_circuit(n_local, eps_W, beta, num_trotter_steps, k_randomized_grid=k_randomized_grid)

    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, circuit_fn, zeno_overlap_probability)


def get_one_step_quantum_walk_queries(n: int | float, eps_TV: float, spectral_gap: float, overlap: float = 1.0 / np.e) -> float:
    """Return the expected walk queries for one warm-start spectral-filter step.

    This is the one-step analogue of ``get_annealing_queries_quantum_walks``.
    It reuses the same error split and spectral-filter degree formula, but uses
    the expected number of direct attempts ``1 / overlap`` rather than the
    schedule-level Zeno-rewind overhead ``1 + 1 / overlap``.

    :param n: Number of spins; currently unused but kept for API compatibility.
    :param eps_TV: Desired total-variation distance error.
    :param spectral_gap: Spectral gap of the Markov-chain discriminant matrix.
    :param overlap: Lower bound on the squared overlap between the input and target coherent Gibbs states.
    :return: Expected total number of calls to ``W`` or ``W-dagger``.
    """
    if not np.isfinite(overlap) or not 0 < overlap <= 1:
        raise ValueError("overlap must satisfy 0 < overlap <= 1.")
    _, _, eps_FLT, _ = split_quantum_error_budget(eps_TV)
    expected_attempts = 1.0 / overlap
    eps_filter = eps_FLT / expected_attempts
    degree_filter = spectral_filter_polynomial_degree(spectral_gap, eps_filter)
    return float(expected_attempts * degree_filter)


def get_annealing_queries_quantum_walks(n: int | float, eps_TV: float, spectral_gaps: list[float], zeno_overlap_probability: float = 1.0 / np.e) -> float:
    """Return the expected number of walk queries in the Zeno-rewind quantum annealing schedule.

    :param n: Number of spins; currently unused but kept for API compatibility.
    :param eps_TV: Desired final total-variation distance error.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :return: Expected total number of calls to ``W`` or ``W-dagger``.
    """
    budget = get_quantum_annealing_error_budget(eps_TV, spectral_gaps, zeno_overlap_probability)
    return budget["total_queries"]


def tight_schedule_annealing(n: int | float, beta: float) -> list[float]:
    """Return the variance-fit annealing grid, capped exactly at the final beta.

    Steps use the fitted SK energy-variance density and scale as 1/sqrt(n).
    This is an empirical schedule, not an instance-wise overlap certificate.
    """
    n = float(_validate_sizes(n))
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if beta == 0.0:
        return []

    current = 0.0
    schedule = []


    def delta_beta(
        n: int | float,
        beta: float | np.ndarray,
    ) -> float | np.ndarray:
        beta_arr = np.asarray(beta, dtype=float)
        step = np.empty_like(beta_arr)
        low_beta = beta_arr < 1.0
        step[low_beta] = np.exp(
            -0.111 * beta_arr[low_beta]
            + 0.615 * beta_arr[low_beta] ** 2
        )
        step[~low_beta] = (
            0.533 * (beta_arr[~low_beta] + 0.286) ** 1.75
        )
        step /= np.sqrt(float(n))
        return float(step) if beta_arr.ndim == 0 else step
    

    while current < beta:
        step = float(delta_beta(n, current))
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"Invalid annealing step: n={n}, beta={current}, step={step}")

        next_beta = current + step
        if next_beta <= current:
            raise ValueError("Annealing step is below floating-point resolution.")

        if current < 1.0 < min(next_beta, beta):
            next_beta = 1.0

        current = min(next_beta, beta)

        schedule.append(current)

    return schedule


