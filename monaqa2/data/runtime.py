import numpy as np
import mpmath as mp
from monaqa2.data.utils_interpolation_cache import interpolation_cache
import monaqa2.data.filename
from monaqa2.mcmc.search import search_monotone

ALLOWED_DEVICES = {"cpu", "gpu", "fpga"}
ALLOWED_ARITHMETIC_TYPES = {"HYBRID", "FULLY_PHASE"}


def _validate_arithmetic_type(arithmetic_type: str) -> str:
    """Return a validated arithmetic type.

    :param arithmetic_type: Arithmetic implementation name.
    :return: The validated arithmetic implementation name.
    """
    if arithmetic_type not in ALLOWED_ARITHMETIC_TYPES:
        raise ValueError(f"Unknown arithmetic_type={arithmetic_type!r}. Allowed: {ALLOWED_ARITHMETIC_TYPES}")
    return arithmetic_type


def cpu_local_step(n: int | float) -> float:
    """Return the fitted CPU time for one local classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one local update.
    """
    # local/op: 5.959e-09 + 1.429e-10 n
    return 5.959e-09 + 1.429e-10 * n


def cpu_uniform_step(n: int | float) -> float:
    """Return the fitted CPU time for one uniform classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one uniform update.
    """
    # uniform/op: 0.000e+00 + 1.173e-08 n + 6.964e-11 n^2
    return 1.173e-08 * n + 6.964e-11 * n**2


def gpu_local_step(n: int | float) -> float:
    """Return the fitted GPU time for one local classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one local update.
    """
    # local/op: 7.837e-07 + 1.459e-09 n
    return 7.837e-07 + 1.459e-09 * n


def gpu_uniform_step(n: int | float) -> float:
    """Return the fitted GPU time for one uniform classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one uniform update.
    """
    # uniform/op: 0.000e+00 + 0.000e+00 n + 2.215e-10 n^2
    return 2.215e-10 * n**2


def fpga_local_step(n: int | float) -> float:
    """Return the fitted FPGA time for one local classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one local update.
    """
    # 0.267900 + 0.001800 log2(N) [μs, multiply by 10^{-6}]
    return (0.267900 + 0.001800 * np.log2(n)) * 1e-6


def fpga_uniform_step(n: int | float) -> float:
    """Return the fitted FPGA time for one uniform classical Metropolis step.

    :param n: Number of spins.
    :return: Runtime in seconds for one uniform update.
    """
    # 0.254100 + 0.004200 log2(N) [μs, multiply by 10^{-6}]
    return (0.254100 + 0.004200 * np.log2(n)) * 1e-6


def split_quantum_error_budget(eps_TV: float) -> tuple[float, float, float, float]:
    """Split the target total-variation error into four resource budgets.

    :param eps_TV: Desired final total-variation distance error.
    :return: Tuple ``(eps_SF, eps_MS, eps_FLT, eps_W_budget)`` for surface-code failures, magic-state failures, spectral-filter approximation, and walk-implementation error.
    """
    eps_SF = eps_TV / 4.0
    eps_MS = eps_TV / 4.0
    eps_FLT = eps_TV / 4.0
    eps_W_budget = eps_TV / 4.0
    return eps_SF, eps_MS, eps_FLT, eps_W_budget


def rotated_surface_code_distance(spacetime_volume: int | float, physical_error_rate: float, eps_SF: float) -> float:
    """Return the smallest odd rotated-surface-code distance satisfying the surface-code budget.

    :param spacetime_volume: Logical spacetime volume, equal to logical time times logical space.
    :param physical_error_rate: Physical Clifford error rate.
    :param eps_SF: Error budget assigned to surface-code logical failures.
    :return: Smallest odd code distance ``d >= 3`` satisfying the error model.
    """
    def distance(index: int) -> int:
        return 2 * index + 3

    def total_error(index: int) -> float:
        d = distance(index)
        logical_error_per_round = d * 0.1 * (100 * physical_error_rate) ** ((d + 1.0) / 2.0)
        return spacetime_volume * logical_error_per_round

    index = search_monotone(total_error, lambda error: error - eps_SF, 0, 1_000_000, info="rotated_surface_code_distance")
    return float(distance(index))


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
    spacetime_volume = logical_time * logical_space
    d = rotated_surface_code_distance(spacetime_volume, physical_error_rate, eps_SF)
    logical_cycle_time = d * (4 * physical_operation_time + physical_measurement_time)
    return logical_time * logical_cycle_time


def _calculate_auxiliary_quantum_circuit_vars(n: int | float, eps_W: float, beta: float) -> tuple[float, float, float, float, float, float, float, float]:
    """Return auxiliary parameters used by the quantum-walk circuit formulas.

    The hybrid-arithmetic resource formula assumes the active-tail regime ``m >= 3``.
    For smaller ``beta`` or ``n``, the active-tail expression is not valid as written;
    we use the conservative floor ``m = 3`` to keep the resource estimate finite.

    :param n: Number of spins.
    :param eps_W: Error budget for a single implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :return: Tuple ``(ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m)`` used in the resource formulas.
    """
    n = float(n)
    beta = float(beta)
    eps_W = float(eps_W)

    if not np.isfinite(n) or n <= 0.0:
        raise ValueError(f"Invalid n={n}")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"Invalid beta={beta}")
    if not np.isfinite(eps_W) or not (0.0 < eps_W < 1.0):
        raise ValueError(f"Invalid eps_W={eps_W}")

    ell_n = np.log2(n)
    ell_eps = np.log2(1.0 / eps_W)
    ell_2n = np.log2(2.0 * n)
    S = 1.0 + np.log2(n) + ell_eps
    ell_S = np.log2(1.0 + 3.5 * S)
    alpha = 2.0 * n**1.5 / np.sqrt(np.pi)

    m_arg = beta * alpha / (2.0 * np.log(1.0 / eps_W))
    m_raw = np.log2(m_arg) if m_arg > 0.0 else -np.inf
    m = max(3.0, float(m_raw))
    ell_m = np.log2(m)

    return ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m


def quantum_walk_local_circuit(n: int | float, eps_W: float, beta: float, arithmetic_type: str = "HYBRID") -> tuple[float, float]:
    """Return resources for one local-proposal Szegedy walk application.

    :param n: Number of spins.
    :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Tuple ``(logical_depth, logical_qubits)`` for one walk application.
    """
    arithmetic_type = _validate_arithmetic_type(arithmetic_type)
    ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)

    proposal_depth, proposal_qubits = 13 * ell_n + 15, 2 * n

    if arithmetic_type == "HYBRID":
        boltz_depth = 162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12
        boltz_qubits = 2 * n + 4 * n**2 + 21 * n**2 * S + 2
        reflection_depth = 14 * np.log2(n + 7 * S + 3) - 13
        reflection_qubits = 2 * n + 7 * S + 6
        accept_depth = 28 * np.log2(4 + 7 * S) - 23
        accept_qubits = 3 * n + 7 * S + 4
    else:
        d = 2.0 + max(np.sqrt(0.5 * beta * alpha * ell_eps) + ell_eps, alpha * ell_eps)
        controlled_qubitized_depth = 58 * n - 58 + 14 * np.log2(6 * n)
        boltz_depth = 3 * d * controlled_qubitized_depth + 3 * (2 * d + 1)
        boltz_qubits = 10 * n + 2
        reflection_depth = 14 * np.log2(n) - 13
        reflection_qubits = 2 * n + 3
        accept_depth = 3
        accept_qubits = 3 * n + 1

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def quantum_walk_uniform_circuit(n: int | float, eps_W: float, beta: float, arithmetic_type: str = "HYBRID") -> tuple[float, float]:
    """Return resources for one uniform-proposal Szegedy walk application.

    :param n: Number of spins.
    :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Tuple ``(logical_depth, logical_qubits)`` for one walk application.
    """
    arithmetic_type = _validate_arithmetic_type(arithmetic_type)
    ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)

    proposal_depth, proposal_qubits = 0, 2 * n

    if arithmetic_type == "HYBRID":
        boltz_depth = 162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12
        boltz_qubits = 2 * n + 4 * n**2 + 21 * n**2 * S + 2
        reflection_depth = 14 * np.log2(n + 7 * S + 3) - 13
        reflection_qubits = 2 * n + 7 * S + 6
        accept_depth = 28 * np.log2(4 + 7 * S) - 23
        accept_qubits = 3 * n + 7 * S + 4
    else:
        d = 2.0 + max(np.sqrt(0.5 * beta * alpha * ell_eps) + ell_eps, alpha * ell_eps)
        controlled_qubitized_depth = 58 * n - 58 + 14 * np.log2(6 * n)
        boltz_depth = 3 * d * controlled_qubitized_depth + 3 * (2 * d + 1)
        boltz_qubits = 10 * n + 2
        reflection_depth = 14 * np.log2(n) - 13
        reflection_qubits = 2 * n + 3
        accept_depth = 3
        accept_qubits = 3 * n + 1

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def quantum_walk_qemc_circuit(n: int | float, eps_W: float, beta: float, num_trotter_steps: int = 50, arithmetic_type: str = "HYBRID") -> tuple[float, float]:
    """Return resources for one QEMC-proposal Szegedy walk application.

    :param n: Number of spins.
    :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
    :param beta: Inverse temperature for the current annealing step.
    :param num_trotter_steps: Number of Trotter steps used by the QEMC proposal.
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Tuple ``(logical_depth, logical_qubits)`` for one walk application.
    """
    arithmetic_type = _validate_arithmetic_type(arithmetic_type)
    ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)

    proposal_depth, proposal_qubits = 1 + num_trotter_steps * (n + 2), 2 * n

    if arithmetic_type == "HYBRID":
        boltz_depth = 162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12
        boltz_qubits = 2 * n + 4 * n**2 + 21 * n**2 * S + 2
        reflection_depth = 14 * np.log2(n + 7 * S + 3) - 13
        reflection_qubits = 2 * n + 7 * S + 6
        accept_depth = 28 * np.log2(4 + 7 * S) - 23
        accept_qubits = 3 * n + 7 * S + 4
    else:
        d = 2.0 + max(np.sqrt(0.5 * beta * alpha * ell_eps) + ell_eps, alpha * ell_eps)
        controlled_qubitized_depth = 58 * n - 58 + 14 * np.log2(6 * n)
        boltz_depth = 3 * d * controlled_qubitized_depth + 3 * (2 * d + 1)
        boltz_qubits = 10 * n + 2
        reflection_depth = 14 * np.log2(n) - 13
        reflection_qubits = 2 * n + 3
        accept_depth = 3
        accept_qubits = 3 * n + 1

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)



def get_time_direct_enumeration(n: int):
    # return 1.504584e-07 * (2**n)
    return 6.323731e-10 * (n**2) * (2**n)


def get_annealing_time_classical_walk_local(n: int | float, vec_queries: list[float], device: str = "cpu") -> float:
    """Return the classical annealing runtime for local Metropolis updates.

    :param n: Number of spins.
    :param vec_queries: Number of classical update queries per annealing step.
    :param device: Classical device model, one of ``"cpu"``, ``"gpu"``, or ``"fpga"``.
    :return: Total runtime in seconds.
    """
    assert device in ALLOWED_DEVICES, f"Device {device} unknown. Allowed: {ALLOWED_DEVICES}"
    if device == "cpu":
        return sum(vec_queries) * cpu_local_step(n)
    if device == "gpu":
        return sum(vec_queries) * gpu_local_step(n)
    if device == "fpga":
        return sum(vec_queries) * fpga_local_step(n)


def get_annealing_time_classical_walk_uniform(n: int | float, vec_queries: list[float], device: str = "cpu") -> float:
    """Return the classical annealing runtime for uniform Metropolis updates.

    :param n: Number of spins.
    :param vec_queries: Number of classical update queries per annealing step.
    :param device: Classical device model, one of ``"cpu"``, ``"gpu"``, or ``"fpga"``.
    :return: Total runtime in seconds.
    """
    assert device in ALLOWED_DEVICES, f"Device {device} unknown. Allowed: {ALLOWED_DEVICES}"
    if device == "cpu":
        return sum(vec_queries) * cpu_uniform_step(n)
    if device == "gpu":
        return sum(vec_queries) * gpu_uniform_step(n)
    if device == "fpga":
        return sum(vec_queries) * fpga_uniform_step(n)


def get_annealing_time_classical_walk_qemc(n: int | float, vec_queries: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, eps_SF: float, num_trotter_steps: int = 50) -> float:
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
    total_time = 0
    for queries in vec_queries:
        logical_time_per_query = 1 + num_trotter_steps * (n + 2)
        logical_space = n
        physical_time = queries * rotated_surface_code_time(logical_time_per_query, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate, eps_SF)
        total_time += physical_time
    return total_time


# @interpolation_cache(monaqa2.data.filename.CACHE_PHASE_GAP_FACTOR_FILE)
# def phase_gap_factor(spectral_gap: float) -> float:
#     """Return the inverse phase gap of a Szegedy walk.
# 
#     :param spectral_gap: Classical spectral gap ``delta``.
#     :return: ``1 / arccos(1 - delta)``, evaluated as ``1 / (2 asin(sqrt(delta / 2)))`` for numerical stability.
#     """
#     with mp.workdps(100):
#         g = mp.mpf(spectral_gap)
#         # 1.0 is wrapped in mp.mpf to force mpmath high-precision division
#         # MUUUUCH SAFER THAN arccos(1 - g)
#         result = mp.mpf(1) / (2 * mp.asin(mp.sqrt(g / 2)))
#         return float(result)

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
    return 2 * np.log2(1.0 / eps_filter) * phase_gap_factor(spectral_gap)


def spectral_filter_polynomial_degree_list(spectral_gaps: list[float], eps_FLT: float, overlap: float = 1.0 / np.e) -> list[float]:
    """Return the QSVT spectral-filter degree for each annealing step.

    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param eps_FLT: Total error budget assigned to all spectral-filter polynomial approximations.
    :param overlap: Lower bound on the squared overlap between consecutive annealing states.
    :return: List of polynomial degrees, one for each spectral gap.
    """
    num_filters = len(spectral_gaps)
    num_zeno_filters = num_filters * (1.0 + 1.0 / overlap)
    eps_filter = eps_FLT / num_zeno_filters
    return [spectral_filter_polynomial_degree(spectral_gap, eps_filter) for spectral_gap in spectral_gaps]


def get_scheduled_filter_queries(degree_filters: list[float], overlap: float = 1.0 / np.e) -> list[float]:
    """Return the expected number of walk queries per annealing step after Zeno-rewind repetitions.

    :param degree_filters: Degree of each unique QSVT spectral filter.
    :param overlap: Lower bound on the squared overlap between consecutive annealing states.
    :return: Expected number of calls to ``W`` or ``W-dagger`` contributed by each annealing step.
    """
    zeno_overhead = 1.0 + 1.0 / overlap
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
    total_queries = float(sum(queries_per_step))
    eps_W = eps_W_budget / total_queries
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


def _get_annealing_time_quantum_walk(n: int | float, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, circuit_fn, zeno_overlap_probability: float = 1.0 / np.e, arithmetic_type: str = "HYBRID") -> float:
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
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    budget = get_quantum_annealing_error_budget(eps_TV, spectral_gaps, zeno_overlap_probability)
    eps_SF = budget["eps_SF"]
    eps_W = budget["eps_W"]
    queries_per_step = budget["queries_per_step"]

    logical_time = 0
    logical_space = 0
    for beta, queries in zip(betas, queries_per_step):
        walk_time, walk_space = circuit_fn(n, eps_W, beta, arithmetic_type)
        logical_time += queries * walk_time
        logical_space = max(logical_space, walk_space)

    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate, eps_SF)


def get_annealing_time_quantum_walk_local(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, *, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = 50, arithmetic_type: str = "HYBRID") -> float:
    """Return the surface-code runtime for quantum annealing with local-proposal Szegedy walks.

    :param n: Number of spins.
    :param eps_TV: Desired final total-variation distance error.
    :param betas: Inverse temperatures along the annealing schedule.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :param num_trotter_steps: Unused for local proposals; kept for signature compatibility.
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, quantum_walk_local_circuit, zeno_overlap_probability, arithmetic_type)


def get_annealing_time_quantum_walk_uniform(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, *, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = 50, arithmetic_type: str = "HYBRID") -> float:
    """Return the surface-code runtime for quantum annealing with uniform-proposal Szegedy walks.

    :param n: Number of spins.
    :param eps_TV: Desired final total-variation distance error.
    :param betas: Inverse temperatures along the annealing schedule.
    :param spectral_gaps: Spectral gaps along the annealing schedule.
    :param physical_operation_time: Time for one physical operation.
    :param physical_measurement_time: Time for one physical measurement.
    :param physical_error_rate: Physical Clifford error rate.
    :param zeno_overlap_probability: Lower bound on the squared overlap used in the Zeno-rewind cost model.
    :param num_trotter_steps: Unused for uniform proposals; kept for signature compatibility.
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, quantum_walk_uniform_circuit, zeno_overlap_probability, arithmetic_type)


def get_annealing_time_quantum_walk_qemc(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, *, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = 50, arithmetic_type: str = "HYBRID") -> float:
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
    :param arithmetic_type: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
    :return: Physical runtime in the same time unit as the physical operation and measurement times.
    """
    def circuit_fn(n_local: int | float, eps_W: float, beta: float, arithmetic_type_local: str = "HYBRID") -> tuple[float, float]:
        """Wrap the QEMC circuit resource function with a fixed Trotter-step count.

        :param n_local: Number of spins.
        :param eps_W: Error budget for one implementation of ``W`` or ``W-dagger``.
        :param beta: Inverse temperature for the current annealing step.
        :param arithmetic_type_local: Arithmetic implementation, either ``"HYBRID"`` or ``"FULLY_PHASE"``.
        :return: Tuple ``(logical_depth, logical_qubits)`` for one QEMC walk application.
        """
        return quantum_walk_qemc_circuit(n_local, eps_W, beta, num_trotter_steps, arithmetic_type_local)

    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, circuit_fn, zeno_overlap_probability, arithmetic_type)



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
    """
    There ...
    """
    beta = float(beta)
    if beta <= 0.0:
        return []

    current = 0.0
    schedule = []

    def delta_beta(n: int | float, beta: float | np.ndarray) -> float | np.ndarray:
        beta = np.asarray(beta, dtype=float)
        step = np.where(
            beta < 1.0,
            np.exp(-0.111 * beta + 0.615 * beta**2),
            0.533 * (beta + 0.286)**1.75,
        ) / np.sqrt(float(n))
        return step

    while current < beta:
        step = float(delta_beta(n, current))
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"Invalid annealing step: n={n}, beta={current}, step={step}")

        next_beta = current + step

        if current < 1.0 < min(next_beta, beta):
            next_beta = 1.0

        current = min(next_beta, beta)

        if not schedule or not np.isclose(current, schedule[-1]):
            schedule.append(current)

    return schedule


def make_prefix_stable_schedule_generator(
    beta_max: float,
    n_ref: int | float,
    base_schedule_generator,
):
    """Return a schedule generator using one fixed master beta grid.

    :param beta_max: Maximum final beta for which the schedule is needed.
    :param n_ref: Reference n used to generate the master schedule, usually n_plot_max.
    :param base_schedule_generator: Original adaptive schedule generator.
    :return: Function ``schedule(n, beta)`` with beta-prefix stability and no n-dependent node motion.
    """
    beta_max = float(beta_max)
    master_schedule = base_schedule_generator(n_ref, beta_max)

    def schedule(n: int | float, beta: float) -> list[float]:
        beta = float(beta)
        if beta <= 0.0:
            return []

        prefix = [b for b in master_schedule if b < beta and not np.isclose(b, beta)]

        if not prefix or not np.isclose(prefix[-1], beta):
            prefix.append(beta)

        return prefix

    return schedule