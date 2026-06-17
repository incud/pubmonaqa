import numpy as np
import mpmath as mp
from monaqa2.data.utils_interpolation_cache import interpolation_cache
import monaqa2.data.filename
from monaqa2.mcmc.search import search_monotone

ALLOWED_DEVICES = {"cpu", "gpu", "fpga"}


def cpu_local_step(n: int | float) -> float:
    # local/op: 5.959e-09 + 1.429e-10 n
    return 5.959e-09 + 1.429e-10 * n


def cpu_uniform_step(n: int | float) -> float:
    # uniform/op: 0.000e+00 + 1.173e-08 n + 6.964e-11 n^2
    return 1.173e-08 * n + 6.964e-11 * n**2


def gpu_local_step(n: int | float) -> float:
    # local/op: 7.837e-07 + 1.459e-09 n
    return 7.837e-07 + 1.459e-09 * n


def gpu_uniform_step(n: int | float) -> float:
    # uniform/op: 0.000e+00 + 0.000e+00 n + 2.215e-10 n^2
    return 2.215e-10 * n**2


def fpga_local_step(n: int | float) -> float:
    # 0.267900 + 0.001800 log2(N) [μs, multiply by 10^{-6}]
    return (0.267900 + 0.001800 * np.log2(n)) * 1e-6


def fpga_uniform_step(n: int | float) -> float:
    # 0.254100 + 0.004200 log2(N) [μs, multiply by 10^{-6}]
    return (0.254100 + 0.004200 * np.log2(n)) * 1e-6


def split_quantum_error_budget(eps_TV: float) -> tuple[float, float, float, float]:
    eps_SF = eps_TV / 4.0
    eps_MS = eps_TV / 4.0
    eps_FLT = eps_TV / 4.0
    eps_W_budget = eps_TV / 4.0
    return eps_SF, eps_MS, eps_FLT, eps_W_budget


def rotated_surface_code_distance(spacetime_volume: int | float, physical_error_rate: float, eps_SF: float) -> float:
    def distance(index: int) -> int:
        return 2 * index + 3

    def total_error(index: int) -> float:
        d = distance(index)
        logical_error_per_round = d * 0.1 * (100 * physical_error_rate) ** ((d + 1.0) / 2.0)
        return spacetime_volume * logical_error_per_round

    index = search_monotone(total_error, lambda error: error - eps_SF, 0, 1_000_000, info="rotated_surface_code_distance")
    return float(distance(index))


def rotated_surface_code_time(logical_time: int | float, logical_space: int | float, physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, eps_SF: float) -> float:
    spacetime_volume = logical_time * logical_space
    d = rotated_surface_code_distance(spacetime_volume, physical_error_rate, eps_SF)
    logical_cycle_time = d * (4 * physical_operation_time + physical_measurement_time)
    return logical_time * logical_cycle_time


def _calculate_auxiliary_quantum_circuit_vars(n: int | float, eps_W: float, beta: float) -> tuple[float, float, float, float, float, float, float, float]:
    ell_n = np.log2(n)
    ell_eps = np.log2(1.0 / eps_W)
    ell_2n = np.log2(2.0 * n)
    S = 1.0 + np.log2(n) + ell_eps
    ell_S = np.log2(1.0 + 3.5 * S)
    alpha = 2.0 * n**1.5 / np.sqrt(np.pi)

    m_arg = beta * alpha / (2.0 * np.log(1.0 / eps_W))
    m = np.log2(m_arg)
    ell_m = np.log2(m)

    return ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m


def quantum_walk_local_circuit(n: int | float, eps_W: float, beta: float) -> tuple[float, float]:
    ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)

    proposal_depth, proposal_qubits = 13 * ell_n + 15, 2 * n
    boltz_depth = 162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12
    boltz_qubits = 2 * n + 4 * n**2 + 21 * n**2 * S + 2
    reflection_depth = 14 * np.log2(n + 7 * S + 3) - 13
    reflection_qubits = 2 * n + 7 * S + 6
    accept_depth = 28 * np.log2(4 + 7 * S) - 23
    accept_qubits = 3 * n + 7 * S + 4

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def quantum_walk_uniform_circuit(n: int | float, eps_W: float, beta: float) -> tuple[float, float]:
    ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)

    proposal_depth, proposal_qubits = 0, 2 * n
    boltz_depth = 162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12
    boltz_qubits = 2 * n + 4 * n**2 + 21 * n**2 * S + 2
    reflection_depth = 14 * np.log2(n + 7 * S + 3) - 13
    reflection_qubits = 2 * n + 7 * S + 6
    accept_depth = 28 * np.log2(4 + 7 * S) - 23
    accept_qubits = 3 * n + 7 * S + 4

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def quantum_walk_qemc_circuit(n: int | float, eps_W: float, beta: float, num_trotter_steps: int = 50) -> tuple[float, float]:
    ell_n, ell_eps, ell_2n, S, ell_S, alpha, m, ell_m = _calculate_auxiliary_quantum_circuit_vars(n, eps_W, beta)

    proposal_depth, proposal_qubits = 1 + num_trotter_steps * (n + 2), 2 * n
    boltz_depth = 162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12
    boltz_qubits = 2 * n + 4 * n**2 + 21 * n**2 * S + 2
    reflection_depth = 14 * np.log2(n + 7 * S + 3) - 13
    reflection_qubits = 2 * n + 7 * S + 6
    accept_depth = 28 * np.log2(4 + 7 * S) - 23
    accept_qubits = 3 * n + 7 * S + 4

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def get_annealing_time_classical_walk_local(n: int | float, vec_queries: list[float], device: str = "cpu") -> float:
    assert device in ALLOWED_DEVICES, f"Device {device} unknown. Allowed: {ALLOWED_DEVICES}"
    if device == "cpu":
        return sum(vec_queries) * cpu_local_step(n)
    if device == "gpu":
        return sum(vec_queries) * gpu_local_step(n)
    if device == "fpga":
        return sum(vec_queries) * fpga_local_step(n)


def get_annealing_time_classical_walk_uniform(n: int | float, vec_queries: list[float], device: str = "cpu") -> float:
    assert device in ALLOWED_DEVICES, f"Device {device} unknown. Allowed: {ALLOWED_DEVICES}"
    if device == "cpu":
        return sum(vec_queries) * cpu_uniform_step(n)
    if device == "gpu":
        return sum(vec_queries) * gpu_uniform_step(n)
    if device == "fpga":
        return sum(vec_queries) * fpga_uniform_step(n)


def get_annealing_time_classical_walk_qemc(n: int | float, vec_queries: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, eps_SF: float, num_trotter_steps: int = 50) -> float:
    total_time = 0
    for queries in vec_queries:
        logical_time_per_query = 1 + num_trotter_steps * (n + 2)
        logical_space = n
        physical_time = queries * rotated_surface_code_time(logical_time_per_query, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate, eps_SF)
        total_time += physical_time
    return total_time


@interpolation_cache(monaqa2.data.filename.CACHE_PHASE_GAP_FACTOR_FILE)
def phase_gap_factor(spectral_gap: float) -> float:
    with mp.workdps(100):
        g = mp.mpf(spectral_gap)
        # 1.0 is wrapped in mp.mpf to force mpmath high-precision division
        # MUUUUCH SAFER THAN arccos(1 - g)
        result = mp.mpf(1) / (2 * mp.asin(mp.sqrt(g / 2)))
        return float(result)


def spectral_gap_to_filter_degree(spectral_gap: float, prec: float) -> float:
    return 2 * prec * phase_gap_factor(spectral_gap)


def get_qsvt_filter_application_overhead(zeno_overlap_probability: float = 1.0 / np.e) -> float:
    return 1.0 + 1.0 / zeno_overlap_probability


def get_total_qsvt_filter_applications(num_filters: int | float, zeno_overlap_probability: float = 1.0 / np.e) -> float:
    return num_filters * get_qsvt_filter_application_overhead(zeno_overlap_probability)


def get_filter_degrees_from_budget(spectral_gaps: list[float], eps_FLT: float, zeno_overlap_probability: float = 1.0 / np.e) -> list[float]:
    total_filter_applications = get_total_qsvt_filter_applications(len(spectral_gaps), zeno_overlap_probability)
    eps_filter = eps_FLT / total_filter_applications
    prec = np.log2(1.0 / eps_filter)
    return [spectral_gap_to_filter_degree(spectral_gap, prec) for spectral_gap in spectral_gaps]


def get_scheduled_filter_queries(degree_filters: list[float], zeno_overlap_probability: float = 1.0 / np.e) -> list[float]:
    overhead = get_qsvt_filter_application_overhead(zeno_overlap_probability)
    return [overhead * degree for degree in degree_filters]


def get_quantum_annealing_error_budget(eps_TV: float, spectral_gaps: list[float], zeno_overlap_probability: float = 1.0 / np.e) -> dict[str, float | list[float]]:
    eps_SF, eps_MS, eps_FLT, eps_W_budget = split_quantum_error_budget(eps_TV)
    degree_filters = get_filter_degrees_from_budget(spectral_gaps, eps_FLT, zeno_overlap_probability)
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


def _get_annealing_time_quantum_walk(n: int | float, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, circuit_fn, zeno_overlap_probability: float = 1.0 / np.e) -> float:
    budget = get_quantum_annealing_error_budget(eps_TV, spectral_gaps, zeno_overlap_probability)
    eps_SF = budget["eps_SF"]
    eps_W = budget["eps_W"]
    queries_per_step = budget["queries_per_step"]

    logical_time = 0
    logical_space = 0
    for beta, queries in zip(betas, queries_per_step):
        walk_time, walk_space = circuit_fn(n, eps_W, beta)
        logical_time += queries * walk_time
        logical_space = max(logical_space, walk_space)

    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate, eps_SF)


def get_annealing_time_quantum_walk_local(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = 50) -> float:
    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, quantum_walk_local_circuit, zeno_overlap_probability)


def get_annealing_time_quantum_walk_uniform(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = 50) -> float:
    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, quantum_walk_uniform_circuit, zeno_overlap_probability)


def get_annealing_time_quantum_walk_qemc(n: int, eps_TV: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, zeno_overlap_probability: float = 1.0 / np.e, num_trotter_steps: int = 50) -> float:
    def circuit_fn(n_local: int | float, eps_W: float, beta: float) -> tuple[float, float]:
        return quantum_walk_qemc_circuit(n_local, eps_W, beta, num_trotter_steps)

    return _get_annealing_time_quantum_walk(n, eps_TV, betas, spectral_gaps, physical_operation_time, physical_measurement_time, physical_error_rate, circuit_fn, zeno_overlap_probability)


def get_annealing_queries_quantum_walks(n: int | float, eps_TV: float, spectral_gaps: list[float], zeno_overlap_probability: float = 1.0 / np.e) -> float:
    budget = get_quantum_annealing_error_budget(eps_TV, spectral_gaps, zeno_overlap_probability)
    return budget["total_queries"]
