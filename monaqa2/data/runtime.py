import math


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
    return (0.267900 + 0.001800 * math.log2(n)) * 1e-6


def fpga_uniform_step(n: int | float) -> float:
    # 0.254100 + 0.004200 log2(N) [μs, multiply by 10^{-6}]
    return (0.254100 + 0.004200 * math.log2(n)) * 1e-6


def rotated_surface_code_distance(spacetime_volume: int, physical_error_rate: float) -> int:
    assert spacetime_volume > 0
    assert 0 < physical_error_rate < 1e-2

    d = 3
    while True:
        logical_error_per_round = d * 0.1 * (100 * physical_error_rate) ** ((d + 1) / 2)
        total_error = spacetime_volume * logical_error_per_round
        if total_error <= 1 / 3:
            return d
        d += 2


def rotated_surface_code_time(logical_time: int, logical_space: int, physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float) -> float:
    spacetime_volume = logical_time * logical_space
    d = rotated_surface_code_distance(spacetime_volume, physical_error_rate)
    logical_cycle_time = d * (4 * physical_operation_time + physical_measurement_time)
    return logical_time * logical_cycle_time


def quantum_walk_local_circuit(n: int, eps: float, beta: float) -> tuple[int, int]:
    ell_n = math.ceil(math.log2(n))
    ell_eps = math.log2(1 / eps)
    ell_2n = math.ceil(math.log2(2 * n))
    S = 1 + math.log2(n) + ell_eps
    ell_S = math.ceil(math.log2(1 + 3.5 * S))
    alpha = 2 * n ** 1.5 / math.sqrt(math.pi)
    m = math.log2(beta * alpha / (2 * math.log(1 / eps)))
    ell_m = math.log2(m)

    proposal_depth, proposal_qubits = 13 * ell_n + 15, 2 * n
    boltz_depth = math.ceil(162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12)
    boltz_qubits = math.ceil(2 * n + 4 * n**2 + 21 * n**2 * S + 2)
    reflection_depth = 14 * math.ceil(math.log2(n + 7 * S + 3)) - 13
    reflection_qubits = math.ceil(2 * n + 7 * S + 6)
    accept_depth = 28 * math.ceil(math.log2(4 + 7 * S)) - 23
    accept_qubits = math.ceil(3 * n + 7 * S + 4)

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def quantum_walk_uniform_circuit(n: int, eps: float, beta: float) -> tuple[int, int]:
    ell_eps = math.log2(1 / eps)
    ell_2n = math.ceil(math.log2(2 * n))
    S = 1 + math.log2(n) + ell_eps
    ell_S = math.ceil(math.log2(1 + 3.5 * S))
    alpha = 2 * n ** 1.5 / math.sqrt(math.pi)
    m = math.log2(beta * alpha / (2 * math.log(1 / eps)))
    ell_m = math.log2(m)

    proposal_depth, proposal_qubits = 0, 2 * n
    boltz_depth = math.ceil(162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12)
    boltz_qubits = math.ceil(2 * n + 4 * n**2 + 21 * n**2 * S + 2)
    reflection_depth = 14 * math.ceil(math.log2(n + 7 * S + 3)) - 13
    reflection_qubits = math.ceil(2 * n + 7 * S + 6)
    accept_depth = 28 * math.ceil(math.log2(4 + 7 * S)) - 23
    accept_qubits = math.ceil(3 * n + 7 * S + 4)

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def quantum_walk_qemc_circuit(n: int, eps: float, beta: float, num_trotter_steps: int = 50) -> tuple[int, int]:
    ell_eps = math.log2(1 / eps)
    ell_2n = math.ceil(math.log2(2 * n))
    S = 1 + math.log2(n) + ell_eps
    ell_S = math.ceil(math.log2(1 + 3.5 * S))
    alpha = 2 * n ** 1.5 / math.sqrt(math.pi)
    m = math.log2(beta * alpha / (2 * math.log(1 / eps)))
    ell_m = math.log2(m)

    proposal_depth, proposal_qubits = 1 + num_trotter_steps * (n + 2), 2 * n
    boltz_depth = math.ceil(162 * ell_eps * ell_S + 54 * ell_S - 93 * ell_eps + 24 * ell_2n + 49 * S + 28 * ell_m - 12)
    boltz_qubits = math.ceil(2 * n + 4 * n**2 + 21 * n**2 * S + 2)
    reflection_depth = 14 * math.ceil(math.log2(n + 7 * S + 3)) - 13
    reflection_qubits = math.ceil(2 * n + 7 * S + 6)
    accept_depth = 28 * math.ceil(math.log2(4 + 7 * S)) - 23
    accept_qubits = math.ceil(3 * n + 7 * S + 4)

    return 2 * proposal_depth + 2 * boltz_depth + reflection_depth + accept_depth, max(proposal_qubits, boltz_qubits, reflection_qubits, accept_qubits)


def get_annealing_time_classical_walk_local(n: int, vec_queries: list[int], device: str = "cpu") -> float:
    assert device in ALLOWED_DEVICES, f"Device {device} unknown. Allowed: {ALLOWED_DEVICES}"
    if device == "cpu":
        return sum(vec_queries) * cpu_local_step(n)
    if device == "gpu":
        return sum(vec_queries) * gpu_local_step(n)
    if device == "fpga":
        return sum(vec_queries) * fpga_local_step(n)


def get_annealing_time_classical_walk_uniform(n: int, vec_queries: list[int], device: str = "cpu") -> float:
    assert device in ALLOWED_DEVICES, f"Device {device} unknown. Allowed: {ALLOWED_DEVICES}"
    if device == "cpu":
        return sum(vec_queries) * cpu_uniform_step(n)
    if device == "gpu":
        return sum(vec_queries) * gpu_uniform_step(n)
    if device == "fpga":
        return sum(vec_queries) * fpga_uniform_step(n)


def get_annealing_time_classical_walk_qemc(n: int, vec_queries: list[int], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, num_trotter_steps: int = 50) -> float:
    total_time = 0
    for queries in vec_queries:
        logical_time_per_query = (1 + num_trotter_steps * (n + 2))
        logical_space = n
        physical_time = queries * rotated_surface_code_time(logical_time_per_query, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate)
        total_time += physical_time
    return total_time


def get_annealing_time_quantum_walk_local(n: int, eps: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, num_trotter_steps: int = 50) -> float:
    assert len(betas) == len(spectral_gaps)
    phase_gaps = [math.acos(1.0 - spectral_gap) for spectral_gap in spectral_gaps]
    degree_filters = [2 * math.ceil(math.log(1 / eps) / phase_gap) for phase_gap in phase_gaps]
    F = sum(degree_filters)

    logical_time = 0
    logical_space = 0
    for beta, degree in zip(betas, degree_filters):
        walk_time, walk_space = quantum_walk_local_circuit(n, eps / F, beta)
        logical_time += degree * walk_time
        logical_space = max(logical_space, walk_space)

    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate)


def get_annealing_time_quantum_walk_uniform(n: int, eps: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, num_trotter_steps: int = 50) -> float:
    assert len(betas) == len(spectral_gaps)
    phase_gaps = [math.acos(1.0 - spectral_gap) for spectral_gap in spectral_gaps]
    degree_filters = [2 * math.ceil(math.log(1 / eps) / phase_gap) for phase_gap in phase_gaps]
    F = sum(degree_filters)

    logical_time = 0
    logical_space = 0
    for beta, degree in zip(betas, degree_filters):
        walk_time, walk_space = quantum_walk_uniform_circuit(n, eps / F, beta)
        logical_time += degree * walk_time
        logical_space = max(logical_space, walk_space)

    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate)


def get_annealing_time_quantum_walk_qemc(n: int, eps: float, betas: list[float], spectral_gaps: list[float], physical_operation_time: float, physical_measurement_time: float, physical_error_rate: float, num_trotter_steps: int = 50) -> float:
    assert len(betas) == len(spectral_gaps)
    phase_gaps = [math.acos(1.0 - spectral_gap) for spectral_gap in spectral_gaps]
    degree_filters = [2 * math.ceil(math.log(1 / eps) / phase_gap) for phase_gap in phase_gaps]
    F = sum(degree_filters)

    logical_time = 0
    logical_space = 0
    for beta, degree_filter in zip(betas, degree_filters):
        walk_time, walk_space = quantum_walk_qemc_circuit(n, eps / F, beta, num_trotter_steps)
        logical_time += degree_filter * walk_time
        logical_space = max(logical_space, walk_space)

    return rotated_surface_code_time(logical_time, logical_space, physical_operation_time, physical_measurement_time, physical_error_rate)

