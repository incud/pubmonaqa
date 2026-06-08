import math
from collections.abc import Callable


CLASSICAL_CPU_MOVE_TIME_FORMULAS: dict[str, Callable[[int, float, float, float], float]] = {
    "uniform": lambda n, a_cpu, b_cpu, c_cpu: (
        a_cpu + b_cpu * n + c_cpu * n * (n - 1) / 4.0
    ),
    "local1": lambda n, a_cpu, b_cpu, c_cpu: (
        a_cpu + b_cpu + c_cpu * (n - 1)
    ),
    "local2": lambda n, a_cpu, b_cpu, c_cpu: (
        a_cpu + b_cpu + c_cpu * 2.0 * (n - 2)
    ),
    "local3": lambda n, a_cpu, b_cpu, c_cpu: (
        a_cpu + b_cpu + c_cpu * 3.0 * (n - 3)
    ),
}


def _as_positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _as_nonnegative_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    return value


def _loglog_argument(name: str, value: float, *, need_logloglog: bool) -> tuple[float, float]:
    log_value = math.log(value)
    loglog_value = math.log(log_value)

    if need_logloglog and loglog_value <= 0.0:
        raise ValueError(
            f"{name} gives log(log({name})) <= 0, so log(log(log({name}))) is undefined "
            f"over the reals. Got {name}={value!r}."
        )

    return loglog_value, math.log(loglog_value) if need_logloglog else 0.0


def get_quantum_walk_uniform_steps(
    n: int,
    U: float,
    beta: float,
    epsilon: float,
) -> float:
    n = int(n)
    U = _as_positive_float("U", U)
    beta = _as_nonnegative_float("beta", beta)
    epsilon = _as_positive_float("epsilon", epsilon)

    if n <= 1:
        raise ValueError("n must be > 1 because log(n) appears.")

    A = U * beta + 8.0 * n**2 / epsilon**2
    log_n = math.log(n)
    log_eps = math.log(8.0 / epsilon)
    loglog_A, logloglog_A = _loglog_argument("A", A, need_logloglog=True)

    return float(
        260.0 * (math.e * U * beta / 4.0 + log_eps + 1.0) * loglog_A
        + 260.0 * log_n * loglog_A
        + 6.0 * logloglog_A
        + 18.0
    )


def get_quantum_walk_local_steps(
    n: int,
    U: float,
    beta: float,
    epsilon: float,
) -> float:
    n = int(n)
    U = _as_positive_float("U", U)
    beta = _as_nonnegative_float("beta", beta)
    epsilon = _as_positive_float("epsilon", epsilon)

    if n <= 1:
        raise ValueError("n must be > 1 because log(n) appears.")

    A = U * beta + 8.0 * n**2 / epsilon**2
    log_n = math.log(n)
    log_eps = math.log(8.0 / epsilon)
    loglog_A, logloglog_A = _loglog_argument("A", A, need_logloglog=True)

    return float(
        (
            64.0 * math.e * U * beta
            + 256.0 * log_eps
            + 259.0 * log_n
            + 256.0
        )
        * loglog_A
        + 24.0 * log_n
        + 6.0 * logloglog_A
        + 70.0
    )


def get_quantum_walk_qemc_steps(
    n: int,
    U: float,
    beta: float,
    epsilon: float,
    alpha: float,
    t: float,
) -> float:
    n = int(n)
    U = _as_positive_float("U", U)
    beta = _as_nonnegative_float("beta", beta)
    epsilon = _as_positive_float("epsilon", epsilon)
    alpha = _as_positive_float("alpha", alpha)
    t = _as_nonnegative_float("t", t)

    if n <= 1:
        raise ValueError("n must be > 1 because log(n) appears.")

    A = U * beta + 18.0 * n**2 / epsilon**2
    log_n = math.log(n)
    log_eps = math.log(12.0 / epsilon)
    loglog_A, _ = _loglog_argument("A", A, need_logloglog=False)

    return float(
        63.0 * math.e * U * beta / 2.0
        + 118.0 * math.e * alpha * t
        + n * (8.0 * math.e * alpha * t + 8.0 * log_eps + 12.0)
        + (23.0 * math.e * U * beta + 92.0 * log_eps + 260.0) * loglog_A
        + (
            208.0 * math.e * alpha * t
            + 208.0 * log_eps
            + 128.0 * loglog_A
            + 648.0
        )
        * log_n
        + 244.0 * log_eps
        + 674.0
    )


def evaluate_classical_cpu_move_time(
    proposal: str,
    num_spins: int,
    a_cpu: float,
    b_cpu: float,
    c_cpu: float,
) -> float:
    if proposal not in CLASSICAL_CPU_MOVE_TIME_FORMULAS:
        raise ValueError(f"No classical CPU formula for proposal={proposal!r}")

    n = int(num_spins)
    if n <= 0:
        raise ValueError("num_spins must be positive.")

    a_cpu = _as_nonnegative_float("a_cpu", a_cpu)
    b_cpu = _as_nonnegative_float("b_cpu", b_cpu)
    c_cpu = _as_nonnegative_float("c_cpu", c_cpu)

    if proposal.startswith("local"):
        k = int(proposal[-1])
        if n < k:
            raise ValueError(f"{proposal} requires num_spins >= {k}, got {n}")

    return float(CLASSICAL_CPU_MOVE_TIME_FORMULAS[proposal](n, a_cpu, b_cpu, c_cpu))


def sk_spherical_energy_difference_upper_bound(num_spins: int) -> float:
    """
    Return the deterministic spherical-SK upper bound
    U = 2 sum_i |h_i| + 2 sum_{i<j} |J_ij|.

    The normalized instances satisfy sum_i h_i^2 + sum_{i<j} J_ij^2 = n.
    There are m = n + n(n-1)/2 = n(n+1)/2 coefficients. Therefore,
    by Cauchy-Schwarz, sum_i |h_i| + sum_{i<j} |J_ij| <= sqrt(m) sqrt(n)
    = n sqrt((n+1)/2), and U <= n sqrt(2(n+1)).
    """
    n = int(num_spins)
    if n <= 0:
        raise ValueError("num_spins must be positive.")
    return float(n * math.sqrt(2.0 * (n + 1)))


def _validate_qec_parameters(
    eps: float,
    time_quantum_gate: float,
    time_quantum_measurement: float,
    num_gate_layers_per_qec_cycle: int,
    prob_phys_error: float,
) -> tuple[float, float, float, int, float]:
    eps = _as_positive_float("eps", eps)
    time_quantum_gate = _as_positive_float("time_quantum_gate", time_quantum_gate)
    time_quantum_measurement = _as_positive_float("time_quantum_measurement", time_quantum_measurement)
    prob_phys_error = _as_positive_float("prob_phys_error", prob_phys_error)

    num_gate_layers_per_qec_cycle = int(num_gate_layers_per_qec_cycle)
    if num_gate_layers_per_qec_cycle <= 0:
        raise ValueError("num_gate_layers_per_qec_cycle must be positive.")

    if prob_phys_error >= 1e-2:
        raise ValueError(
            "prob_phys_error must be < 1e-2 for the heuristic logical-error "
            "formula 0.1 * (100 p)^((d + 1) / 2) to decrease with distance."
        )

    return (
        eps,
        time_quantum_gate,
        time_quantum_measurement,
        num_gate_layers_per_qec_cycle,
        prob_phys_error,
    )


def _surface_code_distance_for_budget(
    *,
    num_locations: float,
    eps_qec: float,
    prob_phys_error: float,
) -> int:
    num_locations = _as_nonnegative_float("num_locations", num_locations)
    eps_qec = _as_positive_float("eps_qec", eps_qec)

    def p_logical(d: int) -> float:
        return 0.1 * (100.0 * prob_phys_error) ** ((d + 1) / 2.0)

    d = 3
    while num_locations * p_logical(d) > eps_qec:
        d += 2
    return d


def calculate_runtime_from_uniform_distribution(
    num_spins: int,
    num_classical_queries: int,
    num_quantum_queries_per_step: list[int],
    beta_quantum_per_step: list[float],
    p_succ_quantum_per_step: list[float],
    U_quantum_upper_bound: float | None = None,
    walk: str = "classical",
    proposal: str = "uniform",
    eps: float = 1e-2,
    a_cpu: float = 1e-6,
    b_cpu: float = 5e-9,
    c_cpu: float = 5e-10,
    num_steps_trotter: int = 100,
    time_quantum_gate: float = 20e-9,
    time_quantum_measurement: float = 100e-9,
    num_gate_layers_per_qec_cycle: int = 4,
    prob_phys_error: float = 1e-5,
) -> float:
    """
    Estimate wall-clock runtime in seconds for either classical MCMC transition
    attempts or a quantum annealing procedure from the uniform distribution.

    In the classical branch, `num_classical_queries` is the number of classical
    MCMC transition attempts. In the quantum branch, `num_quantum_queries_per_step[t]`
    is the QPE projection cost `C_t` in Szegedy-walk queries at annealing inverse
    temperature `beta_quantum_per_step[t]`. Therefore `num_quantum_queries_per_step`
    and `beta_quantum_per_step` must have length `L + 1` for an annealing schedule
    `beta_0, ..., beta_L`, while `p_succ_quantum_per_step[t]` is the adjacent
    transition success probability `p_{t+1}=|<pi_{beta_{t+1}}|pi_{beta_t}>|^2`
    and must have length `L`.

    If `U_quantum_upper_bound` is None, it is set automatically using the
    spherical-SK normalization. Since sum_i h_i^2 + sum_{i<j} J_ij^2 = n and
    there are n(n+1)/2 coefficients, Cauchy-Schwarz gives
    2 sum_i |h_i| + 2 sum_{i<j} |J_ij| <= n sqrt(2(n+1)). This is the default
    value used by the quantum arithmetic and block-encoding resource formulas.

    The quantum branch uses the rewind expected-cost formula. For each transition
    `t -> t + 1`, the expected QPE-projection cost is
    `C_{t+1} + (C_t + C_{t+1}) / (2 p_{t+1})`. Since the walk-query implementation
    cost depends on beta, the same formula is applied after converting `C_t` into
    non-Clifford depth at `beta_t`.

    This is a coarse critical-path timing model. It assumes enough magic-state
    factory throughput, so wall-clock time is controlled by non-Clifford depth rather
    than non-Clifford count. The code-distance estimate is deliberately heuristic and
    uses accumulated non-Clifford depth as a proxy for logical fault locations.

    :param num_spins: Number of Ising spins.
    :param num_classical_queries: Number of classical MCMC transition attempts.
    :param num_quantum_queries_per_step: QPE projection costs `C_t` in walk queries
        for each annealing beta, including the initial beta.
    :param beta_quantum_per_step: Annealing inverse temperatures `beta_t` associated
        with the QPE projection costs.
    :param p_succ_quantum_per_step: Adjacent annealing success probabilities
        `p_{t+1}` used by the rewind formula.
    :param U_quantum_upper_bound: Optional upper bound U used by arithmetic and
        block-encoding resource formulas; if None, use n sqrt(2(n+1)).
    :param walk: Either "classical" or "quantum".
    :param proposal: One of "uniform", "local1", "local2", "local3", "qemc", or "layden".
    :param eps: Total tolerated approximation/logical-error budget.
    :param a_cpu: Fixed CPU overhead per classical transition attempt.
    :param b_cpu: CPU cost coefficient for proposal mask generation or spin-mask manipulation.
    :param c_cpu: CPU cost coefficient for dense SK energy-difference evaluation.
    :param num_steps_trotter: Number of Trotter steps used for classically simulated
        QEMC/Layden proposals.
    :param time_quantum_gate: Physical gate-layer time used in one QEC cycle estimate.
    :param time_quantum_measurement: Physical measurement time used in one QEC cycle estimate.
    :param num_gate_layers_per_qec_cycle: Number of physical gate layers per QEC cycle.
    :param prob_phys_error: Physical error probability used in the heuristic logical-error formula.
    :return: Estimated wall-clock time in seconds.
    """
    if walk not in {"classical", "quantum"}:
        raise ValueError(f"walk must be either 'classical' or 'quantum', got {walk!r}")

    if proposal not in {"uniform", "local1", "local2", "local3", "qemc", "layden"}:
        raise ValueError(f"unknown proposal={proposal!r}")

    n = int(num_spins)
    if n <= 1:
        raise ValueError("num_spins must be > 1.")

    num_classical_queries = int(num_classical_queries)
    if num_classical_queries < 0:
        raise ValueError("num_classical_queries must be nonnegative.")

    (
        eps,
        time_quantum_gate,
        time_quantum_measurement,
        num_gate_layers_per_qec_cycle,
        prob_phys_error,
    ) = _validate_qec_parameters(
        eps,
        time_quantum_gate,
        time_quantum_measurement,
        num_gate_layers_per_qec_cycle,
        prob_phys_error,
    )

    U_quantum_upper_bound = (
        sk_spherical_energy_difference_upper_bound(n)
        if U_quantum_upper_bound is None
        else _as_positive_float("U_quantum_upper_bound", U_quantum_upper_bound)
    )

    if walk == "classical":
        if proposal in CLASSICAL_CPU_MOVE_TIME_FORMULAS:
            return float(
                num_classical_queries
                * evaluate_classical_cpu_move_time(proposal, n, a_cpu, b_cpu, c_cpu)
            )

        if proposal in {"qemc", "layden"}:
            num_steps_trotter = int(num_steps_trotter)
            if num_steps_trotter <= 0:
                raise ValueError("num_steps_trotter must be positive.")

            eps_qec = eps / 2.0
            t_accept = evaluate_classical_cpu_move_time("uniform", n, a_cpu, b_cpu, c_cpu)

            n_rot_per_step = n * (n - 1) / 2.0 + 2.0 * n
            rot_depth_per_step = n + 2.0
            n_rot = num_steps_trotter * n_rot_per_step
            rot_depth = num_steps_trotter * rot_depth_per_step

            eps_per_rot = (eps / 2.0) / n_rot
            t_per_rot = 3.0 * math.log2(1.0 / eps_per_rot) + 10.0

            n_cnot = num_steps_trotter * n * (n - 1)
            n_t = n_rot * t_per_rot
            n_locations = n_cnot + n_rot + n_t

            d = _surface_code_distance_for_budget(
                num_locations=n_locations,
                eps_qec=eps_qec,
                prob_phys_error=prob_phys_error,
            )

            qec_cycle_time = (
                num_gate_layers_per_qec_cycle * time_quantum_gate
                + time_quantum_measurement
            )
            t_quantum_move = rot_depth * t_per_rot * d * qec_cycle_time
            return float(num_classical_queries * (t_accept + t_quantum_move))

    if walk == "quantum":
        if len(num_quantum_queries_per_step) != len(beta_quantum_per_step):
            raise ValueError(
                "num_quantum_queries_per_step and beta_quantum_per_step must have the same length"
            )
        if len(p_succ_quantum_per_step) != max(len(num_quantum_queries_per_step) - 1, 0):
            raise ValueError(
                "p_succ_quantum_per_step must have length len(num_quantum_queries_per_step) - 1"
            )
        if len(num_quantum_queries_per_step) <= 1:
            return 0.0

        p_succ_values = []
        for p_succ_j in p_succ_quantum_per_step:
            p_succ_j = float(p_succ_j)
            if not math.isfinite(p_succ_j) or p_succ_j <= 0.0 or p_succ_j > 1.0:
                raise ValueError(f"invalid quantum success probability: {p_succ_j}")
            p_succ_values.append(p_succ_j)

        qpe_query_values = []
        for n_quantum_queries_j in num_quantum_queries_per_step:
            n_quantum_queries_j = float(n_quantum_queries_j)
            if not math.isfinite(n_quantum_queries_j) or n_quantum_queries_j < 0.0:
                raise ValueError(f"invalid quantum query count: {n_quantum_queries_j}")
            qpe_query_values.append(n_quantum_queries_j)

        beta_values = [_as_nonnegative_float("beta_quantum_per_step", beta_j) for beta_j in beta_quantum_per_step]

        expected_qpe_queries = 0.0
        for j, p_succ_j in enumerate(p_succ_values):
            expected_qpe_queries += (
                qpe_query_values[j + 1]
                + (qpe_query_values[j] + qpe_query_values[j + 1]) / (2.0 * p_succ_j)
            )

        if expected_qpe_queries <= 0.0:
            raise ValueError("expected quantum QPE query count must be positive.")

        eps_qec = eps / 2.0
        eps_per_query = (eps / 2.0) / expected_qpe_queries
        qpe_nc_depth_per_step = []

        for n_quantum_queries_j, beta_j in zip(qpe_query_values, beta_values):
            if proposal == "uniform":
                nc_depth_j = get_quantum_walk_uniform_steps(
                    n=n,
                    U=U_quantum_upper_bound,
                    beta=beta_j,
                    epsilon=eps_per_query,
                )

            elif proposal in {"local1", "local2", "local3"}:
                nc_depth_j = get_quantum_walk_local_steps(
                    n=n,
                    U=U_quantum_upper_bound,
                    beta=beta_j,
                    epsilon=eps_per_query,
                )

            elif proposal in {"qemc", "layden"}:
                # Same convention as the previous version: use t = 1 and
                # alpha = U_quantum_upper_bound for the QEMC/Layden simulation block.
                nc_depth_j = get_quantum_walk_qemc_steps(
                    n=n,
                    U=U_quantum_upper_bound,
                    beta=beta_j,
                    epsilon=eps_per_query,
                    alpha=U_quantum_upper_bound,
                    t=1.0,
                )

            qpe_nc_depth_per_step.append(n_quantum_queries_j * nc_depth_j)

        total_nc_depth = 0.0
        for j, p_succ_j in enumerate(p_succ_values):
            total_nc_depth += (
                qpe_nc_depth_per_step[j + 1]
                + (qpe_nc_depth_per_step[j] + qpe_nc_depth_per_step[j + 1])
                / (2.0 * p_succ_j)
            )

        d = _surface_code_distance_for_budget(
            num_locations=total_nc_depth,
            eps_qec=eps_qec,
            prob_phys_error=prob_phys_error,
        )

        qec_cycle_time = (
            num_gate_layers_per_qec_cycle * time_quantum_gate
            + time_quantum_measurement
        )
        return float(total_nc_depth * d * qec_cycle_time)

    raise ValueError(f"unreachable branch: {walk=}, {proposal=}")
