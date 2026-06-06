from monaqa2.data.single_step_classical import cpu_time_per_local_move, cpu_time_per_uniform_move
import numpy as np


def get_time_logical_quantum_step(n: int, logical_operation_time: float = 1e-6):
    return logical_operation_time # maybe n is not counted here because we use the depth later? or maybe we should consider it to be log(d)?

def get_time_classical_walk_uniform(n: int, num_classical_queries: int):
    return cpu_time_per_uniform_move(n) * num_classical_queries

def get_time_classical_walk_local(n: int, num_classical_queries: int):
    return cpu_time_per_local_move(n) * num_classical_queries

def get_time_classical_walk_qemc(n: int, num_classical_queries: int, trotter_steps: int, logical_operation_time: float = 1e-6):
    # second order trotter gets simplified
    trotter_depth = 1 + trotter_steps * (1 + n + 1) # X + #TR * (Z + ZZ + X)
    return get_time_logical_quantum_step(n, logical_operation_time) * trotter_depth * num_classical_queries

def get_time_quantum_walk_uniform(n: int, num_quantum_queries: int, logical_operation_time: float = 1e-6):
    proposal = 0 
    accept_path = 7 * np.log2(n)
    coin_energy = 185 * np.log2(n)
    coin_sqrt = 2410 * n
    coin = coin_energy + coin_sqrt
    reflection = 4 * np.log2(n)
    depth = 2 * proposal + 2 * coin + accept_path + reflection
    return get_time_logical_quantum_step(n, logical_operation_time) * depth * num_quantum_queries

def get_time_quantum_walk_local(n: int, num_quantum_queries: int, logical_operation_time: float = 1e-6):
    proposal = 12 * np.log2(n)
    accept_path = 7 * np.log2(n)
    coin_energy = 185 * np.log2(n)
    coin_sqrt = 2410 * n
    coin = coin_energy + coin_sqrt
    reflection = 4 * np.log2(n)
    depth = 2 * proposal + 2 * coin + accept_path + reflection
    return get_time_logical_quantum_step(n, logical_operation_time) * depth * num_quantum_queries

def get_time_quantum_walk_qemc(n: int, num_quantum_queries: int, trotter_steps: int, logical_operation_time: float = 1e-6):
    trotter_depth = 1 + trotter_steps * (1 + n + 1)
    proposal = trotter_depth
    accept_path = 7 * np.log2(n)
    coin_energy = 185 * np.log2(n)
    coin_sqrt = 2410 * n
    coin = coin_energy + coin_sqrt
    reflection = 4 * np.log2(n)
    depth = 2 * proposal + 2 * coin + accept_path + reflection
    return get_time_logical_quantum_step(n, logical_operation_time) * depth * num_quantum_queries
