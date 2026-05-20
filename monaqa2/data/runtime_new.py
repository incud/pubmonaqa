from monaqa2.data.single_step_cpu import cpu_time_per_local_move, cpu_time_per_uniform_move
import numpy as np


def get_time_logical_quantum_step(n: int):
    return 1e-6 * n

def get_time_classical_walk_uniform(n: int, num_classical_queries: int):
    return cpu_time_per_uniform_move(n) * num_classical_queries

def get_time_classical_walk_local(n: int, num_classical_queries: int):
    return cpu_time_per_local_move(n) * num_classical_queries

def get_time_classical_walk_qemc(n: int, num_classical_queries: int, trotter_steps):
    # second order trotter gets simplified
    trotter_depth = 1 + trotter_steps * (1 + n + 1) # X + #TR * (Z + ZZ + X)
    return get_time_logical_quantum_step(n) * trotter_depth * num_classical_queries

def get_time_quantum_walk_uniform(n: int, num_quantum_queries: int):
    return 0.0

def get_time_quantum_walk_local(n: int, num_quantum_queries: int):
    return 0.0

def get_time_quantum_walk_qemc(n: int, num_quantum_queries: int):
    return 0.0
