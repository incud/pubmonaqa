from pathlib import Path

MONAQA2_PARENT = Path(__file__).resolve().parents[2]

ISING_INSTANCES_FILE = MONAQA2_PARENT / "data/ising_instances.hdf5"

BEST_HYPERPARAMS_JSON_FILE_LIST = [
    (3, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n3.json"),
    (4, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n4.json"),
    (5, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n5.json"),
    (5, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n5_fine.json"),
    (6, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n6.json"),
    (6, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n6_fine.json"),
    (7, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n7.json"),
    (7, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n7_fine.json"),
    (8, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n8.json"),
    (8, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n8_fine.json"),
    (9, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n9.json"),
    (9, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n9_fine.json"),
    (10, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n10.json"),
    (10, MONAQA2_PARENT / "../quantum-mcmc-main/data/grid_search_n10_fine.json")
]
BEST_HYPERPARAMS_QEMC_FILE = MONAQA2_PARENT / "data/best_hyperparams_qemc.hdf5"

SPECTRAL_GAP_FILE = MONAQA2_PARENT / "data/spectral_gaps_merged.pkl"

ANNEALING_SCHEDULE_FILE = MONAQA2_PARENT / "data/annealing_schedules.hdf5"

CLASSICAL_QUERY_FILE = MONAQA2_PARENT / "data/classical_queries_merged.pkl"

QUANTUM_QUERY_FILE = MONAQA2_PARENT / "data/quantum_queries.pkl"

QUANTUM_RUNTIME_PROPOSAL_UNIFORM = MONAQA2_PARENT / "data/quantum_runtime_proposal_uniform.pkl"

QUANTUM_RUNTIME_PROPOSAL_LOCAL = MONAQA2_PARENT / "data/quantum_runtime_proposal_local.pkl"

QUANTUM_RUNTIME_PROPOSAL_QEMC = MONAQA2_PARENT / "data/quantum_runtime_proposal_qemc.pkl"

QUANTUM_RUNTIME_REFLECTION = MONAQA2_PARENT / "data/quantum_runtime_reflection.pkl"

QUANTUM_RUNTIME_ACCEPT_PATH = MONAQA2_PARENT / "data/quantum_runtime_accept_path.pkl"

