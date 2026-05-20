from pathlib import Path

from monaqa2.data.filename import MONAQA2_PARENT, QUANTUM_RUNTIME_ACCEPT_PATH, QUANTUM_RUNTIME_PROPOSAL_LOCAL, QUANTUM_RUNTIME_PROPOSAL_UNIFORM, QUANTUM_RUNTIME_REFLECTION
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.model import IsingModel
from monaqa2.qiskit.accept_path_gate import AcceptPath
from monaqa2.qiskit.metropolis_hastings_energy_gate import MetropolisHastingsEnergy
from monaqa2.qiskit.proposal_local_gate import ProposalLocal
from monaqa2.qiskit.proposal_qemc_gate import ProposalQemc
from monaqa2.qiskit.proposal_uniform_gate import ProposalUniform
from monaqa2.qiskit.reflection_gate import Reflection
from monaqa2.qiskit.sqrt_exp_arithmetic_gate import SqrtExpArithmetic
from monaqa2.qiskit.utils_qiskit import get_nc_depth, get_rz_count, get_t_count, get_toffoli_count, qiskit_to_clifford_rz
from qiskit.circuit import Gate, QuantumCircuit
import numpy as np
import pandas as pd


COLUMNS = ["component", "n", "idx", "n_plus_c", "beta", "eps_2_minus", "eps", "n_qubits", "depth", "t_count", "ccx_count", "rz_count", "ok", "error_message"]


def get_info(gate: Gate) -> dict:
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))
    qc = qiskit_to_clifford_rz(qc, opt=2)
    return {"n_qubits": gate.num_qubits, "depth": get_nc_depth(qc, transpile=False), "t_count": get_t_count(qc, transpile=False), "ccx_count": get_toffoli_count(qc, transpile=False), "rz_count": get_rz_count(qc, transpile=False)}


def generate_proposal_uniform(n: int):
    return get_info(ProposalUniform(n))


def generate_proposal_local(n: int):
    return get_info(ProposalLocal(n, k=1))


def generate_proposal_qemc(n: int, model: IsingModel, eps: float):
    return get_info(ProposalQemc(n, model.h_rescaled, model.J_rescaled, 0.5, 1.0, eps, mocked_reflection=False, mocked_angles=True))


def generate_reflection(n_plus_c: int): 
    return get_info(Reflection(n=n_plus_c // 2, coins=n_plus_c - (n_plus_c // 2), mocked_circuit=False))


def generate_accept_path(n_plus_c: int): 
    return get_info(AcceptPath(n=n_plus_c // 2, coins=n_plus_c - (n_plus_c // 2), mocked_circuit=False))


generate_coin_cache = {}


def generate_coin(n: int, idx: int, model: IsingModel, beta: float, eps: float):
    key = (int(n), int(idx), float(beta), float(eps))

    if key in generate_coin_cache:
        return generate_coin_cache[key]

    gate_1 = MetropolisHastingsEnergy(n, model.h_rescaled, model.J_rescaled, eps / 2)
    gate_1_info = get_info(gate_1)

    gate_2 = SqrtExpArithmetic(gate_1.signal_bits, beta, gate_1.normalization, eps / 2, mocked_circuit=False, mocked_angles=True)
    gate_2_info = get_info(gate_2)

    generate_coin_cache[key] = (gate_1_info, gate_2_info)
    return generate_coin_cache[key]


def _run(out_file: Path, jobs):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(out_file).reindex(columns=COLUMNS) if out_file.exists() else pd.DataFrame(columns=COLUMNS)
    seen = set((str(r.component), None if pd.isna(r.n) else int(r.n), None if pd.isna(r.idx) else int(r.idx), None if pd.isna(r.n_plus_c) else int(r.n_plus_c), None if pd.isna(r.beta) else float(r.beta), None if pd.isna(r.eps_2_minus) else int(r.eps_2_minus)) for r in df.itertuples())

    for meta, build in jobs:
        key = (meta["component"], meta.get("n"), meta.get("idx"), meta.get("n_plus_c"), meta.get("beta"), meta.get("eps_2_minus"))
        print("Processing", key, flush=True)
        if key in seen:
            print(f"Skipping {key}", flush=True)
            continue

        try:
            info = build()
            row = {**meta, **info, "ok": True, "error_message": ""}
        except Exception as e:
            row = {**meta, "n_qubits": np.nan, "depth": np.nan, "t_count": np.nan, "ccx_count": np.nan, "rz_count": np.nan, "ok": False, "error_message": repr(e)}

        row = {c: row.get(c, np.nan) for c in COLUMNS}
        df = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
        seen.add(key)
        df.to_pickle(out_file)
        print(".", end="", flush=True)

    print("")


def run_experiment_proposal_uniform():
    _run(QUANTUM_RUNTIME_PROPOSAL_UNIFORM, [({"component": "proposal_uniform", "n": n}, 
        lambda n=n: generate_proposal_uniform(n)) for n in range(1, 201)])


def run_experiment_proposal_local():
    _run(QUANTUM_RUNTIME_PROPOSAL_LOCAL, [({"component": "proposal_local1", "n": n}, 
        lambda n=n: generate_proposal_local(n)) for n in range(1, 201)])


def run_experiment_proposal_qemc(eps_2_minus: int):
    eps = 2.0 ** (-int(eps_2_minus))
    out_file = MONAQA2_PARENT / f"data/quantum_runtime_proposal_qemc_eps2minus{int(eps_2_minus)}.pkl"
    jobs = []
    for n in range(3, 10):
        for idx in range(100):
            jobs.append(({"component": "proposal_qemc", "n": n, "idx": idx, "eps_2_minus": int(eps_2_minus), "eps": eps}, 
                lambda n=n, idx=idx: generate_proposal_qemc(n, load_instances(n, idx), eps)))
    _run(out_file, jobs)


def run_experiment_reflection():
    _run(QUANTUM_RUNTIME_REFLECTION, [({"component": "reflection", "n_plus_c": n_plus_c}, 
        lambda n_plus_c=n_plus_c: generate_reflection(n_plus_c)) for n_plus_c in range(1, 201)])


def run_experiment_accept_path():
    _run(QUANTUM_RUNTIME_ACCEPT_PATH, [({"component": "accept_path", "n_plus_c": n_plus_c}, 
        lambda n_plus_c=n_plus_c: generate_accept_path(n_plus_c)) for n_plus_c in range(1, 201)])


def run_experiment_coin(beta: float, eps_2_minus: int):
    beta = float(beta)
    eps = 2.0 ** (-int(eps_2_minus))
    beta_tag = str(beta).replace(".", "p").replace("-", "m")
    out_file = MONAQA2_PARENT / f"data/quantum_runtime_coin_beta{beta_tag}_eps2minus{int(eps_2_minus)}.pkl"
    jobs = []

    for idx in range(100):
        for n in range(3, 11):
            def build_energy(n=n, idx=idx):
                return generate_coin(n, idx, load_instances(n, idx), beta, eps)[0]

            def build_sqrt_exp(n=n, idx=idx):
                return generate_coin(n, idx, load_instances(n, idx), beta, eps)[1]

            jobs.append(({"component": "coin_energy", "n": n, "idx": idx, "beta": beta, "eps_2_minus": int(eps_2_minus), "eps": eps}, build_energy))
            jobs.append(({"component": "coin_sqrt_exp", "n": n, "idx": idx, "beta": beta, "eps_2_minus": int(eps_2_minus), "eps": eps}, build_sqrt_exp))

    _run(out_file, jobs)


def launch_experiment_on_cineca(experiment: str, beta: float):

    assert experiment in ["qemc", "coin"]
    EPS_LIST = [2, 4, 8, 16, 32, 64, 96, 100]

    if experiment == "qemc":
        for eps_2_minus_k in EPS_LIST:
            run_experiment_proposal_qemc(eps_2_minus_k)
    
    if experiment == "coin":
        for eps_2_minus_k in EPS_LIST:
            run_experiment_coin(beta, eps_2_minus_k)