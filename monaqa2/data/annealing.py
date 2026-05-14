import h5py
import numpy as np
from monaqa2.data.filename import ANNEALING_SCHEDULE_FILE
from monaqa2.data.instances import load_instances
from monaqa2.mcmc.distribution import get_gibbs_distribution
from monaqa2.mcmc.model import IsingModel


def get_annealing_betas(model: IsingModel, beta_final: float = 100.0, alpha: float = np.sqrt(1 / np.e), iters: int = 50) -> list[float]:
    
    def overlap(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(np.sqrt(p * q)))

    betas = [0.0]

    while betas[-1] < beta_final:
        beta = betas[-1]
        pi = get_gibbs_distribution(model, beta)

        if overlap(pi, get_gibbs_distribution(model, beta_final)) >= alpha:
            betas.append(beta_final)
            break

        lo, hi = beta, beta_final
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if overlap(pi, get_gibbs_distribution(model, mid)) >= alpha:
                lo = mid
            else:
                hi = mid

        betas.append(lo)

    return betas



def run_experiment_to_generate_annealing_schedules() -> None:
    ANNEALING_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(ANNEALING_SCHEDULE_FILE, "a") as h5:
        beta_group = h5.require_group("beta")

        for n in range(3, 11):
            n_group = beta_group.require_group(str(n))

            for idx in range(100):
                idx_key = str(idx)
                if idx_key in n_group:
                    continue

                model = load_instances(n, idx)
                betas = np.asarray(get_annealing_betas(model), dtype=float)
                n_group.create_dataset(idx_key, data=betas)


def load_annealing_schedule(n: int, idx: int) -> np.ndarray:
    with h5py.File(ANNEALING_SCHEDULE_FILE, "r") as h5:
        return h5[f"beta/{n}/{idx}"][...]