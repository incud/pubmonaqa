import numpy as np
from monaqa2.mcmc.model import IsingModel
from monaqa2.mcmc.search import search_monotone
from monaqa2.mcmc.utils_hamiltonian import energies


def get_gibbs_distribution(model: IsingModel, beta: float) -> np.ndarray:
    """Stable Gibbs distribution p(x) ∝ exp(-beta * E(x))."""
    E = model.energies_rescaled
    pi_beta = get_gibbs_distribution_from_energies(E, beta)
    pi_beta[pi_beta <= 0.0] = 0.0
    return pi_beta


def get_gibbs_distribution_hJ(h, J, beta: float) -> np.ndarray:
    E = energies(h, J)
    pi_beta = get_gibbs_distribution_from_energies(E, beta)
    pi_beta[pi_beta <= 0.0] = 0.0
    return pi_beta


def get_gibbs_distribution_from_energies(energies: np.ndarray, beta: float) -> np.ndarray:
    """Stable Gibbs distribution p(x) ∝ exp(-beta * E(x))."""
    logits = -beta * energies
    m = float(np.max(logits))
    w = np.exp(logits - m)
    w[w <= 0.0] = 0.0
    Z = float(np.sum(w))
    if Z <= 0.0 or not np.isfinite(Z):
        raise RuntimeError("Partition function computation failed (overflow/underflow).")
    return w / Z


def calculate_warmness_parameter_alpha(q: np.ndarray, pi: np.ndarray) -> float:
    """Calculate the warmness parameter alpha such that
    alpha = <psi_q | psi_pi> = sum_x sqrt(q(x) * pi(x))
    where psi_q and psi_pi are the quantum states corresponding 
    to the distributions q and pi, i.e. |psi_q> = sum_x sqrt(q(x)) |x>.
    """
    try:
        with np.errstate(invalid='raise'):
                alpha = np.sum(np.sqrt(q * pi))
    except FloatingPointError:
        print(f"sqrt got invalid values, likely because q * pi contains negatives: {np.min(q)=} {np.min(pi)=}")
        alpha = np.nan
    return float(alpha)


def get_gibbs_distribution_with_bhattacharyya_guarantee(model: IsingModel, beta: float, pi: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """
    Return q such that:
    * q = Gibbs(beta0) with beta0 in [0,target_beta]
    * alpha(q,pi) >= min_alpha = sum_x sqrt(q(x) pi(x)).
    * pi = Gibbs(target_beta)
    """
    if beta < 0:
        raise ValueError("target_beta must be >= 0.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("min_alpha must be in [0,1].")

    energies = model.energies_rescaled

    T = 1 << 20

    def fun(t: int) -> np.ndarray:
        return get_gibbs_distribution_from_energies(energies, beta * (t / T))

    def compare(q: np.ndarray) -> float:
        return alpha - calculate_warmness_parameter_alpha(q, pi)

    t_star = search_monotone(fun=fun, compare=compare, start_iter=0, max_iter=T, info="warm-start(alpha)")
    beta_star = beta * (t_star / T)
    q_ = fun(t_star)
    return q_, beta_star


def get_uniform_distribution(model: IsingModel) -> np.ndarray:
    n = model.n
    return np.ones(2**n) / 2**n
