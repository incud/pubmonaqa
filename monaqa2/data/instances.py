import h5py
from monaqa2.data.filename import ISING_INSTANCES_FILE
from monaqa2.mcmc.model import IsingModel


def load_instances(n: int, idx: int) -> IsingModel:
    n = int(n)
    idx = int(idx)
    assert 3 <= n <= 10, f"The preloaded instances have between 3 and 10 spins (you asked for {n=})"
    assert 0 <= idx <= 99, f"The preloaded instances have index between 0 and 99 included (you asked for {idx=})"
    file = h5py.File(ISING_INSTANCES_FILE)
    coeffs = file['coefficients'][f'{n}'][:, idx]
    return IsingModel.from_coefficients(n=n, coefficients=coeffs)
