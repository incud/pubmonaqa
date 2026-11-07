from monaqa2.data.filename import ISING_INSTANCES_FILE
from monaqa2.mcmc.model import IsingModel


def load_instances(n: int, idx: int) -> IsingModel:
    """Load one stored Ising model; HDF5 support is needed only for this call."""
    import h5py

    n = int(n)
    idx = int(idx)
    if not 3 <= n <= 10 or not 0 <= idx < 100:
        raise ValueError("Stored instances require 3 <= n <= 10 and 0 <= idx < 100.")
    with h5py.File(ISING_INSTANCES_FILE, "r") as file:
        coeffs = file['coefficients'][f'{n}'][:, idx]
    return IsingModel.from_coefficients(n=n, coefficients=coeffs)
