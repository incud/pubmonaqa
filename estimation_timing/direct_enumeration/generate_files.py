# allows us to have visibility on our package without installing it in editing mode
import sys
if "../.." not in sys.path: sys.path.append("../..")

from pathlib import Path
import numpy as np
from monaqa2.mcmc.model import RandomIsingModel

N_INSTANCES = 10
OUT_DIR = Path("ising_inputs")
OUT_DIR.mkdir(exist_ok=True)

for n in range(5, 30 + 1):
    for j in range(N_INSTANCES):
        model = RandomIsingModel(n, seed=n * N_INSTANCES + j)

        h = np.asarray(model.h_rescaled, dtype=float)
        J = np.asarray(model.J_rescaled, dtype=float)

        filename = OUT_DIR / f"model_n{n}_idx{j}.txt"

        with open(filename, "w") as f:
            for i in range(n):
                f.write(f"{h[i]:.17g}\n")

            for i in range(n):
                for k in range(i + 1, n):
                    f.write(f"{J[i, k]:.17g}\n")

