from pathlib import Path
import numpy as np
import pandas as pd

N_VALS = list(range(5, 30 + 1))
BETA_VALS = [0.0, 1.0 / 32.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 8.0, 3.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 2.0, 5.0 / 8.0, 3.0 / 4.0, 13.0 / 16.0, 7.0 / 8.0, 15.0 / 16.0, 31.0 / 32.0, 1.0, 33.0 / 32.0, 17.0 / 16.0, 9.0 / 8.0, 19.0 / 16.0, 5.0 / 4.0, 11.0 / 8.0, 3.0 / 2.0, 7.0 / 4.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
INSTANCE_VALS = list(range(100))
OUTPUT_DIR = Path("variance_outputs")
OUTPUT_PKL = Path("var_energies.pkl")


def parse_variance_file(path: Path) -> dict[float, float]:
    if not path.exists():
        return {}

    try:
        with path.open("r") as f:
            text = f.read()
    except OSError:
        return {}

    values = {}

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        try:
            beta = float(parts[0])
            variance = float(parts[1])
        except ValueError:
            continue

        values[beta] = variance

    return values


def main() -> None:
    data = np.full((len(N_VALS), len(BETA_VALS), len(INSTANCE_VALS)), np.nan, dtype=float)

    for ni, n in enumerate(N_VALS):
        for ii, instance in enumerate(INSTANCE_VALS):
            path = OUTPUT_DIR / f"variance_n{n}_idx{instance}.txt"
            values = parse_variance_file(path)

            if not values:
                continue

            for bi, beta in enumerate(BETA_VALS):
                if beta in values:
                    data[ni, bi, ii] = values[beta]

    index = pd.MultiIndex.from_product([N_VALS, BETA_VALS], names=["n", "beta"])
    columns = pd.Index(INSTANCE_VALS, name="instance")
    table = pd.DataFrame(data.reshape(len(N_VALS) * len(BETA_VALS), len(INSTANCE_VALS)), index=index, columns=columns)
    table.attrs["shape_3d"] = [len(N_VALS), len(BETA_VALS), len(INSTANCE_VALS)]
    table.attrs["n_vals"] = N_VALS
    table.attrs["beta_vals"] = BETA_VALS
    table.attrs["instance_vals"] = INSTANCE_VALS
    table.attrs["missing_value"] = "np.nan"
    table.to_pickle(OUTPUT_PKL)

    present = int(np.isfinite(data).sum())
    total = int(data.size)
    print(f"saved {OUTPUT_PKL} with dataframe shape {table.shape} and tensor shape {data.shape}; finite values {present}/{total}")


if __name__ == "__main__":
    main()
