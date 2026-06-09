"""Collect SK energy-variance outputs into a long-form pandas table.

Expected input files:
    variance_outputs/variance_n{n}_idx{instance}.txt

Each file is read-only opened, fully read, and closed immediately. Missing files,
unreadable files, malformed lines, and missing beta values are represented by NaN.
The output pickle has columns: n, beta, instance, var_energy.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

N_VALS = range(5, 31)
INSTANCE_VALS = range(100)
BETA_VALS = [0.0, 1.0 / 32.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 8.0, 3.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 2.0, 5.0 / 8.0, 3.0 / 4.0, 13.0 / 16.0, 7.0 / 8.0, 15.0 / 16.0, 31.0 / 32.0, 1.0, 33.0 / 32.0, 17.0 / 16.0, 9.0 / 8.0, 19.0 / 16.0, 5.0 / 4.0, 11.0 / 8.0, 3.0 / 2.0, 7.0 / 4.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]


def read_variances(path: Path) -> dict[float, float]:
    """Return {beta: variance}; return empty dict if the file is missing or unreadable."""
    if not path.exists():
        return {}

    try:
        # Open read-only, read everything, then close immediately.
        with path.open("r") as f:
            text = f.read()
    except OSError:
        return {}

    out = {}
    for line in text.splitlines():
        # Expected line format: "beta variance". Ignore extra columns and bad lines.
        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            out[float(parts[0])] = float(parts[1])
        except ValueError:
            continue

    return out


def collect(input_dir: Path, output_pkl: Path) -> pd.DataFrame:
    """Build and save the long table with columns n, beta, instance, var_energy."""
    rows = []

    for n in N_VALS:
        for instance in INSTANCE_VALS:
            values = read_variances(input_dir / f"variance_n{n}_idx{instance}.txt")

            for beta in BETA_VALS:
                # Missing beta values are written as NaN.
                rows.append((n, beta, instance, values.get(beta, np.nan)))

    table = pd.DataFrame(rows, columns=["n", "beta", "instance", "var_energy"])
    table.to_pickle(output_pkl)
    return table


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("variance_outputs"))
    parser.add_argument("--output", type=Path, default=Path("var_energies.pkl"))
    args = parser.parse_args()

    table = collect(args.input_dir, args.output)
    print(f"saved {args.output} with shape {table.shape}; finite values {int(table['var_energy'].notna().sum())}/{len(table)}")


if __name__ == "__main__":
    main()
