"""Collect exact-enumeration timing outputs into a pandas table.

Expected input files:
    enumeration_output/enumeration_n{n}_idx{instance}.txt

Missing files, unreadable files, malformed values, and missing keys are represented by NaN.
The output pickle has columns: n, instance, states, sum, seconds_total,
seconds_per_operation, ns_per_operation, completed.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

N_VALS = range(5, 31)
INSTANCE_VALS = range(10)
KEYS = ["states", "sum", "seconds_total", "seconds_per_operation", "ns_per_operation", "completed"]


def read_timing(path: Path) -> dict[str, float]:
    """Return parsed timing values; return empty dict if the file is missing or unreadable."""
    if not path.exists():
        return {}

    try:
        with path.open("r") as f:
            text = f.read()
    except OSError:
        return {}

    out = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in KEYS:
            continue

        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            out[parts[0]] = np.nan

    return out


def collect(input_dir: Path, output_pkl: Path) -> pd.DataFrame:
    """Build and save the table with one row per (n, instance)."""
    rows = []

    for n in N_VALS:
        for instance in INSTANCE_VALS:
            values = read_timing(input_dir / f"enumeration_n{n}_idx{instance}.txt")
            rows.append((n, instance, *(values.get(key, np.nan) for key in KEYS)))

    table = pd.DataFrame(rows, columns=["n", "instance", *KEYS])
    table.to_pickle(output_pkl)
    return table


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("enumeration_output"))
    parser.add_argument("--output", type=Path, default=Path("estimation_timing.pkl"))
    args = parser.parse_args()

    table = collect(args.input_dir, args.output)
    print(f"saved {args.output} with shape {table.shape}; finite values {int(table['seconds_per_operation'].notna().sum())}/{len(table)}")


if __name__ == "__main__":
    main()
