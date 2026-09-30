"""Collect annealing-schedule outputs into one pandas table.

Expected input files:
    schedule_outputs/schedule_n{n}_idx{idx}.txt

The output pickle has columns:
    n, idx, L, mean_safety, std_safety, max_safety, num_safety_gt_1
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

N_VALS = range(5, 31)
IDX_VALS = range(100)


def read_schedule(path: Path) -> list[float]:
    """Return the safety factors; return an empty list if missing or unreadable."""
    if not path.exists():
        return []

    try:
        with path.open("r") as f:
            text = f.read()
    except OSError:
        return []

    safety = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        try:
            safety.append(float(parts[2]))
        except ValueError:
            continue

    return safety


def collect(input_dir: Path, output_pkl: Path) -> pd.DataFrame:
    """Build and save one row per instance."""
    rows = []

    for n in N_VALS:
        for idx in IDX_VALS:
            safety = read_schedule(input_dir / f"schedule_n{n}_idx{idx}.txt")

            if safety:
                x = np.asarray(safety, dtype=float)
                rows.append((n, idx, len(x), x.mean(), x.std(), x.max(), int((x > 1.0).sum())))
            else:
                rows.append((n, idx, np.nan, np.nan, np.nan, np.nan, np.nan))

    columns = ["n", "idx", "L", "mean_safety", "std_safety", "max_safety", "num_safety_gt_1"]
    table = pd.DataFrame(rows, columns=columns)
    table.to_pickle(output_pkl)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("schedule_outputs"))
    parser.add_argument("--output", type=Path, default=Path("schedule_summary.pkl"))
    args = parser.parse_args()

    table = collect(args.input_dir, args.output)
    print(f"saved {args.output} with shape {table.shape}; complete schedules {int(table['L'].notna().sum())}/{len(table)}")


if __name__ == "__main__":
    main()
