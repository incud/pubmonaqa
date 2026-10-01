"""Collect adaptive, fixed-safety, and conservative annealing schedules."""

from pathlib import Path
import numpy as np
import pandas as pd

N_VALS = range(5, 31)
IDX_VALS = range(100)
MIN_OVERLAP = np.exp(-1.0)

CASES = {
    "schedule_outputs": "schedule_summary.pkl",
    "schedule_outputs_safe": "schedule_summary_safe.pkl",
    "schedule_outputs_conservative": "schedule_summary_conservative.pkl",
}

COLUMNS = [
    "n", "idx", "L",
    "overlap_min", "overlap_num_invalid", "overlap_invalid_mean", "overlap_invalid_std",
    "S_max", "S_num_invalid", "S_invalid_mean", "S_invalid_std",
]


def read_schedule(path: Path):
    if not path.exists():
        return [], []

    try:
        text = path.read_text()
    except OSError:
        return [], []

    overlaps, safety = [], []

    for line in text.splitlines():
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        try:
            overlaps.append(float(parts[1]))
            safety.append(float(parts[2]))
        except ValueError:
            continue

    return overlaps, safety


def invalid_stats(values, mask):
    x = np.asarray(values, dtype=float)
    bad = x[mask(x)]
    return len(bad), bad.mean() if len(bad) else np.nan, bad.std() if len(bad) else np.nan


def collect(input_dir: Path, output: Path):
    rows = []

    for n in N_VALS:
        for idx in IDX_VALS:
            overlaps, safety = read_schedule(input_dir / f"schedule_n{n}_idx{idx}.txt")

            if not overlaps:
                rows.append((n, idx) + (np.nan,) * 9)
                continue

            o = np.asarray(overlaps, dtype=float)
            s = np.asarray(safety, dtype=float)

            onum, omean, ostd = invalid_stats(o, lambda x: x < MIN_OVERLAP)
            snum, smean, sstd = invalid_stats(s, lambda x: x > 1.0)

            rows.append((n, idx, len(o), o.min(), onum, omean, ostd, s.max(), snum, smean, sstd))

    table = pd.DataFrame(rows, columns=COLUMNS)
    table.to_pickle(output)
    print(f"saved {output} with shape {table.shape}; complete schedules {int(table['L'].notna().sum())}/{len(table)}")


for folder, output in CASES.items():
    collect(Path(folder), Path(output))
