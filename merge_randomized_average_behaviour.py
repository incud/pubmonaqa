"""Run: python3 merge_randomized_average_behaviour.py

Merge the old aggregate and per-job files, retaining every T and P approach.
Tables keep the original [n - 3][instance] layout. Non-null job values replace
existing values; overlapping job files are processed in filename order.
Missing entries stay null: rows n=3..10 and instances 0..99 are never compacted.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from monaqa2.data.filename import CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE
from monaqa2.filename import MONAQA2_PARENT

NS = range(3, 11)
IDXS = range(100)
KS = range(1, 6)
APPROACHES = (
    ["T_exact"]
    + [f"T_grid_k{k}" for k in KS]
    + ["P_exact"]
    + [f"P_grid_k{k}" for k in KS]
)


def read_results(path):
    """Read one file, converting the earlier nested format when necessary.

    :param path: Path to an aggregate or per-job JSON file.
    :return: Dictionary of approach names and [n - 3][instance] tables.
    """
    with path.open() as f:
        results = json.load(f)
    for approach, values in results.items():
        if isinstance(values, dict):
            # The first launcher saved P gaps as n=... -> inst... -> gap.
            table = [[None] * len(IDXS) for _ in NS]
            for n, instances in values.items():
                for instance, gap in instances.items():
                    table[int(n.removeprefix("n=")) - NS.start][
                        int(instance.removeprefix("inst"))
                    ] = gap
            results[approach] = table
    return results


def main():
    """Merge matching files into CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE.

    :return: None.
    """
    data_dir = Path(MONAQA2_PARENT) / "data"
    output = Path(CONVERGENCE_RANDOMIZED_HYPERPARAMS_FILE)
    pattern = (
        rf"{re.escape(output.stem)}_n\d+_inst\d+_to_\d+"
        rf"{re.escape(output.suffix)}"
    )
    job_files = sorted(
        path for path in data_dir.glob(f"{output.stem}_*{output.suffix}")
        if re.fullmatch(pattern, path.name)
    )
    # Start with the old aggregate, then apply completed job results.
    files = ([output] if output.exists() else []) + job_files
    if not files:
        print(f"No results found in {data_dir}.")
        return

    # Reserve every position, including sizes and approaches with no results yet.
    combined = {
        approach: [[None] * len(IDXS) for _ in NS] for approach in APPROACHES
    }
    with ThreadPoolExecutor() as pool:
        # Reads run in parallel; map returns results in the order of files.
        for results in pool.map(read_results, files):
            for approach, rows in results.items():
                table = combined.setdefault(
                    approach, [[None] * len(IDXS) for _ in NS]
                )
                while len(table) < len(rows):
                    table.append([None] * len(IDXS))
                for i, row in enumerate(rows):
                    table[i].extend([None] * (len(row) - len(table[i])))
                    for idx, gap in enumerate(row):
                        # Write at the original index; never append just the gaps.
                        # Null placeholders must not erase completed values.
                        if gap is not None:
                            table[i][idx] = gap

    # Replace the aggregate only after the complete JSON has been written.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(combined, f)
    temporary.replace(output)
    print(f"Merged {len(job_files)} job files into {output}.")


if __name__ == "__main__":
    main()
