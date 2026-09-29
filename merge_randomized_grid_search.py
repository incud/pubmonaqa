"""Run: python3 merge_randomized_grid_search.py

Merge grid-search files as k=... -> n=... -> inst... -> [time][gamma] P gaps.
Completed job results replace existing instances; overlapping job files are
processed in filename order. Individual gaps are preserved without averaging.
For each k, sizes n=3..10 contain ordered inst000..inst099 entries, with null
for missing results. The indices of completed instances never shift.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from monaqa2.data.filename import GRIDSEARCH_RANDOMIZED_HYPERPARAMS_FILE
from monaqa2.filename import MONAQA2_PARENT

NS = range(3, 11)
IDXS = range(100)


def read_results(path):
    """Read one grid-search result file.

    :param path: Path to an aggregate or per-job JSON file.
    :return: Nested dictionary of grid-search results.
    """
    with path.open() as f:
        return json.load(f)


def main():
    """Merge matching files into GRIDSEARCH_RANDOMIZED_HYPERPARAMS_FILE.

    :return: None.
    """
    data_dir = Path(MONAQA2_PARENT) / "data"
    output = Path(GRIDSEARCH_RANDOMIZED_HYPERPARAMS_FILE)
    pattern = (
        rf"{re.escape(output.stem)}_k\d+_n\d+_inst\d+_to_\d+"
        rf"{re.escape(output.suffix)}"
    )
    job_files = sorted(
        path for path in data_dir.glob(f"{output.stem}_*{output.suffix}")
        if re.fullmatch(pattern, path.name)
    )
    files = ([output] if output.exists() else []) + job_files
    if not files:
        print(f"No results found in {data_dir}.")
        return

    combined = {}
    with ThreadPoolExecutor() as pool:
        # Read in parallel, but merge the aggregate before the job files.
        for results in pool.map(read_results, files):
            for k, sizes in results.items():
                # Preallocate in numeric order, even for completely missing jobs.
                target_sizes = combined.setdefault(k, {
                    f"n={n}": {f"inst{idx:03d}": None for idx in IDXS} for n in NS
                })
                for n, instances in sizes.items():
                    target = target_sizes.setdefault(
                        n, {f"inst{idx:03d}": None for idx in IDXS}
                    )
                    for instance, gaps in instances.items():
                        # An unfinished instance cannot erase a completed grid.
                        if gaps is not None:
                            instance = f"inst{int(instance.removeprefix('inst')):03d}"
                            target[instance] = gaps

    # Replace the aggregate only after the complete JSON has been written.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(combined, f)
    temporary.replace(output)
    print(f"Merged {len(job_files)} job files into {output}.")


if __name__ == "__main__":
    main()
