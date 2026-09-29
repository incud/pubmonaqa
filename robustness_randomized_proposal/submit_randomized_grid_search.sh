#!/bin/bash
# Place all three launcher files in the repository root, then run this script.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Each sbatch call submits one independent Python process.
# Columns: N, number of processes, memory per process in GB.
while read -r N PROCESSES MEMORY; do
    INSTANCES_PER_TASK=$((100 / PROCESSES))
    for ((MIN_INST = 0; MIN_INST < 100; MIN_INST += INSTANCES_PER_TASK)); do
        MAX_INST=$((MIN_INST + INSTANCES_PER_TASK - 1))
        sbatch --job-name="grid_n${N}_inst${MIN_INST}_to_${MAX_INST}" \
            --mem="${MEMORY}G" \
            launch_randomized_grid_search.slurm "$N" "$MIN_INST" "$MAX_INST"
    done
done <<'JOBS'
4 1 2
5 1 2
6 1 4
7 4 4
8 10 8
9 25 8
10 100 16
JOBS
