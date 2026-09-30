#!/bin/bash
#SBATCH --job-name=annealing_conservative
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=annealing_conservative_%A_%a.out
#SBATCH --error=annealing_conservative_%A_%a.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math \
    -fno-math-errno -funroll-loops -DNDEBUG \
    generation_annealing_schedules.cpp -o generation_annealing_schedules_conservative.x

mkdir -p schedule_outputs_conservative

idx=$SLURM_ARRAY_TASK_ID

for n in $(seq 5 30); do
    in="ising_inputs/model_n${n}_idx${idx}.txt"
    out="schedule_outputs_conservative/schedule_n${n}_idx${idx}.txt"
    tmp="${out}.tmp"

    if [ -f "$out" ]; then
        echo "file <$in>: skipped because present"
        continue
    fi

    echo "file <$in>: processing"
    ./generation_annealing_schedules_conservative.x "$n" "$in" --conservative-fit > "$tmp"
    mv "$tmp" "$out"
done
