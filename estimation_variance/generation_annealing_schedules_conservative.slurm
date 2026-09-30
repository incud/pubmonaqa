#!/bin/bash
#SBATCH --job-name=conservative_annealing
#SBATCH --account=
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --array=0-99
#SBATCH --output=conservative_annealing_%A_%a.out
#SBATCH --error=conservative_annealing_%A_%a.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math \
    -fno-math-errno -funroll-loops -DNDEBUG \
    generation_annealing_schedules.cpp \
    -o generation_annealing_schedules_conservative_${SLURM_ARRAY_TASK_ID}.x

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
    ./generation_annealing_schedules_conservative_${SLURM_ARRAY_TASK_ID}.x "$n" "$in" --conservative-fit > "$tmp"
    mv "$tmp" "$out"
done

rm -f generation_annealing_schedules_conservative_${SLURM_ARRAY_TASK_ID}.x