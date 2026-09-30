#!/bin/bash
#SBATCH --job-name=annealing_schedules_safe
#SBATCH --account=
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --array=0-99
#SBATCH --output=log_annealing_schedules_safe_%A_%a.out
#SBATCH --error=log_annealing_schedules_safe_%A_%a.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

# check that the safety value has been specified
if [ $# -lt 1 ]; then
    echo "usage: sbatch generation_annealing_schedules_safe.slurm <fixed_safety>"
    exit 1
fi

fixed_safety=$1

if ! [[ "$fixed_safety" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "error: fixed_safety must be a positive number"
    exit 1
fi

awk "BEGIN {exit !($fixed_safety > 0)}" || { echo "error: fixed_safety must be > 0"; exit 1; }

# we need many parallel executable otherwise we might race all the compilations
g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math -fno-math-errno -funroll-loops -DNDEBUG generation_annealing_schedules.cpp -o generation_annealing_schedules_${SLURM_ARRAY_TASK_ID}.x

mkdir -p schedule_outputs_fixed_safety
find schedule_outputs_fixed_safety -type f -name "*.txt" -empty -delete
echo "$fixed_safety" > schedule_outputs_fixed_safety/chosen_fixed_safety.txt

idx=$SLURM_ARRAY_TASK_ID

for n in $(seq 5 30); do
    in="ising_inputs/model_n${n}_idx${idx}.txt"
    out="schedule_outputs_fixed_safety/schedule_n${n}_idx${idx}.txt"
    tmp="${out}.tmp"

    if [ -f "$out" ]; then
        echo "file <$in>: skipped because present"
        continue
    fi

    echo "file <$in>: processing"
    ./generation_annealing_schedules_${SLURM_ARRAY_TASK_ID}.x "$n" "$in" --fixed-safety "$fixed_safety" > "$tmp"
    mv "$tmp" "$out"
done

rm generation_annealing_schedules_${SLURM_ARRAY_TASK_ID}.x
