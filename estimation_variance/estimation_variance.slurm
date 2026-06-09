#!/bin/bash
#SBATCH --job-name=variance
#SBATCH --account=IscrC_MONAQA
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --output=log_variance_%j.out
#SBATCH --error=log_variance_%j.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math -fno-math-errno -funroll-loops -DNDEBUG estimation_variance_new.cpp -o estimation_variance_new.x

mkdir -p variance_outputs

for idx in $(seq 0 9); do
    for n in $(seq 5 25); do
        out="variance_outputs/variance_n${n}_idx${idx}.txt"

        if [ -f "$out" ]; then
            echo "skipping n=$n idx=$idx because $out already exists"
            continue
        fi

        echo "running n=$n idx=$idx"
        srun --ntasks=1 --cpus-per-task=1 ./estimation_variance_new.x "$n" "ising_inputs/model_n${n}_idx${idx}.txt" > "$out" & sleep 0.5
    done
done
