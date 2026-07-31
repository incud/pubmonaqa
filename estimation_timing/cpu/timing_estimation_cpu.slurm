#!/bin/bash
#SBATCH --job-name=jcpu
#SBATCH --account=
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=log_timing_estimation_cpu_%j.out
#SBATCH --error=log_timing_estimation_cpu_%j.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

SRC=timing_estimation_cpu.cpp
NS=(64 128 256 512)
N_STEPS=100000
N_MODELS=10
SEED=123456789
BETA=1.0
CXX=g++
CXXFLAGS="-std=c++20 -O3 -march=native -mtune=native -ffast-math -fno-math-errno -funroll-loops -DNDEBUG"


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "===== JOB INFO ====="
date
echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-manual}"
echo "Ns: ${NS[*]} | N_MODELS=$N_MODELS | N_STEPS=$N_STEPS | SEED=$SEED | BETA=$BETA"
echo

echo "===== CPU INFO ====="
lscpu | grep -E 'Model name|Architecture|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|Flags' || true
echo

echo "===== COMPILER ====="
which "$CXX"
"$CXX" --version | head -1
echo "$CXXFLAGS"
echo

for n in "${NS[@]}"; do
    exe="./timing_cpu_N${n}"
    echo "===== N=$n ====="
    "$CXX" $CXXFLAGS -DSK_N="$n" -DN_MODELS="$N_MODELS" -DN_STEPS="$N_STEPS" -DSEED="$SEED" -DBETA="$BETA" "$SRC" -o "$exe"
    srun "$exe"
    rm -f "$exe"
    echo
done
