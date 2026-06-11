#!/bin/bash
#SBATCH --job-name=jgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H100:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=log_timing_estimation_gpu_%j.out
#SBATCH --error=log_timing_estimation_gpu_%j.err

set -euo pipefail

module purge
module load h100 
module load dev2026a 
module load cuda

SRC=timing_estimation_gpu.cu
NS=(64 128 256 512)
N_MODELS=1
CHAINS_PER_MODEL=1
N_STEPS=1000000
SEED=123456789
BETA=1.0f
BLOCK_THREADS=128
RNG_POOL_SIZE=1048576
GPU_ARCH=sm_90

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

printf 'job=%s host=%s src=%s\n' "${SLURM_JOB_ID:-manual}" "$(hostname)" "$SRC"
nvidia-smi -L
nvcc --version | sed -n '1,4p'

NVCCFLAGS=(-std=c++17 -O3 --use_fast_math -arch="$GPU_ARCH" -lineinfo -Xcompiler=-O3 -Xcompiler=-DNDEBUG)
DEFS=(-DN_MODELS="$N_MODELS" -DCHAINS_PER_MODEL="$CHAINS_PER_MODEL" -DN_STEPS="$N_STEPS" -DSEED="$SEED" -DBETA="$BETA" -DBLOCK_THREADS="$BLOCK_THREADS" -DRNG_POOL_SIZE="$RNG_POOL_SIZE")

for n in "${NS[@]}"; do
    exe="timing_gpu_N$n"
    echo "=== N=$n ==="
    nvcc "${NVCCFLAGS[@]}" -DSK_N="$n" "${DEFS[@]}" "$SRC" -o "$exe"
    srun "$exe"
done
