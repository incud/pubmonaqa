#!/bin/bash
#SBATCH --job-name=cpu_time
#SBATCH --account=IscrC_MONAQA
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=cpu_timing_log_%j.out
#SBATCH --error=cpu_timing_log_%j.err

set -euo pipefail

echo "===== JOB INFO ====="
date
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-NA}"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME:-NA}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-NA}"
echo "SLURM_CPU_BIND: ${SLURM_CPU_BIND:-NA}"
echo

echo "===== CPU INFO ====="
lscpu
echo

echo "===== CPU TOPOLOGY ====="
lscpu -e=CPU,SOCKET,CORE,NODE,ONLINE,MAXMHZ,MINMHZ,MHZ || true
echo

echo "===== NUMA INFO ====="
numactl -H || true
echo

echo "===== FREQUENCY GOVERNOR ====="
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "$f: $(cat "$f" 2>/dev/null)"
done | sort -V | head -20
echo

echo "===== CURRENT FREQUENCIES, FIRST 20 CORES ====="
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    echo "$f: $(cat "$f" 2>/dev/null)"
done | sort -V | head -20
echo

echo "===== COMPILER ====="
which g++ || true
g++ --version || true
echo

echo "===== MODULES ====="
module list 2>&1 || true
echo

module purge
module load profile/base
module load gcc/12.2.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SRC=benchmark_dense_sk.cpp
OUTDIR=results_${SLURM_JOB_ID:-manual}
mkdir -p "$OUTDIR"

for n in $(seq 10 100); do
    exe="$OUTDIR/dense_sk_executable_N${n}"
    csv="$OUTDIR/dense_sk_timing_N${n}.csv"
    steps=10000000

    echo "===== COMPILE SK_N=${n} ====="
    g++ -O3 -march=native -DNDEBUG -std=c++20 -DSK_N=${n} "$SRC" -o "$exe"

    echo "===== RUN SK_N=${n} ====="
    echo "Instances 100 | Steps $steps | Seed 123456789 | beta 1.0" || true
    srun "$exe" --instances 100 --steps "$steps" --seed 123456789 --beta 1.0 --out "$csv"
done

echo "===== DONE ====="
date
