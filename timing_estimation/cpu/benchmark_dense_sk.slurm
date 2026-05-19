#!/bin/bash
#SBATCH --job-name=cpu_time
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1G
#SBATCH --output=sk_move_latency_%j.out
#SBATCH --error=sk_move_latency_%j.err

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SRC=benchmark_dense_sk_fixedN_bitset_v5.cpp
OUTDIR=results_${SLURM_JOB_ID:-manual}
mkdir -p "$OUTDIR"

for n in $(seq 10 10); do
    exe="$OUTDIR/dense_sk_bench_N${n}"
    csv="$OUTDIR/dense_sk_single_move_N${n}.csv"

    echo "===== COMPILE SK_N=${n} ====="
    g++ -O3 -march=native -DNDEBUG -std=c++20 -DSK_N=${n} "$SRC" -o "$exe"

    echo "===== RUN SK_N=${n} ====="
    srun "$exe" --instances 100 --steps 1000000 --seed 123456789 --beta 1.0 --out "$csv"
done

echo "===== DONE ====="
date
