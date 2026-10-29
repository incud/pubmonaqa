#!/bin/bash
#SBATCH --job-name=schedule_gaps
#SBATCH --account=Sis26_mazzola
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --array=0-599
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --output=log_schedule_gaps_%A_%a.out
#SBATCH --error=log_schedule_gaps_%A_%a.err

set -euo pipefail
source "$HOME/monaqa_venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# Prevent NumPy/BLAS from spawning additional CPU threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

n=$((5 + SLURM_ARRAY_TASK_ID / 100))
idx=$((SLURM_ARRAY_TASK_ID % 100))
python3 -u validate_schedule_gaps.py \
    "schedule_outputs_conservative/schedule_n${n}_idx${idx}.txt" \
    "schedule_outputs_conservative/gaps_${n}_idx${idx}.txt"
