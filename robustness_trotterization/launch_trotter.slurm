#!/bin/bash
#SBATCH --job-name=trotter_gaps
#SBATCH --account=IscrC_MONAQA
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --array=0-299%200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --output=trotter_%A_%a.out
#SBATCH --error=trotter_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/monaqa_venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
idx=$((SLURM_ARRAY_TASK_ID / 3))
group=$((SLURM_ARRAY_TASK_ID % 3))
python3 spectral_gaps_trotter.py analyze "$idx" "$group"
