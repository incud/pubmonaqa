#!/bin/bash
#SBATCH --job-name=sk_gaps
#SBATCH --account=IscrC_MONAQA
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=spectral_gaps_%A_%a.out
#SBATCH --error=spectral_gaps_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/monaqa_venv/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python3 spectral_gaps_models.py analyze "$SLURM_ARRAY_TASK_ID"
