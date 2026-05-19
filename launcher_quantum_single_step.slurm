#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=IscrC_MONAQA
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --job-name=monaqa_quantum_single_step
#SBATCH --mem=4G

set -euo pipefail

module load python/3.11
module load profile/quantum
source "$HOME/monaqa_venv/bin/activate"

mkdir -p logs

: "${EXPERIMENT:?Set EXPERIMENT=qemc or EXPERIMENT=coin}"
: "${BETA:?Set BETA, e.g. BETA=1.0}"

echo "Running quantum single-step experiment=${EXPERIMENT}, beta=${BETA}"

srun python3 launcher_quantum_single_step.py \
    --experiment "${EXPERIMENT}" \
    --beta "${BETA}"
