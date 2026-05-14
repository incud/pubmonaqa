#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_n%j.out
#SBATCH --error=logs/%x_n%j.err
#SBATCH --account=IscrC_MONAQA
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --job-name=monaqa_spectral_gap
#SBATCH --mem=4G

set -euo pipefail

: "${N:?Need to set N}"
: "${IDX_MIN:?Need to set IDX_MIN}"
: "${IDX_MAX:?Need to set IDX_MAX}"

module load python/3.11
source "$HOME/monaqa_venv/bin/activate"

echo "Running spectral gaps for n=${N}, idx_min=${IDX_MIN}, idx_max=${IDX_MAX}"

srun python3 launcher_spectral_gap.py \
    --n "${N}" \
    --idx-min "${IDX_MIN}" \
    --idx-max "${IDX_MAX}"
