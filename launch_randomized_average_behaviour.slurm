#!/bin/bash
#SBATCH --job-name=randomized_average
#SBATCH --account=Sis26_mazzola
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=2G
#SBATCH --output=log_randomized_average_%x_%j.out
#SBATCH --error=log_randomized_average_%x_%j.err

# Submit all sizes with: bash submit_randomized_average_behaviour.sh
set -euo pipefail
source "$HOME/monaqa_venv/bin/activate"

# Keep NumPy/BLAS within the one CPU allocated to this process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

N=$1
MIN_INST=$2
MAX_INST=$3

exec python3 -u launch_randomized_average_behaviour.py "$N" "$MIN_INST" "$MAX_INST"
