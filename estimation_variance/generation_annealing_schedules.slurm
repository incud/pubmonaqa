#!/bin/bash
#SBATCH --job-name=annealing_schedules
#SBATCH --account=
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --output=log_annealing_schedules_%j.out
#SBATCH --error=log_annealing_schedules_%j.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --ntasks=1 --cpus-per-task=1 bash generation_annealing_schedules.sh
