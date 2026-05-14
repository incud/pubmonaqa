#!/bin/bash
set -euo pipefail

mkdir -p logs

# n = 9, chunks of 10
sbatch --export=ALL,N=9,IDX_MIN=5,IDX_MAX=10 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=10,IDX_MAX=20 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=20,IDX_MAX=30 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=30,IDX_MAX=40 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=40,IDX_MAX=50 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=50,IDX_MAX=60 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=60,IDX_MAX=70 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=70,IDX_MAX=80 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=80,IDX_MAX=90 launcher_spectral_gap.slurm
sbatch --export=ALL,N=9,IDX_MIN=90,IDX_MAX=100 launcher_spectral_gap.slurm

# n = 10, chunks of 5
#sbatch --export=ALL,N=10,IDX_MIN=0,IDX_MAX=5 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=5,IDX_MAX=10 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=10,IDX_MAX=15 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=15,IDX_MAX=20 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=20,IDX_MAX=25 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=25,IDX_MAX=30 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=30,IDX_MAX=35 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=35,IDX_MAX=40 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=40,IDX_MAX=45 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=45,IDX_MAX=50 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=50,IDX_MAX=55 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=55,IDX_MAX=60 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=60,IDX_MAX=65 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=65,IDX_MAX=70 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=70,IDX_MAX=75 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=75,IDX_MAX=80 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=80,IDX_MAX=85 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=85,IDX_MAX=90 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=90,IDX_MAX=95 launcher.slurm
#sbatch --export=ALL,N=10,IDX_MIN=95,IDX_MAX=100 launcher.slurm
