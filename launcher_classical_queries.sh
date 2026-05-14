#!/bin/bash
set -euo pipefail

mkdir -p logs

# n = 3, 4, 5, 6: one process each
sbatch --export=ALL,N=3,IDX_MIN=0,IDX_MAX=100 launcher_classical_queries.slurm
sbatch --export=ALL,N=4,IDX_MIN=0,IDX_MAX=100 launcher_classical_queries.slurm
sbatch --export=ALL,N=5,IDX_MIN=0,IDX_MAX=100 launcher_classical_queries.slurm
sbatch --export=ALL,N=6,IDX_MIN=0,IDX_MAX=100 launcher_classical_queries.slurm

# n = 7: two chunks of 50
sbatch --export=ALL,N=7,IDX_MIN=0,IDX_MAX=50 launcher_classical_queries.slurm
sbatch --export=ALL,N=7,IDX_MIN=50,IDX_MAX=100 launcher_classical_queries.slurm

## n = 8: chunks of 20
#sbatch --export=ALL,N=8,IDX_MIN=0,IDX_MAX=20 launcher_classical_queries.slurm
#sbatch --export=ALL,N=8,IDX_MIN=20,IDX_MAX=40 launcher_classical_queries.slurm
#sbatch --export=ALL,N=8,IDX_MIN=40,IDX_MAX=60 launcher_classical_queries.slurm
#sbatch --export=ALL,N=8,IDX_MIN=60,IDX_MAX=80 launcher_classical_queries.slurm
#sbatch --export=ALL,N=8,IDX_MIN=80,IDX_MAX=100 launcher_classical_queries.slurm
#
## n = 9: chunks of 10
#sbatch --export=ALL,N=9,IDX_MIN=0,IDX_MAX=10 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=10,IDX_MAX=20 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=20,IDX_MAX=30 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=30,IDX_MAX=40 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=40,IDX_MAX=50 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=50,IDX_MAX=60 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=60,IDX_MAX=70 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=70,IDX_MAX=80 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=80,IDX_MAX=90 launcher_classical_queries.slurm
#sbatch --export=ALL,N=9,IDX_MIN=90,IDX_MAX=100 launcher_classical_queries.slurm
#
## n = 10: chunks of 5
#sbatch --export=ALL,N=10,IDX_MIN=0,IDX_MAX=5 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=5,IDX_MAX=10 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=10,IDX_MAX=15 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=15,IDX_MAX=20 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=20,IDX_MAX=25 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=25,IDX_MAX=30 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=30,IDX_MAX=35 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=35,IDX_MAX=40 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=40,IDX_MAX=45 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=45,IDX_MAX=50 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=50,IDX_MAX=55 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=55,IDX_MAX=60 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=60,IDX_MAX=65 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=65,IDX_MAX=70 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=70,IDX_MAX=75 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=75,IDX_MAX=80 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=80,IDX_MAX=85 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=85,IDX_MAX=90 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=90,IDX_MAX=95 launcher_classical_queries.slurm
#sbatch --export=ALL,N=10,IDX_MIN=95,IDX_MAX=100 launcher_classical_queries.slurm