#!/bin/bash
set -euo pipefail

BETAS=(1.0 2.0 4.0 8.0 10.0 20.0 100.0)

mkdir -p logs

echo "Submitting qemc, beta=0.0"
sbatch --job-name=mon_qss_qemc_b0p0 --export=ALL,EXPERIMENT=qemc,BETA=0.0 launcher_quantum_single_step.slurm

for BETA in "${BETAS[@]}"; do
    BETA_TAG="${BETA//./p}"
    echo "Submitting coin, beta=${BETA}"
    sbatch --job-name="mon_qss_coin_b${BETA_TAG}" --export=ALL,EXPERIMENT=coin,BETA="${BETA}" launcher_quantum_single_step.slurm
done
