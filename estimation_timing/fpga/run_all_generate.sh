#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: generate headers."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_STAGE="generate"
echo "RUN_STAGE=$RUN_STAGE"
export SK_N_VAL="all"
echo "SK_N_VAL=$SK_N_VAL"
source scripts/run_phase.sh

