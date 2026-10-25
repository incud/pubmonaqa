#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: test uniform moves at all configured precisions."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_STAGE="csim"
echo "RUN_STAGE=$RUN_STAGE"
export SK_N_VAL="all"
echo "SK_N_VAL=$SK_N_VAL"
export MOVE="uniform"
echo "MOVE=$MOVE"
source scripts/run_phase.sh

