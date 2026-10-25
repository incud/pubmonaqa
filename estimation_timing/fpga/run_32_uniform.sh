#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: synthesize uniform moves at n=32."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_STAGE="synth"
echo "RUN_STAGE=$RUN_STAGE"
export SK_N_VAL="32"
echo "SK_N_VAL=$SK_N_VAL"
export MOVE="uniform"
echo "MOVE=$MOVE"
source scripts/run_phase.sh

