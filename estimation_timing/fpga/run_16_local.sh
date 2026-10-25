#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: synthesize local moves at n=16."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_STAGE="synth"
echo "RUN_STAGE=$RUN_STAGE"
export SK_N_VAL="16"
echo "SK_N_VAL=$SK_N_VAL"
export MOVE="local"
echo "MOVE=$MOVE"
source scripts/run_phase.sh

