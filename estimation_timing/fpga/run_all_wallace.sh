#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: test Wallace."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_STAGE="wallace"
echo "RUN_STAGE=$RUN_STAGE"
export SK_N_VAL="all"
echo "SK_N_VAL=$SK_N_VAL"
source scripts/run_phase.sh

