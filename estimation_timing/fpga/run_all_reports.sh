#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: copy reports and print latencies."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_STAGE="reports"
echo "RUN_STAGE=$RUN_STAGE"
export SK_N_VAL="all"
echo "SK_N_VAL=$SK_N_VAL"
source scripts/run_phase.sh

