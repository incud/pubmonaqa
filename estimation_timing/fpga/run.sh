#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: run all phases in order."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 0 ]]; then
    echo "This script takes no parameters."
    exit 1
fi
for SCRIPT in run_all_generate.sh run_all_wallace.sh \
    run_all_test_local.sh run_all_test_uniform.sh \
    run_8_local.sh run_8_uniform.sh run_16_local.sh run_16_uniform.sh \
    run_32_local.sh run_32_uniform.sh run_64_local.sh run_64_uniform.sh \
    run_all_reports.sh; do
    echo "SCRIPT=$SCRIPT"
    bash "$SCRIPT"
done
