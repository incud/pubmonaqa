#!/usr/bin/env bash
set -u -o pipefail

echo "Sourcing..."
source /scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh

export SK_N_VAL=20
export F_VAL=64
export HLS_PART=xcvu19p-fsva3824-2-e
export CLOCK_NS=3.333

vitis-run --mode hls --tcl scripts/run_csim_local.tcl