#!/usr/bin/env bash
set +u -o pipefail

echo "Sourcing Vitis..."
source /scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh

export FRAC_VAL=64
export HLS_PART=xcvu19p-fsva3824-2-e
export CLOCK_NS=3.333

export TEST_N=10
export TEST_BETA=4
export TEST_DEGREE=12
export TEST_LAMBDA_VAL=142.702511707779

echo "Generating exp_cheby.hpp... (needs python3.11 + numpy + scipy)"
python3.11 src/make_exp_cheby.py --n ${TEST_N} --beta ${TEST_BETA} --degree ${TEST_DEGREE} --out src/exp_cheby.hpp 

echo "Running C simulation..."
vitis-run --mode hls --tcl scripts/run_csim_local_n10.tcl
