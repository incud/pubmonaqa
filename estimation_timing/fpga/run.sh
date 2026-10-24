#!/usr/bin/env bash
set -euo pipefail
echo "FPGA Metropolis: generate headers, test Wallace and moves, synthesize, copy reports, parse timings."
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 0 ]]; then
    echo "This script takes no parameters; edit the constants below."
    exit 1
fi

PYTHON="python3"
echo "PYTHON=$PYTHON"
CXX="g++"
echo "CXX=$CXX"
VITIS_SETTINGS="/scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh"
echo "VITIS_SETTINGS=$VITIS_SETTINGS"
export HLS_PART="xcvu19p-fsva3824-2-e"
echo "HLS_PART=$HLS_PART"
export CLOCK_NS="3.000"
echo "CLOCK_NS=$CLOCK_NS"
BETA="1"
echo "BETA=$BETA"
DEGREE="12"
echo "DEGREE=$DEGREE"
EPS_B="0.0001"
echo "EPS_B=$EPS_B"
EPS_OP="0.0001"
echo "EPS_OP=$EPS_OP"
TEST_N="10"
echo "TEST_N=$TEST_N"
SIZES=(8 16 32 64)
echo "SIZES=${SIZES[*]}"

"$PYTHON" src/make_exp_cheby.py --beta "$BETA" --degree "$DEGREE" \
    --eps-b "$EPS_B" --eps-op "$EPS_OP" --test-n "$TEST_N" --sizes "${SIZES[@]}"

# Vitis writes its own journals internally; discard them at exit, including after a failure.
trap '"$PYTHON" scripts/reports.py clean' EXIT
echo "Compiling and running the standalone Wallace testbench."
"$CXX" -std=c++17 -O2 -Wno-unknown-pragmas -Isrc tb/tb_wallace.cpp -o build/test_wallace
./build/test_wallace

if [[ -f "$VITIS_SETTINGS" ]]; then
    set +u
    source "$VITIS_SETTINGS"
    set -u
fi
if ! command -v vitis-run >/dev/null; then
    echo "vitis-run is unavailable. Install/source Vitis 2025.2 or edit VITIS_SETTINGS."
    exit 1
fi

export RUN_STAGE="csim"
echo "RUN_STAGE=$RUN_STAGE"
echo "Compiling and running both move testbenches at the actual configured precisions."
for SK_N_VAL in "$TEST_N" "${SIZES[@]}"; do
    export SK_N_VAL
    echo "SK_N_VAL=$SK_N_VAL"
    for MOVE in local uniform; do
        export MOVE
        echo "MOVE=$MOVE"
        vitis-run --mode hls --tcl scripts/run_hls.tcl
    done
done

echo "Synthesizing both moves for every synthesis size."
export RUN_STAGE="synth"
echo "RUN_STAGE=$RUN_STAGE"
for SK_N_VAL in "${SIZES[@]}"; do
    export SK_N_VAL
    echo "SK_N_VAL=$SK_N_VAL"
    for MOVE in local uniform; do
        export MOVE
        echo "MOVE=$MOVE"
        vitis-run --mode hls --tcl scripts/run_hls.tcl
    done
done

echo "Copying synthesis XML/text reports into reports/."
"$PYTHON" scripts/reports.py collect
echo "Reading the transition-kernel latency from each XML report."
"$PYTHON" scripts/reports.py parse
echo "Done: reports/summary.csv."
