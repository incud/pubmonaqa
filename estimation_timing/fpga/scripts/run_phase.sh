# Shared hardcoded settings and actions for the parameterless run_*.sh scripts.
if [[ $# -ne 0 ]]; then
    echo "These scripts take no parameters; edit their hardcoded settings."
    exit 1
fi
FPGA_ROOT="$PWD"
echo "FPGA_ROOT=$FPGA_ROOT"
PYTHON="python3.11"
echo "PYTHON=$PYTHON"

run_hls() {
    FPGA_COMPONENT="$FPGA_ROOT/build/${RUN_STAGE}_${MOVE}_N${SK_N_VAL}"
    echo "FPGA_COMPONENT=$FPGA_COMPONENT"
    # Clean only this job's files, including after a failure.
    trap '"$PYTHON" "$FPGA_ROOT/scripts/reports.py" clean "$FPGA_COMPONENT" "$FPGA_LAUNCH"' EXIT
    vitis-run --mode hls --tcl "$FPGA_ROOT/scripts/run_hls.tcl"
    "$PYTHON" "$FPGA_ROOT/scripts/reports.py" clean "$FPGA_COMPONENT" "$FPGA_LAUNCH"
}

case "$RUN_STAGE" in
    generate)
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
        ;;
    wallace)
        CXX="g++"
        echo "CXX=$CXX"
        mkdir -p build
        "$CXX" -std=c++17 -O2 -Wno-unknown-pragmas -Isrc tb/tb_wallace.cpp -o build/test_wallace
        ./build/test_wallace
        ;;
    csim|synth)
        if [[ ! -f build/run_config.json ]]; then
            echo "Run bash run_all_generate.sh first."
            exit 1
        fi
        VITIS_SETTINGS="/scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh"
        echo "VITIS_SETTINGS=$VITIS_SETTINGS"
        export HLS_PART="xcvu19p-fsva3824-2-e"
        echo "HLS_PART=$HLS_PART"
        export CLOCK_NS="3.000"
        echo "CLOCK_NS=$CLOCK_NS"
        if [[ -f "$VITIS_SETTINGS" ]]; then
            set +u
            source "$VITIS_SETTINGS"
            set -u
        fi
        if ! command -v vitis-run >/dev/null; then
            echo "vitis-run is unavailable. Source Vitis 2025.2 or edit VITIS_SETTINGS."
            exit 1
        fi
        if [[ "$RUN_STAGE" == synth ]] && ! command -v v++ >/dev/null; then
            echo "v++ is unavailable. Source Vitis 2025.2 or edit VITIS_SETTINGS."
            exit 1
        fi
        # Separate launch directories also keep tool journals local to each job.
        FPGA_LAUNCH="$FPGA_ROOT/build/launch_${RUN_STAGE}_${MOVE}_N${SK_N_VAL}"
        echo "FPGA_LAUNCH=$FPGA_LAUNCH"
        mkdir -p "$FPGA_LAUNCH"
        cd "$FPGA_LAUNCH"
        if [[ "$RUN_STAGE" == csim ]]; then
            for SK_N_VAL in 10 8 16 32 64; do
                export SK_N_VAL
                echo "SK_N_VAL=$SK_N_VAL"
                run_hls
            done
        else
            run_hls
        fi
        ;;
    reports)
        "$PYTHON" scripts/reports.py collect
        "$PYTHON" scripts/reports.py parse
        echo "Done: reports/summary.csv."
        ;;
    *)
        echo "Unknown phase: $RUN_STAGE"
        exit 1
        ;;
esac
