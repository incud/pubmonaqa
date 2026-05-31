#!/usr/bin/env bash
set +o pipefail

echo "Sourcing Vitis... (note you need set +u because the following script needs PYTHONPATH)"
source /scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh

export HLS_PART="${HLS_PART:-xcvu19p-fsva3824-2-e}"
export CLOCK_NS="${CLOCK_NS:-3.000}"

# Parameters used to generate src/exp_cheby.hpp for each N.
export BETA_VAL="${BETA_VAL:-4}"
export DEGREE_VAL="${DEGREE_VAL:-12}"
export EPS_OP="${EPS_OP:-1e-4}"
export EPS_DISCR="${EPS_DISCR:-1e-4}"

# Path to the piecewise Chebyshev generator. It should write src/exp_cheby.hpp.
export CHEBY_SCRIPT="${CHEBY_SCRIPT:-src/make_exp_cheby.py}"

mkdir -p logs reports

echo "From now on python3.11 + numpy + scipy is required."

echo "top,source,n,frac,beta,degree,eps_op,status,report,log" > reports/synthesis_status.csv

#for n in 8 16 32; do
for n in 8 16 32 64; do
    frac="$(python3.11 - <<PY
import math
n = ${n}
eps = float("${EPS_DISCR}")
print(math.ceil(3.5 * math.log2(n) - math.log2(eps)))
PY
)"
    export SK_N_VAL="${n}"
    export FRAC_VAL="${frac}"

    echo "============================================================"
    echo "Generating exp_cheby.hpp for N=${SK_N_VAL}"
    echo "beta=${BETA_VAL}, degree=${DEGREE_VAL}, eps_op=${EPS_OP}, FRAC=${FRAC_VAL}"
    echo "============================================================"

    python3 "${CHEBY_SCRIPT}" \
        --n "${SK_N_VAL}" \
        --beta "${BETA_VAL}" \
        --degree "${DEGREE_VAL}" \
        --eps-op "${EPS_OP}" \
        --out src/exp_cheby.hpp

    for move in local uniform; do
        if [[ "${move}" == "local" ]]; then
            export TOP_NAME="local_spin_flip_operation"
            export SRC_FILE="local_move_hls.cpp"
        else
            export TOP_NAME="uniform_move_operation"
            export SRC_FILE="uniform_move_hls.cpp"
        fi

        log="logs/${TOP_NAME}_N${SK_N_VAL}_FRAC${FRAC_VAL}.log"

        echo "============================================================"
        echo "Synthesizing TOP_NAME=${TOP_NAME}"
        echo "Source: src/${SRC_FILE}"
        echo "N=${SK_N_VAL}, FRAC=${FRAC_VAL}"
        echo "Log: ${log}"
        echo "============================================================"

        if vitis-run --mode hls --tcl scripts/run_synthesis_move.tcl 2>&1 | tee "${log}"; then
            comp="hls_${TOP_NAME}_N${SK_N_VAL}_FRAC${FRAC_VAL}"
            rpt="$(find "${comp}" -name "*csynth*.rpt" | head -1 || true)"

            if [[ -n "${rpt}" && -f "${rpt}" ]]; then
                out_rpt="reports/${TOP_NAME}_N${SK_N_VAL}_FRAC${FRAC_VAL}_csynth.rpt"
                cp "${rpt}" "${out_rpt}"
                echo "${TOP_NAME},${SRC_FILE},${SK_N_VAL},${FRAC_VAL},${BETA_VAL},${DEGREE_VAL},${EPS_OP},ok,${out_rpt},${log}" >> reports/synthesis_status.csv
                echo "OK: copied report to ${out_rpt}"
            else
                echo "${TOP_NAME},${SRC_FILE},${SK_N_VAL},${FRAC_VAL},${BETA_VAL},${DEGREE_VAL},${EPS_OP},no_report,,${log}" >> reports/synthesis_status.csv
                echo "WARNING: synthesis succeeded but no csynth report found."
            fi
        else
            echo "${TOP_NAME},${SRC_FILE},${SK_N_VAL},${FRAC_VAL},${BETA_VAL},${DEGREE_VAL},${EPS_OP},failed,,${log}" >> reports/synthesis_status.csv
            echo "FAILED: ${TOP_NAME}, N=${SK_N_VAL}. Continuing."
        fi

        echo
    done
done

echo "Done. Status file:"
echo "reports/synthesis_status.csv"
