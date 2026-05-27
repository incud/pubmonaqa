#!/usr/bin/env bash
set -u -o pipefail

echo "Sourcing..."
source /scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh

export HLS_PART="${HLS_PART:-xcvu19p-fsva3824-2-e}"
export CLOCK_NS="${CLOCK_NS:-3.333}"
export COEFF_W="${COEFF_W:-40}"
export COEFF_I="${COEFF_I:-16}"

mkdir -p logs reports

echo "top,n,status,report,log" > reports/scan_status.csv

for n in $(seq 10 30); do
    for top in local_metropolis_step uniform_metropolis_step; do
        export TOP_NAME="$top"
        export SK_N_VAL="$n"

        log="logs/${top}_N${n}.log"

        echo "============================================================"
        echo "Synthesizing TOP_NAME=${TOP_NAME}, SK_N_VAL=${SK_N_VAL}"
        echo "Log: ${log}"
        echo "============================================================"

        if vitis-run --mode hls --tcl scripts/run_hls_one.tcl 2>&1 | tee "$log"; then
            rpt="$(find "hls_${top}_N${n}" -name "*csynth*.rpt" | head -1 || true)"

            if [[ -n "$rpt" && -f "$rpt" ]]; then
                out_rpt="reports/${top}_N${n}_csynth.rpt"
                cp "$rpt" "$out_rpt"
                echo "${top},${n},ok,${out_rpt},${log}" >> reports/scan_status.csv
                echo "OK: copied report to ${out_rpt}"
            else
                echo "${top},${n},no_report,,${log}" >> reports/scan_status.csv
                echo "WARNING: synthesis succeeded but no csynth report found."
            fi
        else
            echo "${top},${n},failed,,${log}" >> reports/scan_status.csv
            echo "FAILED: ${top}, N=${n}. Continuing."
        fi

        echo
    done
done

echo "Done. Status file:"
echo "reports/scan_status.csv"
