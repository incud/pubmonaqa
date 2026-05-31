#!/usr/bin/env bash
set -euo pipefail

RPT_PATH="hls/syn/report"

copy_reports() {
    local folder="$1"
    local name="${folder#hls_}"
    local dst="reports/syn_report_${name}"

    mkdir -p "${dst}"
    cp "${folder}/${RPT_PATH}"/* "${dst}/"
    echo "Copied ${folder}/${RPT_PATH} -> ${dst}"
}

copy_reports hls_local_spin_flip_operation_N8_FRAC24
copy_reports hls_local_spin_flip_operation_N16_FRAC28
copy_reports hls_local_spin_flip_operation_N32_FRAC31
copy_reports hls_local_spin_flip_operation_N64_FRAC35
copy_reports hls_uniform_move_operation_N8_FRAC24
copy_reports hls_uniform_move_operation_N16_FRAC28
copy_reports hls_uniform_move_operation_N32_FRAC31
copy_reports hls_uniform_move_operation_N64_FRAC35
