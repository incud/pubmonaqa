#!/usr/bin/env bash
set +u -o pipefail

echo "To run this you need to have installed some tools: sudo apt update; sudo apt install yosys graphviz librsvg2-bin"
echo "Check you have them with: yosys -V; dot -V; rsvg-convert --version"

echo "Sourcing Vitis..."
source /scratch/mincudin/xlnx/2025.2/Vitis/settings64.sh
echo "Done sourcing Vitis"

rm -rf visualization
mkdir -p visualization/temp visualization/out

cp src/local_move_hls.cpp visualization/temp/
cp src/uniform_move_hls.cpp visualization/temp/
cp src/make_exp_cheby.py visualization/temp/
cp scripts/run_synthesis_move.tcl visualization/temp/

echo "Start running make_exp_cheby.py"
python3 visualization/temp/make_exp_cheby.py \
    --n 8 \
    --beta 4 \
    --degree 4 \
    --eps-op 1e-4 \
    --out visualization/temp/exp_cheby.hpp

echo "Start Python patch on the C++ files"
python3 - <<'PY'
from pathlib import Path

for name in ["local_move_hls.cpp", "uniform_move_hls.cpp"]:
    p = Path("visualization/temp") / name
    s = p.read_text()
    s = s.replace("#pragma HLS INLINE off", "#pragma HLS INLINE")
    s = s.replace("#pragma HLS INLINE", "#pragma HLS INLINE off")
    p.write_text(s)
PY

echo "Start Python patch on the TCL file"
python3 - <<'PY'
from pathlib import Path

p = Path("visualization/temp/run_synthesis_move.tcl")
s = p.read_text()

s = s.replace(
    'set src_path "src/${src_file}"',
    'set src_path "../temp/${src_file}"',
)
s = s.replace("-Isrc", "-I../temp")
s = s.replace("csynth_design", "csynth_design\nexport_design -format ip")

p.write_text(s)
PY

echo "Now move to visualization/out and start synthesis"
cd visualization/out

TOP_NAME=local_spin_flip_operation \
SRC_FILE=local_move_hls.cpp \
SK_N_VAL=8 \
FRAC_VAL=8 \
HLS_PART=xcvu19p-fsva3824-2-e \
CLOCK_NS=3.333 \
vitis-run --mode hls --tcl ../temp/run_synthesis_move.tcl

TOP_NAME=uniform_move_operation \
SRC_FILE=uniform_move_hls.cpp \
SK_N_VAL=8 \
FRAC_VAL=8 \
HLS_PART=xcvu19p-fsva3824-2-e \
CLOCK_NS=3.333 \
vitis-run --mode hls --tcl ../temp/run_synthesis_move.tcl

for folder in hls_local_spin_flip_operation_N8_FRAC8 hls_uniform_move_operation_N8_FRAC8; do
    impl_dir="$(find "${folder}" -type d -path "*/impl/verilog" | head -1)"
    if [[ -z "${impl_dir}" ]]; then
        echo "ERROR: could not find impl/verilog inside ${folder}" >&2
        exit 1
    fi

    rm -rf "../${folder}_verilog"
    cp -r "${impl_dir}" "../${folder}_verilog"
    echo "Copied ${impl_dir} -> visualization/${folder}_verilog"
done

echo "Done."
echo "Temporary sources: visualization/temp"
echo "HLS outputs:       visualization/out"
echo "Verilog folders:   visualization/*_verilog"

mkdir -p visualization/schematics

yosys -p "
  read_verilog visualization/hls_local_spin_flip_operation_N8_FRAC8_verilog/*.v;
  hierarchy -check -top local_spin_flip_operation;
  proc; opt;
  show -viewer none -format svg -prefix visualization/schematics/local_N8 local_spin_flip_operation
"

yosys -p "
  read_verilog visualization/hls_uniform_move_operation_N8_FRAC8_verilog/*.v;
  hierarchy -check -top uniform_move_operation;
  proc; opt;
  show -viewer none -format svg -prefix visualization/schematics/uniform_N8 uniform_move_operation
"

rsvg-convert -f pdf -o visualization/schematics/local_N8.pdf visualization/schematics/local_N8.svg
rsvg-convert -f pdf -o visualization/schematics/uniform_N8.pdf visualization/schematics/uniform_N8.svg