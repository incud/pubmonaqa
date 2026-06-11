# scripts/run_synthesis_move.tcl
#
# Generic synthesis script for the current local/uniform HLS move kernels.
#
# Expected repository layout:
#   scripts/run_synthesis_move.tcl
#   ./launch_synthesis_many_n.sh
#   src/make_exp_cheby.py
#   src/local_move_hls.cpp
#   src/uniform_move_hls.cpp
#   src/exp_cheby.hpp
#
# Required environment variables:
#   TOP_NAME   top-level HLS function, e.g. local_spin_flip_operation or uniform_move_operation
#   SRC_FILE   source file under src/, e.g. local_move_hls.cpp or uniform_move_hls.cpp
#   SK_N_VAL   number of spins
#   FRAC_VAL   fractional bits, FRAC = ceil(3.5 * log2(SK_N / eps_discr))
#   HLS_PART   FPGA part, e.g. xcvu19p-fsva3824-2-e
#   CLOCK_NS   target clock period in ns

set top_name $::env(TOP_NAME)
set src_file $::env(SRC_FILE)
set n_spins  $::env(SK_N_VAL)
set frac     $::env(FRAC_VAL)
set part     $::env(HLS_PART)
set clk_ns   $::env(CLOCK_NS)

set src_path "src/${src_file}"
set comp_name "hls_${top_name}_N${n_spins}_FRAC${frac}"

puts "============================================================"
puts "Top function : ${top_name}"
puts "Source       : ${src_path}"
puts "Header       : src/exp_cheby.hpp"
puts "SK_N         : ${n_spins}"
puts "FRAC         : ${frac}"
puts "Part         : ${part}"
puts "Clock ns     : ${clk_ns}"
puts "Component    : ${comp_name}"
puts "============================================================"

open_component -reset ${comp_name} -flow_target vivado

add_files ${src_path} -cflags "-std=c++17 -Isrc -DSK_N=${n_spins} -DFRAC=${frac}"

set_top ${top_name}
set_part ${part}
create_clock -period ${clk_ns} -name default

csynth_design

exit
