# scripts/run_synthesis_local.tcl
#
# Synthesize the local single-spin-flip HLS kernel.
#
# Expected repository layout:
#   scripts/run_synthesis_local.tcl
#   src/local_move_hls.cpp
#   src/exp_cheby.hpp
#
# Required environment variables:
#   SK_N_VAL   number of spins
#   F_VAL      fractional bits, F = ceil(3.5 * log2(SK_N / eps_discr))
#   HLS_PART   FPGA part, e.g. xcvu19p-fsva3824-2-e
#   CLOCK_NS   target clock period in ns

set top_name "local_spin_flip_operation"
set n_spins  $::env(SK_N_VAL)
set f_bits   $::env(F_VAL)
set part     $::env(HLS_PART)
set clk_ns   $::env(CLOCK_NS)

set comp_name "hls_local_spin_flip_N${n_spins}_F${f_bits}"

puts "============================================================"
puts "Top function : ${top_name}"
puts "Source       : src/local_move_hls.cpp"
puts "Header       : src/exp_cheby.hpp"
puts "SK_N         : ${n_spins}"
puts "F            : ${f_bits}"
puts "Part         : ${part}"
puts "Clock ns     : ${clk_ns}"
puts "Component    : ${comp_name}"
puts "============================================================"

open_component -reset ${comp_name} -flow_target vivado

add_files src/local_move_hls.cpp -cflags "-std=c++17 -Isrc -DSK_N=${n_spins} -DF=${f_bits}"

set_top ${top_name}
set_part ${part}
create_clock -period ${clk_ns} -name default

csynth_design

exit
