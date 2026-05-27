# scripts/run_csim_local.tcl
#
# C simulation for the local single-spin-flip HLS kernel.
#
# Expected layout:
#   scripts/run_csim_local.tcl
#   src/local_move_hls.cpp
#   src/exp_cheby.hpp
#   tb/tb_local_move.cpp
#
# Required environment variables:
#   SK_N_VAL
#   F_VAL
#   HLS_PART
#   CLOCK_NS

set top_name "local_spin_flip_operation"
set n_spins  $::env(SK_N_VAL)
set f_bits   $::env(F_VAL)
set part     $::env(HLS_PART)
set clk_ns   $::env(CLOCK_NS)

set comp_name "csim_local_spin_flip_N${n_spins}_F${f_bits}"

puts "============================================================"
puts "C simulation"
puts "Top function : ${top_name}"
puts "SK_N         : ${n_spins}"
puts "F            : ${f_bits}"
puts "Part         : ${part}"
puts "Clock ns     : ${clk_ns}"
puts "Component    : ${comp_name}"
puts "============================================================"

open_component -reset ${comp_name} -flow_target vivado

add_files src/local_move_hls.cpp -cflags "-std=c++17 -Isrc -DSK_N=${n_spins} -DFRAC=${f_bits}"
add_files -tb tb/tb_local_move.cpp -cflags "-std=c++17 -Isrc -DSK_N=${n_spins} -DFRAC=${f_bits}"

set_top ${top_name}
set_part ${part}
create_clock -period ${clk_ns} -name default

csim_design

exit
