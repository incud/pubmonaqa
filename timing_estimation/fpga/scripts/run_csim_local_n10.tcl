# scripts/run_csim_local_n10.tcl
#
# C simulation for SK_N=10. It checks all 10 one-hot local spin flips.
#
# Expected layout:
#   scripts/run_csim_local_n10.tcl
#   src/local_move_hls.cpp
#   src/exp_cheby.hpp
#   tb/tb_local_move_n10_all.cpp
#
# Required environment variables:
#   FRAC_VAL
#   TEST_LAMBDA_VAL   lambda = beta * alpha used by std::exp reference
#   HLS_PART
#   CLOCK_NS

set top_name "local_spin_flip_operation"
set n_spins  10
set frac     $::env(FRAC_VAL)
set lambda   $::env(TEST_LAMBDA_VAL)
set part     $::env(HLS_PART)
set clk_ns   $::env(CLOCK_NS)

set cflags "-std=c++17 -Isrc -DSK_N=${n_spins} -DFRAC=${frac} -DTEST_LAMBDA=${lambda}"

set comp_name "csim_local_spin_flip_N${n_spins}_FRAC${frac}"

puts "============================================================"
puts "C simulation: all local flips against std::exp"
puts "Top function : ${top_name}"
puts "SK_N         : ${n_spins}"
puts "FRAC         : ${frac}"
puts "lambda       : ${lambda}"
puts "Part         : ${part}"
puts "Clock ns     : ${clk_ns}"
puts "CFLAGS       : ${cflags}"
puts "Component    : ${comp_name}"
puts "============================================================"

open_component -reset ${comp_name} -flow_target vivado

add_files src/local_move_hls.cpp -cflags ${cflags}
add_files -tb tb/tb_local_move_n10_all.cpp -cflags ${cflags}

set_top ${top_name}
set_part ${part}
create_clock -period ${clk_ns} -name default

csim_design

exit
