set top_name $::env(TOP_NAME)
set n_spins  $::env(SK_N_VAL)
set part     $::env(HLS_PART)
set clk_ns   $::env(CLOCK_NS)
set coeff_w  $::env(COEFF_W)
set coeff_i  $::env(COEFF_I)

set comp_name "hls_${top_name}_N${n_spins}"

puts "============================================================"
puts "Top function : ${top_name}"
puts "SK_N         : ${n_spins}"
puts "Part         : ${part}"
puts "Clock ns     : ${clk_ns}"
puts "Coeff        : ap_fixed<${coeff_w},${coeff_i}>"
puts "Component    : ${comp_name}"
puts "============================================================"

open_component -reset ${comp_name} -flow_target vivado

add_files src/sk_step.cpp -cflags "-Isrc -DSK_N=${n_spins} -DCOEFF_W=${coeff_w} -DCOEFF_I=${coeff_i}"

set_top ${top_name}
set_part ${part}
create_clock -period ${clk_ns} -name default

csynth_design

exit
