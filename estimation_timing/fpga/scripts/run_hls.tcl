# One driver for move tests and move synthesis.
set root [file normalize [file join [file dirname [info script]] ..]]
cd $root
set stage $::env(RUN_STAGE)
set move $::env(MOVE)
set n $::env(SK_N_VAL)
set component [file join $root build "${stage}_${move}_N${n}"]
set flags "-std=c++17 -I\"[file join $root src]\" -DSK_N=${n}"

switch -- $move {
    local {
        set top local_spin_flip_operation
        set source local_move_hls.cpp
        set bench tb_local.cpp
    }
    uniform {
        set top uniform_move_operation
        set source uniform_move_hls.cpp
        set bench tb_uniform.cpp
    }
    default { error "Unknown move: $move" }
}

puts "stage=$stage move=$move n=$n top=$top component=$component"
open_component -reset $component -flow_target vivado
add_files [file join $root src $source] -cflags $flags
set_top $top
set_part $::env(HLS_PART)
create_clock -period $::env(CLOCK_NS) -name default
if {$stage eq "csim"} {
    add_files -tb [file join $root tb $bench] -cflags $flags
    csim_design
} elseif {$stage eq "synth"} {
    csynth_design
} else {
    error "Unsupported stage/move: $stage/$move"
}
exit
