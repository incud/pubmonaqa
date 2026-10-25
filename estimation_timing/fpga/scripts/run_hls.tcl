# One driver for C simulation and synthesis.
set root [file normalize [file join [file dirname [info script]] ..]]
cd $root
set stage $::env(RUN_STAGE)
set move $::env(MOVE)
set n $::env(SK_N_VAL)
set component [file join $root build "${stage}_${move}_N${n}"]
# Vitis parses -cflags itself; embedded quotes become part of the include path.
set flags "-std=c++17 -I[file join $root src] -DSK_N=${n}"

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

if {$stage ni {csim synth}} { error "Unsupported stage: $stage" }

set source [file join $root src $source]
set bench [file join $root tb $bench]
if {![file isfile $source]} { error "Missing kernel: $source" }

puts "stage=$stage move=$move n=$n top=$top component=$component"
# Recreate only the generated project for this stage, move and size.
file delete -force $component
if {$stage eq "csim"} {
    open_project -reset $component
    set_top $top
    open_solution -reset solution -flow_target vivado
    set_part $::env(HLS_PART)
    create_clock -period $::env(CLOCK_NS) -name default
    # Both files are compiled explicitly in this simulation-only project.
    add_files -tb $source -cflags $flags
    add_files -tb $bench -cflags $flags
    csim_design -clean
} else {
    # Bypass Tcl add_files path rewriting: give v++ absolute paths in a config.
    file mkdir $component
    set config [file join $component hls_config.cfg]
    set output [open $config w]
    puts $output "part=$::env(HLS_PART)"
    puts $output {[hls]}
    puts $output "flow_target=vivado"
    puts $output "clock=$::env(CLOCK_NS)ns"
    puts $output "syn.top=$top"
    puts $output "syn.file=$source"
    puts $output "syn.cflags=$flags"
    close $output
    cd $component
    exec v++ -c --mode hls --config $config --work_dir $component >@ stdout 2>@ stderr
}
exit
