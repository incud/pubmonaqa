This folder contains the code and data used in Appendix G, "Details in the Gibbs sampling with quantum walks", subsection 1, "Annealing schedule".

The script `generate_files.py` creates the folder `ising_inputs` and fills it with Sherrington-Kirkpatrick model instances ranging from `n=5` to `n=30`. For each `n`, we generate 100 different instances.

The instances are generated using the `monaqa2` package and are therefore of the same type as those used in the other experiments. Technically, the instances from `n=5` to `n=10` could have been the same as those used elsewhere. However, for consistency in the generation procedure and seed selection, we chose to generate them ex novo.

The code `estimation_variance_new.cpp` reads a single instance and computes the energy variance `Var_{pi_beta}` for the following hardcoded values of beta:

    0.0, 1.0 / 32.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 8.0, 3.0 / 16.0,
    1.0 / 4.0, 3.0 / 8.0, 1.0 / 2.0, 5.0 / 8.0, 3.0 / 4.0,
    13.0 / 16.0, 7.0 / 8.0, 15.0 / 16.0, 31.0 / 32.0,
    1.0, 33.0 / 32.0, 17.0 / 16.0, 9.0 / 8.0,
    19.0 / 16.0, 5.0 / 4.0, 11.0 / 8.0, 3.0 / 2.0, 7.0 / 4.0,
    2.0, 3.0, 4.0, 6.0, 8.0, 16.0

The code was run on the CINECA cluster. The corresponding SLURM file is included. Results are saved in the folder `variance_outputs`, with one file per instance containing pairs of beta and energy variance.

The file `collect_var_energies.py` collects all variance output files into `var_energies.pkl`. No information is discarded and no statistics are computed at this stage.

The code `generation_annealing_schedules.cpp` reads a single Ising instance and constructs an instance-specific annealing schedule. The schedule starts from the average fitted variance density with safety factor `s=1`. Whenever the exact squared overlap between two consecutive Gibbs states is smaller than `exp(-1)`, the code increases `s` until the overlap condition is satisfied. Partition functions are evaluated by streaming over the `2^n` configurations without storing all energies in memory.

The script `generation_annealing_schedules.sh` is intended for local execution. It compiles `generation_annealing_schedules.cpp`, creates the folder `schedule_outputs` if needed, removes empty output files, and processes all instances with `n=5,...,30` and `idx=0,...,99`. Existing output files are skipped.

The file `generation_annealing_schedules.slurm` runs the same workflow on the CINECA cluster.

The adaptive schedule results are saved in `schedule_outputs`, with one file per instance. Each file contains the starting beta value of each schedule step, the corresponding squared overlap, and the safety factor required for that step.

The executable also supports a fixed-safety mode through the option `--fixed-safety <s>`. In this mode, the same safety factor is used at every annealing step and no adaptive correction is applied.

The script `generation_annealing_schedules_safe.sh` is intended for local execution of the fixed-safety analysis. For example,

    bash generation_annealing_schedules_safe.sh 2

generates schedules with fixed safety factor `s=2`.

The file `generation_annealing_schedules_safe.slurm` runs the same fixed-safety analysis on the CINECA cluster. For example,

    sbatch generation_annealing_schedules_safe.slurm 2

runs the fixed-safety calculation with `s=2`.

The fixed-safety results are saved in `schedule_outputs_fixed_safety`. At the beginning of the calculation, the selected safety factor is written to `schedule_outputs_fixed_safety/chosen_fixed_safety.txt`. The remaining output files contain one fixed schedule per instance.

The file `collect_schedule.py` collects the adaptive schedule outputs into `schedule_summary.pkl`. For each instance, it stores `n`, `idx`, the schedule length `L`, the mean safety factor, its standard deviation, the maximum safety factor, and the number of schedule steps with safety factor greater than 1.
