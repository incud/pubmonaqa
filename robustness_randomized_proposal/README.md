The analysis on the averaged matrix has been done with 

    launch_spectral_gap.py
    launch_spectral_gap.slurm

This is fully sequential and write on a single file 'convergence_randomized_hyperparams.json'
(actually, the filename is a constant in monaqa2.data.filename). 

The analysis on the grid search stuff has been done with

    launch_randomized_grid_search.py
    launch_randomized_grid_search.slurm
    submit_randomized_grid_search.sh
    merge_randomized_grid_search.py

in parallel, and write on many parallel files 'gridsearch_randomized_hyperparams_[...].json' that later on have been merged. The single files have been deleted and only the merged one has been kept. This is all for grid search with k=3, meaning 2^3 values of t and 2^3 values of gamma. 

I have later parallelized the averaged matrix calculation because it was simply too slow to wait for it to finish. This has been done with 

    launch_randomized_average_behaviour[_2].py
    launch_randomized_average_behaviour.slurm
    submit_randomized_average_behaviour.sh
    merge_randomized_average_behaviour.py

but actually this has not been used because Cineca cluster is stuck and no nodes are available. Thus I have fallen back to the sequential calculation.
