#!/bin/bash
# This script includes examples using the cuBNM CLI.
# Usage:
# source ./cli_examples.sh
# This will load the test functions into the current shell. E.g.:
# cubnm_run_grid

cubnm_example_grid () {
    cubnm grid \
        --model rWW --sc example --emp_bold example \
        --out_dir ./grid_cli \
        --TR 1 --duration 60 --window_size 10 --window_step 2 --states_ts \
        --params G=0.001:10.0,w_p=0:2.0,J_N=0.001:0.5 --grid_shape G=4,w_p=5,J_N=5 --sim_verbose
}

cubnm_example_cmaes_homo () {
    cubnm optimize \
        --model rWW --sc example --emp_bold example \
        --out_dir ./cmaes_homo_cli \
        --TR 1 --duration 60 --window_size 10 --window_step 2 --states_ts \
        --params G=0.001:10.0,w_p=0:2.0,J_N=0.001:0.5 \
        --optimizer CMAES --optimizer_seed 0 --n_iter 2 --popsize 10
}

cubnm_example_cmaes_het () {
    cubnm optimize \
        --model rWW --sc example --emp_bold example \
        --out_dir ./cmaes_het_cli \
        --TR 1 --duration 60 --window_size 10 --window_step 2 --states_ts \
        --params G=0.001:10.0,w_p=0:2.0,J_N=0.001:0.5 \
        --optimizer CMAES --optimizer_seed 0 --n_iter 2 --popsize 10 \
        --het_params w_p J_N --maps example
}