#!/bin/bash
cd $(dirname $0)/..
# Builds wheels for main Python versions using temporary
# conda environments
# this is a temporary (and fast) solution to build linux_x86_64 wheels
# before setting up cibuildwheel / github actions
# this is also faster than cibuildwheel / github actions
# because nvcc and GSL are already available on the system I'm running
# this script (juseless head node)

for version in "3.7" "3.8" "3.9" "3.10" "3.11" "3.12"; do
    env_dir=$(mktemp -d -u)
    echo "creating tmp env in $env_dir"
    conda create --prefix $env_dir python=$version -y
    eval "$(conda shell.bash hook)" && \
    conda activate $env_dir && \
    $env_dir/bin/pip install build setuptools && \
    $env_dir/bin/python -m build .
done