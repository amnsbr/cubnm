#!/bin/bash
# Clones the HCP open access dataset
# Note: This script has "datalad" as a dependency, which is not installed as one of the package
# dependencies, but should instead be installed e.g. via miniforge.

cd $(dirname $0)

if [ -f ../../.env ]; then
    source ../../.env
fi

if [ -z "$HCP_INPUT_DATASET_PATH" ]; then
    echo "HCP_INPUT_DATASET_PATH is not set"
    exit 1
fi


# ensure datalad is installed
if ! command -v datalad &> /dev/null; then
    echo "datalad could not be found"
    exit 1
fi

# clone the dataset
if [ ! -d $HCP_INPUT_DATASET_PATH ]; then
    datalad install -s https://github.com/datalad-datasets/human-connectome-project-openaccess.git $HCP_INPUT_DATASET_PATH
fi