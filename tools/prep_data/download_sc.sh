#!/bin/bash
# Clones the HCP SC output dataset. The source dataset is currently stored on a
# local cluster and is not accessible publicly. Therefore this script can only be
# run by maintainers who have access to the cluster.
# The SC output was created by running https://jugit.fz-juelich.de/inm7/public/vbc-mri-pipeline.git
# on the HCP open access data (github.com/datalad-datasets/human-connectome-project-openaccess), using
# parameters defined in e.g. "vbc_mri_pipeline_example_input.txt".
# Note: This script has "datalad" as a dependency, which is not installed as one of the package
# dependencies, but should instead be installed e.g. via miniforge.

PARCS=("Schaefer2018_100Parcels_7Networks" "Schaefer2018_200Parcels_7Networks" "Schaefer2018_400Parcels_7Networks" "DesikanKilliany_68Parcels")

cd $(dirname $0)

if [ -f ../../.env ]; then
    source ../../.env
fi

if [ -z "$HCP_SC_DATASET_PATH" ]; then
    echo "HCP_SC_DATASET_PATH is not set"
    exit 1
fi

if [ -z "$HCP_SC_DATASET_URL" ]; then
    echo "HCP_SC_DATASET_URL is not set"
    exit 1
fi

# ensure datalad is installed
if ! command -v datalad &> /dev/null; then
    echo "datalad could not be found"
    exit 1
fi

# clone the dataset
if [ ! -d $HCP_SC_DATASET_PATH ]; then
    datalad install -s $HCP_SC_DATASET_URL $HCP_SC_DATASET_PATH
fi

cd $HCP_SC_DATASET_PATH

for parc in ${PARCS[@]}; do
    datalad get output/SC/HCP/*/${parc}*
done