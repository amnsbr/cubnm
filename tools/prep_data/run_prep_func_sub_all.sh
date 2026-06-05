#!/bin/bash

usage() {
    echo "Usage: $(basename "$0") [-j MAX_JOBS] [-r MAX_RETRIES]"
}

MAX_JOBS=${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}
MAX_RETRIES=${MAX_RETRIES:-3}

while getopts ":hj:r:" opt; do
    case $opt in
        h)
            usage
            exit 0
            ;;
        j)
            MAX_JOBS=$OPTARG
            ;;
        r)
            MAX_RETRIES=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer (got: $MAX_JOBS)" >&2
    exit 1
fi
if ! [[ "$MAX_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_RETRIES must be a positive integer (got: $MAX_RETRIES)" >&2
    exit 1
fi

cd $(dirname $0)
prep_data_dir=$(pwd)

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

process_subject() {
    local subject=$1
    echo "Processing $subject..."
    datalad get -n "$subject" || return $?
    uv run --directory "$prep_data_dir" "$prep_data_dir/prep_func_sub.py" --subjects "$subject" || return $?
}

process_subject_with_retry() {
    local subject=$1 attempt=1
    while (( attempt <= MAX_RETRIES )); do
        if process_subject "$subject"; then
            return 0
        fi
        if (( attempt < MAX_RETRIES )); then
            echo "Attempt $attempt/$MAX_RETRIES failed for $subject; retrying..."
        fi
        ((attempt++))
    done
    echo "Failed $subject after $MAX_RETRIES attempt(s)"
    return 1
}

cd $HCP_INPUT_DATASET_PATH/HCP1200

failed=0
for subject in */; do
    subject="${subject%/}"
    if [[ -f $HCP_OUTPUT_DIR/bold/$subject/REST2_RL/ctx_parc-schaefer-400_desc-bold.npz ]]; then
        echo "Skipping $subject because it already exists"
        continue
    fi
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n || ((failed++))
    done
    process_subject_with_retry "$subject" &
done
while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || ((failed++))
done
(( failed > 0 )) && exit 1
