#!/bin/bash

cd $(dirname $0)

for parc in "aparc" "schaefer-100" "schaefer-200" "schaefer-400"; do
    for measure in "strength" "length"; do
        echo "Processing $parc $measure..."
        uv run prep_sc_sub.py --parc $parc --measure $measure
    done
done