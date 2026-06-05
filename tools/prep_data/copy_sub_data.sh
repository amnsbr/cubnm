#!/bin/bash
# Copy per-subject BOLD and SC outputs into src/cubnm/data/hcp/

cd "$(dirname "$0")"

if [ -f ../../.env ]; then
    source ../../.env
fi

SUBJECTS=(100206 100307)
CUBNM_DATA_DIR="${CUBNM_DATA_DIR:-$(cd ../../src/cubnm/data && pwd)}"

while [ $# -gt 0 ]; do
    case "$1" in
        --subjects)
            shift
            SUBJECTS=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                SUBJECTS+=("$1")
                shift
            done
            ;;
        --cubnm-data-dir)
            shift
            CUBNM_DATA_DIR=$(cd "$1" && pwd)
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$HCP_OUTPUT_DIR" ]; then
    echo "HCP_OUTPUT_DIR is not set" >&2
    exit 1
fi

BOLD_SESSIONS=(REST1_LR REST2_LR)

for sub in "${SUBJECTS[@]}"; do
    bold_copied=0
    for ses in "${BOLD_SESSIONS[@]}"; do
        ses_src="$HCP_OUTPUT_DIR/bold/$sub/$ses"
        if [ -d "$ses_src" ]; then
            mkdir -p "$CUBNM_DATA_DIR/hcp/bold/$sub/$ses"
            cp -a "$ses_src/." "$CUBNM_DATA_DIR/hcp/bold/$sub/$ses/"
            bold_copied=1
        fi
    done
    if [ "$bold_copied" -eq 1 ]; then
        echo "Copied BOLD for $sub (REST1_LR, REST2_LR)"
    else
        echo "Warning: BOLD not found for $sub (REST1_LR, REST2_LR)" >&2
    fi

    sc_src="$HCP_OUTPUT_DIR/SC/$sub"
    if [ -d "$sc_src" ]; then
        mkdir -p "$CUBNM_DATA_DIR/hcp/sc/$sub"
        sc_copied=0
        for f in "$sc_src"/ctx_parc-*_desc-*.csv; do
            [ -e "$f" ] || continue
            cp -a "$f" "$CUBNM_DATA_DIR/hcp/sc/$sub/"
            sc_copied=1
        done
        if [ "$sc_copied" -eq 1 ]; then
            echo "Copied SC for $sub"
        else
            echo "Warning: no ctx_parc-*_desc-*.csv files for $sub" >&2
        fi
    else
        echo "Warning: SC not found for $sub ($sc_src)" >&2
    fi
done
