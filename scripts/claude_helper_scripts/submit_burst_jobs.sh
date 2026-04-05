#!/bin/bash
# Submit burst_frequency_size experiment jobs from TSV
# Usage: bash scripts/claude_helper_scripts/submit_burst_jobs.sh [--dry-run] [--filter PATTERN]
#
# Reads docs/notebooks/model_comparisons/burst_jobs.tsv and submits each job via bsub+papermill.
# Columns with value "-" are skipped (not passed to papermill).

set -euo pipefail

TSV_FILE="docs/notebooks/model_comparisons/burst_jobs.tsv"
DRY_RUN=false
FILTER=""
QUEUE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --filter) FILTER="$2"; shift 2 ;;
        --tsv) TSV_FILE="$2"; shift 2 ;;
        --queue) QUEUE_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$TSV_FILE" ]]; then
    echo "ERROR: TSV file not found: $TSV_FILE"
    exit 1
fi

# Read header to get column names
IFS=$'\t' read -ra HEADER < <(head -1 "$TSV_FILE")

# Fixed bsub/papermill settings
CONDA_ENV="regularizedvi"
N_CPU=8
GPU_SPEC="mode=shared:j_exclusive=yes:gmem=80000:num=1"

submitted=0
skipped=0

# Process each data row (process substitution avoids subshell counter bug)
while IFS=$'\t' read -ra COLS; do
    # Map columns to variables
    declare -A ROW
    for i in "${!HEADER[@]}"; do
        ROW["${HEADER[$i]}"]="${COLS[$i]:-}"
    done

    name="${ROW[name]}"
    template="${ROW[template]}"
    output="${ROW[output]}"
    queue="${QUEUE_OVERRIDE:-${ROW[queue]}}"
    mem="${ROW[mem]}"
    priority="${ROW[priority]}"

    # Apply filter if specified
    if [[ -n "$FILTER" && ! "$name" =~ $FILTER ]]; then
        continue
    fi

    # Build papermill -r arguments from remaining columns
    PM_ARGS=""
    for col in "${HEADER[@]}"; do
        # Skip meta columns (used for bsub, not papermill)
        case "$col" in
            name|template|output|queue|mem|priority) continue ;;
        esac
        val="${ROW[$col]}"
        # Skip columns with "-" (use notebook default)
        if [[ "$val" == "-" || -z "$val" ]]; then
            continue
        fi
        PM_ARGS="$PM_ARGS -r $col $val"
    done

    # Build bsub command — add GPU flag only for gpu-* queues
    if [[ "$queue" == gpu-* ]]; then
        BSUB_CMD="bsub -q ${queue} -n ${N_CPU} -M ${mem} -R\"select[mem>${mem}] rusage[mem=${mem}] span[hosts=1]\" -gpu \"${GPU_SPEC}\" -sp ${priority} -e ./%J.gpu.err -o ./%J.gpu.out -J ${name} 'PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate ${CONDA_ENV} && papermill ${template} ${output}${PM_ARGS}'"
    else
        BSUB_CMD="bsub -q ${queue} -n 4 -M ${mem} -R\"select[mem>${mem}] rusage[mem=${mem}] span[hosts=1]\" -sp ${priority} -e ./%J.gpu.err -o ./%J.gpu.out -J ${name} 'PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate ${CONDA_ENV} && papermill ${template} ${output}${PM_ARGS}'"
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] $name:"
        echo "  $BSUB_CMD"
        echo ""
        ((skipped++)) || true
    else
        echo "Submitting: $name (queue=$queue, mem=${mem}MB, priority=$priority)"
        eval "$BSUB_CMD"
        ((submitted++)) || true
    fi
done < <(tail -n +2 "$TSV_FILE")

if $DRY_RUN; then
    echo "Dry run complete. $skipped jobs would be submitted."
else
    echo "Done. $submitted jobs submitted."
fi
