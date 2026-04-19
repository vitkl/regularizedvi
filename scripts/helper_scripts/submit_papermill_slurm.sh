#!/bin/bash
# Submit papermill training jobs from a TSV via sbatch on Crick Slurm.
# TSV schema matches the existing zloc_init_jobs.tsv / plate_fix_jobs.tsv
# format: name, template, output, queue, mem, priority, <papermill param columns>.
# Columns with value "-" are skipped. wandb_project / wandb_group / wandb_name
# are emitted as -r (raw-string papermill parameter); the rest as -p.
#
# Usage:
#   bash scripts/helper_scripts/submit_papermill_slurm.sh \
#       --tsv docs/notebooks/model_comparisons/embryo_slurm_jobs.tsv \
#       [--partition ga100] [--gres gpu:1] [--cpus 8] [--time 7-00:00:00] \
#       [--name-filter <regex>] [--dry-run]
#
# Per-job Slurm logs: <results_dir>/<jobid>.slurm.out, <jobid>.slurm.err
# (results_dir = dirname of the TSV 'output' column).

set -euo pipefail

# --- Defaults ---
# GPU allocation:
#   --gres=gpu:1 on Crick ga100 is already exclusive per GPU card (no MPS/MIG
#   sharing). Slurm GRES scheduling binds each allocation to a distinct GPU via
#   CUDA_VISIBLE_DEVICES; this matches the LSF `-gpu "...:j_exclusive=yes"`
#   contract used on Sanger. Use --exclusive-node to additionally reserve the
#   whole node (blocks other users' jobs from CPU/mem/other GPUs on the node).
TSV=""
PARTITION="ga100"
GRES="gpu:1"
CPUS=8
TIME="7-00:00:00"
NAME_FILTER=".*"
DRY_RUN=false
EXCLUSIVE_NODE=false
ENV_PATH="/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tsv)          TSV="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --gres)         GRES="$2"; shift 2 ;;
        --cpus)         CPUS="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --name-filter)  NAME_FILTER="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --exclusive-node) EXCLUSIVE_NODE=true; shift ;;
        -h|--help)
            sed -n '2,17p' "$0"; exit 0 ;;
        *)  echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$TSV" ]]; then
    echo "ERROR: --tsv is required" >&2
    exit 2
fi
if [[ ! -f "$TSV" ]]; then
    echo "ERROR: TSV not found: $TSV" >&2
    exit 2
fi

# --- Auto-detect usable reservation on PARTITION (ported from srun.sh) ---
detect_reservation() {
    local part="$1"
    scontrol show reservations --oneliner 2>/dev/null | while IFS= read -r line; do
        local nodes name accts users
        nodes=$(echo "$line" | grep -oP 'Nodes=\K\S+' || true)
        [[ -z "$nodes" ]] && continue
        local overlap
        overlap=$(comm -12 \
            <(sinfo -N -o "%N" --noheader -p "$part" 2>/dev/null | sort -u) \
            <(scontrol show hostname "$nodes" 2>/dev/null | sort -u))
        [[ -z "$overlap" ]] && continue

        name=$(echo "$line"  | grep -oP 'ReservationName=\K\S+' || true)
        accts=$(echo "$line" | grep -oP 'Accounts=\K\S+' || true)
        users=$(echo "$line" | grep -oP 'Users=\K\S+' || true)

        local has_access=0
        local user_name="${USER:-$(id -un)}"
        if [[ -n "$users" ]] && echo ",$users," | grep -q ",$user_name,"; then
            has_access=1
        fi
        if [[ $has_access -eq 0 && -n "$accts" ]]; then
            while read -r a; do
                if echo ",$accts," | grep -q ",$a,"; then
                    has_access=1; break
                fi
            done < <(sacctmgr -nP show assoc where user="$user_name" format=Account 2>/dev/null | sort -u)
        fi
        [[ $has_access -eq 1 ]] && echo "$name" && break
    done
}

RESERVATION="$(detect_reservation "$PARTITION" || true)"

# --- Read TSV header + build column-name -> index map ---
IFS=$'\t' read -ra HEADER < "$TSV"
declare -A IDX
for i in "${!HEADER[@]}"; do
    IDX["${HEADER[$i]}"]=$i
done

# Require these columns
for col in name template output mem; do
    if [[ -z "${IDX[$col]+set}" ]]; then
        echo "ERROR: required column missing from TSV header: $col" >&2
        exit 3
    fi
done

FIXED_COLS=(name template output queue mem priority)
RAW_COLS=(wandb_project wandb_group wandb_name)

is_fixed_col() {
    local c="$1"
    for f in "${FIXED_COLS[@]}"; do [[ "$c" == "$f" ]] && return 0; done
    return 1
}
is_raw_col() {
    local c="$1"
    for f in "${RAW_COLS[@]}"; do [[ "$c" == "$f" ]] && return 0; done
    return 1
}

# --- Truncate per-logdir "last submitted" ledgers up-front.
# Before the per-row loop we don't yet know which log dirs will be touched;
# we truncate lazily inside the loop by tracking which have been seen this run.
declare -A SEEN_LOGDIR=()

# --- Iterate data rows via process substitution so counters stay in the
# current shell (a `tail | while` pipe runs the loop body in a subshell).
while IFS= read -r line; do
    # Split the row (use same field separator)
    IFS=$'\t' read -ra ROW <<< "$line"

    name="${ROW[${IDX[name]:-0}]:-}"
    [[ -z "$name" ]] && continue

    # --- Filter by --name-filter regex ---
    if ! [[ "$name" =~ $NAME_FILTER ]]; then
        continue
    fi

    template="${ROW[${IDX[template]:-1}]:-}"
    output="${ROW[${IDX[output]:-2}]:-}"
    mem="${ROW[${IDX[mem]:-4}]:-}"

    if [[ -z "$template" || -z "$output" || -z "$mem" || "$mem" == "-" ]]; then
        echo "WARN: skipping '$name' — missing required fields" >&2
        continue
    fi

    # --- Log dir = dirname(output) ---
    LOGDIR="$(dirname "$output")"
    mkdir -p "$LOGDIR"

    # First time we see this logdir in THIS run, truncate the ledgers so
    # .last_submitted_*.txt reflects only the current invocation, not history.
    if ! $DRY_RUN && [[ -z "${SEEN_LOGDIR[$LOGDIR]:-}" ]]; then
        : > "$LOGDIR/.last_submitted_jobs.txt"
        : > "$LOGDIR/.last_submitted_summary.tsv"
        SEEN_LOGDIR[$LOGDIR]=1
    fi

    # --- Build papermill args from all non-fixed columns ---
    PM_ARGS=()
    for i in "${!HEADER[@]}"; do
        col="${HEADER[$i]}"
        is_fixed_col "$col" && continue
        val="${ROW[$i]:-}"
        [[ -z "$val" || "$val" == "-" ]] && continue

        if is_raw_col "$col"; then
            PM_ARGS+=(-r "$col" "$val")
        else
            PM_ARGS+=(-p "$col" "$val")
        fi
    done

    # --- Build sbatch command ---
    SBATCH_ARGS=(
        --job-name="$name"
        --output="${LOGDIR}/%j.slurm.out"
        --error="${LOGDIR}/%j.slurm.err"
        --partition="$PARTITION"
        --gres="$GRES"
        --cpus-per-task="$CPUS"
        --time="$TIME"
        --mem="${mem}M"
    )
    if $EXCLUSIVE_NODE; then
        SBATCH_ARGS+=(--exclusive)
    fi
    if [[ -n "$RESERVATION" ]]; then
        SBATCH_ARGS+=(--reservation="$RESERVATION")
    fi

    # --- Build wrapped command ---
    # Use printf %q to quote each papermill arg safely into the single --wrap string.
    PM_ARGS_QUOTED=""
    for a in "${PM_ARGS[@]}"; do
        PM_ARGS_QUOTED+=" $(printf '%q' "$a")"
    done
    # PYTHONNOUSERSITE=TRUE + PIP_CACHE_DIR are set EXPLICITLY immediately
    # before `conda activate` (not relied on from ~/.bashrc). source ~/.bashrc
    # brings in conda init + module. PIP_CACHE_DIR redirect prevents any
    # incidental `pip install` inside the notebook from blowing the 5 GB
    # homefs quota.
    WRAP="source ~/.bashrc && export PYTHONNOUSERSITE=TRUE && export PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip && conda activate $(printf '%q' "$ENV_PATH") && echo \"CONDA_PREFIX=\$CONDA_PREFIX PYTHONNOUSERSITE=\$PYTHONNOUSERSITE python=\$(which python)\" && papermill $(printf '%q' "$template") $(printf '%q' "$output")${PM_ARGS_QUOTED}"

    echo "----------------------------------------"
    echo "Job: $name"
    echo "Template: $template"
    echo "Output:   $output"
    echo "Logs:     $LOGDIR/<jobid>.slurm.{out,err}"
    echo "sbatch ${SBATCH_ARGS[*]} --wrap=\"${WRAP}\""

    if ! $DRY_RUN; then
        OUT=$(sbatch "${SBATCH_ARGS[@]}" --wrap="$WRAP" 2>&1)
        echo "$OUT"
        JID=$(echo "$OUT" | grep -oP 'Submitted batch job \K\d+' || true)
        if [[ -n "$JID" ]]; then
            echo "$JID" >> "$LOGDIR/.last_submitted_jobs.txt"
            printf '%s\t%s\t%s\n' "$JID" "$name" "$output" >> "$LOGDIR/.last_submitted_summary.tsv"
        fi
    fi
done < <(tail -n +2 "$TSV")

if ! $DRY_RUN; then
    echo ""
    echo "=========================================="
    echo "Submission summary"
    echo "=========================================="
    # SEEN_LOGDIR was populated by the loop above (same shell, thanks to
    # the process-substitution read-from). Iterate its keys safely.
    for d in "${!SEEN_LOGDIR[@]}"; do
        if [[ -f "$d/.last_submitted_summary.tsv" ]]; then
            echo "Logdir: $d"
            cat "$d/.last_submitted_summary.tsv"
            echo ""
        fi
    done
    echo "Monitor with:"
    for d in "${!SEEN_LOGDIR[@]}"; do
        if [[ -f "$d/.last_submitted_jobs.txt" ]]; then
            JIDS=$(tr '\n' ' ' < "$d/.last_submitted_jobs.txt")
            echo "  bash ~/.claude/shared-skills/scripts/check_jobs_slurm.sh --log-dir $d $JIDS"
        fi
    done
    echo ""
    echo "Quick status:  squeue -u \$USER"
fi
