#!/bin/bash
# Submit per-sample snapatac2 import jobs on Crick Slurm.
#
# Wraps cell2state's run_snapatac_one_sample.sh (which does conda activation,
# /var/tmp staging, calls load_atac_snapatac2.py CLI, copies result back).
# One sbatch per sample. Skips samples whose atac_fragments.h5ad already exists
# (the worker also re-checks this and exits cleanly).
#
# Usage:
#   bash submit_snapatac_slurm.sh <FRAGMENTS_DIR> <GENOME_REF> <SAMPLE> [SAMPLE...]
#
# Arguments:
#   FRAGMENTS_DIR  Directory containing per-sample subdirs, each with atac_fragments.tsv.gz
#                  e.g. /nemo/.../gastrulation_multiome_anndata/latest/data/original_with_atac
#   GENOME_REF     Path to 10X reference genome directory (must end with /)
#                  e.g. /nemo/.../genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/
#   SAMPLE         Sample name; fragments expected at $FRAGMENTS_DIR/$SAMPLE/atac_fragments.tsv.gz
#
# Optional env overrides:
#   PARTITION  (default: ncpu)
#   MEM        (default: 40G)
#   CPUS       (default: 8)
#   TIME       (default: 08:00:00)
#   CONDA_ENV  (default: cell2state_v2026_cuda124_torch25)
#   LOG_DIR    (default: <FRAGMENTS_DIR>/snapatac_logs)
#   DRY_RUN    (default: 0; set to 1 to print sbatch commands without submitting)
#
# Example (mouse gastrulation, all 11 samples):
#   bash scripts/submit_snapatac_slurm.sh \
#     /nemo/lab/briscoej/home/users/kleshcv/large_data/gastrulation_multiome_anndata/latest/data/original_with_atac \
#     /nemo/lab/briscoej/home/users/kleshcv/large_data/genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/ \
#     E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep1 E8.0_rep2 \
#     E8.5_rep1 E8.5_rep2 E8.5_CRISPR_T_KO E8.5_CRISPR_T_WT \
#     E8.75_rep1 E8.75_rep2

set -eo pipefail

if [ "$#" -lt 3 ]; then
    sed -n '2,28p' "$0"
    exit 1
fi

FRAGMENTS_DIR="$1"
GENOME_REF="$2"
shift 2
SAMPLES=("$@")

PARTITION="${PARTITION:-ncpu}"
MEM="${MEM:-40G}"
CPUS="${CPUS:-8}"
TIME="${TIME:-08:00:00}"
CONDA_ENV="${CONDA_ENV:-cell2state_v2026_cuda124_torch25}"
DRY_RUN="${DRY_RUN:-0}"

# Use a Crick-tailored worker that assumes conda env is already active
# (the upstream cell2state worker does `module load ISG/conda; conda activate`,
# which doesn't work in a sbatch --wrap'd bash subshell on Crick — conda is a
# shell function and doesn't propagate). The Crick worker mirrors the staging
# + python-call logic of the cell2state worker but trusts the caller for env.
WORKER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_snapatac_one_sample_crick.sh"
if [ ! -f "$WORKER" ]; then
    echo "ERROR: Crick worker script not found at $WORKER" >&2
    exit 2
fi

# Locate the cluster-aware conda activator. run_conda_bash.sh activates the
# right conda env on Crick/Sanger/Mac. We invoke it as the entry point of the
# sbatch --wrap so the worker runs inside the correct env without needing the
# worker's own Sanger-only `module load ISG/conda` to succeed.
RUN_CONDA_BASH="/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/scripts/helper_scripts/run_conda_bash.sh"
if [ ! -f "$RUN_CONDA_BASH" ]; then
    echo "ERROR: run_conda_bash.sh not found at $RUN_CONDA_BASH" >&2
    exit 2
fi

LOG_DIR="${LOG_DIR:-${FRAGMENTS_DIR}/snapatac_logs}"
mkdir -p "$LOG_DIR"

echo "Worker:        $WORKER"
echo "Fragments dir: $FRAGMENTS_DIR"
echo "Genome ref:    $GENOME_REF"
echo "Log dir:       $LOG_DIR"
echo "Resources:     partition=$PARTITION mem=$MEM cpus=$CPUS time=$TIME"
echo "Conda env:     $CONDA_ENV"
echo "Samples:       ${#SAMPLES[@]}"
echo

SUBMITTED=0
SKIPPED_EXIST=0
SKIPPED_MISSING=0
FAILED=0

for SAMPLE in "${SAMPLES[@]}"; do
    FRAG_PATH="${FRAGMENTS_DIR}/${SAMPLE}/atac_fragments.tsv.gz"
    OUTPUT_H5AD="${FRAGMENTS_DIR}/${SAMPLE}/atac_fragments.h5ad"

    if [ ! -f "$FRAG_PATH" ]; then
        echo "[SKIP] $SAMPLE — fragments missing at $FRAG_PATH"
        SKIPPED_MISSING=$((SKIPPED_MISSING + 1))
        continue
    fi
    if [ -f "$OUTPUT_H5AD" ]; then
        echo "[SKIP] $SAMPLE — h5ad already exists at $OUTPUT_H5AD"
        SKIPPED_EXIST=$((SKIPPED_EXIST + 1))
        continue
    fi

    JOB_NAME="snap_${SAMPLE}"
    OUT_FILE="${LOG_DIR}/${SAMPLE}.slurm.out"
    ERR_FILE="${LOG_DIR}/${SAMPLE}.slurm.err"
    # Wrap the Crick worker through run_conda_bash.sh so conda activation is
    # handled by the regularizedvi cluster-aware helper. run_conda_bash.sh
    # exec's the command after `conda activate`, so the worker inherits the
    # full env (PATH + the conda function — though the Crick worker doesn't
    # need the function, it just trusts `which python` resolves to the env).
    WRAP="bash ${RUN_CONDA_BASH} --env cell2state -- bash ${WORKER} ${FRAG_PATH} ${SAMPLE} ${GENOME_REF}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] sbatch --job-name=$JOB_NAME --partition=$PARTITION --mem=$MEM --cpus-per-task=$CPUS --time=$TIME --output=$OUT_FILE --error=$ERR_FILE --wrap=\"$WRAP\""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    if JID=$(sbatch --parsable \
                    --job-name="$JOB_NAME" \
                    --partition="$PARTITION" \
                    --mem="$MEM" \
                    --cpus-per-task="$CPUS" \
                    --time="$TIME" \
                    --output="$OUT_FILE" \
                    --error="$ERR_FILE" \
                    --wrap="$WRAP"); then
        echo "[SUBMIT] $SAMPLE -> JobId=$JID  (logs: $OUT_FILE, $ERR_FILE)"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "[FAIL]   $SAMPLE — sbatch returned non-zero"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "=== Summary ==="
echo "Submitted:        $SUBMITTED"
echo "Skipped (exists): $SKIPPED_EXIST"
echo "Skipped (no frag):$SKIPPED_MISSING"
echo "Failed sbatch:    $FAILED"
