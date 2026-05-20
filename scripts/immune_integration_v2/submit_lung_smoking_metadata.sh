#!/usr/bin/env bash
# Submit lung_smoking Seurat meta.data extraction as a Slurm job on Crick.
# Matches the convention of scripts/intestine_hickey/submit_download.sh:
# uses --wrap, source ~/.bashrc + conda activate seurat (no `module load`),
# pins R_LIBS to the seurat env to avoid picking up user-level R libs.
#
# Phase 0 step 0.4 of immune_integration_v2.
#
# Usage:
#   bash scripts/immune_integration_v2/submit_lung_smoking_metadata.sh           # submit
#   bash scripts/immune_integration_v2/submit_lung_smoking_metadata.sh --dry-run # echo sbatch only

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
TAG="lung_smoking_meta"
DATA_DIR="/nemo/lab/briscoej/home/users/kleshcv/large_data/lung_smoking"
LOG_DIR="${DATA_DIR}/logs"
mkdir -p "${LOG_DIR}"

RDS_PATH="${DATA_DIR}/annotations/GSE241468_share_seur.rds"
CSV_OUT="${DATA_DIR}/annotations/lung_smoking_meta.csv"
R_SCRIPT="${REPO_DIR}/scripts/immune_integration_v2/extract_lung_smoking_metadata.R"

if [[ ! -f "${RDS_PATH}" ]] && ! $DRY_RUN; then
    echo "ERROR: ${RDS_PATH} not found." >&2
    exit 1
fi

WRAP="source ~/.bashrc
set -eo pipefail
export PYTHONNOUSERSITE=TRUE
conda activate seurat
export R_LIBS_USER=\${CONDA_PREFIX}/lib/R/library
export R_LIBS=\${CONDA_PREFIX}/lib/R/library
echo \"CONDA_PREFIX=\$CONDA_PREFIX  R: \$(which Rscript) \$(Rscript --version 2>&1)\"
export RDS_PATH=${RDS_PATH}
export CSV_OUT=${CSV_OUT}
Rscript --vanilla ${R_SCRIPT}
"

SBATCH_ARGS=(
    --job-name=${TAG}_extract
    --output="${LOG_DIR}/%j.${TAG}_extract.out"
    --error="${LOG_DIR}/%j.${TAG}_extract.err"
    --partition=ncpu
    --cpus-per-task=4
    --mem=64G
    --time=02:00:00
)

echo "sbatch ${SBATCH_ARGS[*]} --wrap=<see below>"
echo "---WRAP---"
echo "${WRAP}"
echo "----------"

if $DRY_RUN; then
    echo "[DRY RUN] not submitting"
    exit 0
fi

OUT=$(sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}" 2>&1)
echo "$OUT"
JID=$(echo "$OUT" | grep -oP 'Submitted batch job \K\d+' || true)
if [[ -n "$JID" ]]; then
    echo "$JID" > "${LOG_DIR}/.last_submitted_jobs.txt"
    echo "Monitor: bash ~/.claude/claude-shared-skills/scripts/check_jobs_slurm.sh --log-dir ${LOG_DIR} $JID"
fi
