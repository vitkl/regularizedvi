#!/bin/bash
# Submit Roy et al. Down syndrome fetal liver multiome download as a Slurm job on Crick.
# Disomic donors only (H_5, H_6, H_7 → 6 samples, ~15 GB) from EBI BioStudies E-MTAB-13070.
# Usage:
#   bash scripts/down_fetal_blood/submit_download.sh           # submit
#   bash scripts/down_fetal_blood/submit_download.sh --dry-run # echo sbatch only

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
TAG="down_fetal_blood"
MANIFEST="${REPO_DIR}/data/E-MTAB-13070_${TAG}_disomic_manifest.tsv"
OUTPUT_DIR="/nemo/lab/briscoej/home/users/kleshcv/large_data/${TAG}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

ENV_PATH="/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi"

WRAP="source ~/.bashrc
set -eo pipefail
export PYTHONNOUSERSITE=TRUE
conda activate ${ENV_PATH}
echo \"CONDA_PREFIX=\$CONDA_PREFIX  python=\$(which python)\"

python -u ${REPO_DIR}/scripts/geo_download/download_multiome.py \\
    --manifest ${MANIFEST} \\
    --output-dir ${OUTPUT_DIR}
"

SBATCH_ARGS=(
    --job-name=${TAG}_download
    --output="${LOG_DIR}/%j.${TAG}_download.out"
    --error="${LOG_DIR}/%j.${TAG}_download.err"
    --partition=ncpu
    --cpus-per-task=1
    --mem=4G
    --time=12:00:00
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
