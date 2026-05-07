#!/bin/bash
# Submit the HDMA Zenodo download as a long-running Slurm job on Crick.
# Pulls ~336 GB across 14 records (12 organs + cell metadata + caCREs +
# ChromBPNet training regions), then materialises the sample mapping CSV
# and extracts all_training_regions.tar.gz.
#
# Usage:
#   bash scripts/zenodo_download/submit_download.sh           # submit
#   bash scripts/zenodo_download/submit_download.sh --dry-run # echo sbatch only

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
RECORDS="${REPO_DIR}/scripts/zenodo_download/zenodo_records.tsv"
OUTPUT_DIR="/nemo/lab/briscoej/home/users/kleshcv/large_data/HDMA"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

ENV_PATH="/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi"
TRAINING_TARBALL="${OUTPUT_DIR}/caCREs/all_training_regions.tar.gz"
TRAINING_DEST="${OUTPUT_DIR}/caCREs/chrombpnet_training_regions"

WRAP="source ~/.bashrc
set -eo pipefail
export PYTHONNOUSERSITE=TRUE
export PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip
conda activate ${ENV_PATH}
echo \"CONDA_PREFIX=\$CONDA_PREFIX  python=\$(which python)\"

echo
echo '--- 1. Download from Zenodo (6 parallel workers) ---'
python -u ${REPO_DIR}/scripts/zenodo_download/download_hdma.py \\
    --records ${RECORDS} \\
    --output-dir ${OUTPUT_DIR} \\
    --workers 6

echo
echo '--- 2. Build sample mapping ---'
python -u ${REPO_DIR}/scripts/zenodo_download/create_sample_mapping.py \\
    --output-dir ${OUTPUT_DIR}

echo
echo '--- 3. Extract ChromBPNet training regions ---'
if [ -s ${TRAINING_TARBALL} ] && gzip -t ${TRAINING_TARBALL} 2>/dev/null; then
    mkdir -p ${TRAINING_DEST}
    tar -xzf ${TRAINING_TARBALL} -C ${TRAINING_DEST}
    echo \"Extracted training regions to ${TRAINING_DEST}\"
else
    echo \"WARN: ${TRAINING_TARBALL} missing or corrupt; skipping extract\"
fi
"

SBATCH_ARGS=(
    --job-name=hdma_download
    --output="${LOG_DIR}/%j.hdma_download.out"
    --error="${LOG_DIR}/%j.hdma_download.err"
    --partition=ncpu
    --cpus-per-task=4
    --mem=8G
    --time=2-00:00:00
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
    echo "Monitor: bash ~/.claude/shared-skills/scripts/check_jobs_slurm.sh --log-dir ${LOG_DIR} $JID"
fi
