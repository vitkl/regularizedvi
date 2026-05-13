#!/bin/bash
# Submit Hickey/Becker intestine multiome download as a Slurm job on Crick.
# Multiome donors only (B006, B008, B009, B010, B011, B012 → 45 samples, ~115 GB)
# from Dryad deposits 10.5061/dryad.8pk0p2ns8 (scRNA) + 10.5061/dryad.0zpc8672f (scATAC).
#
# Prereqs: ~/.dryad_credentials (chmod 600) with [dryad] app_id/secret.
#   See https://datadryad.org/ → ORCID login → My account → API accounts.
#
# Usage:
#   bash scripts/intestine_hickey/submit_download.sh           # submit
#   bash scripts/intestine_hickey/submit_download.sh --dry-run # echo sbatch only

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
TAG="intestine_hickey"
MANIFEST="${REPO_DIR}/data/dryad_hickey_intestine_multiome_manifest.tsv"
OUTPUT_DIR="/nemo/lab/briscoej/home/users/kleshcv/large_data/${TAG}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CREDENTIALS="${HOME}/.dryad_credentials"
if [[ ! -f "${CREDENTIALS}" ]] && ! $DRY_RUN; then
    echo "ERROR: ${CREDENTIALS} not found. Set up a Dryad API account first." >&2
    echo "  https://datadryad.org/ → ORCID login → My account → API accounts" >&2
    echo "  Then create the file:" >&2
    echo "    [dryad]" >&2
    echo "    app_id = <your_app_id>" >&2
    echo "    secret = <your_secret>" >&2
    echo "  chmod 600 ${CREDENTIALS}" >&2
    exit 1
fi

ENV_PATH="/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi"

WRAP="source ~/.bashrc
set -eo pipefail
export PYTHONNOUSERSITE=TRUE
conda activate ${ENV_PATH}
echo \"CONDA_PREFIX=\$CONDA_PREFIX  python=\$(which python)\"

echo
echo '--- 1. Download from Dryad (4 parallel workers) ---'
python -u ${REPO_DIR}/scripts/intestine_hickey/download_dryad.py \\
    --manifest ${MANIFEST} \\
    --output-dir ${OUTPUT_DIR} \\
    --credentials ${CREDENTIALS} \\
    --workers 4

echo
echo '--- 2. Extract per-compartment cell metadata from Seurat .rds objects ---'
ANNOT_DIR=${OUTPUT_DIR}/annotations
if compgen -G \"\${ANNOT_DIR}/clustered_*_object.rds\" > /dev/null; then
    Rscript ${REPO_DIR}/scripts/intestine_hickey/extract_seurat_metadata.R \\
        \${ANNOT_DIR}
else
    echo \"WARN: no clustered_*_object.rds files in \${ANNOT_DIR}; skipping Seurat metadata extraction\"
fi
"

SBATCH_ARGS=(
    --job-name=${TAG}_download
    --output="${LOG_DIR}/%j.${TAG}_download.out"
    --error="${LOG_DIR}/%j.${TAG}_download.err"
    --partition=ncpu
    --cpus-per-task=4
    --mem=16G
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
    echo "Monitor: bash ~/.claude/claude-shared-skills/scripts/check_jobs_slurm.sh --log-dir ${LOG_DIR} $JID"
fi
