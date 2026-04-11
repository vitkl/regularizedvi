#!/bin/bash
# Submit HTAN multiome download as a long queue job.
# Usage: bash scripts/pan_cancer_data/submit_htan_download.sh

set -euo pipefail

# Resolve repo root from this script's location (scripts/pan_cancer_data/)
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOG_DIR=${REPO_DIR}/logs
mkdir -p "${LOG_DIR}"

DOWNLOAD_SCRIPT=${REPO_DIR}/scripts/pan_cancer_data/download_htan_multiome.py
SYMLINK_SCRIPT=${REPO_DIR}/scripts/pan_cancer_data/create_10x_symlinks.py
MANIFEST=${REPO_DIR}/data/htan_download_manifest.tsv

bsub -q long \
  -n 1 \
  -M 8000 \
  -R "select[mem>8000] rusage[mem=8000]" \
  -e "${LOG_DIR}/%J.htan_download.err" \
  -o "${LOG_DIR}/%J.htan_download.out" \
  -J htan_download \
  "PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && python ${DOWNLOAD_SCRIPT} --manifest ${MANIFEST} && python ${SYMLINK_SCRIPT} 2>&1"
