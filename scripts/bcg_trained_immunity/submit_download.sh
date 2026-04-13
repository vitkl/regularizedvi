#!/bin/bash
# Submit BCG Trained Immunity multiome download as a long queue job.
# Usage: bash scripts/bcg_trained_immunity/submit_download.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOG_DIR=${REPO_DIR}/logs
mkdir -p "${LOG_DIR}"

GSE_TAG="gse295277_gse295308"
MANIFEST=${REPO_DIR}/data/${GSE_TAG}_download_manifest.tsv
OUTPUT_DIR="/nfs/team205/vk7/sanger_projects/large_data/bcg_trained_immunity"

bsub -q long \
  -n 1 \
  -M 8000 \
  -R "select[mem>8000] rusage[mem=8000]" \
  -e "${LOG_DIR}/%J.${GSE_TAG}_download.err" \
  -o "${LOG_DIR}/%J.${GSE_TAG}_download.out" \
  -J ${GSE_TAG}_download \
  "PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && python ${REPO_DIR}/scripts/geo_download/download_multiome.py --manifest ${MANIFEST} --output-dir ${OUTPUT_DIR} && python ${REPO_DIR}/scripts/geo_download/create_sample_mapping.py --manifest ${MANIFEST} --data-dir ${OUTPUT_DIR} 2>&1"
