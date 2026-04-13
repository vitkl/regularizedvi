#!/bin/bash
# Submit GSE249572 multiome download as a long queue job.
# Usage: bash scripts/neuron_patterning_data/submit_download.sh

set -euo pipefail

# Resolve repo root from this script's location (scripts/neuron_patterning_data/)
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOG_DIR=${REPO_DIR}/logs
mkdir -p "${LOG_DIR}"

DOWNLOAD_SCRIPT=${REPO_DIR}/scripts/neuron_patterning_data/download_gse249572.py
MAPPING_SCRIPT=${REPO_DIR}/scripts/neuron_patterning_data/create_sample_mapping.py
MANIFEST=${REPO_DIR}/data/gse249572_download_manifest.tsv

bsub -q long \
  -n 1 \
  -M 8000 \
  -R "select[mem>8000] rusage[mem=8000]" \
  -e "${LOG_DIR}/%J.gse249572_download.err" \
  -o "${LOG_DIR}/%J.gse249572_download.out" \
  -J gse249572_download \
  "PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && python ${DOWNLOAD_SCRIPT} --manifest ${MANIFEST} && python ${MAPPING_SCRIPT} --manifest ${MANIFEST} 2>&1"
