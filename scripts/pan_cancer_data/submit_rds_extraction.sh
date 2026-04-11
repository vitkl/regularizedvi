#!/bin/bash
# Submit R metadata extraction for HTAN multiome .rds files.
# Usage: bash scripts/pan_cancer_data/submit_rds_extraction.sh

set -euo pipefail

# Resolve repo root from this script's location (scripts/pan_cancer_data/)
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOG_DIR=${REPO_DIR}/logs
mkdir -p "${LOG_DIR}"

DATA_DIR=/nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome
OUT_DIR=${DATA_DIR}/annotations
mkdir -p "${OUT_DIR}"

SCRIPT=${REPO_DIR}/scripts/pan_cancer_data/convert_rds_annotations.R

# ATAC extraction (139 files, ~2 GB each)
bsub -q normal \
  -n 2 \
  -M 32000 \
  -R "select[mem>32000] rusage[mem=32000] span[hosts=1]" \
  -e "${LOG_DIR}/%J.rds_atac.err" \
  -o "${LOG_DIR}/%J.rds_atac.out" \
  -J rds_atac \
  "module load HGI/softpack/groups/jaguar_analysis/seurat5_signac/1 && Rscript ${SCRIPT} ${DATA_DIR}/level4/atac ${OUT_DIR}/pan_cancer_multiome_atac_annotations.csv atac"

# RNA extraction (124 files, mostly smaller)
bsub -q normal \
  -n 2 \
  -M 32000 \
  -R "select[mem>32000] rusage[mem=32000] span[hosts=1]" \
  -e "${LOG_DIR}/%J.rds_rna.err" \
  -o "${LOG_DIR}/%J.rds_rna.out" \
  -J rds_rna \
  "module load HGI/softpack/groups/jaguar_analysis/seurat5_signac/1 && Rscript ${SCRIPT} ${DATA_DIR}/level4/rna ${OUT_DIR}/pan_cancer_multiome_rna_annotations.csv rna"
