#!/bin/bash
# Crick-tailored per-sample snapatac2 import_data wrapper.
#
# Identical purpose to cell2state/scripts/run_snapatac_one_sample.sh but
# assumes conda env is already active (e.g. via run_conda_bash.sh). The
# cell2state worker tries `module load ISG/conda` (Sanger-only) and then
# `conda activate $CONDA_ENV` — both fail on Crick when invoked from a
# bash subshell that doesn't inherit the conda shell function.
#
# This wrapper:
#   - Skips conda activation (caller must activate via run_conda_bash.sh).
#   - Resolves the load_atac_snapatac2.py CLI path via cell2state package.
#   - Stages fragments to /var/tmp for fast IO, copies h5ad back to NFS.
#
# Usage:
#   bash run_snapatac_one_sample_crick.sh <FRAG_PATH> <SAMPLE> <GENOME_REF>
#
# Typical invocation (from submit_snapatac_slurm.sh):
#   bash $RUN_CONDA_BASH --env cell2state -- bash $THIS_SCRIPT FRAG SAMPLE GENOME

set -eo pipefail

FRAG_PATH="$1"
SAMPLE="$2"
GENOME_REF="$3"

if [ -z "$FRAG_PATH" ] || [ -z "$SAMPLE" ] || [ -z "$GENOME_REF" ]; then
    echo "Usage: $0 <FRAG_PATH> <SAMPLE> <GENOME_REF>" >&2
    exit 1
fi

FRAG_DIR=$(dirname "$FRAG_PATH")
OUTPUT_H5AD="${FRAG_DIR}/atac_fragments.h5ad"

# Skip if h5ad already exists
if [ -f "$OUTPUT_H5AD" ]; then
    echo "h5ad already exists for ${SAMPLE}, skipping: ${OUTPUT_H5AD}"
    exit 0
fi

# Verify python from active env can find cell2state and locate the CLI
LOAD_SCRIPT=$(python -c "import cell2state, os; print(os.path.join(os.path.dirname(cell2state.__file__), 'utils', 'load_atac_snapatac2.py'))" 2>/dev/null || true)
if [ -z "$LOAD_SCRIPT" ] || [ ! -f "$LOAD_SCRIPT" ]; then
    echo "ERROR: cannot locate cell2state load_atac_snapatac2.py — is conda env active?" >&2
    python -c "import cell2state; print(cell2state.__file__)" >&2 || true
    exit 2
fi

echo "Worker: run_snapatac_one_sample_crick.sh"
echo "Sample: ${SAMPLE}"
echo "Frag:   ${FRAG_PATH}"
echo "Genome: ${GENOME_REF}"
echo "Out:    ${OUTPUT_H5AD}"
echo "Python: $(which python)"
echo "Load script: ${LOAD_SCRIPT}"

# 1. Stage fragments to /var/tmp (node-local) for fast IO
LOCAL_TMPDIR="/var/tmp/snapatac_${SAMPLE}_$$"
cleanup() {
    echo "Cleaning up ${LOCAL_TMPDIR} ..."
    rm -rf "$LOCAL_TMPDIR"
}
trap cleanup EXIT

echo "Copying fragments to ${LOCAL_TMPDIR} ..."
mkdir -p "$LOCAL_TMPDIR"
cp "$FRAG_PATH" "$LOCAL_TMPDIR/atac_fragments.tsv.gz"
if [ -f "${FRAG_PATH}.tbi" ]; then
    cp "${FRAG_PATH}.tbi" "$LOCAL_TMPDIR/atac_fragments.tsv.gz.tbi"
else
    echo "Warning: .tbi index not found, proceeding without it"
fi

# 2. Run snapatac2 import via the cell2state CLI
echo "Running snapatac2 import_data for ${SAMPLE} ..."
export PYTHONNOUSERSITE=TRUE

python "$LOAD_SCRIPT" \
    --path "$LOCAL_TMPDIR/atac_fragments.tsv.gz" \
    --sample "$SAMPLE" \
    --overwrite True \
    --use_complete_path True \
    --path_to_reference "$GENOME_REF"

# 3. Copy h5ad back to NFS
echo "Copying h5ad to ${OUTPUT_H5AD} ..."
cp "$LOCAL_TMPDIR/atac_fragments.h5ad" "$OUTPUT_H5AD"

# 4. EXIT trap handles cleanup
echo "Done: ${SAMPLE}"
