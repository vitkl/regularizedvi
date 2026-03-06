#!/bin/bash
# Extract library_log_means/vars and key parameter stats from saved models.
# Usage: bash _extract_lib_values.sh [model_filter]
export PYTHONNOUSERSITE="TRUE"
module load ISG/conda 2>/dev/null
conda run -n regularizedvi python _extract_lib_values.py "$@" 2>/dev/null
