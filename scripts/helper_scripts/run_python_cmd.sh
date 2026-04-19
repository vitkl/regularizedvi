#!/bin/bash
# Cluster-aware Python launcher. Auto-detects Crick vs Sanger and activates
# the appropriate regularizedvi conda env before executing Python.
# Usage:
#   bash scripts/helper_scripts/run_python_cmd.sh <script.py> [args...]
#   bash scripts/helper_scripts/run_python_cmd.sh -c "<python code>"
#   bash scripts/helper_scripts/run_python_cmd.sh --env <name> <script> [args...]

set -euo pipefail

# --- Parse --env if present (must be the first arg) ---
ENV_NAME="regularizedvi"
if [[ "${1:-}" == "--env" ]]; then
    shift
    ENV_NAME="${1:-}"
    if [[ -z "$ENV_NAME" ]]; then
        echo "ERROR: --env requires a value" >&2; exit 2
    fi
    shift
fi

# --- Cluster detection ---
if [[ -d /nemo/lab/briscoej && -d /camp/apps/eb/software/Miniconda3 ]]; then
    CLUSTER="crick"
elif [[ -d /software/conda/users/vk7 ]]; then
    CLUSTER="sanger"
else
    echo "ERROR: unrecognized cluster. Add a detection rule to run_python_cmd.sh." >&2; exit 3
fi

# --- Resolve env path ---
case "$CLUSTER:$ENV_NAME" in
    crick:regularizedvi)
        ENV_PATH="/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi" ;;
    sanger:regularizedvi)
        ENV_PATH="/software/conda/users/vk7/regularizedvi" ;;
    sanger:cell2state)
        ENV_PATH="/software/conda/users/vk7/cell2state_v2026_cuda124_torch25" ;;
    sanger:hashfrag)
        ENV_PATH="/software/conda/users/vk7/hashfrag_env" ;;
    crick:cell2state|crick:hashfrag)
        echo "ERROR: env '$ENV_NAME' not yet installed on Crick." >&2; exit 4 ;;
    *)
        echo "ERROR: unknown env '$ENV_NAME' on cluster '$CLUSTER'." >&2; exit 4 ;;
esac

# --- Ensure conda + module are available in this shell ---
if [[ "$CLUSTER" == "crick" ]]; then
    # User's ~/.bashrc has conda init; source it so we inherit the function
    # in non-interactive shells. module is provided by /etc/bashrc.
    # Disable -u briefly — /etc/bashrc references unset $BASHRCSOURCED.
    set +u
    # shellcheck source=/dev/null
    source "$HOME/.bashrc"
    set -u
else
    # Sanger: ISG/conda module defines the conda function on load.
    if ! type module >/dev/null 2>&1; then
        # shellcheck source=/dev/null
        source /etc/profile.d/modules.sh 2>/dev/null || true
    fi
    if ! type conda 2>/dev/null | head -1 | grep -q "function"; then
        module load ISG/conda 2>/dev/null || true
    fi
fi

if ! type conda 2>/dev/null | head -1 | grep -q "function"; then
    echo "ERROR: conda is not available as a shell function; cannot run 'conda activate'." >&2
    echo "Hint: on Crick, run scripts/helper_scripts/setup_bashrc_crick.sh to check prerequisites." >&2
    exit 5
fi

# --- Activate (PYTHONNOUSERSITE + PIP_CACHE_DIR explicit, before conda activate) ---
export PYTHONNOUSERSITE=TRUE
if [[ "$CLUSTER" == "crick" && -d /nemo/lab/briscoej/home/users/kleshcv ]]; then
    export PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip
    mkdir -p "$PIP_CACHE_DIR" 2>/dev/null || true
fi
conda activate "$ENV_PATH"

# --- Execute ---
if [[ "${1:-}" == "-c" ]]; then
    shift
    CODE="${1:-}"
    if [[ -z "$CODE" ]]; then
        echo "ERROR: -c requires a code string" >&2; exit 2
    fi
    exec python -c "$CODE"
elif [[ $# -eq 0 ]]; then
    echo "ERROR: no script or -c command given" >&2
    echo "Usage: bash $0 [--env NAME] <script.py> [args...]" >&2
    echo "       bash $0 [--env NAME] -c \"<code>\"" >&2
    exit 2
else
    exec python "$@"
fi
