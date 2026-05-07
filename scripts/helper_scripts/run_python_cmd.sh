#!/bin/bash
# Cluster-aware Python launcher. Thin wrapper on top of run_conda_bash.sh that
# adds Python-script-path translation (local → cluster project root) and
# forwards `python <script> [args...]` or `python -c <code>` to the foundation.
#
# Usage:
#   bash scripts/helper_scripts/run_python_cmd.sh <script.py> [args...]
#   bash scripts/helper_scripts/run_python_cmd.sh -c "<python code>"
#   bash scripts/helper_scripts/run_python_cmd.sh --env <name> <script> [args...]
#   bash scripts/helper_scripts/run_python_cmd.sh --remote [crick|sanger] [--project NAME] <script> [args...]
#
# Env activation, SSH dispatch, and SSH-host probing live in run_conda_bash.sh.

set -euo pipefail

# --- Project map (used only for path translation here) ---
declare -A PROJECTS_LOCAL=(
    [cell2state]=/Users/kleshcv/Desktop/my_packages/cell2state
    [cell2state_embryo]=/Users/kleshcv/Desktop/my_packages/cell2state_embryo
    [regularizedvi]=/Users/kleshcv/Desktop/my_packages/regularizedvi
)
declare -A PROJECTS_CRICK=(
    [cell2state]=/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state
    [cell2state_embryo]=/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state_embryo
    [regularizedvi]=/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi
)
declare -A PROJECTS_SANGER=(
    [cell2state]=/nfs/team205/vk7/sanger_projects/my_packages/cell2state
    [cell2state_embryo]=/nfs/team205/vk7/sanger_projects/cell2state_embryo
    [regularizedvi]=/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi
)

# --- Parse leading flags ---
ENV_NAME="regularizedvi"
REMOTE_HOST=""
PROJECT=""

while [[ $# -gt 0 ]]; do
    case "${1:-}" in
        --env)
            shift
            ENV_NAME="${1:-}"
            if [[ -z "$ENV_NAME" ]]; then
                echo "ERROR: --env requires a value" >&2; exit 2
            fi
            shift
            ;;
        --remote)
            shift
            if [[ "${1:-}" == "crick" || "${1:-}" == "sanger" ]]; then
                REMOTE_HOST="$1"
                shift
            else
                REMOTE_HOST="crick"
            fi
            ;;
        --project)
            shift
            PROJECT="${1:-}"
            if [[ -z "$PROJECT" ]]; then
                echo "ERROR: --project requires a value" >&2; exit 2
            fi
            shift
            ;;
        *)
            break
            ;;
    esac
done

# --- Auto-detect remote target from script-path prefix (Mac only) ---
if [[ "$(uname)" == "Darwin" && -z "${REMOTE_HOST:-}" && "${1:-}" != "-c" && -n "${1:-}" ]]; then
    case "${1:-}" in
        /nemo/*|/camp/*) REMOTE_HOST=crick ;;
        /nfs/team205/*|/software/conda/*) REMOTE_HOST=sanger ;;
    esac
fi

# --- Translate a script-path arg from local to cluster project root.
# Refuses to translate paths inside the local cell2state repo (per plan).
translate_script_path() {
    local path="$1"
    local local_root="${PROJECTS_LOCAL[$PROJECT]}"
    local cell2state_local="${PROJECTS_LOCAL[cell2state]}"
    local remote_root
    if [[ "$REMOTE_HOST" == "crick" ]]; then
        remote_root="${PROJECTS_CRICK[$PROJECT]}"
    else
        remote_root="${PROJECTS_SANGER[$PROJECT]}"
    fi

    local resolved=""
    case "$path" in
        /nemo/*|/camp/*|/nfs/team205/*)
            echo "$path"; return 0 ;;
        "~"|"~/"*)
            echo "$path"; return 0 ;;
        "$local_root"/*|"$local_root")
            resolved="$path" ;;
        /*)
            echo "ERROR: absolute path $path is outside project root $local_root and not a known cluster path." >&2
            exit 7 ;;
        *)
            resolved="$local_root/$path" ;;
    esac

    if [[ "$resolved" == "$cell2state_local" || "$resolved" == "$cell2state_local"/* ]]; then
        echo "ERROR: $path is inside the local cell2state repo. Use the local path directly" >&2
        echo "       (without --remote) to read project source files. --remote is for accessing" >&2
        echo "       cluster-resident data or for cross-project reads (--project NAME)." >&2
        exit 7
    fi

    echo "${remote_root}${resolved#$local_root}"
    return 0
}

# --- Auto-detect project from cwd's git toplevel (only needed when remote) ---
detect_project() {
    [[ -n "${PROJECT:-}" ]] && return 0
    local toplevel
    toplevel="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -z "$toplevel" ]]; then
        echo "ERROR: --project not given and cwd is not in a git repo. Pass --project NAME." >&2; exit 6
    fi
    for name in "${!PROJECTS_LOCAL[@]}"; do
        if [[ "$toplevel" == "${PROJECTS_LOCAL[$name]}" ]]; then
            PROJECT="$name"; return 0
        fi
    done
    echo "ERROR: cwd's git toplevel ($toplevel) does not match any known project. Pass --project NAME." >&2
    exit 6
}

# --- Build the python invocation ---
if [[ "${1:-}" == "-c" ]]; then
    shift
    CODE="${1:-}"
    if [[ -z "$CODE" ]]; then
        echo "ERROR: -c requires a code string" >&2; exit 2
    fi
    shift || true
    PY_ARGS=(python -c "$CODE" "$@")
elif [[ $# -eq 0 ]]; then
    echo "ERROR: no script or -c command given" >&2
    echo "Usage: bash $0 [--remote [crick|sanger]] [--project NAME] [--env NAME] <script.py> [args...]" >&2
    echo "       bash $0 [--env NAME] -c \"<code>\"" >&2
    exit 2
else
    SCRIPT="$1"; shift
    # Translate path only when dispatching remotely from Mac.
    if [[ "$(uname)" == "Darwin" && -n "${REMOTE_HOST:-}" ]]; then
        detect_project
        SCRIPT="$(translate_script_path "$SCRIPT")"
    fi
    PY_ARGS=(python "$SCRIPT" "$@")
fi

# --- Delegate to run_conda_bash.sh ---
DIR="$(cd "$(dirname "$0")" && pwd)"
DELEGATE_FLAGS=(--env "$ENV_NAME")
if [[ -n "${REMOTE_HOST:-}" ]]; then
    DELEGATE_FLAGS+=(--remote "$REMOTE_HOST")
fi
if [[ -n "${PROJECT:-}" ]]; then
    DELEGATE_FLAGS+=(--project "$PROJECT")
fi

exec bash "$DIR/run_conda_bash.sh" "${DELEGATE_FLAGS[@]}" -- "${PY_ARGS[@]}"
