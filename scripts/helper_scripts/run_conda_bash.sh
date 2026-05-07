#!/bin/bash
# Cluster-aware conda env activator + bash command runner. Auto-detects
# Mac / Crick / Sanger and activates the appropriate conda env, then exec's
# an arbitrary bash command. Supports remote dispatch from Mac to Crick or
# Sanger via SSH.
#
# Usage:
#   bash scripts/helper_scripts/run_conda_bash.sh [--remote crick|sanger] \
#        [--env NAME] [--project NAME] -- <bash command...>
#
# Examples:
#   bash run_conda_bash.sh -- git commit -m "msg"
#   bash run_conda_bash.sh --env cell2state -- gh pr list
#   bash run_conda_bash.sh --remote crick -- nvidia-smi
#
# This script is the FOUNDATION layer. `run_python_cmd.sh` is a thin wrapper
# on top that adds Python-script-path translation + python invocation.

set -euo pipefail

# --- Project map ---
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

# --- Parse leading flags up to `--` ---
ENV_NAME="regularizedvi"
REMOTE_HOST=""
PROJECT=""
SSH_ALIAS=""

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
        --)
            shift
            break
            ;;
        *)
            echo "ERROR: unknown flag or missing '--' separator before bash command: $1" >&2
            echo "Usage: bash $0 [--remote crick|sanger] [--env NAME] [--project NAME] -- <bash command...>" >&2
            exit 2
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "ERROR: no bash command given after --" >&2
    echo "Usage: bash $0 [--remote crick|sanger] [--env NAME] [--project NAME] -- <bash command...>" >&2
    exit 2
fi

# --- Cluster detection ---
if [[ "$(uname)" == "Darwin" ]]; then
    CLUSTER="mac"
elif [[ -d /nemo/lab/briscoej && -d /camp/apps/eb/software/Miniconda3 ]]; then
    CLUSTER="crick"
elif [[ -d /software/conda/users/vk7 ]]; then
    CLUSTER="sanger"
else
    echo "ERROR: unrecognized cluster. Add a detection rule to run_conda_bash.sh." >&2; exit 3
fi

# --- Resolve env spec (name on Crick, full path on Sanger, name on Mac) ---
case "$CLUSTER:$ENV_NAME" in
    mac:cell2state)       ENV_SPEC="cell2state_cuda124_torch25" ;;
    mac:regularizedvi)
        echo "ERROR: env 'regularizedvi' not on Mac. Either:" >&2
        echo "  (1) create it locally on Mac, OR" >&2
        echo "  (2) call with --remote (auto-routes to cluster)." >&2; exit 4 ;;
    mac:hashfrag)
        echo "ERROR: env 'hashfrag' not on Mac. Use --remote crick or --remote sanger." >&2; exit 4 ;;
    crick:cell2state)     ENV_SPEC="cell2state_v2026_cuda124_torch25" ;;
    crick:regularizedvi)  ENV_SPEC="regularizedvi" ;;
    sanger:cell2state)    ENV_SPEC="/software/conda/users/vk7/cell2state_v2026_cuda124_torch25" ;;
    sanger:regularizedvi) ENV_SPEC="/software/conda/users/vk7/regularizedvi" ;;
    sanger:hashfrag)      ENV_SPEC="/software/conda/users/vk7/hashfrag_env" ;;
    crick:hashfrag)
        echo "ERROR: env 'hashfrag' not installed on Crick." >&2; exit 4 ;;
    *)
        echo "ERROR: unknown env '$ENV_NAME' on cluster '$CLUSTER'." >&2; exit 4 ;;
esac

# --- SSH connection multiplexing ---
# Reuse a single control socket per script invocation. The user's ~/.ssh/config
# already has ControlMaster for crick-default-shared, but we set explicit
# per-process options to (1) work even without that config, (2) avoid
# socket-name collisions across concurrent invocations, (3) be deterministic
# about cleanup.
SSH_CONTROL_PATH="${TMPDIR:-/tmp}/run_conda_bash_ssh_$$.sock"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$SSH_CONTROL_PATH" -o "ControlPersist=600")

cleanup_ssh_socket() {
    if [[ -S "$SSH_CONTROL_PATH" ]]; then
        ssh "${SSH_OPTS[@]}" -O exit "${SSH_ALIAS:-dummy}" 2>/dev/null || true
    fi
    rm -f "$SSH_CONTROL_PATH"
}
trap cleanup_ssh_socket EXIT

# --- SSH host probes (Mac → cluster) ---
ensure_crick_default() {
    if ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=5 crick-default-shared true 2>/dev/null; then
        SSH_ALIAS=crick-default-shared; return
    fi
    echo "[crick-default-shared unavailable; starting vscode job via login...]" >&2
    ssh login "~/vscode.sh" >&2
    for i in {1..30}; do
        sleep 10
        ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=5 crick-default-shared true 2>/dev/null && {
            SSH_ALIAS=crick-default-shared; return
        }
    done
    echo "ERROR: crick-default-shared never came up. Falling back to login (limited)." >&2
    SSH_ALIAS=login
}

ensure_sanger_head() {
    for host in farm22-head2 farm22-head1 gen22-head2; do
        if ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=5 "$host" true 2>/dev/null; then
            SSH_ALIAS="$host"; return
        fi
    done
    echo "ERROR: no Sanger head node reachable. Configure Teleport tsh first." >&2; exit 5
}

# Auto-detect project from cwd's git toplevel (only needed for remote dispatch).
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

# --- Remote dispatch (Mac + --remote) ---
if [[ "$CLUSTER" == "mac" && -n "${REMOTE_HOST:-}" ]]; then
    detect_project

    if [[ "$REMOTE_HOST" == "crick" ]]; then
        ensure_crick_default
        REMOTE_PROJ="${PROJECTS_CRICK[$PROJECT]}"
    elif [[ "$REMOTE_HOST" == "sanger" ]]; then
        ensure_sanger_head
        REMOTE_PROJ="${PROJECTS_SANGER[$PROJECT]}"
    else
        echo "ERROR: unknown --remote target '$REMOTE_HOST' (use crick or sanger)" >&2; exit 4
    fi

    # Build the remote bash command, quoting each arg with %q.
    # Tilde-prefixed paths (~/..., ~) are left unquoted so the remote bash
    # performs tilde expansion against the cluster's $HOME — used by shared
    # scripts like ~/.claude/claude-shared-skills/scripts/_inspect_notebook.py
    # whose path is identical relative to $HOME on every machine.
    QUOTED_CMD=""
    for a in "$@"; do
        case "$a" in
            "~"|"~/"*) QUOTED_CMD+="$a " ;;
            *)         QUOTED_CMD+="$(printf '%q' "$a") " ;;
        esac
    done

    REMOTE_CMD="cd $(printf '%q' "$REMOTE_PROJ") && bash scripts/helper_scripts/run_conda_bash.sh --env $(printf '%q' "$ENV_NAME") --project $(printf '%q' "$PROJECT") -- $QUOTED_CMD"

    echo "[remote:$REMOTE_HOST project=$PROJECT host=$SSH_ALIAS] running $REMOTE_CMD" >&2
    exec ssh "${SSH_OPTS[@]}" -T "$SSH_ALIAS" "$REMOTE_CMD"
fi

# --- Local execution path (Mac without --remote, or Crick/Sanger directly) ---

# --- Ensure conda + module are available in this shell ---
if [[ "$CLUSTER" == "mac" ]]; then
    set +u
    if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    else
        eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    fi
    set -u
elif [[ "$CLUSTER" == "crick" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "$HOME/.bashrc"
    set -u
else
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
    exit 5
fi

# --- Activate ---
export PYTHONNOUSERSITE=TRUE
if [[ "$CLUSTER" == "crick" ]]; then
    export PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip
    mkdir -p "$PIP_CACHE_DIR" 2>/dev/null || true
fi
conda activate "$ENV_SPEC"

# --- Exec the bash command verbatim ---
exec "$@"
