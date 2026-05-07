#!/bin/bash
# Cluster-aware pytest runner. Local by default; --remote routes to cluster
# (sync + reinstall via pull_reinstall.sh, then ssh + pytest).
#
# Usage:
#   bash scripts/helper_scripts/run_tests.sh [pytest args...]                       # --local default
#   bash scripts/helper_scripts/run_tests.sh --local [pytest args...]
#   bash scripts/helper_scripts/run_tests.sh --remote [crick|sanger] [pytest args...]
#   bash scripts/helper_scripts/run_tests.sh --env NAME [pytest args...]
#   bash scripts/helper_scripts/run_tests.sh --remote crick --allow-dirty [pytest args...]
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse our flags; everything else is forwarded to pytest.
MODE=local
HOST=crick
ENV_NAME=""
ALLOW_DIRTY=""
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) MODE=local; shift ;;
        --remote)
            MODE=remote
            shift
            if [[ ${1:-} == "crick" || ${1:-} == "sanger" ]]; then
                HOST="$1"; shift
            fi
            ;;
        --env)
            shift; ENV_NAME="${1:-}"
            [[ -z "$ENV_NAME" ]] && { echo "ERROR: --env requires a value" >&2; exit 2; }
            shift ;;
        --allow-dirty) ALLOW_DIRTY="--allow-dirty"; shift ;;
        *) PYTEST_ARGS+=("$1"); shift ;;
    esac
done

ENV_FLAG=()
[[ -n "$ENV_NAME" ]] && ENV_FLAG=(--env "$ENV_NAME")

if [[ "$MODE" == "local" ]]; then
    # Existing behaviour: just call run_python_cmd.sh with -m pytest.
    exec bash "$DIR/run_python_cmd.sh" "${ENV_FLAG[@]}" -m pytest "${PYTEST_ARGS[@]}"
fi

# Remote mode: pull_reinstall first, then ssh+pytest on cluster.
echo "[run_tests --remote $HOST] syncing repo + reinstalling..." >&2
bash "$DIR/pull_reinstall.sh" $ALLOW_DIRTY

# SSH connection multiplexing for the pytest run (single ssh call but matches
# pull_reinstall.sh's pattern; ControlPersist keeps the socket alive briefly
# after exec so any pull_reinstall.sh socket can be reused if mtimes align).
SSH_CONTROL_PATH="${TMPDIR:-/tmp}/run_tests_ssh_$$.sock"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$SSH_CONTROL_PATH" -o "ControlPersist=300")
trap 'rm -f "$SSH_CONTROL_PATH"' EXIT

# Determine SSH alias and remote project path.
case "$HOST" in
    crick)
        # Use crick-default-shared (multiplexed) for the test run.
        SSH_ALIAS="crick-default-shared"
        REMOTE_PROJ="/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi"
        ;;
    sanger)
        SSH_ALIAS="farm22-head2"
        REMOTE_PROJ="/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi"
        ;;
    *) echo "ERROR: unknown --remote host '$HOST'" >&2; exit 4 ;;
esac

# Build forwarded args, escaped.
FWD=""
for a in "${PYTEST_ARGS[@]}"; do
    FWD+=" $(printf '%q' "$a")"
done

ENV_FLAG_STR=""
[[ -n "$ENV_NAME" ]] && ENV_FLAG_STR="--env $(printf '%q' "$ENV_NAME")"

echo "[run_tests --remote $HOST] running pytest on $SSH_ALIAS..." >&2
exec ssh "${SSH_OPTS[@]}" -T "$SSH_ALIAS" "cd $REMOTE_PROJ && bash scripts/helper_scripts/run_python_cmd.sh $ENV_FLAG_STR -m pytest$FWD"
