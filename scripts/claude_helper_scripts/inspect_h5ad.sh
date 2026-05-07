#!/bin/bash
# Auto-activate env + dispatch remote for cluster paths.
# Usage: bash inspect_h5ad.sh [--remote crick|sanger] [--env NAME] [--project NAME] FILE [flags]
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

RPC=""
if [[ -n "${RUN_PYTHON_CMD:-}" ]]; then
    RPC="$RUN_PYTHON_CMD"
elif git_root=$(git rev-parse --show-toplevel 2>/dev/null) && [[ -f "$git_root/scripts/helper_scripts/run_python_cmd.sh" ]]; then
    RPC="$git_root/scripts/helper_scripts/run_python_cmd.sh"
elif [[ -f "$HOME/Desktop/my_packages/cell2state/scripts/helper_scripts/run_python_cmd.sh" ]]; then
    RPC="$HOME/Desktop/my_packages/cell2state/scripts/helper_scripts/run_python_cmd.sh"
elif [[ -f "/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state/scripts/helper_scripts/run_python_cmd.sh" ]]; then
    RPC="/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state/scripts/helper_scripts/run_python_cmd.sh"
fi
[[ -z "$RPC" ]] && { echo "ERROR: cannot locate run_python_cmd.sh" >&2; exit 1; }

PY_SCRIPT="$DIR/inspect_h5ad.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }

LIB="$HOME/.claude/claude-shared-skills/scripts/_remote_passthrough.sh"
[[ -f "$LIB" ]] || { echo "ERROR: cannot locate _remote_passthrough.sh at $LIB" >&2; exit 1; }
# shellcheck disable=SC1090
source "$LIB"
parse_remote_args "$@"
exec bash "$RPC" "${RPT_LEADING[@]}" "$PY_SCRIPT" "${RPT_REST[@]}"
