#!/bin/bash
# Auto-activate env + dispatch remote for cluster paths.
# Usage: bash check_imports.sh [--remote crick|sanger] [--env NAME] [--project NAME] FILE [flags]
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

RPC=""
if [[ -n "${RUN_PYTHON_CMD:-}" ]]; then
    RPC="$RUN_PYTHON_CMD"
elif git_root=$(git rev-parse --show-toplevel 2>/dev/null) && [[ -f "$git_root/scripts/helper_scripts/run_python_cmd.sh" ]]; then
    RPC="$git_root/scripts/helper_scripts/run_python_cmd.sh"
elif [[ -f "$HOME/Desktop/my_packages/regularizedvi/scripts/helper_scripts/run_python_cmd.sh" ]]; then
    RPC="$HOME/Desktop/my_packages/regularizedvi/scripts/helper_scripts/run_python_cmd.sh"
elif [[ -f "/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/scripts/helper_scripts/run_python_cmd.sh" ]]; then
    RPC="/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/scripts/helper_scripts/run_python_cmd.sh"
fi
[[ -z "$RPC" ]] && { echo "ERROR: cannot locate run_python_cmd.sh" >&2; exit 1; }

PY_SCRIPT="$DIR/check_imports.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }

exec bash "$RPC" "$PY_SCRIPT" "$@"
