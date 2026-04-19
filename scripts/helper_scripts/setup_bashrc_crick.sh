#!/bin/bash
# Sanity-check Crick shell prerequisites and install a guarded block in
# ~/.bashrc that exports PYTHONNOUSERSITE=TRUE and redirects PIP_CACHE_DIR
# onto the lab NEMO filesystem (home is capped at 5 GB).
#
# Usage:
#   bash scripts/helper_scripts/setup_bashrc_crick.sh
#
# Idempotent — re-running replaces the existing managed block with the
# current version rather than duplicating it.

set -euo pipefail

BASHRC="${HOME}/.bashrc"
BEGIN="# >>> regularizedvi PYTHONNOUSERSITE >>>"
END="# <<< regularizedvi PYTHONNOUSERSITE <<<"

BLOCK=$(cat <<'EOF'
# >>> regularizedvi PYTHONNOUSERSITE >>>
# Managed by scripts/helper_scripts/setup_bashrc_crick.sh — do not edit by hand.
# Ensure no user-site packages leak into conda envs. Also set explicitly
# right before every `conda activate` in submission scripts (belt-and-braces).
export PYTHONNOUSERSITE=TRUE
# Load Miniconda3 module so `conda` is on PATH in every shell (including
# non-interactive sbatch --wrap shells that re-source ~/.bashrc). Silent
# fallthrough on non-Crick hosts.
if [ -d /camp/apps/eb/software/Miniconda3/22.11.1-1 ]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load Miniconda3/22.11.1-1 2>/dev/null || true
fi
# Redirect pip cache off the 5 GB homefs quota onto the lab NEMO filesystem.
# Without this, pip install of torch/scvi-tools blows the home quota and the
# install aborts with `OSError: [Errno 122] Disk quota exceeded`.
if [ -d /nemo/lab/briscoej/home/users/kleshcv ]; then
    export PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip
    mkdir -p "$PIP_CACHE_DIR" 2>/dev/null
fi
# <<< regularizedvi PYTHONNOUSERSITE <<<
EOF
)

echo "=== Crick shell prerequisite check ==="

FAIL=0
if [ -f "$BASHRC" ] && grep -q "conda initialize" "$BASHRC"; then
    echo "[ok] ~/.bashrc contains conda init block"
else
    echo "[FAIL] ~/.bashrc is missing 'conda initialize' block."
    echo "       Run:  module load Miniconda3/22.11.1-1 && conda init bash"
    FAIL=1
fi

# Disable -u around source: /etc/bashrc references unset $BASHRCSOURCED.
# Keep stderr visible so real ~/.bashrc syntax errors surface.
set +u
# shellcheck source=/dev/null
source "$BASHRC" || echo "[warn] sourcing $BASHRC returned non-zero — continuing sanity checks" >&2
set -u

if type -t conda 2>/dev/null | grep -q function; then
    echo "[ok] conda is a shell function in this shell"
else
    echo "[FAIL] conda is not a shell function. Open a fresh shell or fix conda init."
    FAIL=1
fi

if type -t module 2>/dev/null | grep -q function; then
    echo "[ok] module is a shell function in this shell"
else
    echo "[FAIL] module command unavailable. Ensure /etc/profile.d/modules.sh is sourced."
    FAIL=1
fi

if [ -d /camp/apps/eb/software/Miniconda3/22.11.1-1 ]; then
    echo "[ok] Miniconda3/22.11.1-1 module tree exists"
else
    echo "[FAIL] /camp/apps/eb/software/Miniconda3/22.11.1-1 not found — not on Crick?"
    FAIL=1
fi

if [ $FAIL -ne 0 ]; then
    echo ""
    echo "One or more prerequisites missing. Skipping managed-block install."
    exit 1
fi

echo ""
echo "=== Ensure managed env vars are exported in ~/.bashrc ==="
BACKUP="${BASHRC}.regularizedvi-pyusersite.bak"
if [ ! -f "$BACKUP" ]; then
    cp -p "$BASHRC" "$BACKUP"
    echo "[info] Backed up ~/.bashrc -> $BACKUP"
fi

# If an old managed block exists, strip it out first — then append the
# current block. Safe re-run: old block is replaced with the new contents.
if grep -qF "$BEGIN" "$BASHRC"; then
    tmp=$(mktemp)
    awk -v b="$BEGIN" -v e="$END" '
        index($0, b) { skip=1; next }
        skip && index($0, e) { skip=0; next }
        !skip { print }
    ' "$BASHRC" > "$tmp"
    mv "$tmp" "$BASHRC"
    echo "[info] Removed existing managed block"
fi
printf '\n%s\n' "$BLOCK" >> "$BASHRC"
echo "[ok] Appended PYTHONNOUSERSITE + PIP_CACHE_DIR block to ~/.bashrc"

echo ""
echo "All prerequisites met. You can now run:"
echo "    bash scripts/helper_scripts/build_conda_env_crick.sh"
echo ""
echo "Managed env vars now exported by ~/.bashrc:"
echo "  PYTHONNOUSERSITE=TRUE  — no user-site leakage into conda envs"
echo "  PIP_CACHE_DIR=/nemo/lab/briscoej/home/users/kleshcv/.cache/pip  — off the 5 GB homefs"
echo "Also set explicitly in every submission entry point immediately before \`conda activate\`"
echo "(run_python_cmd.sh, build_conda_env_crick.sh, submit_papermill_slurm.sh)."
