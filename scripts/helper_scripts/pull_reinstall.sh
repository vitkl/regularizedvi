#!/bin/bash
# Daily sync helper for cell2state: laptop <-> Crick git pull/push + reinstall.
#
# Run from the laptop. Bundles the safe order:
#   1. Refuse if laptop tree is dirty (or stash with --allow-dirty).
#   2. git pull --ff-only on laptop.
#   3. git push on laptop.
#   4. ssh login "cd <CRICK_PROJ> && git pull --ff-only".
#   5. cluster reinstall via ssh login "... pip install . --no-deps -q".
#   6. local reinstall via run_python_cmd.sh -m pip install . --no-deps -q.
#
# CLI:
#   bash pull_reinstall.sh [--allow-dirty] [--no-remote] [--no-local]
#
#   --allow-dirty   Stash uncommitted changes (push -u) and restore on EXIT
#                   instead of refusing.
#   --no-remote     Skip the cluster pull + cluster reinstall.
#   --no-local      Skip the laptop reinstall.

set -euo pipefail

# --- Constants ---------------------------------------------------------------
LAPTOP_PROJ=/Users/kleshcv/Desktop/my_packages/regularizedvi
CRICK_PROJ=/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi
CRICK_SSH=login   # login node for git ops; crick-default-shared is compute-only

# --- SSH connection multiplexing --------------------------------------------
# This script makes 3-5 ssh calls in succession (status, pull, reinstall);
# multiplex them onto a single connection so the login node doesn't see
# repeated handshakes.
SSH_CONTROL_PATH="${TMPDIR:-/tmp}/pull_reinstall_ssh_$$.sock"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$SSH_CONTROL_PATH" -o "ControlPersist=300")

cleanup_ssh_socket() {
    if [[ -S "$SSH_CONTROL_PATH" ]]; then
        ssh "${SSH_OPTS[@]}" -O exit "$CRICK_SSH" 2>/dev/null || true
    fi
    rm -f "$SSH_CONTROL_PATH"
}

# --- Flags -------------------------------------------------------------------
ALLOW_DIRTY=0
DO_REMOTE=1
DO_LOCAL=1

while [[ $# -gt 0 ]]; do
    case "${1:-}" in
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --no-remote)   DO_REMOTE=0;   shift ;;
        --no-local)    DO_LOCAL=0;    shift ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "ERROR: unknown flag '$1'" >&2
            echo "Usage: bash $0 [--allow-dirty] [--no-remote] [--no-local]" >&2
            exit 2 ;;
    esac
done

# --- State for trap ----------------------------------------------------------
STEP="init"
STASHED=0
LAPTOP_SHA_BEFORE=""
LAPTOP_SHA_AFTER=""
CRICK_SHA_AFTER=""
LOCAL_REINSTALL_OK="skipped"
REMOTE_REINSTALL_OK="skipped"

cleanup() {
    local rc=$?
    if [[ "$STASHED" == "1" ]]; then
        echo "[trap] restoring stashed changes via 'git stash pop'" >&2
        if ! git -C "$LAPTOP_PROJ" stash pop 2>&1; then
            echo "WARNING: 'git stash pop' did not apply cleanly. Your changes are still" >&2
            echo "         in the stash. Run 'git -C $LAPTOP_PROJ stash list' to inspect," >&2
            echo "         then 'git stash pop' / 'git stash apply' / 'git stash drop'" >&2
            echo "         once you have resolved any conflicts." >&2
        fi
    fi
    if [[ "$rc" -ne 0 ]]; then
        echo "[FAIL] pull_reinstall.sh failed at step: $STEP (exit $rc)" >&2
    fi
    exit "$rc"
}
trap 'cleanup; cleanup_ssh_socket' EXIT

# --- Step 1: laptop git repo + branch ---------------------------------------
STEP="verify-laptop-repo"
cd "$LAPTOP_PROJ"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: $LAPTOP_PROJ is not a git repository." >&2
    exit 10
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    echo "ERROR: laptop repo is not on a named branch (detached HEAD?)." >&2
    exit 11
fi
LAPTOP_SHA_BEFORE="$(git rev-parse HEAD)"
echo "[1/7] laptop branch=$BRANCH  HEAD=$LAPTOP_SHA_BEFORE"

# --- Step 2: dirty-tree check ------------------------------------------------
STEP="check-dirty-tree"
DIRTY="$(git status --porcelain)"
if [[ -n "$DIRTY" ]]; then
    if [[ "$ALLOW_DIRTY" == "1" ]]; then
        echo "[2/7] tree is dirty; --allow-dirty given -> stashing"
        echo "$DIRTY" | sed 's/^/      /'
        git stash push -u -m "pull_reinstall.sh auto-stash"
        STASHED=1
    else
        echo "ERROR: laptop repo has uncommitted changes. Commit or stash first," >&2
        echo "       or rerun with --allow-dirty to auto-stash + restore." >&2
        echo "       Dirty files:" >&2
        echo "$DIRTY" | sed 's/^/         /' >&2
        exit 12
    fi
else
    echo "[2/7] tree is clean"
fi

# --- Step 3: laptop git pull --ff-only --------------------------------------
STEP="laptop-git-pull"
echo "[3/7] git pull --ff-only origin $BRANCH (laptop)"
if ! git pull --ff-only origin "$BRANCH"; then
    echo "ERROR: laptop 'git pull --ff-only origin $BRANCH' failed." >&2
    echo "       Likely the branch has diverged. Resolve manually with:" >&2
    echo "         cd $LAPTOP_PROJ && git fetch origin && git status" >&2
    echo "       and either rebase or merge before rerunning." >&2
    exit 13
fi

# --- Step 4: laptop git push ------------------------------------------------
STEP="laptop-git-push"
echo "[4/7] git push origin $BRANCH (laptop)"
if ! git push origin "$BRANCH"; then
    echo "ERROR: laptop 'git push origin $BRANCH' failed." >&2
    exit 14
fi
LAPTOP_SHA_AFTER="$(git rev-parse HEAD)"

# --- Step 5: cluster git pull -----------------------------------------------
if [[ "$DO_REMOTE" == "1" ]]; then
    STEP="cluster-clean-check"
    echo "[5/7] cluster: verifying clean tree at $CRICK_PROJ"
    REMOTE_DIRTY="$(ssh "${SSH_OPTS[@]}" "$CRICK_SSH" "cd '$CRICK_PROJ' && git status --porcelain" || true)"
    if [[ -n "$REMOTE_DIRTY" ]]; then
        echo "ERROR: cluster repo has divergent state — log in via 'ssh $CRICK_SSH' and resolve." >&2
        echo "       Dirty files on cluster:" >&2
        echo "$REMOTE_DIRTY" | sed 's/^/         /' >&2
        exit 15
    fi

    STEP="cluster-git-pull"
    echo "      cluster: git pull --ff-only"
    if ! ssh "${SSH_OPTS[@]}" "$CRICK_SSH" "cd '$CRICK_PROJ' && git pull --ff-only"; then
        echo "ERROR: cluster 'git pull --ff-only' failed." >&2
        echo "       Resolve manually: ssh $CRICK_SSH; cd $CRICK_PROJ; git status" >&2
        exit 16
    fi
    CRICK_SHA_AFTER="$(ssh "${SSH_OPTS[@]}" "$CRICK_SSH" "cd '$CRICK_PROJ' && git rev-parse HEAD" | tr -d '\r\n')"

    # --- Step 6: cluster reinstall ------------------------------------------
    STEP="cluster-reinstall"
    echo "[6/7] cluster: pip install . --no-deps -q"
    # Direct ssh form: run_python_cmd.sh on the cluster handles env activation.
    # We pipe through bash on the login node; cd into project so '.' resolves
    # to the cluster repo. (--remote crick from the wrapper would attempt to
    # translate the script-arg '.', which is positional for pip, not a script
    # path — so we go direct.)
    if ! ssh "${SSH_OPTS[@]}" "$CRICK_SSH" "cd '$CRICK_PROJ' && bash scripts/helper_scripts/run_python_cmd.sh -m pip install . --no-deps -q"; then
        echo "ERROR: cluster reinstall failed." >&2
        REMOTE_REINSTALL_OK="FAIL"
        exit 17
    fi
    REMOTE_REINSTALL_OK="ok"
else
    echo "[5/7] --no-remote: skipping cluster pull + reinstall"
    echo "[6/7] --no-remote: skipping cluster reinstall"
fi

# --- Step 7: laptop reinstall -----------------------------------------------
if [[ "$DO_LOCAL" == "1" ]]; then
    STEP="laptop-reinstall"
    echo "[7/7] laptop: pip install . --no-deps -q"
    if ! bash "$LAPTOP_PROJ/scripts/helper_scripts/run_python_cmd.sh" -m pip install . --no-deps -q; then
        echo "ERROR: laptop reinstall failed." >&2
        LOCAL_REINSTALL_OK="FAIL"
        exit 18
    fi
    LOCAL_REINSTALL_OK="ok"
else
    echo "[7/7] --no-local: skipping laptop reinstall"
fi

# --- Summary -----------------------------------------------------------------
STEP="summary"
echo
echo "=== pull_reinstall.sh summary ==="
echo "  branch:           $BRANCH"
echo "  laptop HEAD:      $LAPTOP_SHA_BEFORE -> $LAPTOP_SHA_AFTER"
if [[ "$DO_REMOTE" == "1" ]]; then
    echo "  cluster HEAD:     $CRICK_SHA_AFTER"
    if [[ "$LAPTOP_SHA_AFTER" == "$CRICK_SHA_AFTER" ]]; then
        echo "  laptop == cluster: yes"
    else
        echo "  WARNING: laptop and cluster HEADs differ after pull — investigate." >&2
    fi
else
    echo "  cluster HEAD:     (skipped)"
fi
echo "  reinstall local:  $LOCAL_REINSTALL_OK"
echo "  reinstall remote: $REMOTE_REINSTALL_OK"
echo "================================="
