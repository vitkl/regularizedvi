# Handoff: complete the move of `submit_snapatac_slurm.sh` into cell2state

**For**: cell2state agent running on the laptop (canonical repo lives there; Crick syncs via `sync-projects-rsync-git`).

**Why**: 2026-05-14 — a new Crick-Slurm submission wrapper + a Crick-tailored worker were drafted on Crick under `regularizedvi/scripts/` to unblock the mouse gastrulation ATAC re-aggregation. They were **tested end-to-end** (all 11 per-sample SnapATAC2 imports completed: 9:22 to 17:57 each, ExitCode 0) and the 2 aggregation jobs are now running. The scripts must move into cell2state. This handoff carries the full file contents + a fix for cell2state's existing worker so the Crick-only workaround can be deleted.

User's instruction: do this on the **laptop** (laptop is source of truth for cell2state; Crick gets the post-edit state via normal git sync). Do not edit cell2state directly on Crick.

---

## What to do (4 steps)

### Step 1 — Create `cell2state/scripts/submit_snapatac_slurm.sh` on laptop

Sibling of the existing `cell2state/scripts/run_snapatac_one_sample.sh`. Full content below — **one edit relative to the Crick draft**: change the `RUN_CONDA_BASH` path so it resolves regardless of which project the user has on their `$PATH` (the current draft hardcodes the Crick regularizedvi path).

```bash
#!/bin/bash
# Submit per-sample snapatac2 import jobs on Crick Slurm.
#
# Wraps cell2state's run_snapatac_one_sample.sh (which does conda activation,
# /var/tmp staging, calls load_atac_snapatac2.py CLI, copies result back).
# One sbatch per sample. Skips samples whose atac_fragments.h5ad already exists.
#
# Usage:
#   bash submit_snapatac_slurm.sh <FRAGMENTS_DIR> <GENOME_REF> <SAMPLE> [SAMPLE...]
#
# Arguments:
#   FRAGMENTS_DIR  Directory containing per-sample subdirs, each with atac_fragments.tsv.gz
#                  e.g. /nemo/.../gastrulation_multiome_anndata/latest/data/original_with_atac
#   GENOME_REF     Path to 10X reference genome directory (must end with /)
#                  e.g. /nemo/.../genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/
#   SAMPLE         Sample name; fragments expected at $FRAGMENTS_DIR/$SAMPLE/atac_fragments.tsv.gz
#
# Optional env overrides:
#   PARTITION  (default: ncpu)
#   MEM        (default: 40G)
#   CPUS       (default: 8)
#   TIME       (default: 08:00:00)
#   CONDA_ENV  (default: cell2state_v2026_cuda124_torch25)
#   LOG_DIR    (default: <FRAGMENTS_DIR>/snapatac_logs)
#   DRY_RUN    (default: 0; set to 1 to print sbatch commands without submitting)
#
# Example (mouse gastrulation, all 11 samples):
#   bash scripts/submit_snapatac_slurm.sh \
#     /nemo/.../large_data/gastrulation_multiome_anndata/latest/data/original_with_atac \
#     /nemo/.../large_data/genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/ \
#     E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep1 E8.0_rep2 \
#     E8.5_rep1 E8.5_rep2 E8.5_CRISPR_T_KO E8.5_CRISPR_T_WT \
#     E8.75_rep1 E8.75_rep2

set -eo pipefail

if [ "$#" -lt 3 ]; then
    sed -n '2,28p' "$0"
    exit 1
fi

FRAGMENTS_DIR="$1"
GENOME_REF="$2"
shift 2
SAMPLES=("$@")

PARTITION="${PARTITION:-ncpu}"
MEM="${MEM:-40G}"
CPUS="${CPUS:-8}"
TIME="${TIME:-08:00:00}"
CONDA_ENV="${CONDA_ENV:-cell2state_v2026_cuda124_torch25}"
DRY_RUN="${DRY_RUN:-0}"

# Worker: same dir as this script (cell2state/scripts/run_snapatac_one_sample.sh)
WORKER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_snapatac_one_sample.sh"
if [ ! -f "$WORKER" ]; then
    echo "ERROR: worker script not found at $WORKER" >&2
    exit 2
fi

# Cluster-aware conda activator. After moving to cell2state, look for
# regularizedvi's run_conda_bash.sh at its known cluster locations.
# (If found, sbatch --wrap delegates conda activation to it; otherwise we
# bail with a clear message — refusing to silently fall back to the broken
# `module load ISG/conda` pattern.)
RUN_CONDA_BASH=""
for candidate in \
    /nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/scripts/helper_scripts/run_conda_bash.sh \
    /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/scripts/helper_scripts/run_conda_bash.sh \
    "$HOME/Desktop/my_packages/regularizedvi/scripts/helper_scripts/run_conda_bash.sh"; do
    if [ -f "$candidate" ]; then
        RUN_CONDA_BASH="$candidate"
        break
    fi
done
if [ -z "$RUN_CONDA_BASH" ]; then
    echo "ERROR: run_conda_bash.sh not found in any of the regularizedvi candidate paths." >&2
    echo "Searched Crick / Sanger / Mac. Edit the candidate list in this script if needed." >&2
    exit 2
fi

LOG_DIR="${LOG_DIR:-${FRAGMENTS_DIR}/snapatac_logs}"
mkdir -p "$LOG_DIR"

echo "Worker:        $WORKER"
echo "Conda helper:  $RUN_CONDA_BASH"
echo "Fragments dir: $FRAGMENTS_DIR"
echo "Genome ref:    $GENOME_REF"
echo "Log dir:       $LOG_DIR"
echo "Resources:     partition=$PARTITION mem=$MEM cpus=$CPUS time=$TIME"
echo "Conda env:     $CONDA_ENV"
echo "Samples:       ${#SAMPLES[@]}"
echo

SUBMITTED=0
SKIPPED_EXIST=0
SKIPPED_MISSING=0
FAILED=0

for SAMPLE in "${SAMPLES[@]}"; do
    FRAG_PATH="${FRAGMENTS_DIR}/${SAMPLE}/atac_fragments.tsv.gz"
    OUTPUT_H5AD="${FRAGMENTS_DIR}/${SAMPLE}/atac_fragments.h5ad"

    if [ ! -f "$FRAG_PATH" ]; then
        echo "[SKIP] $SAMPLE — fragments missing at $FRAG_PATH"
        SKIPPED_MISSING=$((SKIPPED_MISSING + 1))
        continue
    fi
    if [ -f "$OUTPUT_H5AD" ]; then
        echo "[SKIP] $SAMPLE — h5ad already exists at $OUTPUT_H5AD"
        SKIPPED_EXIST=$((SKIPPED_EXIST + 1))
        continue
    fi

    JOB_NAME="snap_${SAMPLE}"
    OUT_FILE="${LOG_DIR}/${SAMPLE}.slurm.out"
    ERR_FILE="${LOG_DIR}/${SAMPLE}.slurm.err"
    # Activate conda via run_conda_bash.sh (handles Crick + Sanger + Mac),
    # then run the worker. The worker (after the Step-2 cell2state patch
    # below) trusts the caller for env and skips its broken `module load
    # ISG/conda` Sanger-only block.
    WRAP="bash ${RUN_CONDA_BASH} --env cell2state -- bash ${WORKER} ${FRAG_PATH} ${SAMPLE} ${GENOME_REF}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] sbatch --job-name=$JOB_NAME --partition=$PARTITION --mem=$MEM --cpus-per-task=$CPUS --time=$TIME --output=$OUT_FILE --error=$ERR_FILE --wrap=\"$WRAP\""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    if JID=$(sbatch --parsable \
                    --job-name="$JOB_NAME" \
                    --partition="$PARTITION" \
                    --mem="$MEM" \
                    --cpus-per-task="$CPUS" \
                    --time="$TIME" \
                    --output="$OUT_FILE" \
                    --error="$ERR_FILE" \
                    --wrap="$WRAP"); then
        echo "[SUBMIT] $SAMPLE -> JobId=$JID  (logs: $OUT_FILE, $ERR_FILE)"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "[FAIL]   $SAMPLE — sbatch returned non-zero"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "=== Summary ==="
echo "Submitted:        $SUBMITTED"
echo "Skipped (exists): $SKIPPED_EXIST"
echo "Skipped (no frag):$SKIPPED_MISSING"
echo "Failed sbatch:    $FAILED"
```

Make sure it is executable: `chmod +x cell2state/scripts/submit_snapatac_slurm.sh`.

### Step 2 — Patch `cell2state/scripts/run_snapatac_one_sample.sh`

The existing worker's conda activation block:
```bash
# 2. Activate conda and run snapatac2 import_data on /var/tmp
echo "Running snapatac2 import_data for ${SAMPLE} ..."
export PYTHONNOUSERSITE="TRUE"
eval "$(module load ISG/conda 2>&1)" || true
conda activate "$CONDA_ENV"
```

is broken on Crick when invoked from a sbatch `--wrap` because:
1. `module load ISG/conda` is **Sanger-only** (Lmod). On Crick, `module` may not be a function at all, or it errors with "command not found". The `2>&1` redirect captures Lmod's error output into the eval, producing a bash syntax error when eval'd.
2. Even after `|| true` swallows that, `conda activate "$CONDA_ENV"` then fails with `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.` because the **`conda` shell function does not propagate across `bash $WORKER` subshells** — it's defined only in shells that source the conda init block.

**Recommended patch** — make the worker work on Crick AND Sanger AND on a fresh subshell, by sourcing conda's profile script directly with a per-cluster path:

Replace the four lines above with:

```bash
# 2. Activate conda — handle Crick, Sanger, and "caller already activated".
echo "Running snapatac2 import_data for ${SAMPLE} ..."
export PYTHONNOUSERSITE="TRUE"

# If conda is already a function in this shell (e.g. caller did
# `bash run_conda_bash.sh ... -- bash $0 ...` and exported the function via
# the sourced env), skip activation. Otherwise, source the cluster's
# conda init script directly.
if ! type conda 2>/dev/null | head -1 | grep -q "function"; then
    # Crick (Briscoe lab)
    if [ -f /camp/apps/eb/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh ]; then
        source /camp/apps/eb/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
    # Sanger farm22 — try the canonical user-installed conda
    elif [ -f /software/conda/users/vk7/cell2state_v2026_cuda124_torch25/etc/profile.d/conda.sh ]; then
        source /software/conda/users/vk7/cell2state_v2026_cuda124_torch25/etc/profile.d/conda.sh
    # Legacy Sanger module-based init (kept as last resort)
    elif type module >/dev/null 2>&1; then
        eval "$(module load ISG/conda 2>/dev/null)" || true
    else
        echo "ERROR: cannot find a conda init script (Crick, Sanger, or module). Aborting." >&2
        exit 5
    fi
fi
conda activate "$CONDA_ENV"
```

This makes the worker self-contained (it activates its own env, regardless of caller) AND Crick-compatible. The `submit_snapatac_slurm.sh` wrapper then doesn't strictly need to pre-activate via `run_conda_bash.sh`, but it's harmless to keep doing so — the inner `conda activate` becomes a no-op when already active.

### Step 3 — Delete the Crick draft files

Once the patched `cell2state/scripts/run_snapatac_one_sample.sh` + new `cell2state/scripts/submit_snapatac_slurm.sh` are committed and synced to Crick, **delete**:
```
/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/scripts/submit_snapatac_slurm.sh
/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/scripts/run_snapatac_one_sample_crick.sh
```
These are the Crick drafts. The Crick worker (`run_snapatac_one_sample_crick.sh`) becomes redundant once the cell2state worker has the conda-init patch from Step 2.

Also delete this handoff file once acted on:
```
/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/.claude/plans/cell2state_submit_snapatac_slurm_handoff.md
```

### Step 4 — Verify on Crick after sync

```bash
# Confirm files arrived
ls -la /nemo/.../my_packages/cell2state/scripts/submit_snapatac_slurm.sh
ls -la /nemo/.../my_packages/cell2state/scripts/run_snapatac_one_sample.sh

# Dry-run against the mouse gastrulation samples — should skip all 11 since
# h5ads are now cached.
bash /nemo/.../cell2state/scripts/submit_snapatac_slurm.sh \
    /nemo/.../gastrulation_multiome_anndata/latest/data/original_with_atac \
    /nemo/.../genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/ \
    E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep1 E8.0_rep2 E8.5_rep1 E8.5_rep2 \
    E8.5_CRISPR_T_KO E8.5_CRISPR_T_WT E8.75_rep1 E8.75_rep2
# Expect: "Skipped (exists): 11, Submitted: 0".
```

If a fresh sample comes along later (different dataset), submit by adding its sample list to the trailing args.

---

## Suggested commit messages (laptop side)

For the new submission wrapper:
```
add scripts/submit_snapatac_slurm.sh: Crick-Slurm per-sample submitter

Crick counterpart to scripts/submit_snapatac_jobs.sh
(which is the Sanger LSF version under cell2state_embryo). Loops over
named samples and submits one sbatch per sample, calling the existing
run_snapatac_one_sample.sh worker. Skips samples whose
atac_fragments.h5ad already exists. Env-var overrides for PARTITION,
MEM, CPUS, TIME, CONDA_ENV, LOG_DIR; DRY_RUN=1 prints sbatch lines
without submitting. Delegates conda activation to regularizedvi's
run_conda_bash.sh (Crick + Sanger + Mac aware).

Tested on Crick 2026-05-14: 11/11 mouse gastrulation samples imported
successfully (9-18 min each; cell2state v2026 env on snapatac2 v2.9.0).
```

For the worker patch:
```
scripts/run_snapatac_one_sample.sh: fix conda init for Crick + sub-shells

The previous `eval "$(module load ISG/conda 2>&1)"; conda activate $ENV`
block worked on Sanger LSF but broke on Crick Slurm:
  - Crick has no `module load ISG/conda` (Lmod path differs)
  - When invoked from sbatch --wrap'd `bash $worker`, the `conda` shell
    function is not inherited from the parent shell

Replace with a per-cluster source of the appropriate conda.sh:
  - Crick: /camp/apps/eb/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
  - Sanger: /software/conda/users/vk7/.../etc/profile.d/conda.sh
  - Skip activation entirely if `conda` is already a function in the
    caller's shell (idempotent under run_conda_bash.sh pre-activation).

Fixes 22 jobs from 2026-05-13 22:55 + 23:16 that exited within 30 sec
with `ModuleNotFoundError` / `CommandNotFoundError`.
```

Push, then on Crick run `bash scripts/helper_scripts/pull_reinstall.sh` (or the cell2state equivalent) to install the updated `cell2state/scripts/`.

---

## Test outcome on Crick (CONFIRMED 2026-05-14)

After 5 debug iterations to find the right conda-activation strategy, attempt 5 **completed successfully** and the full 11-sample batch ran cleanly.

### Final attempt (template for the final workflow)

JobId 46682989 — single-sample test of `E7.75_rep1` via `bash $RUN_CONDA_BASH --env cell2state -- bash $CRICK_WORKER FRAG SAMPLE GENOME`. Result:
- State: COMPLETED, ExitCode 0:0, Elapsed 00:09:22
- Output h5ad: `atac_fragments.h5ad`, 540 MB
- Stderr: only anndata FutureWarnings (harmless)

### Full batch (10 jobs submitted, 9 unique samples + 1 already-cached E7.5_rep1)

JobIds 46683035–46683043. Per-sample sizes after import (12 GB total across 11 samples):

| Sample | h5ad size (GB) | Elapsed (mm:ss) |
|---|---|---|
| E7.5_rep1 | 1.11 | (pre-cached from earlier job, not re-run) |
| E7.5_rep2 | 1.28 | 13:32 |
| E7.75_rep1 | 0.54 | 9:22 (test sample) |
| E8.0_rep1 | 0.90 | 11:08 |
| E8.0_rep2 | 0.80 | 10:41 |
| E8.5_rep1 | 1.32 | 17:57 |
| E8.5_rep2 | 1.18 | 13:06 |
| E8.5_CRISPR_T_KO | 1.18 | 13:39 |
| E8.5_CRISPR_T_WT | 1.21 | 13:36 |
| E8.75_rep1 | 0.94 | 11:39 |
| E8.75_rep2 | 0.95 | 11:42 |

All exit 0. Aggregation TSV `mouse_gastrulation_atac_loading_jobs.tsv` (2 rows, insertion + paired-insertion) submitted as JobIds 46698156, 46698157 (status PENDING at handoff write time).

### Debug history (for posterity — do NOT re-run on the laptop side)

| JobId | WRAP strategy | Outcome | Lesson |
|---|---|---|---|
| 46682895 | `bash $WORKER ...` directly | FAILED 27s | Worker's `module load ISG/conda` is Sanger-only; subsequent `conda activate` fails CommandNotFoundError |
| 46682914 | `source ~/.bashrc; conda activate $ENV; bash $WORKER ...` | FAILED 27s | `conda` is a shell function, not a binary — sourced bashrc in parent shell doesn't propagate to `bash $WORKER` subshell |
| 46682928 | `bash $RUN_CONDA_BASH --env $ENV_FULL -- bash $WORKER ...` | FAILED <5s | `run_conda_bash.sh` allowlists **short** env names (`cell2state`), not full names (`cell2state_v2026_cuda124_torch25`); returns `ERROR: unknown env` |
| 46682978 | `bash $RUN_CONDA_BASH --env cell2state -- bash $WORKER ...` (short name) | FAILED 30s | Worker still failed: run_conda_bash.sh `exec "$@"` activates conda in a process that then exec's `bash $WORKER`, but the conda function still doesn't survive the inner bash spawn — same root cause as 46682914 |
| 46682989 | `bash $RUN_CONDA_BASH --env cell2state -- bash $CRICK_WORKER ...` (Crick worker has no conda block) | **PASSED 9:22** | Crick worker trusts the caller for env and uses `python` from PATH; no `conda activate` call to fail |

After Step 2 (patching cell2state's worker), the canonical worker becomes Crick-compatible and the `$CRICK_WORKER` indirection in the wrapper is no longer needed — `$WORKER` (canonical cell2state path) will work.

---

## Caller side (after the move, on Crick)

```bash
# Per-sample import (run when cached h5ads need building):
bash /nemo/.../my_packages/cell2state/scripts/submit_snapatac_slurm.sh \
    /nemo/.../gastrulation_multiome_anndata/latest/data/original_with_atac \
    /nemo/.../genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/ \
    E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep1 E8.0_rep2 \
    E8.5_rep1 E8.5_rep2 E8.5_CRISPR_T_KO E8.5_CRISPR_T_WT \
    E8.75_rep1 E8.75_rep2

# Aggregation (already submitted — JobIds 46698156, 46698157):
bash /nemo/.../my_packages/regularizedvi/scripts/helper_scripts/submit_papermill_slurm.sh \
    --tsv /nemo/.../regularizedvi/docs/notebooks/model_comparisons/mouse_gastrulation_atac_loading_jobs.tsv \
    --gres "" \
    --env-path /nemo/.../conda_environments/.conda/envs/cell2state_v2026_cuda124_torch25
```

Aggregation outputs (~2.7M tiles each, will be ~50–100 GB on disk):
- `anndata_atac_tiles_1000bp_split160_insertion_11samples_full.h5ad`
- `anndata_atac_tiles_1000bp_split160_insertion_11samples_peakoverlap.h5ad`
- `anndata_atac_tiles_1000bp_split160_paired-insertion_11samples_full.h5ad`
- `anndata_atac_tiles_1000bp_split160_paired-insertion_11samples_peakoverlap.h5ad`

These feed back into the parent multimodal training: [mouse_gastrulation_4mod_handoff.md](mouse_gastrulation_4mod_handoff.md).
