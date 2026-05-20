# Handoff: update cell2state for snapatac2 v2.9.0 API

**PATH_TO_PLAN**: /nemo/lab/briscoej/home/users/kleshcv/.claude/plans/applying-regularizedvi-to-mouse-twinkling-token.md
**PATH_TO_CONVERSATION**: /camp/home/kleshcv/.claude/projects/-nemo-lab-briscoej-home-users-kleshcv-my-packages-regularizedvi/ab3c4a8b-441f-411b-909e-e52ea5b5d726.jsonl

## Goal

Make `cell2state.utils.load_atac_snapatac2` work with snapatac2 **v2.9.0** (currently installed in the `cell2state_v2026_cuda124_torch25` conda env). Without this fix, none of the cell2state ATAC entrypoints (`load_atac`, `load_atac_tiles`, `load_atac_peaks`, `concatenate_h5ad(variable_type="atac_tiles"|"peaks")`) can run — they fail at import time with `ModuleNotFoundError: No module named 'snapatac2.pp'`.

This blocks the **mouse gastrulation 4-mod RegularizedMultimodalVI** pipeline (see parent plan) at Step 1: rebuilding the per-sample ATAC fragment caches.

## Root cause

Two breaking changes in snapatac2 between the version that last successfully ran the immune integration pipeline (~March 2026) and v2.9.0:

1. **`snapatac2.pp` is no longer a real submodule** — it is now an attribute alias set in `snapatac2/__init__.py` via:
   ```python
   from . import preprocessing as pp
   ```
   So `snapatac2.pp.import_data` works as attribute access in interactive use (`hasattr(snapatac2, 'pp')` returns `True`), but `from snapatac2.pp import X` fails because Python's `from a.b import c` machinery looks up `sys.modules['a.b']` — and `sys.modules['snapatac2.pp']` is never registered.

2. **`import_data` was renamed to `import_fragments`** — `snapatac2.pp.import_data` no longer exists in v2.9.0. The replacement `snapatac2.pp.import_fragments` has the same signature (verified: `fragment_file, chrom_sizes, *, file=None, sorted_by_barcode=True, chunk_size=2000` etc. — all kwargs cell2state passes are still accepted).

`add_tile_matrix` and `make_peak_matrix` are still named the same in v2.9.0 — only the `from snapatac2.pp import …` syntax breaks them.

Verification (in cell2state env):
```bash
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state -c "
import snapatac2
# attribute access works:
print('pp:', hasattr(snapatac2, 'pp'))
# submodule-style import fails:
import snapatac2.pp
"
# → ModuleNotFoundError: No module named 'snapatac2.pp'
```

snapatac2 v2.9.0 path: `/nemo/lab/briscoej/home/users/kleshcv/conda_environments/.conda/envs/cell2state_v2026_cuda124_torch25/lib/python3.11/site-packages/snapatac2/` (symlinked at `/camp/home/kleshcv/.conda/envs/cell2state_v2026_cuda124_torch25/...`).

## Required changes to cell2state

File: `/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state/cell2state/utils/load_atac_snapatac2.py`

Three call sites — all use the broken `from snapatac2.pp import …` syntax. Each patch must keep **backwards compatibility** with the previous snapatac2 API (≤ v2.8, where `pp` was a real submodule and `import_data` had its old name). The pattern: try the new (v2.9+) import first, fall back to legacy locations on `ImportError`.

### Patch 1 — `load_atac` (around line 32, inside the function body)

Current (broken on v2.9.0):
```python
from snapatac2 import read
from snapatac2.pp import import_data
```

Replacement (works on v2.9+ AND ≤ v2.8):
```python
from snapatac2 import read
# snapatac2 v2.9 renamed import_data -> import_fragments and `pp` became an
# attribute alias rather than a real submodule. Try the new API first, then
# fall back to the legacy submodule path for older installs.
try:
    from snapatac2.preprocessing import import_fragments as import_data
except ImportError:
    try:
        from snapatac2.preprocessing import import_data  # snapatac2 ≤ v2.8 (real submodule, old name)
    except ImportError:
        from snapatac2.pp import import_data  # very old: `pp` was the submodule
```

### Patch 2 — `load_atac_tiles` (around line 100, inside the function body)

Current (broken):
```python
from snapatac2.pp import add_tile_matrix
```

Replacement (`add_tile_matrix` was NOT renamed in v2.9; only the import path needs the v2.9 fallback):
```python
try:
    from snapatac2.preprocessing import add_tile_matrix
except ImportError:
    from snapatac2.pp import add_tile_matrix
```

### Patch 3 — `load_atac_peaks` (around line 137, inside the function body)

Current (broken):
```python
from snapatac2.pp import make_peak_matrix
```

Replacement (`make_peak_matrix` also unchanged in v2.9 — only the import path):
```python
try:
    from snapatac2.preprocessing import make_peak_matrix
except ImportError:
    from snapatac2.pp import make_peak_matrix
```

### Notes
- Body call sites (e.g. `data = import_data(fragment_file, chrom_sizes=..., file=..., sorted_by_barcode=False, chunk_size=2000)`) need NO changes. `import_fragments` has the same kwargs as the old `import_data`, and the local `as import_data` alias keeps the rest of the function code untouched.
- `from snapatac2 import read` (line 31, above the patched block) is unchanged — it still works on v2.9 because `read` is a top-level export listed in `snapatac2.__all__`.
- The fallback chain is deliberately **outer = v2.9-and-newer, inner = legacy** so newer installs short-circuit before the slower fallback paths.
- Consider adding a one-line `pyproject.toml` constraint such as `snapatac2>=2.8` to make the requirement explicit, but DO NOT require `>=2.9` — older installs (e.g. the Sanger env that successfully ran the immune integration in March 2026) must still work.

## Verification after patch

Three checks — first the import resolves cleanly under the current snapatac2, then an end-to-end smoke test that the cache writes correctly, then a guard that the backwards-compat fallback is reachable.

**1. Imports clean** (in the cell2state env, with snapatac2 v2.9.0 installed):
```bash
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state -c "
from cell2state.utils.load_atac_snapatac2 import load_atac, load_atac_tiles, load_atac_peaks
print('Imports OK')
"
```

**2. Backwards-compat fallback is exercised** (simulates an older snapatac2 by hiding the new attribute):
```bash
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state -c "
import snapatac2.preprocessing as _p
# Hide the v2.9+ name so the import statement falls through to the legacy branches
del _p.import_fragments
from cell2state.utils.load_atac_snapatac2 import load_atac
print('Fallback path imports OK')
"
```
This is a smoke test that the try/except chain compiles and that the second branch is reachable. If snapatac2 ever ships a hybrid version that has BOTH `import_fragments` and the old `import_data`, this also confirms the fallback prefers the right one. If both branches are unreachable on a future install, this command surfaces it loudly.

**3. End-to-end smoke test** — single per-sample import on the smallest sample:

```bash
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state -c "
from cell2state.utils.load_atac_snapatac2 import load_atac
adata = load_atac(
    sample='E7.5_rep1',
    fragment_file='/nemo/lab/briscoej/home/users/kleshcv/large_data/gastrulation_multiome_anndata/latest/data/original_with_atac/E7.5_rep1/atac_fragments.tsv.gz',
    path_to_reference='/nemo/lab/briscoej/home/users/kleshcv/large_data/genome_references/refdata-cellranger-arc-mm10-2020-A-2.0.0/',
    overwrite=False,
)
print('Cached:', adata.shape)
"
```

Expected: `atac_fragments.h5ad` lands at `/nemo/.../latest/data/original_with_atac/E7.5_rep1/atac_fragments.h5ad`, ~10 minutes wall clock, ~6k cells × tile-less anndata (tiles get added later by `add_tile_matrix`).

## What worked

- Identified the regression by running `bash ~/.claude/claude-shared-skills/scripts/check_jobs_slurm.sh --compact …` on the 11 failed jobs — the compact output surfaced the exact `ModuleNotFoundError` traceback line per job.
- `inspect.signature(snapatac2.pp.import_fragments)` confirmed the new function exposes the same call shape as the old `import_data` — so cell2state callers won't need argument changes, only the import name.
- `pkgutil.walk_packages` over the installed `snapatac2` was the fastest way to find where `import_fragments` (and what else) lives in v2.9.0.

## What didn't work

- **Monkey-patching `sys.modules['snapatac2.pp'] = snapatac2.preprocessing`** before importing cell2state — this resolves the submodule-lookup issue (problem 1) but does NOT fix the rename (problem 2). The downstream `import_data(...)` call still fails with `ImportError: cannot import name 'import_data' from 'snapatac2.preprocessing'`. Rejected because it doesn't actually work.
- **Pinning to an older snapatac2** — not attempted, but would be a heavier intervention than the 3-line API migration. Only consider if other cell2state callers need API behaviour that changed silently between v2.7/2.8 and v2.9.0 (none surfaced in this conversation).

## Next steps

1. Apply the 3-line patch above to `cell2state/utils/load_atac_snapatac2.py`. Open a branch + PR on the cell2state repo so the change is reviewable; commit message e.g. `fix: migrate snapatac2 imports to v2.9.0 API (import_data → import_fragments)`.
2. Re-run the verification command (single per-sample import) — confirm a `.h5ad` cache lands next to the fragments file with non-empty `adata.shape[0]`.
3. **Resubmit the 11 per-sample import jobs** from the mouse gastrulation pipeline:
   ```bash
   bash scripts/helper_scripts/submit_papermill_slurm.sh \
     --tsv docs/notebooks/model_comparisons/mouse_gastrulation_atac_per_sample_import_jobs.tsv \
     --env-path /nemo/lab/briscoej/home/users/kleshcv/conda_environments/.conda/envs/cell2state_v2026_cuda124_torch25 \
     --gres "" \
     --partition ncpu
   ```
   The previous attempt (job IDs 46682491–46682501) all FAILED with the snapatac2.pp error; the cron `5cae425a` was monitoring them — delete it via CronDelete after resubmitting with new job IDs, or update its embedded job-ID list.
4. After all 11 imports complete (~2–4h each, parallel), submit the 2 concat jobs:
   ```bash
   bash scripts/helper_scripts/submit_papermill_slurm.sh \
     --tsv docs/notebooks/model_comparisons/mouse_gastrulation_atac_loading_jobs.tsv \
     --env-path /nemo/lab/briscoej/home/users/kleshcv/conda_environments/.conda/envs/cell2state_v2026_cuda124_torch25 \
     --gres "" \
     --partition ncpu
   ```
   Then continue from Step 2 of the parent plan (QC exploration → Step 5 training sweep).

## Files in scope

- `/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state/cell2state/utils/load_atac_snapatac2.py` — **the only file to patch**.
- `/nemo/lab/briscoej/home/users/kleshcv/my_packages/cell2state/cell2state/utils/aggregation_v2.py` — uses the patched functions; no changes here (just calls `load_one_sample_tiles` → `load_atac` etc.).
- `/nemo/lab/briscoej/home/users/kleshcv/cell2state_embryo/notebooks/benchmark/regularizedvi/mouse_gastrulation_atac_per_sample_import.ipynb` — papermill template; needs no edits, just the upstream library fix.
- `/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/docs/notebooks/immune_integration/bm_pbmc_atac_loading_out.ipynb` (executed Mar 14 2026) — predates v2.9.0; was the proof of concept the mouse gastrulation rebuild is mirroring.

## Cross-reference

This handoff was extracted from the conversation that built the mouse-gastrulation 4-mod RegularizedMultimodalVI pipeline (see PATH_TO_PLAN). All other plan steps (training notebook updates, TSV expansion to 6 rows, QC exploration notebook) are unblocked and complete — the only thing holding back the training submission is the upstream snapatac2 fix described here.
