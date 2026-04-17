# Handover: Neighbourhood Correlation — Remaining Fixes

## Context

The neighbourhood correlation metrics module (`src/regularizedvi/plt/_neighbourhood_correlation.py`, 2,661 lines, 32 public functions) has been implemented and tests pass (104/104). Two rounds of `/verify-implementation` revealed 12 remaining issues. This handover documents the fixes needed.

**Source materials** (all in project `.claude/plans/`):
- Main plan: `neighbourhood_correlation_plan.md`
- User feedback: `neighbourhood_correlation_user_feedback.md`
- Sub-plans: `neighbourhood_correlation_subplan_01_*.md` through `_11_*.md`

**Test command**: `bash run_tests.sh tests/test_neighbourhood_correlation.py -x -q`

---

## Critical fixes

### F1: Normalisation order bug (CRITICAL — affects Pearson correctness when n_genes is small)

**Problem**: `compute_marker_correlation` (line 692-707) subsets to markers BEFORE normalisation. Plan requires normalisation on FULL gene matrix. With small marker counts (~180-2000), this changes Pearson values across cells with different total counts.

**Fix**: Compute `total_counts` on full matrix, pass to modified `normalise_counts`.

**Functions to change**:
1. `normalise_counts` (line 60) — add optional `total_counts` parameter
2. `compute_marker_correlation` (line 692) — compute totals on full X first, then subset

**Code changes**:

```python
# normalise_counts (~line 60)
def normalise_counts(X, n_vars: int | None = None, total_counts=None):
    """Normalise counts so per-gene average = 1.

    If ``total_counts`` is provided, it is used directly (allows pre-computing
    on a full gene matrix before subsetting to markers).
    """
    if n_vars is None:
        n_vars = X.shape[1]
    if total_counts is None:
        total_counts = np.asarray(X.sum(axis=1)).flatten()
    else:
        total_counts = np.asarray(total_counts).flatten()
        if total_counts.shape[0] != X.shape[0]:
            raise ValueError(
                f"total_counts length {total_counts.shape[0]} != X.shape[0] {X.shape[0]}"
            )
    scale = np.zeros(len(total_counts), dtype=np.float32)
    mask = total_counts > 0
    scale[mask] = n_vars / total_counts[mask]
    if sp.issparse(X):
        return X.multiply(scale[:, None]).tocsr()
    return X * scale[:, None]

# compute_marker_correlation (~line 695)
marker_idx = adata.var_names.isin(marker_genes)
if marker_idx.sum() == 0:
    raise ValueError("No marker genes found in adata.var_names")

# Compute total counts on FULL gene matrix BEFORE marker subset
X_full = adata.layers[layer] if layer is not None else adata.X
total_counts_full = np.asarray(X_full.sum(axis=1)).flatten()

X = adata[:, marker_idx].layers[layer] if layer is not None else adata[:, marker_idx].X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
X = normalise_counts(X, n_vars=adata.n_vars, total_counts=total_counts_full).astype(np.float32)
```

**Test additions** (`tests/test_neighbourhood_correlation.py`):
- Equivalence: `normalise_counts(X_full)[:, marker_idx]` vs `normalise_counts(X_full[:, marker_idx], n_vars=full_n, total_counts=X_full.sum(1))` must be allclose
- Validation: total_counts length mismatch raises ValueError
- n_total cell with total=0 → row of zeros (no NaN/inf)

**Note**: Does NOT directly fix F7/F8 (those are `_sparse_pearson_row_stats` divide-by-zero in single-cell adata). Add separate guard.

---

### F3: cross_library isolation should restrict to within-dataset (CRITICAL — wrong baseline)

**Problem**: `compute_isolation_norm` (line 1418-1423) for `cross_library` mask uses global `n_total`. The mask requires "different library AND same dataset", so the baseline must restrict to within-dataset cells.

**Correct per-mask formulas** (where `n_x` = cells with same value as cell i, k = degree):

| Mask | P(neighbour qualifies) | P(isolated) |
|------|------------------------|-------------|
| `same_library` | `(n_lib - 1) / (n_total - 1)` | `(1 - p)^k` |
| `between_libraries` | `(n_total - n_lib) / (n_total - 1)` | `(1 - p)^k` |
| **`cross_library`** | **`(n_dataset - n_lib) / (n_total - 1)`** | `(1 - p)^k` |
| `cross_dataset` | `(n_total - n_dataset) / (n_total - 1)` | `(1 - p)^k` |
| `within_{tech}` | `(n_tech - 1) / (n_total - 1)` | `(1 - p)^k` |
| `between_{tech}` | `(n_total - n_tech) / (n_total - 1)` | `(1 - p)^k` |

**Functions to change**:
1. `compute_isolation_norm` (line 1388-1431) — accept `library_key`, `dataset_key`, `technical_key` separately; per-mask dispatch
2. Caller in `summarise_marker_correlation` (search for `compute_isolation_norm(`) — pass all three keys

**Code**: see investigation report (already drafted). Use `np.errstate` guard for `n_total == 1` edge case.

**Tests**: per-mask formula validation against analytical computation on a small fixture.

---

### F2: Isolation baseline divisor (MINOR but plan-divergent)

**Problem**: `compute_analytical_isolation_baseline` (line 1011) uses `(n_same - 1) / (n_total - 1)`; main plan writes `(n_same / n_total)^k_i`.

**Fix decision**: Update plan to match code (code is more correct). Edit `/nfs/users/nfs_v/vk7/.claude/plans/sprightly-prancing-cray.md` lines 220 and 447 to use `((n_same - 1) / (n_total - 1))^k_i`.

This fix happens together with F1 since the formulas above already use `(n_total - 1)` denominator.

---

## Code quality fixes

### F5: Remove unused `threshold_low` parameter from `classify_failure_modes`

**Verdict**: DEAD CODE. Sub-plan 06 defines `threshold_low` only as a default-suggestion (5th percentile) but no leaf condition uses it. The leaf trees use only `th_high`, `th_std`, `above_random`. `threshold_low` IS correctly used in `compute_integration_failure_rate` (different function — sub-plan 08).

**Fix**: Remove `threshold_low` parameter from `classify_failure_modes` (line 1067) and its computation (line 1080-1081 with the noqa).

---

### F6: Remove unused `has_xd` from `compute_tissue_group_integration`

**Verdict**: DEAD CODE. Variable is computed but never used. The plan does not require this precondition check; `finite_mask` on the correlation column already filters cells without cross-dataset neighbours.

**Fix**: Remove `has_xd` and `n_xd_col` lines (line 1861-1862).

---

### F9: Silent NaN from missing covariate values

**Plan-faithful approach**: **Warn and exclude** (Option C). Matches existing patterns in `select_marker_genes` (silent dropna) and `validate_covariate_hierarchy` (raises only for structural violations, not data hygiene).

**Functions to change**:
1. `compute_analytical_isolation_baseline` (line 999-1018) — detect NaN, log warning, return per-cell NaN for affected rows
2. `compute_isolation_norm` (line 1388-1431) — same NaN-aware logic; use `np.nanmean(p_iso)`; exclude NaN-covariate cells from observed numerator/denominator
3. `validate_covariate_hierarchy` (line 416-450) — add NaN-count warning at setup

**Code pattern**:
```python
nan_mask = adata.obs[covariate_key].isna().to_numpy()
n_nan = int(nan_mask.sum())
if n_nan > 0:
    _logger.warning(
        "compute_X: %d/%d cells have NaN '%s'; excluded from calculation.",
        n_nan, adata.n_obs, covariate_key,
    )
valid = ~nan_mask
# compute on valid cells only; return NaN for excluded
```

---

### F10: Misleading default 'ideal' in `_compute_combined_failure_mode` (CRITICAL — silent misclassification)

**Problem**: Line 1308 initialises `worst_label = np.full(n, 'ideal')`. When `np.select` defaults to `WL-unknown`/`XL-unknown`/`XD-unknown`, those leaf names are absent from severity maps, so loops never overwrite them — cells are reported as `failure_mode='ideal'` falsely.

**Fix**: Change default from `'ideal'` to `'unknown'`. Add `'unknown'` to severity maps as the highest severity (so it always overwrites if any unknown leaf is found).

```python
worst_label = np.full(n, 'unknown', dtype=object)
# In severity loops: also handle 'WL-unknown' etc. → set 'unknown'
```

---

### F11: Remove phantom metrics

**Problem**: `_NEIGHBOURHOOD_METRICS` in `_integration_metrics.py` (line 339-365) lists `integration_failure_rate` and `tissue_group_integration` but these are NEVER produced by `summarise_marker_correlation` or `compute_composite_score`.

**User feedback**: "tissue_group_integration doesn't make sense. I don't [know] what integration_failure_rate is supposed to be."

**Fix**:
1. Remove `integration_failure_rate` and `tissue_group_integration` from `_NEIGHBOURHOOD_METRICS` (lines 353-354)
2. Also remove the corresponding prefix entries from `_NEIGHBOURHOOD_PREFIXES` (lines 376-377)
3. Consider deprecating `compute_tissue_group_integration` and `compute_integration_failure_rate` functions if they have no clear use case — but the user used these terms in the plan (sub-plan 08), so:
   - `compute_integration_failure_rate` is well-defined (cells that CAN be integrated but this model fails) and useful for cross-model comparison
   - `compute_tissue_group_integration` is poorly defined per user — consider removing or asking for clarification

---

### F12: NaN poisoning of composite score

**Problem**: `compute_composite_score` arithmetic propagates NaN — any single NaN component makes total NaN, even when other components are valid.

**Fix decision**: Two options:
- **A (current behaviour, conservative)**: Keep NaN poisoning. Document: "If any component is NaN, total is NaN by design — don't compare incomplete scores."
- **B (graceful)**: Use NaN-aware weighted reductions; compute partial scores from available components.

**Recommended**: Option A with explicit docstring note. Plan does not specify graceful degradation here.

**Action**: Add docstring warning to `compute_composite_score`: "Single-component NaN poisons the total. Score components should all be present for meaningful comparison."

---

## Documentation fixes

### Sparse Pearson docstring note

**Add to `_sparse_pearson_row_stats` (line ~540)**:
```
No existing package provides sparse Pearson correlation:
- sklearn.metrics.pairwise.cosine_similarity handles sparse but is cosine, not Pearson
- scipy.stats.pearsonr requires dense arrays
- sklearn.metrics.pairwise_distances(metric='correlation') requires dense

This custom implementation uses the identity ``var = E[X^2] - E[X]^2`` to
avoid centering (which would destroy sparsity). Validated against
``np.corrcoef`` in TestApproachA.test_matches_np_corrcoef and
TestApproachB.test_matches_np_corrcoef.
```

### Test additions for sparse Pearson on realistic sparsity

Current tests use `density=0.3` (70% zeros). scRNA-seq markers typically have >90% zeros. Add:
- `test_matches_np_corrcoef_high_sparsity`: density=0.05 (95% zeros)
- `test_matches_np_corrcoef_extreme_sparsity`: cells with only 1-2 non-zero markers

---

## Evaluation notebook (sub-plan 10)

Sub-plan 10 file exists at `.claude/plans/neighbourhood_correlation_subplan_10_evaluation_notebook.md` but the actual notebook at `docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb` was never created.

**Required**: Follow sub-plan 10 instructions to create the parameterised papermill notebook with:
- Setup + load adata
- Gene selection (curated + data-driven)
- Per-model loop: diagnostics + correlation + random baseline + classification
- Per-model summaries + composite scores
- Cross-model comparison (failure rate, OVL, consensus isolated)
- Visualisations (V1-V14 from main plan)
- TEA-seq case study (scvi_baseline vs regularizedVI)
- Single-dataset bone marrow comparison

Note: depends on F1, F3, F9, F10, F11 being fixed first (otherwise notebook will use buggy metrics).

---

## Suggested execution order

1. **F11** (remove phantom metrics) — trivial, do first
2. **F5, F6** (dead code) — trivial cleanup
3. **F10** (failure mode default) — simple but critical
4. **F9** (NaN handling) — moderate, sets pattern for F1/F3
5. **F2 + F1** (normalisation fix + plan update) — affects test reference values
6. **F3** (isolation cross_library) — depends on F2 numerator/denominator convention
7. **F12 docstring + sparse Pearson docstring + tests** — documentation
8. **Evaluation notebook** — last, after all metrics are correct

After each fix: run `bash run_tests.sh tests/test_neighbourhood_correlation.py -x -q` to verify no regressions.

---

## Files affected

| File | Sections |
|------|----------|
| `src/regularizedvi/plt/_neighbourhood_correlation.py` | normalise_counts (~60), compute_marker_correlation (~692), classify_failure_modes (~1067), _compute_combined_failure_mode (~1308), compute_analytical_isolation_baseline (~999), compute_isolation_norm (~1388), compute_tissue_group_integration (~1812), validate_covariate_hierarchy (~416), _sparse_pearson_row_stats (~540), compute_composite_score (~1534) |
| `src/regularizedvi/plt/_integration_metrics.py` | _NEIGHBOURHOOD_METRICS (339-365), _NEIGHBOURHOOD_PREFIXES (367-378) |
| `tests/test_neighbourhood_correlation.py` | New tests for F1, F3, F9, F10; high-sparsity Pearson tests |
| `/nfs/users/nfs_v/vk7/.claude/plans/sprightly-prancing-cray.md` | Lines 220, 447 (isolation formula) |
| `docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb` | CREATE per sub-plan 10 |

---

## Outstanding user questions

1. **`tissue_group_integration` semantics**: User said "doesn't make sense". Should the function be removed entirely, or is there a useful version (e.g., per-tissue-pair mean cross-dataset correlation)?
2. **`integration_failure_rate` semantics**: User said unclear what it's supposed to be. The implementation matches sub-plan 08 ("fraction of cells that CAN be integrated but this model fails"). Should it remain as-is or be renamed/redefined?
3. **F12 — NaN composite policy**: Conservative (poison) or graceful (partial score)?
