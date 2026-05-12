# Neighbourhood Correlation Metrics — Code Review

**Repository**: regularizedvi
**File under review**: `src/regularizedvi/plt/_neighbourhood_correlation.py` (2,840 lines, 32 public functions)
**Supporting file**: `src/regularizedvi/plt/_integration_metrics.py`
**Plans reviewed**: `neighbourhood_correlation_plan.md` + subplans SP-01 through SP-09 + handover (F1–F12) + user feedback
**Review methodology**: Section-by-section, four evaluation lenses per finding:
- **[PLAN]** — design soundness issues in the main plan itself
- **[PLAN→SUBPLAN]** — where a subplan diverges from or misinterprets the main plan
- **[SUBPLAN→IMPL]** — where the implementation diverges from the subplan
- **[IMPL]** — implementation bugs not covered by subplans

**Severity levels**: Critical · Major · Minor

---

## Status of Known Handover Issues (F1–F12)

| ID | Description | Status |
|----|-------------|--------|
| F1 | Normalisation order bug — total_counts computed on marker subset | ✅ **Fixed** — `normalise_counts` now accepts `total_counts`; `compute_marker_correlation` pre-computes on full matrix |
| F2 | Isolation baseline divisor — plan says `n_same/n_total`, code uses `(n_same-1)/(n_total-1)` | ✅ **Resolved** — code is more correct; plan should be updated |
| F3 | `cross_library` isolation uses global `n_total` instead of within-dataset restriction | ✅ **Fixed** — `compute_isolation_norm` now uses `(n_dataset - n_lib) / (n_total - 1)` for `cross_library` |
| F5 | `threshold_low` dead code in `classify_failure_modes` | ✅ **Fixed** — parameter removed |
| F6 | `has_xd` dead code in `compute_tissue_group_integration` | ⚠️ **Partially resolved** — `compute_tissue_group_integration` removed entirely (see S8-2) |
| F9 | Silent NaN from missing covariate values | ✅ **Fixed** — `validate_covariate_hierarchy` warns; `compute_isolation_norm` excludes NaN cells |
| F10 | `_compute_combined_failure_mode` initializes `worst_label` to `'ideal'` | ✅ **Fixed** — now initializes to `'unknown'` |
| F11 | Phantom metrics `integration_failure_rate` and `tissue_group_integration` in `_NEIGHBOURHOOD_METRICS` | ⚠️ **Partially resolved** — `tissue_group_integration` removed; `integration_failure_rate` still present (see S7-5) |
| F12 | NaN poisoning of composite score | ⚠️ **Diverged** — implementation uses graceful `_nanweighted` degradation instead of recommended conservative poisoning (see S7-4) |

---

## Category 1: Critical Findings

---

### C-1 — [IMPL] — Critical
**Boolean polarity inversion between `flag_consensus_isolated` output and `classify_failure_modes` input**

`flag_consensus_isolated` returns `True` meaning "no model integrates this cell" (consensus isolated = candidate dataset-specific biology). The docstring explicitly says to pass this to `classify_failure_modes(..., model_comparison_result=consensus_flag)`. However, inside `classify_failure_modes`:

```python
mc_bool = np.asarray(mc, dtype=bool)
other_connects = mc_bool & mc_valid      # True → XD-0b (under-integration)
no_model_connects = ~mc_bool & mc_valid  # False → XD-0a (dataset-enriched)
```

So `True` from `flag_consensus_isolated` (= "no model connects") maps to `other_connects` → **XD-0b** (under-integration failure), when it should map to **XD-0a** (dataset-enriched, not a failure). The XD-0a/XD-0b distinction is the entire point of model comparison for isolated cells — this inversion silently swaps the two most important outcomes.

Subplan §8 correction #6 correctly specifies: *"True (consensus isolated, no model connects) → XD-0a; False (some model connects) → XD-0b."* The implementation inverts this.

**Recommended fix**: Either (a) invert the flag in `flag_consensus_isolated` so `True` = "some model connects", or (b) negate the flag at the call site: `model_comparison_result = ~flag_consensus_isolated(...)`, or rename the flag to `some_model_integrates` and return `~all_fail`.

---

### C-2 — [IMPL] — Critical
**`XD-0_isolated_unknown` treated as severity-4 failure despite being unresolvable**

`XD-0_isolated_unknown` is emitted when `model_comparison_result=None` (the default — no model comparison data available). Its severity is set to 4, identical to `XD-0b_under_integration`. This means that in the default case (no model comparison), ALL cells with no cross-dataset neighbours and ideal cross-library will get `failure_mode='XD-0_isolated_unknown'` with severity 4 — the same severity as a confirmed under-integration failure. Since `XD-0_isolated_unknown` is NOT in `_NON_FAILURE_LEAVES`, it is treated as a failure in the combined `failure_mode`. But the plan explicitly states that without model comparison, we cannot distinguish dataset-enriched biology (XD-0a, not a failure) from under-integration (XD-0b, a failure). Treating the unknown case as a failure of severity 4 is misleading.

**Recommended fix**: Add `XD-0_isolated_unknown` to `_NON_FAILURE_LEAVES` (or give it a distinct severity clearly marked as "ambiguous, not a confirmed failure"), and set `failure_mode` to `'ambiguous_isolation'` for these cells rather than treating them as severity-4 failures.

---

### C-3 — [IMPL] — Critical
**V5: `plot_failure_mode_scatter` missing `corr_avg vs corr_mean` panels**

The main plan V5 specifies "hist2d: corr_avg vs corr_mean per mask (shows discrepancy structure)." The subplan correctly translates this: "avg×mean per mask (3 plots)." The implementation's `panel_specs` in `plot_failure_mode_scatter` contains **zero** `corr_avg_{mask}` vs `corr_mean_{mask}` panels. Instead it has discrepancy panels (`corr_avg` vs `corr_discrepancy`), which is a different comparison. The discrepancy is a derived column (`corr_avg - corr_mean`); plotting `corr_avg` vs `corr_discrepancy` does not show the raw avg-vs-mean relationship that reveals whether Approach B and Approach A agree.

**Recommended fix**: Add three panels to `panel_specs`:
```python
("corr_avg_same_library",  "corr_mean_same_library",  "avg vs mean (same library)"),
("corr_avg_cross_library", "corr_mean_cross_library", "avg vs mean (cross library)"),
("corr_avg_cross_dataset", "corr_mean_cross_dataset", "avg vs mean (cross dataset)"),
```

---

### C-4 — [SUBPLAN→IMPL] — Critical
**`compute_random_knn_baseline` drops `technical_covariate_keys` — random masks entirely absent for technical covariates**

The subplan specifies that `construct_neighbour_masks` should be called on the random graph to produce random masks for all active covariate levels. The implementation calls both `list_active_masks` and `construct_neighbour_masks` without passing `technical_covariate_keys`. The function signature does not even accept the parameter, so callers cannot fix this at the call site.

Result: `corr_avg_random_within_{tech}`, `corr_avg_random_between_{tech}`, and `corr_avg_random_cross_technical` columns are never produced. When the decision tree (Dimension 5) compares model correlation to random baseline for technical-covariate masks, there is no random baseline to compare against.

**Recommended fix**: Add `technical_covariate_keys: list[str] | None = None` to `compute_random_knn_baseline`'s signature. Pass it to both `list_active_masks(...)` and `construct_neighbour_masks(...)` inside the function.

---

### C-5 — [PLAN→SUBPLAN] — Critical
**Subplan correction #1 gives wrong isolation formula for `cross_library` mask**

The subplan's "Review Correction #1" states the expected random isolation probability for the `cross_library` mask is `P = (n_same_library_i / (n_total - 1))^k_i` — i.e., the probability that all k random picks land in the same library. This is the probability of being isolated from all other libraries, not from cross-library-same-dataset neighbours specifically. The main plan correctly states `p_match = (n_dataset - n_lib) / (n_total - 1)`, so `P(isolated) = (1 - p_match)^k`. These are numerically different formulas.

For a library with 1000 cells in a dataset of 5000 out of 50000 total, the subplan formula gives `(1000/49999)^k` while the correct formula gives `(1 - 4000/49999)^k = (45999/49999)^k` — orders of magnitude different.

**Impact**: The implementation correctly follows the main plan formula, not the subplan. The subplan correction #1 is wrong and could mislead future implementers.

**Recommended fix**: Correct subplan correction #1 to use `(1 - p_match)^k` with `p_match = (n_dataset - n_lib) / (n_total - 1)`, consistent with the main plan and implementation.

---

## Category 2: Major Findings

---

### M-1 — [PLAN] — Major
**`between_libraries` absent from the Masks table but required by Graceful Degradation**

The plan's "Masks (no abbreviations)" table lists five masks: `same_library`, `cross_library`, `cross_dataset`, `within_{technical_name}`, `between_{technical_name}`. The `between_libraries` mask is not in this table. Yet the "Graceful degradation" section explicitly states: *"When dataset not provided, only `same_library` and `between_libraries` available."* This internal contradiction means a reader following only the Masks table would not know `between_libraries` exists or should be implemented.

**Recommended fix**: Add `between_libraries` to the Masks table with definition: *"Different library, regardless of dataset (aggregate cross-library view; always computed)."*

---

### M-2 — [PLAN] — Major
**Decision tree leaf count inconsistency: plan says 21, subplan corrects to 25, implementation has 26**

The plan's architecture section states "producing 21 leaves." The subplan corrects this to 25 (WL: 6, XL: 7, XD: 12). The implementation has 26 leaves — it adds `XD-0_isolated_unknown` (emitted when `model_comparison_result=None`), which does not appear anywhere in the main plan. The plan was never updated to reflect the subplan's correction or the implementation's addition.

**Recommended fix**: Update the main plan to state 26 leaves and document `XD-0_isolated_unknown` as the leaf emitted when model comparison data is unavailable.

---

### M-3 — [PLAN] — Major
**XD-0d condition "Cross-library FAILED" is never precisely defined**

The plan's XD tree for the "no cross-dataset neighbours" branch shows: *"Cross-library FAILED → LEAF XD-0d: Compounded failure."* But "cross-library FAILED" is never defined. The implementation uses `xl_was_failure = ~isin(["XL-0a_under_integration", "XL-1_ideal", "XL-2_partial"])`, which means XL-3, XL-4, XL-5, XL-0b, and `XL-unknown` all map to XD-0d. Critically, `XL-unknown` cells (cells where no XL condition fired, e.g., due to NaN correlation) also fall into `xl_was_failure=True` and get `XD-0d_compounded` — a silent misclassification of cells with missing data.

**Recommended fix**: In the plan, explicitly enumerate which XL leaves constitute "failed" for the XD-0d condition. In the implementation, exclude `XL-unknown` from `xl_was_failure` and route those cells to a separate leaf or propagate the unknown status.

---

### M-4 — [PLAN] — Major
**Specificity formula axis direction is wrong for the `(n_genes × n_labels)` matrix orientation**

The main plan specifies `mean_per_gene_per_label / mean_per_gene_per_label.sum(axis=0, keepdims=True)`. On a `(n_genes, n_labels)` matrix, `axis=0` sums **across rows (genes)** per column (label), producing a `(1, n_labels)` denominator. This normalises column-wise (per label), making each label's gene distribution sum to 1 — i.e., "what fraction of this label's total expression is gene X?" That is **not** per-gene specificity. Per-gene specificity requires `axis=1` (sum across labels per gene), so each gene's row sums to 1. The user feedback wrote `sum(0, keep_dims=True)` apparently assuming a transposed `(n_labels, n_genes)` orientation; the plan copied the formula verbatim without correcting for the actual matrix orientation.

The subplan already has the correct formula; the plan itself is wrong.

**Recommended fix**: Update the main plan formula to `(label_averages.T / label_averages.sum(axis=1)).T` with a worked example.

---

### M-5 — [IMPL] — Major
**`top_n_per_label` not applied to `subtype_markers`, making it potentially larger than `cell_type_markers`**

In `_run_specificity_filter_per_level`, `top_n_per_label` is applied to `cell_type_markers` but **not** to `subtype_markers`. With `top_n_per_label=200`, `cell_type_markers` is capped at 200 genes per label while `subtype_markers` retains all genes passing the 0.3 threshold — potentially 400+ genes. This inverts the intended hierarchy: `subtype_markers` is supposed to be the finest-resolution, most selective set, but ends up larger than `cell_type_markers`. The decision tree (Dimension 6) uses these sets to distinguish "broadly correct lineage but wrong subtype" from "completely wrong cell type" — if `subtype_markers ⊄ cell_type_markers`, the comparison is meaningless.

**Recommended fix**: Apply `top_n_per_label` to `subtype_markers` using the same per-column top-N logic. Alternatively, define `subtype_markers` explicitly as `cell_type_markers ∩ {genes passing subtype_specificity_threshold}` to guarantee the subset relationship.

---

### M-6 — [IMPL] — Major
**`top_n_per_label: int = 500` added by implementation with no plan/subplan authorisation**

The main plan's `select_marker_genes` signature has no `top_n_per_label` parameter. The subplan's signature also has no `top_n_per_label`. The implementation adds `top_n_per_label: int | None = 500` with a default of 500, silently capping gene selection to the top 500 genes per label column per dataset. With 5 label columns and 7 datasets, this can dramatically reduce the gene set without any indication in the plan that such capping was intended. The plan's intent was that **all** genes passing both `mean_threshold` and `specificity_threshold` should be selected.

Note: M-5 and M-6 interact — `top_n_per_label=500` (unauthorised default) is applied to `cell_type_markers` but not `subtype_markers`, which can invert the intended gene-set hierarchy and corrupt the Dimension 6 decision tree comparisons.

**Recommended fix**: Either (a) remove `top_n_per_label` and rely solely on `mean_threshold` + `specificity_threshold` as the plan specifies, or (b) change the default to `None` (disabled) and add it to the subplan with explicit justification. If retained, it must also be applied consistently to `subtype_markers` (see M-5).

---

### M-7 — [IMPL] — Major
**`nan_to_num(..., nan=0.0)` in `compute_random_knn_baseline` biases the random baseline for isolated cells**

When a cell has no neighbours in a given random mask, `_approach_B_per_mask` returns NaN. The implementation converts these NaNs to 0.0 before accumulating:

```python
corr_mask = _approach_B_per_mask(X, random_masks[mask_name], mean_x, std_x)
corr_mask = np.nan_to_num(corr_mask, nan=0.0)
accumulators[f"corr_avg_random_{mask_name}"] += corr_mask
```

The accumulator is then divided by `n_random_graphs` (not by the number of graphs where the cell actually had neighbours). Result: cells that are isolated in the random graph contribute 0.0 to the average, pulling the random baseline toward zero. This makes the random baseline artificially low for cells in small covariate groups, distorting the normalised isolation metric (`isolation_norm`). The plan's conceptual model is explicit: "zero neighbours → all weights zero → undefined (NaN as computational consequence)."

**Recommended fix**: Track per-cell counts of graphs where the cell had ≥1 neighbour in each mask. Use `np.nansum` for accumulation and divide by the per-cell count. Return NaN for cells that had zero neighbours in all random graphs for a given mask.

---

### M-8 — [IMPL] — Major
**CSR index array in `compute_random_knn_baseline` built with a Python `for` loop — defeats the bulk-sampling optimisation**

The subplan's correction #4 specifies bulk vectorised sampling to avoid per-cell Python loops. The implementation correctly samples in bulk and rejects self-hits in bulk. However, the CSR `indices` array is built with a Python loop:

```python
indices = np.empty(total_nnz, dtype=np.int64)
for i in range(n_cells):
    start = indptr[i]
    end = indptr[i + 1]
    indices[start:end] = samples[i, : degree_per_cell[i]]
```

For n_cells = 100,000+ this loop is slow (seconds to minutes in Python). The subplan's correction #4 shows `np.concatenate([samples[i, :degree_per_cell[i]] for i in range(n_cells)])` — also a Python loop and equally slow. Neither achieves the vectorised construction the correction promises.

**Recommended fix**: Use a flat boolean mask over the `(n_cells, max_k)` samples array:
```python
col_idx = np.arange(max_k)[None, :]
valid_mask = col_idx < degree_per_cell[:, None]  # (n_cells, max_k)
indices = samples[valid_mask]  # flat, length total_nnz, no Python loop
```

---

### M-9 — [IMPL] — Major
**`compute_neighbourhood_diagnostics` unconditionally adds `between_libraries` penetration — not in plan or subplan, and semantically misleading**

The main plan specifies penetration for "cross-library / cross-dataset / cross-technical" masks. The subplan's correction #6 lists explicit return keys: `penetration_cross_library`, `penetration_cross_dataset`, `penetration_between_{tech_name}`. Neither includes `penetration_between_libraries`.

The implementation unconditionally adds `between_libraries` to `penetration_masks` regardless of whether `dataset_key` is provided. `between_libraries` is a superset of `cross_library` — it includes neighbours from different libraries in *different* datasets too. Including it silently produces a metric that conflates cross-library-within-dataset with cross-dataset neighbours, which is misleading. The plan explicitly distinguishes these two masks.

**Recommended fix**: Remove the unconditional `penetration_masks["between_libraries"] = library_key` line. If `between_libraries` penetration is genuinely useful, add it to the plan explicitly.

---

### M-10 — [SUBPLAN→IMPL] — Major
**`_stratified_summary_single` missing 50th percentile (p50) column**

Both the main plan and subplan explicitly require percentiles `(10th, 25th, 50th, 75th, 90th)`. The implementation computes `np.nanpercentile(vals, [10, 25, 75, 90])` — the 50th percentile is missing. The median is stored separately as `{metric}_median`, but the plan/subplan specify a `p50` column as part of the percentile set.

**Recommended fix**: Change the `nanpercentile` call to `np.nanpercentile(vals, [10, 25, 50, 75, 90])` and add `row[f"{metric}_p50"] = float(pcts[2])`, shifting the indices for p75 and p90.

---

### M-11 — [IMPL] — Major
**`compute_composite_score` uses graceful NaN degradation, contradicting the plan and handover F12 recommendation**

The plan specifies a fixed-weight formula: `batch_correction = 0.5 * library_integration + 0.5 * dataset_integration`. When `has_dataset=True` but `dataset_integration` is NaN (e.g., all cells isolated from cross-dataset neighbours), the plan formula yields NaN for `batch_correction` and `total`. The implementation uses `_nanweighted([lib, ds], [0.5, 0.5])`, which silently re-normalises to `batch_correction = library_integration` when `dataset_integration` is NaN. Handover F12 explicitly recommends Option A (conservative/poison) with a docstring note, stating "Plan does not specify graceful degradation here." The implementation chose Option B (graceful) without documenting the divergence.

**Impact**: A model with `has_dataset=True` but zero cross-dataset integration (all cells isolated) will receive a `batch_correction` equal to its `library_integration` alone — the same score as a model that genuinely only has library-level data. This inflates scores for models that fail cross-dataset integration entirely.

**Recommended fix**: Either (a) implement NaN poisoning as recommended in F12 and document it, or (b) explicitly document the graceful degradation choice and add a warning log when `dataset_integration` is NaN but `has_dataset=True`.

---

### M-12 — [IMPL] — Major (F11 not fully fixed)
**`integration_failure_rate` remains as phantom metric in `_NEIGHBOURHOOD_METRICS`**

Handover F11 identifies `integration_failure_rate` as a phantom metric in `_integration_metrics.py` that is never produced by `summarise_marker_correlation` or `compute_composite_score`. The fix was to remove it from `_NEIGHBOURHOOD_METRICS`. The current code still contains `"integration_failure_rate"` in `_NEIGHBOURHOOD_METRICS` (line 353) and `"integration_failure_"` in `_NEIGHBOURHOOD_PREFIXES` (line 369). The function `compute_integration_failure_rate` exists in `_neighbourhood_correlation.py` but is a cross-model function that returns a scalar per model — it is never added to the per-model headline `pd.Series` returned by `summarise_marker_correlation`. If a caller passes a headline DataFrame to `plot_integration_heatmap`, the heatmap will attempt to color-classify a column named `integration_failure_rate` as a neighbourhood metric, but that column will never exist in the data.

**Recommended fix**: Remove `"integration_failure_rate"` from `_NEIGHBOURHOOD_METRICS` and `"integration_failure_"` from `_NEIGHBOURHOOD_PREFIXES` in `_integration_metrics.py`. The function `compute_integration_failure_rate` can remain as a utility for cross-model comparison, but it should not be listed as a per-model headline metric.

---

### M-13 — [IMPL] — Major
**H11/H12 OVL compares distributions over different cell subsets**

In `summarise_marker_correlation`, H3 (`corr_cross_library`) is computed on `xl_filtered = xl_vals[mask_xl]` (cells with `n_neighbours_cross_library >= min_neighbours`). But H11 (`distrib_overlap_library`) is computed using `xl_vals` (the full unfiltered array). Cells with zero cross-library neighbours have `NaN` in `xl_vals`, which `compute_distribution_overlap` drops. This means H11 is effectively computed on the same subset as H3 (NaN-dropped = zero-neighbour cells excluded). However, `sl_vals` for the same-library side is the full unfiltered array (including cells with zero same-library neighbours, which also produce NaN). The asymmetry means the two distributions being compared in OVL are filtered on different criteria: `sl_vals` drops cells with no same-library neighbours, while `xl_vals` drops cells with no cross-library neighbours. These are different cell subsets, so the OVL is comparing distributions over non-overlapping cell populations.

**Recommended fix**: Restrict both distributions to cells that have both same-library AND cross-library neighbours for a fair comparison.

---

### M-14 — [IMPL] — Major
**`compute_analytical_isolation_baseline` API cannot compute the `cross_library` baseline correctly**

The function takes a single `covariate_key` and computes `((n_same - 1) / (n_total - 1))^k_i`. This is correct for `cross_dataset` (where `n_same = n_dataset`) and `within_{tech}` isolation. However, the `cross_library` baseline requires `(n_dataset - n_lib) / (n_total - 1)` as the numerator (cells in same dataset but different library), which requires both `library_key` and `dataset_key`. The single-key API makes it structurally impossible to compute the `cross_library` baseline correctly. This is related to handover F3 but is a distinct structural observation: F3 identifies the wrong formula in `compute_isolation_norm`; this finding identifies that `compute_analytical_isolation_baseline` cannot be fixed for `cross_library` without an API change.

**Recommended fix**: Add a `mask_name` parameter (or create a dispatcher) that selects the correct numerator formula per mask type. For `cross_library`, accept both `library_key` and `dataset_key` and compute `n_dataset - n_lib` as the numerator.

---

### M-15 — [IMPL] — Major
**`assemble_cross_model_metrics` silently drops models with no `adata.uns` entry**

The function iterates over `model_names` and calls `adata.uns[model_name]` without a `.get()` guard. If a model name is not present in `adata.uns` (e.g., due to a typo or a model that failed to run), the function raises a `KeyError` rather than emitting a warning and continuing. In a multi-model comparison workflow, one missing model should not abort the entire assembly.

**Recommended fix**: Replace `adata.uns[model_name]` with `adata.uns.get(model_name)` and emit a `_logger.warning` for missing models, then `continue`.

---

### M-16 — [PLAN→SUBPLAN] — Major
**Subplan adds `n_neighbours_within_technical_cross_dataset` column not in main plan and never implemented**

Subplan correction #8 states: *"Output column for `n_neighbours_within_technical_cross_dataset`: add this when both technical and dataset keys are provided — sub-plan 08 needs it for H14."* This column does not appear in the main plan's "Per-Cell Output Columns" table, and the mask `within_technical_cross_dataset` is not defined anywhere in the main plan's mask definitions. The implementation does not produce this column (it was never implemented), so there is no code bug — but the subplan added a requirement not in the main plan.

**Recommended fix**: Remove correction #8 from the subplan, or if this column is genuinely needed for H14, add it to the main plan's output table with a clear definition of what the mask means.

---

## Category 3: Minor Findings

---

### m-1 — [PLAN] — Minor
**`"Sets per-gene average to 1"` is mathematically imprecise**

The main plan states: *"Formula: `normalised = count * (n_vars / total_count)`. Sets per-gene average to 1."* This is wrong. The formula sets the **per-cell total** to `n_vars`. The per-gene average across cells equals 1 only if all cells have identical total counts — which is never true in scRNA-seq.

**Recommended fix**: Correct the plan description to: *"Sets per-cell total to `n_vars` (depth normalisation). Per-gene average across cells ≈ 1 when cell depths are similar."*

---

### m-2 — [PLAN] — Minor
**`threshold_high` = 25th percentile is semantically confusing naming**

The plan and subplan specify `threshold_high = np.nanpercentile(corr_avg_same_library, 25)`. This means 75% of cells will have `corr_avg_same_library >= threshold_high` and be classified as "HIGH." The threshold is called `threshold_high` but it is actually a very permissive lower bound — the bottom 25% are "LOW", the top 75% are "HIGH."

**Recommended fix**: Rename to `threshold_min_acceptable` or `threshold_low_quartile` to clarify its meaning.

---

### m-3 — [PLAN] — Minor
**`integration_failure_rate` denominator is ambiguous**

The plan Step 3 says *"fraction of cells where best_corr > threshold_high AND model fails."* This is ambiguous: is the denominator all cells, or only achievable cells (those where `best_corr > threshold_high`)? The implementation uses total cell count as denominator. For a dataset where only 30% of cells are achievable, a model failing all achievable cells would score 0.30, not 1.0. The metric name "integration failure rate" implies a rate over the relevant population (achievable cells), not over all cells.

**Recommended fix**: Either rename to `achievable_failure_fraction` (fraction of all cells) or change denominator to `(best > threshold_high).sum()` and keep the name `integration_failure_rate` (fraction of achievable cells that fail). Document the choice explicitly.

---

### m-4 — [PLAN] — Minor
**Plan does not specify key-existence check in `validate_covariate_hierarchy`**

The plan specifies that `validate_covariate_hierarchy` should raise `ValueError` for hierarchy violations and warn for NaN values, but says nothing about missing keys. If `library_key` or `dataset_key` is not a column in `adata.obs`, the current code raises an uninformative `KeyError` from pandas rather than a clear `ValueError`.

**Recommended fix**: Add to the plan an explicit pre-check: *"Raise `ValueError` if `library_key` or `dataset_key` is not in `adata.obs.columns`."*

---

### m-5 — [PLAN] — Minor
**H2 (`corr_consistency`, lower=better) excluded from composite score without justification**

The plan defines H2 `corr_consistency` as "median of corr_std_same_library, lower = better" — a measure of within-library neighbourhood homogeneity. The composite score formula uses only H1 for `bio_conservation` and does not incorporate H2. A model that achieves high H1 but high H2 (inconsistent within-library correlations, suggesting merged cell types) would receive the same `bio_conservation` score as a model with high H1 and low H2 (clean within-library correlations).

**Recommended fix**: Either (a) incorporate H2 into `bio_conservation` as `bio_conservation = H1 * (1 - clamp(H2, 0, 1))` or similar, or (b) explicitly document in the plan why H2 is a diagnostic-only metric excluded from the composite.

---

### m-6 — [PLAN] — Minor
**Concatenating label columns before computing specificity dilutes per-label specificity scores — undocumented limitation**

The main plan specifies concatenating averages across all label columns before computing specificity. When label columns are concatenated, a gene highly specific to one label in `harmonized_annotation` has its row sum inflated by the additional columns from `level_2`/`level_3`. This reduces its specificity score below what it would be if computed on `harmonized_annotation` alone. The `specificity_threshold=0.1` therefore has a different effective meaning depending on how many label columns are provided — adding more label columns inadvertently tightens the filter. This is not documented in the plan.

**Recommended fix**: Document in the plan and docstring that the concatenated specificity threshold is relative to the total number of label columns provided. Consider recommending that users adjust `specificity_threshold` proportionally.

---

### m-7 — [PLAN→SUBPLAN] — Minor
**Subplan silently overrides "Do NOT reimplement `compute_cluster_averages`" without flagging the contradiction**

The main plan explicitly states `"Copy compute_cluster_averages from cell2location — Do NOT reimplement"`. The subplan introduces a new private helper `_cluster_averages_from_matrix(X_norm, labels, var_names)` that reimplements the per-cluster mean logic using sparse one-hot matrix multiplication. The stated reason is valid (AnnData view materialisation), but the subplan does not flag this as a contradiction of the main plan's explicit instruction.

**Recommended fix**: Add a note to the subplan (and ideally the main plan) explaining that `_cluster_averages_from_matrix` is the preferred path for `select_marker_genes` because `compute_cluster_averages` requires writing to `.layers`, which materialises AnnData views.

---

### m-8 — [PLAN→SUBPLAN] — Minor
**Return dict key `"union"` diverges from main plan's `"all_markers"` name**

The main plan's "Multiple gene groups" table names the primary gene set `all_markers`. The subplan's return dict uses `"union"` as the key for this set. The implementation follows the subplan and returns `"union"`. Any caller that reads the main plan and writes `result["all_markers"]` will get a `KeyError`.

**Recommended fix**: Either rename the key to `"all_markers"` in both subplan and implementation to match the main plan's terminology, or add `"all_markers"` as an alias key alongside `"union"`.

---

### m-9 — [PLAN→SUBPLAN] — Minor
**Subplan has internal contradiction about `avg_profiles` being dense vs sparse**

The subplan contains two directly contradictory statements about `avg_profiles` in Approach B. Correction #3 states: *"Note: `avg_profiles` is dense (unavoidable — weighted mean fills zeros) but the cross-term uses `X.multiply(dense)` which stays sparse."* But the function docstring section states: *"avg_profiles STAYS SPARSE (`sparse @ sparse = sparse`; per-row scaling preserves sparsity)."* The implementation correctly follows the sparse path (no `.todense()` call), but the subplan's correction #3 is wrong and could mislead future implementers.

**Recommended fix**: Remove the incorrect statement from correction #3. The correct statement is in the function docstring: `avg_profiles` is sparse because `sparse @ sparse = sparse` and element-wise scaling preserves sparsity.

---

### m-10 — [PLAN→SUBPLAN] — Minor
**Subplan `compute_random_knn_baseline` docstring retains the approximate (non-self-excluded) isolation formula**

The subplan's "Review Corrections" section (item 3) correctly states that the self-excluded form `((n_same - 1) / (n_total - 1))^k_i` should be used. The `compute_analytical_isolation_baseline` docstring in the subplan correctly uses this form. But the **`compute_random_knn_baseline` docstring** in the same subplan still shows the old approximate formula: `P(isolated_cross_dataset | k_i) = (n_same_dataset / n_total)^k_i`. The subplan is internally inconsistent.

**Recommended fix**: Update the `compute_random_knn_baseline` docstring in the subplan to use `((n_same_dataset - 1) / (n_total - 1))^k_i`.

---

### m-11 — [IMPL] — Minor
**`_approach_B_per_mask` applies a redundant/inconsistent zero-variance guard**

`_sparse_pearson_row_stats` clips variance at `1e-12` before sqrt, so `std_x` is never below `~3.16e-7`. Then `_approach_B_per_mask` applies a second guard `zero_var_mask = (std_x < 1e-6) | (std_a < 1e-6)` and sets those to NaN. Meanwhile, `_approach_A_per_mask` uses `std_i > 1e-6` as its valid-pair guard. The two approaches are inconsistent in their zero-variance handling: pairs where `3.16e-7 < std < 1e-6` are treated as NaN in Approach A but would be computed (then NaN'd by the guard) in Approach B.

**Recommended fix**: Standardise the zero-variance threshold across both approaches and document the choice explicitly.

---

### m-12 — [IMPL] — Minor
**`corr_norm_by_library_same_library` is trivially 1.0 by construction**

The plan specifies `corr_norm_by_library_{mask} = corr_avg_{mask} / corr_avg_same_library` for all masks including `same_library` itself. When `mask = same_library`, this is `corr_avg_same_library / corr_avg_same_library = 1.0` for every cell (except NaN). The plan even notes this: *"`corr_norm_by_library_same_library` ≈ 1.0 by construction (trivial)"* in the subplan test cases. This column wastes storage and could confuse users.

**Recommended fix**: Either skip computing `corr_norm_by_library_same_library` (it is always 1.0), or document it explicitly as a sanity-check column.

---

### m-13 — [IMPL] — Minor
**No warning emitted when curated genes from CSV fail to map to `adata.var_names`**

When `curated_marker_csv` is provided, the implementation silently drops any gene symbol not found in `symbol_to_var`. If 42 of 192 genes fail to map (e.g., due to symbol version mismatches or a wrong `symbol_col`), they are silently discarded. The only indication is the summary print `"Curated markers (in adata): 150"` — which requires the user to know the expected count. The main plan says `"Check marker gene CSV genes present in adata.var"` — implying a check, not a silent drop.

**Recommended fix**: After computing `mapped`, emit a `_logger.warning` listing the count and first 20 unmatched gene symbols when any are missing.

---

### m-14 — [IMPL] — Minor
**`compute_random_knn_baseline` recomputes row statistics already available from `compute_marker_correlation`; no API guard against wrong normalisation**

The function accepts `X_normalised_markers` as a parameter (correct — caller passes it in). However, it recomputes `mean_x, std_x = _sparse_pearson_row_stats(X)` internally, even though `compute_marker_correlation` already computed these. More importantly, there is no API guard ensuring the caller passes the correctly normalised matrix (normalised on full gene count before marker subsetting, per F1 fix). A caller could accidentally pass a differently-normalised matrix and get silently wrong results.

**Recommended fix**: Document in `compute_random_knn_baseline`'s docstring that `X_normalised_markers` must be the exact output of `normalise_counts(X_markers, n_vars=adata.n_vars, total_counts=full_total_counts)`. Optionally accept `mean_x=None, std_x=None` to allow callers to pass pre-computed row statistics.

---

### m-15 — [IMPL] — Minor
**`weighted_median` subplan version has a latent crash bug that the implementation correctly fixed**

The subplan specifies: `return v[np.searchsorted(cw, 0.5 * cw[-1])]` — no bounds guard. When `half` exactly equals `cw[-1]` (all weight on the last element), `searchsorted` returns `len(v)`, which would be out-of-bounds. The implementation correctly adds `min(idx, len(v) - 1)` guard. This is a case where the implementation is more correct than the subplan.

**Recommended fix**: Add a comment in the implementation explaining why `min(idx, len(v) - 1)` is needed. No code change required.

---

### m-16 — [PLAN] — Minor
**Plan does not specify the return value for degree-0 cells in the random baseline**

The plan specifies "Create random KNN with same N neighbours per cell (variable N to match actual per-cell degree)" but does not address cells with degree 0. The implementation handles the case where ALL cells have degree 0 (early return of all-NaN), but if only some cells have degree 0, those cells get `corr_avg_random_* = 0.0` due to the `nan_to_num` issue (see M-7) rather than NaN.

**Recommended fix**: Add to the plan: "Cells with degree 0 produce NaN random baseline values (undefined, not 0), consistent with the missing-neighbours conceptual model."

---

### m-17 — [PLAN→SUBPLAN] — Minor
**Subplan's penetration return-key list omits `within_{tech}` — plan's "ALL 3 covariate levels" is ambiguous about scope**

The main plan says "Integration penetration (applied to ALL 3 covariate levels)". The subplan's correction #6 lists `penetration_cross_library`, `penetration_cross_dataset`, `penetration_between_{tech_name}` — only the cross-level masks. The plan's phrasing "ALL 3 covariate levels" could mean all six masks (same + cross at each level) or just the three cross-level masks. The subplan resolves this ambiguity by choosing only cross-level masks, but without justification.

**Recommended fix**: Clarify in the plan whether "ALL 3 covariate levels" means only the three cross-level masks or all six. If only cross-level, the subplan is correct and the plan should be made explicit.

---

### m-18 — [IMPL] — Minor
**`corr_discrepancy_same_library` has no clear interpretation per the plan's own framing**

The plan defines `corr_discrepancy_{mask} = corr_avg_{mask} - corr_mean_{mask}` and interprets it as: *"Large positive: averaging masks heterogeneity — 'looks integrated but mixes distinct populations'."* This interpretation is meaningful for `cross_library` and `cross_dataset` masks. For `same_library`, the discrepancy has no clear integration failure interpretation. The plan's headline metrics (H6, H10) only use `corr_discrepancy_cross_library` and `corr_discrepancy_cross_dataset`, implicitly acknowledging that `corr_discrepancy_same_library` is not a headline metric — but the plan never explicitly says it is uninformative.

**Recommended fix**: The plan should note that `corr_discrepancy_same_library` is computed but not used as a headline metric, and clarify whether it has any diagnostic value.

---

### m-19 — [PLAN→SUBPLAN] — Minor
**Subplan `compute_isolation_norm` docstring uses outdated formula**

The subplan's section 3 docstring template says `"Expected random: analytical P(all k_i random neighbours same group) = (n_same_group_cell_i / n_total)^k_i averaged over cells"`. The implementation's docstring correctly uses `(1 - p_match)^k` with the per-mask `p_match` table. The subplan's docstring contradicts both the main plan and the implementation.

**Recommended fix**: Update the subplan docstring template to use `(1 - p_match)^k` with the per-mask table.

---

### m-20 — [PLAN] — Minor
**`"Keep sparse throughout"` is self-contradictory**

The main plan states: *"Keep sparse throughout … Only densify when a dense matrix is needed for per-cell correlation (at that point subsetting to ~180 markers gives a dense ~300 MB float32)."* This is internally contradictory — it says "keep sparse throughout" then immediately says densification is needed. The implementation correctly keeps `avg_profiles` sparse in Approach B.

**Recommended fix**: Clarify the plan: *"Keep sparse throughout Approach B — `avg_profiles` remains sparse because per-row scaling of a sparse matrix preserves sparsity. Approach A densifies per-batch for pairwise correlation computation."*

---

## Summary Table

### Critical (5)

| ID | Tag | Title |
|----|-----|-------|
| C-1 | [IMPL] | Boolean polarity inversion: `flag_consensus_isolated` True → XD-0b instead of XD-0a |
| C-2 | [IMPL] | `XD-0_isolated_unknown` treated as severity-4 failure — inflates failure rate when no model comparison data |
| C-3 | [IMPL] | V5 scatter plot missing `corr_avg vs corr_mean` panels — replaced by discrepancy panels |
| C-4 | [SUBPLAN→IMPL] | `compute_random_knn_baseline` drops `technical_covariate_keys` — random masks absent for technical covariates |
| C-5 | [PLAN→SUBPLAN] | Subplan correction #1 gives wrong `cross_library` isolation formula (orders of magnitude off) |

### Major (16)

| ID | Tag | Title |
|----|-----|-------|
| M-1 | [PLAN] | `between_libraries` absent from Masks table but required by Graceful Degradation |
| M-2 | [PLAN] | Decision tree leaf count: plan says 21, subplan says 25, implementation has 26 |
| M-3 | [PLAN] | XD-0d condition "Cross-library FAILED" never precisely defined — `XL-unknown` silently maps to failure |
| M-4 | [PLAN] | Specificity formula `axis=0` wrong for `(n_genes, n_labels)` orientation |
| M-5 | [IMPL] | `top_n_per_label` not applied to `subtype_markers` — subtype set can be larger than `cell_type_markers` |
| M-6 | [IMPL] | `top_n_per_label: int = 500` added with no plan/subplan authorisation — silently caps gene selection |
| M-7 | [IMPL] | `nan_to_num(..., nan=0.0)` biases random baseline toward zero for isolated cells |
| M-8 | [IMPL] | CSR index array built with Python `for` loop — defeats bulk-sampling optimisation |
| M-9 | [IMPL] | `compute_neighbourhood_diagnostics` adds `between_libraries` penetration not in plan/subplan |
| M-10 | [SUBPLAN→IMPL] | `_stratified_summary_single` missing 50th percentile (p50) column |
| M-11 | [IMPL] | `compute_composite_score` uses graceful NaN degradation, contradicting plan and F12 recommendation |
| M-12 | [IMPL] | `integration_failure_rate` remains as phantom metric in `_NEIGHBOURHOOD_METRICS` (F11 not fully fixed) |
| M-13 | [IMPL] | H11/H12 OVL compares distributions over different cell subsets |
| M-14 | [IMPL] | `compute_analytical_isolation_baseline` API cannot compute `cross_library` baseline correctly |
| M-15 | [IMPL] | `assemble_cross_model_metrics` raises `KeyError` for missing models instead of warning and continuing |
| M-16 | [PLAN→SUBPLAN] | Subplan adds `n_neighbours_within_technical_cross_dataset` column not in main plan and never implemented |

### Minor (20)

| ID | Tag | Title |
|----|-----|-------|
| m-1 | [PLAN] | "Sets per-gene average to 1" is mathematically imprecise |
| m-2 | [PLAN] | `threshold_high` = 25th percentile is semantically confusing naming |
| m-3 | [PLAN] | `integration_failure_rate` denominator is ambiguous (all cells vs achievable cells) |
| m-4 | [PLAN] | `validate_covariate_hierarchy` missing key-existence check |
| m-5 | [PLAN] | H2 (`corr_consistency`) excluded from composite score without justification |
| m-6 | [PLAN] | Concatenating label columns before specificity dilutes per-label scores — undocumented |
| m-7 | [PLAN→SUBPLAN] | Subplan silently overrides "Do NOT reimplement `compute_cluster_averages`" |
| m-8 | [PLAN→SUBPLAN] | Return dict key `"union"` diverges from main plan's `"all_markers"` name |
| m-9 | [PLAN→SUBPLAN] | Subplan internal contradiction: `avg_profiles` dense vs sparse |
| m-10 | [PLAN→SUBPLAN] | Subplan `compute_random_knn_baseline` docstring retains approximate isolation formula |
| m-11 | [IMPL] | Inconsistent zero-variance threshold between `_sparse_pearson_row_stats` and approach guards |
| m-12 | [IMPL] | `corr_norm_by_library_same_library` is trivially 1.0 by construction |
| m-13 | [IMPL] | No warning when curated genes from CSV fail to map to `adata.var_names` |
| m-14 | [IMPL] | `compute_random_knn_baseline` recomputes row stats; no API guard against wrong normalisation |
| m-15 | [IMPL] | `weighted_median` subplan has latent crash bug — implementation correctly fixed it |
| m-16 | [PLAN] | Plan silent on degree-0 cells in random baseline |
| m-17 | [PLAN→SUBPLAN] | Subplan penetration return-key list omits `within_{tech}` — plan ambiguous on scope |
| m-18 | [IMPL] | `corr_discrepancy_same_library` has no clear interpretation per plan's own framing |
| m-19 | [PLAN→SUBPLAN] | Subplan `compute_isolation_norm` docstring uses outdated formula |
| m-20 | [PLAN] | "Keep sparse throughout" is self-contradictory |

---

## Prioritised Fix Order

The following ordering is recommended based on correctness impact:

1. **C-1** — Fix boolean polarity inversion in `classify_failure_modes` (XD-0a/XD-0b swapped — all model comparison results are wrong)
2. **C-4** — Add `technical_covariate_keys` to `compute_random_knn_baseline` (technical random baselines entirely absent)
3. **M-7** — Fix `nan_to_num(nan=0.0)` bias in random baseline accumulation (corrupts `isolation_norm` for small groups)
4. **M-5 + M-6** — Fix `top_n_per_label` inconsistency and document the parameter (gene-set hierarchy inversion corrupts Dimension 6)
5. **C-2** — Fix `XD-0_isolated_unknown` severity (inflates failure rate in default no-model-comparison case)
6. **M-11** — Document or fix graceful NaN degradation in composite score (inflates scores for models failing cross-dataset integration)
7. **M-12** — Remove `integration_failure_rate` from `_NEIGHBOURHOOD_METRICS` (F11 not fully fixed)
8. **M-10** — Add p50 to `_stratified_summary_single` (one-line fix)
9. **M-14** — Fix `compute_analytical_isolation_baseline` API for `cross_library` mask
10. **C-3** — Add `corr_avg vs corr_mean` panels to V5 scatter plot
11. **M-3** — Define "Cross-library FAILED" precisely in plan and exclude `XL-unknown` from `xl_was_failure`
12. **M-8** — Vectorise CSR index construction in `compute_random_knn_baseline` (performance)
13. **M-9** — Remove spurious `between_libraries` penetration from `compute_neighbourhood_diagnostics`
14. **M-13** — Align cell subsets for H11/H12 OVL comparison
15. **M-15** — Add `.get()` guard in `assemble_cross_model_metrics`
