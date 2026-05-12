# Neighbourhood Correlation Plan & Implementation Review

## 1. Problematic decisions in the main plan

### 1.1 Decision tree leaf count is 25 but plan specifies 21

The plan text says "6 dimensions, producing 21 leaves" (line 203) but the tree actually has 25 leaves: WL has 6 (WL-0 through WL-5), XL has 7 (XL-0a, XL-0b, XL-1 through XL-5), XD has 12 (XD-0a through XD-0d, XD-1 through XD-6, plus XD-5a/5b). The implementation adds `XD-0_isolated_unknown` as a 26th. This inconsistency makes it hard to verify completeness.

### 1.2 Specificity formula is transposed relative to biological intent

The plan specifies (line 101):
```
specificity = mean_per_gene_per_label / mean_per_gene_per_label.sum(axis=0, keepdims=True)
```
This sums across **labels** (axis=0) per gene, giving a per-label fraction that sums to 1 across labels for each gene. But the user feedback (line 39) corrects this to `.sum(0, keep_dims=True)`, which is the same axis. However, the actual implementation (line 278) does:
```python
specificity = label_averages.div(safe_row_sums, axis=0)
```
where `row_sums = label_averages.sum(axis=1)` — this sums across labels (columns) per gene (rows), which is `axis=1`. The plan says `axis=0` but means `axis=1` — the DataFrame is genes-by-labels, so summing across columns (labels) per gene is `axis=1`. The plan formula is written in matrix notation where axis=0 is columns, creating confusion. The implementation is correct but the plan is misleading.

### 1.3 Composite score NaN policy was left ambiguous

The plan (lines 459-478) specifies a composite score formula but never addresses what happens when components are NaN (e.g. no dataset_key → all cross-dataset metrics NaN). The handover (F12) flagged this as an open question. The plan should have specified graceful degradation upfront since single-dataset scenarios are an explicitly supported use case (lines 509-519). This forced an implementation decision without user guidance — the implementation chose graceful NaN-skip with weight renormalisation, which may silently produce non-comparable scores across models with different NaN patterns.

### 1.4 `compute_cluster_averages` copied rather than imported

The plan explicitly says (line 93) "Copy `compute_cluster_averages` from cell2location... Do NOT reimplement." But copying a function creates maintenance burden — if cell2location fixes a bug, the copy diverges. The implementation actually has TWO versions: the copied `compute_cluster_averages` (line 96, using the old concatenation pattern) AND a new `_cluster_averages_from_matrix` (line 127, using sparse one-hot matmul). The plan forced a copy when the better decision was either (a) add cell2location as a dependency, or (b) write a clean implementation (which happened anyway as `_cluster_averages_from_matrix`). The original copy is used nowhere in the actual pipeline — `select_marker_genes` calls `_cluster_averages_from_matrix`.

### 1.5 Isolation normalisation denominator uses global `n_total` for `cross_library` mask

The plan (line 454) specifies:
```
cross_library (restricted to within-dataset): n_dataset - n_lib over n_total - 1
```
This formula is conceptually wrong: the `cross_library` mask selects neighbours from a *different library but same dataset*. The probability of a random neighbour qualifying should be `(n_dataset - n_lib) / (n_dataset - 1)` — drawing from the within-dataset pool, not the global pool — because the mask itself restricts to within-dataset. Using `n_total - 1` in the denominator underestimates the random baseline when datasets are much smaller than the total, making models look worse than they are on the `cross_library` isolation metric. The handover (F3) flags this as critical but the fix still uses `n_total - 1`, which is the wrong reference frame.

**Counter-argument**: If random KNN draws from the global pool (all cells), then `n_total - 1` is correct — a random neighbour is any cell, and qualifying as cross_library means "different library AND same dataset". The formula `(n_dataset - n_lib) / (n_total - 1)` is the probability of a *globally random* neighbour falling in the same dataset but different library. This is a valid interpretation but conflates two things: the random baseline measures "what if neighbours were random across all cells" rather than "what if neighbours were random within the dataset". The plan should have been explicit about which null model is intended.

### 1.6 Decision tree thresholds derived from data rather than specified

The plan (line 203-295) defines a detailed 6-dimension decision tree but never specifies threshold values — only structural splits (high/low, homogeneous/mixed). The implementation derives `threshold_high` from the 25th percentile of within-library correlation and `std_threshold` from the median within-library std. This means the same cell can be classified differently depending on the overall quality of the dataset, making cross-dataset comparisons of failure mode distributions non-comparable. The plan should have either specified absolute thresholds or explicitly stated the adaptive approach with its limitations.

### 1.7 H14 cross_technical uses union of between masks, but this conflates technical axes

The plan (lines 349-354) defines H14 as median correlation on `cross_technical = union of between_{tech}`. When multiple technical covariates exist (e.g., `tissue` and `10x_kit`), a neighbour that differs in kit but shares tissue is pooled with one that differs in tissue but shares kit. These are very different integration challenges. A single H14 number masks whether the model handles tissue integration but fails kit integration. The plan should have kept per-technical-key metrics as separate headlines.

### 1.8 Discrepancy metric interpretation is under-specified for negative values

The plan (line 154) says negative discrepancy is "rare — distinct equally-correlated subtypes cancel in average." But mathematically, `corr_avg - corr_mean` can be negative when the average-then-correlate profile cancels out diverse neighbours' signal, while per-neighbour correlations remain individually positive. This happens when neighbours are from two distinct but equally good cell types — the cell matches each individually but their average looks like neither. The plan calls this "rare" without quantifying expected frequency or providing decision-tree integration. The user feedback (line 131-133) connects this to debris/multiplets but the plan doesn't incorporate that into the failure mode tree.

---

## 2. Problematic implementation (plan is good but implementation deviates)

### 2.1 CRITICAL: `_approach_A_per_mask` uses a Python for-loop over unique cells per batch

The plan and subplan 04 emphasise vectorised operations for 416k cells. But `_approach_A_per_mask` (line 683) has:
```python
for _k, (rl, start, end) in enumerate(zip(unique_rows_local, row_starts, row_ends)):
```
This iterates per-cell within each batch. For a batch of 2000 cells with ~50 neighbours each, this is 2000 Python iterations per batch, ~200k iterations total for 416k cells. The pairwise correlation computation (lines 664-678) is vectorised, but the aggregation (mean/median/weighted_mean/weighted_median/std/cv per cell) is not. The `weighted_median` call (line 701) is especially slow — it sorts values per cell. This makes Approach A 10-100x slower than it needs to be.

**Fix**: Aggregate using `np.add.reduceat` for mean/std, and use a vectorised weighted median (sort once, use segment boundaries).

### 2.2 `normalise_counts` returns float32 implicitly from sparse multiply

`normalise_counts` (line 91) calls `X.multiply(scale[:, None]).tocsr()` where `scale` is float32. The result dtype depends on the input X dtype — if X is int32 (raw counts), the multiply upcasts to float64 on some scipy versions. The function doesn't enforce output dtype. The caller in `compute_marker_correlation` (line 757) then does `.astype(np.float32)`, but `select_marker_genes` (line 236) calls `normalise_counts` without casting, so gene selection uses whatever dtype scipy chose. This is not a correctness bug but wastes memory when float64 propagates through the gene selection path on sparse 200k-cell matrices.

### 2.3 `compute_random_knn_baseline` replaces NaN with 0 before averaging

Line 1018: `corr_all = np.nan_to_num(corr_all, nan=0.0)`. When a cell has zero neighbours of a given type in the random graph (possible for small groups), NaN is replaced with 0 and counted in the average. This biases the random baseline downward for cells in small covariate groups, making the model look better by comparison. The plan says (line 218-220) "random KNN with same N neighbours per cell" — the NaN happens because the mask isolates some cells even in the random graph, but replacing with 0 is not the same as excluding from the average.

### 2.4 `compute_neighbourhood_diagnostics` calls `construct_neighbour_masks` redundantly

`compute_neighbourhood_diagnostics` (line 926) constructs masks that are also constructed by `compute_marker_correlation`. In a typical workflow, both are called on the same data. This doubles the mask construction cost (two sparse matrix multiplications per mask, O(nnz) each). The masks should be constructed once and passed to both functions.

### 2.5 `classify_failure_modes` uses `corr_avg` but applies `corr_std` thresholds from the same distribution

The threshold_high (line 1138) is derived from `corr_avg_same_library` 25th percentile, but `std_threshold` (line 1141) is derived from `corr_std_same_library` median. These are different distributions on different scales. The plan defines "homogeneous" vs "mixed" conceptually but the implementation's use of the std median as threshold means exactly half of cells are "homogeneous" and half are "mixed" by construction — this is a tautological split that doesn't measure actual heterogeneity. A principled threshold would be based on expected std under a null model (e.g., std of correlations with random neighbours).

### 2.6 `_compute_combined_failure_mode` skips non-failure leaves, losing diagnostic information

Line 1367-1368: `if label in _NON_FAILURE_LEAVES: continue`. This means a cell that is `WL-1_ideal` but `XD-4a_wrong_pairing` gets `failure_mode = "XD-4a_wrong_pairing"` — correct. But a cell that is `WL-1_ideal` and `XL-1_ideal` and `XD-0a_dataset_enriched` gets `failure_mode = "ideal"` (line 1381). `XD-0a_dataset_enriched` IS in `_NON_FAILURE_LEAVES` but carries real diagnostic meaning — it means the cell is dataset-specific. Collapsing this to "ideal" loses the information that drove the plan's model comparison design. The plan (line 271) calls XD-0a "NOT a failure. Biologically informative." — but the implementation treats it as equivalent to "all levels ideal" which hides this biologically informative signal.

### 2.7 `flag_consensus_isolated` inverts the boolean for `model_comparison_result`

`flag_consensus_isolated` returns `True` for cells NO model integrates (line 2101). But `classify_failure_modes` (line 1264-1268) interprets `model_comparison_result` as:
```python
mc_bool = np.asarray(mc, dtype=bool)
other_connects = mc_bool & mc_valid  # True = some model connects
```
So `flag_consensus_isolated=True` → `mc_bool=True` → `other_connects=True` → routes to XD-0b ("under-integration, other models prove matching IS possible"). This is exactly backwards: consensus_isolated=True means NO model connects, which should route to XD-0a ("dataset-enriched, likely real"). The caller must negate the flag: `model_comparison_result=~consensus_flag`. The API is confusing and error-prone — the plan's intent (lines 269-274) is that `model_comparison_result` answers "do alternative models connect this cell?" (True=yes), but `flag_consensus_isolated` answers the opposite question.

### 2.8 `plot_per_library_distributions` hardcodes `library_key="batch"` in the default

Line 2469: `library_key: str = "batch"`. The entire plan and CLAUDE.md insist on purpose-based terminology (`library_key` not `batch_key`). Defaulting to `"batch"` means the plotting function silently uses the wrong column if the caller doesn't override, or crashes if `obs["batch"]` doesn't exist. All other functions in the module correctly require `library_key` with no default.

### 2.9 `select_marker_genes` prints to stdout instead of using the logger

Lines 341-354: `print("Gene selection summary:")` followed by multiple `print()` calls. Every other function in the module uses `_logger.info()`. In a notebook this is fine, but in a pipeline with logging configured, these prints bypass log level control and routing. The plan doesn't specify logging requirements but the implementation is inconsistent with the rest of the module.

### 2.10 `compute_composite_score` uses `_nanweighted` (graceful) but the handover recommended NaN-poisoning

The handover (F12, line 192) recommends "Option A (current behaviour, conservative): Keep NaN poisoning." But the implementation uses `_nanweighted` (line 1685-1702) which skips NaN components and renormalises weights — this is Option B. This was changed silently without the user deciding. The handover explicitly says "Plan does not specify graceful degradation here" and "Recommended: Option A." The implementation went the opposite direction.

### 2.11 Missing `between_libraries` mask from `list_active_masks` but present in `construct_neighbour_masks`

`construct_neighbour_masks` (line 513) always creates `between_libraries` mask. `list_active_masks` (line 559) always includes it. But the plan (lines 36-43) lists masks as: `same_library`, `cross_library`, `cross_dataset`, `within_{tech}`, `between_{tech}`. The plan does NOT list `between_libraries` — it was added by the implementation. `between_libraries` means "any different library regardless of dataset", which overlaps with `cross_library` (different library, same dataset) + `cross_dataset` (different dataset). The plan's masks are mutually exclusive within a level; `between_libraries` breaks this. It adds computation cost without clear diagnostic value beyond what `cross_library + cross_dataset` provides.

---

## 3. Subplan-implementation mismatches relative to main plan intent

### 3.1 Subplan 04 drops `max-gap` metric but doesn't replace it with what the user asked for

The user feedback (line 53-56) questioned `max-gap`, and the plan dropped it (line 147): "max-gap dropped — std/cv + decision tree + multiple gene groups handle multimodality better." But the user's concern was about multimodal effects and incoming edges — the user wanted to detect cells with bimodal neighbour correlation distributions (some neighbours correct, some wrong). `corr_std` captures spread but not bimodality. The implementation provides std and cv but no bimodality indicator. The plan's claim that "decision tree handles this" is only partially true — the decision tree uses homogeneous/mixed as a binary split, not a bimodality test.

### 3.2 Subplan 06 implements 25 leaves but main plan tree has implicit dependencies not captured

The main plan's decision tree (lines 224-295) shows XD-level decisions conditioned on WL and XL results (e.g., "Within-library was HIGH" at XD level). The implementation correctly captures this (lines 1221-1234: `wl_was_ideal_xd`, `xl_was_ideal`, etc.). However, subplan 06 introduces a 26th implicit leaf `XD-0_isolated_unknown` (line 1297: `cond_xd_0_unk`) for cells where model comparison data is unavailable. This leaf is absent from the main plan. It has severity 4 in `_XD_SEVERITY` — same as `XD-0b_under_integration`. This means cells without model comparison data are treated as under-integrated by default, which is a strong assumption. The main plan's intent (lines 269-274) was that model comparison distinguishes XD-0a from XD-0b — without model comparison, the cell should be "unknown", not assigned the severity of under-integration.

### 3.3 Subplan 07's isolation_norm uses a single scalar but the plan implies per-cell normalisation

The plan (lines 446-457) describes isolation normalisation as:
```
isolation_norm = isolation_frac(model) / isolation_frac(random)
```
where both are scalar fractions (fraction of cells isolated). But the analytical baseline `compute_analytical_isolation_baseline` returns per-cell probabilities. The subplan 07 implementation of `compute_isolation_norm` (line 1558) takes `mean(per_cell_p_isolated)` as the expected fraction, then divides the observed fraction by it. This works mathematically (expected fraction = mean of per-cell expectations) but loses per-cell resolution. A per-cell isolation_norm (observed_isolated / p_isolated_random for each cell) would be more informative and consistent with the rest of the per-cell metric framework. The implementation chose scalar summarisation matching the headline metric intent, but this means the per-cell analytical baseline computation is partially wasted — only its mean is used.

### 3.4 Subplan 08 implements `compute_tissue_group_integration` but it's not in the main plan

The main plan mentions "tissue-group integration" only in the context of immune data (lines 354-355). Subplan 08 defined and implemented `compute_tissue_group_integration()` but the user feedback (handover line 177) says "tissue_group_integration doesn't make sense." The function exists in the module but is NOT in `__all__` (not exported) and is not called by any other function. It's dead code that was implemented from a subplan that went beyond the main plan's scope. The handover flagged it for removal but it's still present.

### 3.5 Subplan 05's random baseline doesn't pass `technical_covariate_keys` through

`compute_random_knn_baseline` (line 960-1036) accepts `library_key` and `dataset_key` but NOT `technical_covariate_keys`. It calls `list_active_masks(library_key, dataset_key)` without technical keys and `construct_neighbour_masks(adata, random_conn, library_key, dataset_key)` without technical keys. This means no random baseline is computed for `within_{tech}`, `between_{tech}`, or `cross_technical` masks. The main plan (lines 216-220) defines the random baseline for "Dimension 5" which is used in both WL-level (line 239) and XD-level (line 280-283) but the plan doesn't restrict it to only library/dataset masks. The implementation silently omits technical-mask baselines — the decision tree's `above_random` check at XD level works only because it falls back to `np.ones(n, dtype=bool)` (line 1241: "assume above random if no baseline"), which defeats the purpose of the random baseline test for cross-technical integration.

### 3.6 Subplan 09 visualisation doesn't implement V11 (benchmarker heatmap extension)

The main plan (line 502) specifies "V11: heatmap — Benchmarker heatmap with H1-H14 columns" and subplan 09's description includes "extend benchmarker heatmap with H1-H14 columns." But the implementation has no heatmap function. `plot_isolation_bars` and `plot_leaf_distribution` are implemented but the integration with `_integration_metrics.py`'s existing benchmarker heatmap is not done. The handover (F11) notes that `_NEIGHBOURHOOD_METRICS` in `_integration_metrics.py` lists phantom metrics that are never produced — this is the remnant of the planned but unimplemented heatmap integration.

### 3.7 Subplan 10 evaluation notebook was never created

The handover (line 224) confirms: "the actual notebook at `docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb` was never created." The main plan (line 581) lists it as subplan 10 with dependencies on all previous subplans. Without this notebook, the entire pipeline has never been run end-to-end on real data. The known bugs (F1: normalisation order, F3: isolation formula, F10: failure mode default) mean the metrics have never been validated against biological ground truth. This is the most consequential gap — 2841 lines of code that have only been unit-tested on synthetic fixtures.

### 3.8 `__init__.py` exports only plotting functions, hiding the computational API

`plt/__init__.py` exports 7 functions, all plot_* functions. The 25+ computational functions (`compute_marker_correlation`, `classify_failure_modes`, `summarise_marker_correlation`, etc.) are not exported from the package. Users must do `from regularizedvi.plt._neighbourhood_correlation import compute_marker_correlation` — accessing a private module. The main plan (line 605) says "Register in `plt/__init__.py`" without specifying which functions. The subplan implementation only registered the visualisation functions, which means the core computational API is technically private/internal. This is inconsistent with how `_integration_metrics.py` functions are exported.

---

## Summary of severity

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 2.7 | `flag_consensus_isolated` boolean inversion | **Critical** | Implementation bug |
| 2.1 | Approach A per-cell Python loop | **High** | Performance |
| 1.5 | Isolation norm denominator ambiguity | **High** | Plan design |
| 2.5 | Tautological std threshold | **High** | Implementation design |
| 3.7 | No evaluation notebook / no real-data validation | **High** | Missing deliverable |
| 3.5 | Random baseline missing technical masks | **High** | Subplan gap |
| 2.3 | NaN→0 in random baseline | **Medium** | Implementation bug |
| 2.6 | XD-0a collapsed to "ideal" | **Medium** | Information loss |
| 2.10 | Composite NaN policy contradicts handover | **Medium** | Undecided design |
| 3.8 | Computational API not exported | **Medium** | Usability |
| 2.8 | Hardcoded `batch` default | **Low** | Naming inconsistency |
| 2.9 | print() vs logger | **Low** | Code quality |
| 1.4 | Copied + unused `compute_cluster_averages` | **Low** | Dead code |
| 3.4 | `compute_tissue_group_integration` dead code | **Low** | Dead code |
