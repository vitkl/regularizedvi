# Neighbourhood Correlation Plan + Implementation Review

## Context
Review of `.claude/plans/neighbourhood_correlation_plan.md` and its implementation
`src/regularizedvi/plt/_neighbourhood_correlation.py` (2840 lines), assessing: (1) problematic
decisions in the main plan itself, (2) implementation that diverges from or mis-executes a
good plan, and (3) subplan–main-plan mismatches.

---

## 1. Problematic Decisions in the Main Plan

### 1a. Isolation formula is ambiguous / self-contradictory

The "Isolation score normalisation" section gives:

```
P(isolated | random, k_i) = ((n_same_group - 1) / (n_total - 1))^{k_i}
```

followed by a table describing `n_same_group` values per mask. But the table values are the
**probability of a successful match** `p_match`, not the group size you must **avoid**:

| mask | table value | what it actually is |
|---|---|---|
| `cross_library` | `n_dataset - n_lib / n_total - 1` | `p_match` (prob. of cross-library pick) |
| `cross_dataset` | `n_total - n_dataset / n_total - 1` | `p_match` (prob. of cross-dataset pick) |

Correct formula is `P(isolated) = (1 - p_match)^k`, NOT `(p_match - 1/...)^k`. For `cross_library`
the literal application of the header formula gives `((n_dataset - n_lib - 1)/(n_total-1))^k`,
which is wrong. Sub-plan 07 went with a third interpretation. The implementation ended up
doing the right thing (`(1 - p_match)^k`) by ignoring both formulas and reasoning from scratch,
but the plan contains two contradictory encodings.

**Impact**: any reader re-implementing from the plan alone will get wrong isolation baselines.

### 1b. Composite score ignores `corr_consistency` (H2)

```
bio_conservation = corr_within_library
```

H2 `corr_consistency` (median of `corr_std_same_library`) is listed as a headline metric but
excluded from the composite. A model with high within-library correlation but very high std
(mixed populations merged) receives the same bio score as one with homogeneous results. The plan
provides no justification for dropping H2 from the composite.

### 1c. Decision tree WL-2 claims gene-group evidence it doesn't require

Plan says `LEAF WL-2: "Merged related types" (broad markers agree, specific disagree)`. The
parenthetical implies DIM 6 (gene group comparison) was evaluated. But the decision tree assigns
WL-2 purely via `high correlation + mixed std` — no gene-group check. The sub-leaf claim
"broad HIGH + specific LOW" is asserted without a DIM 6 branch at the WL level.

### 1d. Poor quality cell detection declares a dependency on an unimplemented function

"Dependency: ambient_frac and recon_perplexity require `get_latent_qc_metrics()` from
fuzzy-percolating-conway plan (Phase 1). Compute before neighbourhood correlation."
`get_latent_qc_metrics()` does not exist in the codebase (search confirms absence). The plan
never accounts for this missing upstream function, making the full QC classification pipeline
unreachable.

### 1e. Main plan states "21 leaves" but the tree has 25

The text says "21 leaves" but counting the full tree: 6 WL + 7 XL + 12 XD = 25 leaves. Sub-plan
06 corrects this to 25 internally, creating inconsistency between the plan overview and the tree
itself.

### 1f. `cross_technical` isolation probability is undefined

`compute_isolation_norm` handles `cross_technical` as if it were a single-key `between_{tech}`
mask (using `p_match = (n_total - n_tech)/(n_total-1)` for a single `technical_key`). But
`cross_technical` = **union** of all `between_{tech}` masks — a cell in a tissue group of size
`n_t1` that also has a tech covariate of size `n_t2` has a different isolation probability
than either `between_{t1}` or `between_{t2}` alone. The plan gives no formula for the union case.

---

## 2. Problematic Implementation

### 2a. `flag_consensus_isolated` / `classify_failure_modes` semantics are inverted — **bug**

`flag_consensus_isolated` returns `True` = "no model integrates this cell" (consensus-isolated →
should be **XD-0a**, dataset-enriched, NOT a failure).

`classify_failure_modes(model_comparison_result=...)` treats `True` = "other model connects" →
**XD-0b** (under-integration, IS a failure).

The function docstring says: "Designed to feed into `classify_failure_modes(...,
model_comparison_result=consensus_flag)`". Passing the flag directly yields the exact opposite
classification: consensus-isolated cells (XD-0a candidates) are classified as under-integration
(XD-0b) and vice versa. This requires `~flag_consensus_isolated(...)` at the call site, which
the API nowhere documents.

Relevant lines: [`_neighbourhood_correlation.py:2088-2100`](src/regularizedvi/plt/_neighbourhood_correlation.py#L2088)
(`flag_consensus_isolated`), [`_neighbourhood_correlation.py:1263-1278`](src/regularizedvi/plt/_neighbourhood_correlation.py#L1263)
(`classify_failure_modes` model comparison branch).

### 2b. `compute_random_knn_baseline` ignores `technical_covariate_keys`

[Lines 980–984 and 1021–1022](src/regularizedvi/plt/_neighbourhood_correlation.py#L980) call `list_active_masks` and
`construct_neighbour_masks` without `technical_covariate_keys`. Random baseline correlations
for `within_{tech}`, `between_{tech}`, and `cross_technical` are never computed even when the
user provides technical covariate keys. DIM 5 (random baseline comparison) therefore silently
does nothing for the technical hierarchy.

### 2c. `classify_cell_quality` missing `quality_local_ambient_deviation` column

Plan specifies per-cell output column `quality_local_ambient_deviation` (raw ambient_frac minus
KNN-smoothed ambient_frac). The function accepts no connectivity matrix, never computes the
KNN-smoothed value, and does not return this column. The step that distinguishes "debris near a
real cluster" from "rare cell in a coherent cluster" is absent.

### 2d. `compute_tissue_group_integration` entirely absent

Sub-plan 08 (task 6) specifies `compute_tissue_group_integration()` — the "pure technical
baseline" metric that computes mean cross-dataset correlation restricted to cells where same
technical group membership makes integration unambiguous. The function is not implemented, not in
`__all__`, and the sub-plan 08 task list includes it in the planned `__all__` export. The H14
headline metric is instead served by `compute_cross_technical_correlation`, which answers a
related but distinct question (median cross-technical correlation, not restricted to same-group
cells).

### 2e. `plot_failure_mode_scatter` silently omits random-baseline panels

The function includes panels for `corr_avg_random_same_library` and
`corr_avg_random_cross_dataset` but these columns live in `random_baseline_df`, not `metrics_df`.
There is no merging step. The panels are silently absent (the `if x in metrics_df.columns`
guard hides the failure). Nothing in the docstring or calling convention documents that the
caller must merge the two DataFrames before passing.

### 2f. `plot_marker_correlation_umap` `leaf_df` column mismatch

The function reads `leaf_df["leaf"]` ([line ~2248](src/regularizedvi/plt/_neighbourhood_correlation.py#L2248)).
`classify_failure_modes` returns a DataFrame with columns `leaf_within_library`,
`leaf_cross_library`, `leaf_cross_dataset`, `failure_mode` — no column named `"leaf"`. The
caller must create/rename a column before calling, but this is undocumented and produces a
silent `KeyError` at runtime.

### 2g. `compute_isolation_norm` for `same_library` mask is computed but never used

The function correctly computes P(isolated from same_library neighbours), but this metric is
not referenced in H1–H12, not in the composite score, and not in `summarise_marker_correlation`.
The headline metrics use only `cross_library` (H5) and `cross_dataset` (H9) isolation norms.
The `same_library` case is a dead computation.

---

## 3. Subplan–Main-Plan Mismatches

### 3a. Sub-plan 07 isolation formula for `cross_library` contradicts main plan and is wrong

Sub-plan 07 correction #1 states:
> "Mask `cross_library`: Expected random: `P = (n_same_library_i / (n_total - 1))^k_i` — chance all random picks are same library (thus no cross-library)."

This formula ignores cells from **other datasets** (which also don't count as cross-library
neighbours). Correct P(isolated from cross_library) = `((n_total - n_ds + n_lib - 1) / (n_total - 1))^k`.
The implementation uses this correct form (via `p_match = (n_ds - n_lib)/(n_total-1)` then
`(1 - p_match)^k`), which matches the main plan's table but contradicts sub-plan 07's written
formula. Sub-plan 07 is wrong; main plan table + implementation are correct.

### 3b. Sub-plan 06 specifies `threshold_low` for `classify_failure_modes`; parameter was dropped

Sub-plan 06 signature includes `threshold_low: float | None = None`. The implementation's
[`classify_failure_modes` signature (line 1124–1131)](src/regularizedvi/plt/_neighbourhood_correlation.py#L1124)
has no `threshold_low` parameter. The decision tree is purely binary on `threshold_high`. The
`threshold_low` parameter appears only in `compute_integration_failure_rate` and
`compute_contingency_per_cell`, where it's used differently. This was silently dropped without
noting that it was intentional.

### 3c. V5 "avg vs mean" plots absent — V6 "discrepancy" plots implemented instead

Main plan: "V5 | hist2d | corr_avg vs corr_mean per mask (shows discrepancy structure)".
Sub-plan 09 correction #2 says `plot_failure_mode_scatter` must include avg×mean panels.
Implementation's `plot_failure_mode_scatter` includes `(corr_avg_{mask}, corr_discrepancy_{mask})`
panels (V6-style) but no `(corr_avg_{mask}, corr_mean_{mask})` panel. V5 is unimplemented.

### 3d. Sub-plan 06 DIM 6 sub-labels specified for WL-4 / XL-3 / XD-4a but only XD-5a handled

Sub-plan 06 task 4: add `leaf_{level}_sublabel` column for WL-4, XL-3, XD-4a, XD-5a using gene
group comparison. Implementation applies gene group logic only to route XD-5a vs XD-5b. WL-4,
XL-3, and XD-4a receive no sub-label. The `leaf_{level}_sublabel` column is not produced at all.

### 3e. `plot_leaf_distribution` docstring references non-existent `assign_leaf_labels`

[Docstring (line 2610)](src/regularizedvi/plt/_neighbourhood_correlation.py#L2610): "leaf label (from
`:func:assign_leaf_labels`)". No function named `assign_leaf_labels` exists; the correct
function is `classify_failure_modes`. Breaks auto-generated docs.

### 3f. Sub-plan 08 `compute_tissue_group_integration` listed in `__all__` export list but absent

Sub-plan 08 task 9: `__all__` should include `compute_tissue_group_integration`. Neither the
function nor the export is present. The existing [`__all__` (lines 27–60)](src/regularizedvi/plt/_neighbourhood_correlation.py#L27)
does not include it.

---

## Critical Files
- [`src/regularizedvi/plt/_neighbourhood_correlation.py`](src/regularizedvi/plt/_neighbourhood_correlation.py) — main module (2840 lines)
- [`.claude/plans/neighbourhood_correlation_plan.md`](.claude/plans/neighbourhood_correlation_plan.md) — main plan
- [`.claude/plans/neighbourhood_correlation_subplan_06_decision_tree_classification.md`](.claude/plans/neighbourhood_correlation_subplan_06_decision_tree_classification.md)
- [`.claude/plans/neighbourhood_correlation_subplan_07_summary_and_composite.md`](.claude/plans/neighbourhood_correlation_subplan_07_summary_and_composite.md)
- [`.claude/plans/neighbourhood_correlation_subplan_08_model_comparison.md`](.claude/plans/neighbourhood_correlation_subplan_08_model_comparison.md)
- [`.claude/plans/neighbourhood_correlation_subplan_09_visualisation_and_heatmap.md`](.claude/plans/neighbourhood_correlation_subplan_09_visualisation_and_heatmap.md)
