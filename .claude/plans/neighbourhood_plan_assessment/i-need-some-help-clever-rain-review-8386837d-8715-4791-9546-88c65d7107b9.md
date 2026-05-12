# Review of `neighbourhood_correlation_plan.md` and Implementation

Scope: `[main plan](.claude/plans/neighbourhood_correlation_plan.md)`, 11 sub-plans, [src/regularizedvi/plt/_neighbourhood_correlation.py](src/regularizedvi/plt/_neighbourhood_correlation.py) (2,840 lines, 32 public functions), heatmap hooks in [src/regularizedvi/plt/_integration_metrics.py](src/regularizedvi/plt/_integration_metrics.py), evaluation notebook [docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb](docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb), and the [handover file](.claude/plans/neighbourhood_correlation_handover.md). 104/104 tests pass; many issues below are still latent because tests don't exercise them.

---

## 1. Problematic decisions in the main plan

### 1.1 Composite score weights are uncalibrated and inherit scIB's flaws
Plan §"Composite score" (lines 461–477) hard-codes `0.6 * bio + 0.4 * batch` and `0.4*H3 + 0.3*(1-H5) + 0.3*H11`. These are direct borrows from scIB, yet the introduction explicitly cites *"Recent work (Nat Biotech 2025) showed silhouette-based metrics can mislead"* as motivation for replacing scIB. Re-using scIB's weighting scheme reintroduces the same arbitrariness the metric was designed to avoid. There is no calibration procedure (e.g. agreement with a held-out reference, or sensitivity analysis on weights).

### 1.2 Internally inconsistent isolation formula across the plan
- Line 220: `((n_same_dataset - 1) / (n_total - 1))^k_i`
- Line 449: `((n_same_group - 1) / (n_total - 1))^{k_i}`
- Lines 451–455 list per-mask numerators/denominators that don't form a partition (`same_library` uses `n_same-1`, `cross_library` uses `n_dataset - n_lib`, etc.).

The handover (F2/F3) acknowledges these mismatches, but the plan still ships with three different versions of the same formula. New readers cannot tell which is canonical.

### 1.3 Decision tree leaf classification is bootstrap-circular
Leaves XD-0a ("dataset-enriched, real biology") vs XD-0b ("under-integration") cannot be assigned without **other models' outputs** (`model_comparison_result`). But the plan presents a decision tree run per-model, and it is unclear who orchestrates the cross-model dependency. The implementation falls back to a third leaf `XD-0_isolated_unknown` (not in the plan tree at all) when `model_comparison_result is None`, so when the notebook runs the decision tree before all models are computed, every isolated cell goes to "unknown" — silently undermining the headline interpretation.

### 1.4 25 leaves is too granular for a benchmarker
Plan line 203 promises "21 leaves" but corrections in sub-plan 06 expand to 25, distributed over 3 levels × 6 dimensions. With 416k immune cells / 16 models, many leaves end up with <1% of cells, providing no statistical power for between-model comparison. There is no guidance on aggregating leaves into actionable categories or comparing leaf distributions between models statistically. The plan should specify which leaves are headline-relevant and which are diagnostic-only.

### 1.5 Best-achievable envelope (H13) is biased toward overconfident integrators
`best_achievable = max_models corr_avg_cross_dataset[cell]` (plan §"Best-achievable envelope"). A model that wrongly merges distinct cell types with high correlation pulls up `best_achievable`, marking cells as "failures" for models that *correctly* isolate them. The plan never gates by "above random KNN" or by the discrepancy metric, so positive failures (false merges) inflate the envelope. The decision tree dimension 5 ("vs random KNN") is not propagated into H13.

### 1.6 No minimum-marker filter for Pearson on sparse data
scRNA-seq markers typically have >90% zeros per cell. Pearson correlation on cells with 1–3 non-zero markers is effectively noise. The plan only specifies "zero-variance rows → NaN" (handover docstring note), which catches the *trivial* degenerate case but not cells with 1–3 nonzero markers (variance is non-zero but stat is meaningless). Similar concern for the random baseline.

### 1.7 H6/H10 (`discrepancy_*`) are orphan headlines
Plan §"Headline Metrics" lists H6/H10 with direction "lower = better", yet they are not used in the composite score (lines 461–477) and the plan never explains the sign convention. `corr_discrepancy = corr_avg − corr_mean` is symmetric; "lower" could mean "more negative" or "closer to zero". The plan §"Derived metric: A-vs-B discrepancy" (lines 151–155) actually says **large positive** = bad (debris/multiplets) and **near zero** = good — but the headline says "lower = better", which would prefer large *negative* discrepancies. Direction is internally contradictory.

### 1.8 Penetration thresholds 10/25 are hard-coded with no tie to k
Plan §Part 1.4 fixes thresholds at 10 and 25 because k=50. With k≠50 (the connectivity files in the repo are k=50, but the parameter is configurable in `compute_neighbourhood_diagnostics(k_reference=50, penetration_thresholds=(10, 25))`) these thresholds become meaningless. They should be expressed as fractions of `k_reference`.

### 1.9 `corr_norm_by_library_{mask}` divides correlations
`corr_avg / corr_avg_same_library` (plan §"Per-Cell Output Columns"). Pearson correlations are bounded in [−1, 1] and can be near zero, where the ratio is unstable; signs can flip. Implementation guards with `|denom| > 1e-12 → NaN`, but for cells with `corr_avg_same_library ≈ 0.05` the ratio of 0.04 cross-dataset gives ~0.8, while 0.06 gives ~1.2 — the ranking is dominated by noise, not integration quality. Plan should use a difference (`corr_avg − corr_avg_same_library`) or only normalise when the baseline correlation is above a floor.

### 1.10 `marker_gene_total_expression` is dead output
Plan §"Per-Cell Output Columns" lists `marker_gene_total_expression` as a base column. It is computed in [_neighbourhood_correlation.py:760](src/regularizedvi/plt/_neighbourhood_correlation.py#L760) and emitted in the DataFrame but never used by any failure mode classification, composite score, or visualisation. Either describe its diagnostic purpose or remove.

### 1.11 Plan asks `cross_technical` to be the union of `between_{tech}` masks
Lines 351–354. When two technical keys are correlated (e.g. embryo `Embryo` and `Experiment` are nested), the union is dominated by the broader covariate; the metric loses interpretability. The plan should specify intersection vs union explicitly per use case, or warn when keys are highly correlated (Cramér's V test).

### 1.12 Quality classification cascade has wrong ordering
Plan §"Multi-axis classification" Step 1 says "if corr_deviation ≥ 0 → not flagged". Cells matching their library median can still be debris if the whole library has elevated ambient. Step 1 dominates over Step 3 (poor_quality from ambient_frac), so a uniformly-bad library is labelled "good". Order should be: poor_quality (ambient/perplexity) → rare → good → uncertain.

---

## 2. Problematic implementation (plan is reasonable, implementation isn't)

### 2.1 WL-2 ("merged related types") is treated as ideal in cross-library/cross-dataset gating — silently cascades wrong leaves
[_neighbourhood_correlation.py:1188](src/regularizedvi/plt/_neighbourhood_correlation.py#L1188) and [:1221](src/regularizedvi/plt/_neighbourhood_correlation.py#L1221):
```python
wl_was_ideal = np.isin(wl_leaves, ["WL-1_ideal", "WL-2_merged_related"])
```
The plan tree (lines 232–243) makes WL-2 a *failure* leaf ("Merged related types — broad markers agree, specific disagree"). Including it in `wl_was_ideal` causes XL-3, XL-4, XD-4a, XD-4b, XD-5a, XD-5b conditions to fire for cells that already failed at within-library — they get re-classified as cross-library/cross-dataset failures rather than the cascaded "compounded failure" leaves (XL-0b, XD-0d). This silently shifts the leaf distribution.

### 2.2 `compute_random_knn_baseline` ignores `technical_covariate_keys`
[_neighbourhood_correlation.py:981–984](src/regularizedvi/plt/_neighbourhood_correlation.py#L981) calls `list_active_masks(library_key, dataset_key)` with no `technical_covariate_keys`. As a result no random baseline columns are produced for `within_{tech}`, `between_{tech}`, or `cross_technical`. Decision tree DIM 5 ("vs random KNN") cannot evaluate technical-axis leaves; H14 (`cross_technical_correlation`) has no random reference to normalise against. The plan and sub-plan 05 require this baseline for all masks.

### 2.3 `compute_marker_correlation` does not run `compute_random_knn_baseline` itself
The user must wire the random baseline into `classify_failure_modes` manually (the eval notebook does this at [neighbourhood_correlation_metrics.ipynb cell `per-model-loop`](docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb)). Pipeline composition is the user's responsibility, with no API-level glue. A minor convenience wrapper `compute_all_per_model(...)` would prevent silent misuse where someone forgets the baseline and ends up with `above_random = np.ones(n, bool)` (all-True default at [_neighbourhood_correlation.py:1159](src/regularizedvi/plt/_neighbourhood_correlation.py#L1159)).

### 2.4 `_compute_combined_failure_mode` severity table is fragile
[_neighbourhood_correlation.py:1099–1119](src/regularizedvi/plt/_neighbourhood_correlation.py#L1099). The XD severity dict puts `XD-0a_dataset_enriched` at severity 11 (best) and `XD-0_isolated_unknown` at severity 4 — meaning "we don't know if this cell is integratable" gets ranked worse than "wrong pairing" but better than "compounded failure". With the bootstrap problem in §1.3, every first-run cell becomes `XD-0_isolated_unknown`, so the "failure_mode" headline is anchored at severity 4 by default. Severity numbers should be exposed as a parameter (the plan acknowledges these are tentative).

### 2.5 `compute_isolation_norm` mixes NaN-aware and NaN-naive aggregations
[_neighbourhood_correlation.py:1558–1566](src/regularizedvi/plt/_neighbourhood_correlation.py#L1558):
```python
expected_frac = float(np.nanmean(p_iso))         # NaN-skip
observed_frac = (n_mask[valid] == 0).sum() / n_valid  # naive division
```
`expected` averages over per-cell probabilities (cells with NaN covariate are skipped). `observed` uses `n_valid` (cells where covariate is finite) but `n_mask` is the full array — if a cell has NaN covariate but `n_neighbours_{mask}=5`, it still counts toward the numerator. Numerator/denominator are not from the same population.

### 2.6 `compute_random_knn_baseline` rejection sampling has no max-iteration bound
[_neighbourhood_correlation.py:996–1002](src/regularizedvi/plt/_neighbourhood_correlation.py#L996). The `while self_mask.any():` loop is monte-carlo guaranteed to terminate but has no safety cap. With small `n_cells` (unit tests) it will hammer the same indices repeatedly. Add `max_attempts=10` and fall through to a deterministic fallback (e.g. shift by 1).

### 2.7 `compute_neighbourhood_diagnostics` returns per-cell Series for degree but not summary stats
[_neighbourhood_correlation.py:872–875](src/regularizedvi/plt/_neighbourhood_correlation.py#L872). Plan §Part 1.1 asks for "histogram. Flag cells with degree >> k". The implementation returns the raw Series; high-degree cells are auto-flagged at threshold `k_reference * 1.5` for the `high_degree_obs_means` table, but no histogram bin counts, no warning when a fraction >5% of cells exceed threshold. Visualisation is left to the user.

### 2.8 `_approach_A_per_mask` relies on COO row order without assertion
[_neighbourhood_correlation.py:680–683](src/regularizedvi/plt/_neighbourhood_correlation.py#L680). `np.unique(rows_local, return_index=True)` correctly recovers row groups only because CSR→COO preserves row order. This is a scipy implementation detail, not an API guarantee. A defensive `assert np.all(np.diff(rows_local) >= 0)` would prevent silent corruption if scipy ever changes.

### 2.9 `flag_consensus_isolated` and `compute_integration_failure_rate` use unaligned thresholds
[_neighbourhood_correlation.py:2052](src/regularizedvi/plt/_neighbourhood_correlation.py#L2052) uses `min_corr=0.3`; [:1864](src/regularizedvi/plt/_neighbourhood_correlation.py#L1864) uses `threshold_high=0.4`, `threshold_low=0.2`. Both feed into the XD-0a/0b decision branch (via `model_comparison_result`), but a cell with cross-dataset correlation 0.35 is "consensus-isolated" by `flag_consensus_isolated` and "integratable" by `compute_integration_failure_rate`. Pick one threshold and document the convention; expose as a single `default_corr_threshold` constant.

### 2.10 `compute_distribution_overlap` uses fixed `range_=(-1, 1)`
[_neighbourhood_correlation.py:1421](src/regularizedvi/plt/_neighbourhood_correlation.py#L1421). On sparse marker data, observed Pearson sits in roughly [0, 0.7]; binning over [−1, 1] wastes ~65% of bins on empty regions. OVL detection of subtle distribution differences is degraded. Plan and sub-plan 07 do not specify the range, so this is an implementation choice — auto-range over `[min(x,y), max(x,y)]` would be better.

### 2.11 `select_marker_genes` silently degrades when `dataset_col` not in obs
[_neighbourhood_correlation.py:217–220](src/regularizedvi/plt/_neighbourhood_correlation.py#L217):
```python
if per_dataset and dataset_col in adata.obs.columns:
    datasets = adata.obs[dataset_col].unique()
else:
    datasets = ["__all__"]
```
No warning is emitted when the user asked for `per_dataset=True` but the column is missing. The CLAUDE.md project rule "always ask whether decisions are the same for similar requests" is silently violated.

### 2.12 `compute_marker_correlation` densifies `avg_profiles` for some sparse paths
[_neighbourhood_correlation.py:610](src/regularizedvi/plt/_neighbourhood_correlation.py#L610). `weighted_sum.multiply(1.0 / safe_sums[:, None])` produces a sparse matrix, then `X.multiply(avg_profiles).sum(axis=1)` is sparse × sparse. Good. But `avg_profiles.power(2).sum(axis=1)` is computed even for cells with zero neighbours, where row_sum=0 forces `avg_profiles[i] = 0/1 = 0` — wasted work. Mostly cosmetic, but on 416k cells × many masks this is a measurable inefficiency.

### 2.13 The DEFAULT_LIBRARY_KEY is hard-coded as `batch` in plot helpers
[_neighbourhood_correlation.py:2469](src/regularizedvi/plt/_neighbourhood_correlation.py#L2469) `plot_per_library_distributions(..., library_key="batch", dataset_key="dataset")`. The whole point of the new module (per CLAUDE.md and sub-plan 11) is to *not* use `batch_key`. Defaults inside this module should be `library_key="library"` or have no default, forcing the caller to supply.

### 2.14 H13 in `_NEIGHBOURHOOD_METRICS` registry but never produced by `summarise_marker_correlation`
[_integration_metrics.py:353](src/regularizedvi/plt/_integration_metrics.py#L353) registers `integration_failure_rate`. `summarise_marker_correlation` ([:1571](src/regularizedvi/plt/_neighbourhood_correlation.py#L1571)) does not include it; the eval notebook adds it post-hoc in cell `cross-model`. If a user calls only `summarise_marker_correlation`, H13 is silently absent from the heatmap. Either drop the registration or have the function compute H13 when a `cross_model_df` is passed.

### 2.15 `compute_composite_score` graceful degradation diverges from the plan formula
[_neighbourhood_correlation.py:1685–1702](src/regularizedvi/plt/_neighbourhood_correlation.py#L1685). The implementation chose option B (NaN-skip + renormalise) per handover F12, but the plan §"Composite score" specifies fixed weights without any renormalisation. This is a unilateral design decision. Worse: a single-dataset run (no `dataset_integration`) and a multi-dataset run with an entirely missing `distrib_overlap_dataset` column produce *the same composite score formula*, but with renormalised weights — the comparison is no longer apples-to-apples. The plan's stated convention "When dataset level unavailable: `batch_correction = library_integration`" (line 477) is implemented, but partial-NaN cases are silently rescaled.

### 2.16 No test coverage for the WL-2 conflation, technical-axis random baseline, or model-comparison cascade
Test file is 2,922 lines / 42 tests, but I see no test asserting:
- A cell with WL-2 leaf cascades to XL-0b / XD-0d (not XL-3/XD-4a).
- Random baseline DataFrame contains `corr_avg_random_within_{tech}` columns when `technical_covariate_keys` are passed.
- `XD-0_isolated_unknown` is correctly distinguished from `XD-0a_dataset_enriched` once `model_comparison_result` is provided.

These are exactly the failure modes the plan claims to detect; the absence of regression tests means the bugs above are silent.

---

## 3. Sub-plans / implementation drift from the main plan

### 3.1 `between_libraries` mask: extra mask not in the main plan
- Sub-plan 03 §"Review Correction 1" (lines 7–10) keeps `between_libraries` as a base mask.
- Main plan §"Masks (no abbreviations)" (lines 36–47) lists only `same_library, cross_library, cross_dataset, within_{tech}, between_{tech}` — no `between_libraries`.

The implementation produces `corr_avg_between_libraries`, `n_neighbours_between_libraries` etc. ([_neighbourhood_correlation.py:511–513](src/regularizedvi/plt/_neighbourhood_correlation.py#L511)) — columns the main plan never declares. The decision tree never branches on `between_libraries`. The headline metrics never aggregate it. It's a sub-plan-only concept that adds output columns and confusion.

### 3.2 Leaf count: 21 (main plan) vs 25 (sub-plan 6 / impl)
Main plan line 203: "*producing 21 leaves*". Sub-plan 06 §"Review Correction 1": "**Leaf count: 25 total**". Implementation has 6 WL + 7 XL + 13 XD ≈ 26 names (counting `XD-0_isolated_unknown`). The user reading the main plan will count and not match the implementation.

### 3.3 `XD-0_isolated_unknown` leaf is implementation-only
Main plan tree (lines 222–295) does not contain this leaf — only XD-0a / XD-0b / XD-0c / XD-0d. The implementation adds it ([_neighbourhood_correlation.py:1310](src/regularizedvi/plt/_neighbourhood_correlation.py#L1310)) to handle the case where `model_comparison_result is None`. Sub-plan 06 §"Review Correction 6" mentions it once: "When None, emit `XD-0_isolated_unknown`" — but the main plan was not updated. A reader of the main plan cannot reconstruct the full leaf taxonomy.

### 3.4 `tissue_group_integration` deletion was not propagated
- Sub-plan 08 §"6. Tissue-group integration" (lines 146–164) defines `compute_tissue_group_integration`.
- Handover F11 (lines 169–179) records: user said "*tissue_group_integration doesn't make sense*"; function should be removed and replaced by `cross_technical_correlation`.
- Implementation: the function is **gone** (no `tissue_group` matches in [_neighbourhood_correlation.py](src/regularizedvi/plt/_neighbourhood_correlation.py)).
- But sub-plan 08 still describes the original API. A new contributor reading sub-plan 08 will look for a function that was deleted.

### 3.5 `n_neighbours_within_{tech}_cross_dataset` composite column never produced
- Sub-plan 04 §"Review Correction 8" promises this column.
- Sub-plan 08 §"Review Correction 5" requires it for H14.
- Implementation ([_neighbourhood_correlation.py:799–823](src/regularizedvi/plt/_neighbourhood_correlation.py#L799)) only produces single-axis masks (`within_{tech}`, `between_{tech}`, `cross_technical`) — no `within_{tech}_cross_dataset`.
- H14 was redefined to use only `cross_technical` (handover F11), so the composite column was abandoned. Sub-plan 04 still promises it.

### 3.6 H14 random baseline was abandoned without documentation
The headline H14 (`cross_technical_correlation`) is meant as the "pure technical" reference — *"between which we expect high integration"* (plan line 351). Yet `compute_random_knn_baseline` does not compute the random reference for the technical mask (§2.2 above), so H14 has no normalisation. The plan and sub-plans 05/07/08 do not call out this gap. The ratio "model H14 / random H14" — which is the meaningful integration signal — is unavailable.

### 3.7 Sub-plan 10 says BM/immune comparison is required but the notebook doesn't do it
- Sub-plan 10 §"3. Single-dataset bone marrow comparison": "*Compare BM within-library distributions with immune within-library distributions (should be comparable)*".
- The eval notebook ([neighbourhood_correlation_metrics.ipynb](docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb)) instructs the user to invoke it twice with different parameters — but the cross-run comparison is left as the user's exercise. Validation of the "comparability" claim is not actually performed.

### 3.8 Sub-plan 10 references a stale models TSV
- Sub-plan 10 §"1. Notebook structure" defaults to `models_tsv = "docs/notebooks/model_comparisons/z_init_sigma_jobs.tsv"`.
- Eval notebook params cell uses `integration_metrics_v2.tsv` (different file).
- Main plan §"Comparison: Single-Dataset vs Multi-Dataset" line 517: "**Models to evaluate**: all 16 from z_init_sigma_jobs.tsv".

The plan file references and the notebook params have diverged; nothing is wired to enforce 16 models or any specific provenance.

### 3.9 Plan asks to "copy" `compute_cluster_averages`; implementation copies AND adds a parallel helper
- Sub-plan 01 §"3. Copy cluster_averages": copy verbatim from cell2location.
- Implementation has `compute_cluster_averages` ([:96](src/regularizedvi/plt/_neighbourhood_correlation.py#L96)) **plus** `_cluster_averages_from_matrix` ([:127](src/regularizedvi/plt/_neighbourhood_correlation.py#L127)).
- Sub-plan 02 §"Review Correction 4" introduces `_cluster_averages_from_matrix` to avoid layer mutation but does not retire the verbatim copy.
- Result: the verbatim function is dead code (`select_marker_genes` calls only `_cluster_averages_from_matrix`). Plan says "copy"; implementation has two copies, one unused.

### 3.10 `corr_norm_by_library_{mask}` and `corr_norm_by_all_{mask}` produced for every mask, including `same_library`
[_neighbourhood_correlation.py:829–849](src/regularizedvi/plt/_neighbourhood_correlation.py#L829) iterates over all masks. So `corr_norm_by_library_same_library = corr_avg_same_library / corr_avg_same_library = 1` by construction. Sub-plan 04 §"Test cases" treats this as a sanity check, not a useful output. Eval notebook does not filter the column, so the output parquet contains a constant-1 column. Not in main plan tables. Cosmetic but adds noise to UMAP grids.

### 3.11 Sub-plan 11 / CLAUDE.md update — purpose-based-key transition only partial
- Sub-plan 11 added purpose-based-key section to CLAUDE.md (verified — lines 17–30 of CLAUDE.md visible).
- Main plan declares `_neighbourhood_correlation.py` "uses `library_key`/`dataset_key`/`technical_covariate_keys` exclusively".
- But [_neighbourhood_correlation.py:2469](src/regularizedvi/plt/_neighbourhood_correlation.py#L2469) has `library_key="batch", dataset_key="dataset"` defaults inside `plot_per_library_distributions` — in clear violation of the stated convention.

### 3.12 Sub-plan 09 V1 default columns — implementation broader than plan, eval notebook narrower
- Main plan §V1: "*UMAP / Colored by each per-cell metric + decision tree leaf*".
- Sub-plan 09 §"Review Correction 1": "include all `corr_avg_{mask}, corr_std_{mask}, corr_discrepancy_{mask}, n_neighbours_{mask}, corr_norm_by_library_{mask}, corr_norm_by_all_{mask}`".
- Implementation default selects all of those ([:2160–2173](src/regularizedvi/plt/_neighbourhood_correlation.py#L2160)).
- Eval notebook calls with `columns=["corr_avg_same_library"]` — i.e. one panel only.

The plan-design intent (broad multi-panel V1) is functional in the implementation, but the only consumer (the eval notebook) bypasses it. Result: V1 in production output is one panel per model.

### 3.13 Sub-plan 06 severity ordering "tentative" but hard-coded as the only ordering
Sub-plan 06 §"Review Correction 7": "*The subjective orderings above are TENTATIVE — open to adjustment based on what correlates with model rank in scIB benchmarks.*" Implementation hard-codes `_WL_SEVERITY` / `_XL_SEVERITY` / `_XD_SEVERITY` dicts at module level ([:1078–1119](src/regularizedvi/plt/_neighbourhood_correlation.py#L1078)) with no parameter override. Re-ranking requires editing source.

### 3.14 Curated CSV `category_col` defaults differ between plan and code
- Main plan §"Multiple gene groups for decision tree Dimension 6" (lines 108–119): `per_category_markers` from `category` column.
- Sub-plan 02 §"Review Correction 3": `category_col: str = "category"`.
- Implementation: `category_col="category"` ([:175](src/regularizedvi/plt/_neighbourhood_correlation.py#L175)) ✓ matches.
- But [docs/notebooks/known_marker_genes.csv](docs/notebooks/known_marker_genes.csv) actually has columns `gene, cell_type, lineage, category` (per CLAUDE.md). I haven't verified the live file, but the eval notebook does not assert column presence, and `select_marker_genes` only emits `per_category_markers` if `category_col in marker_df.columns` ([:207](src/regularizedvi/plt/_neighbourhood_correlation.py#L207)). If the column is renamed in a future CSV update, gene groups silently disappear.

---

## Summary table

| Severity | Count |
|----------|-------|
| Plan-level (§1) | 12 |
| Implementation-level (§2) | 16 |
| Sub-plan ↔ plan ↔ impl drift (§3) | 14 |
| **Total findings** | **42** |

### Highest-impact fixes (priority order)

1. **§2.1** — Remove `WL-2_merged_related` from `wl_was_ideal`. Single-line fix, but materially changes XL/XD leaf distributions.
2. **§2.2** — Pass `technical_covariate_keys` through `compute_random_knn_baseline`. Required for H14 to have any random reference.
3. **§1.3 / §3.3** — Decide and document: how is XD-0a vs XD-0b assigned in the first model run? Either rerun classification after `cross_model_df` is assembled, or change the plan to admit a 3-state classification.
4. **§1.7** — Resolve the H6/H10 sign convention contradiction in the plan; remove or fix.
5. **§2.5** — `compute_isolation_norm` consistency: one population for both numerator and denominator.
6. **§3.1, §3.4, §3.5** — Update sub-plans 04 and 08 to match the implementation reality (or remove the implementation's `between_libraries`); update sub-plan 08 to remove the deleted `compute_tissue_group_integration` description.
7. **§1.6** — Add a `min_nonzero_markers` filter before computing per-cell Pearson; flag affected cells.
8. **§1.5** — Gate `best_achievable` (H13) by `above_random_xd` to avoid rewarding overconfident bad integrations.
