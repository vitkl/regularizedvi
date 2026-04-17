# Cell-Level Neighbourhood Marker Gene Correlation Metrics

## Context

Current integration assessment (scIB, LISI, kBET, silhouette) uses **labels or latent distances** but never checks whether neighbours actually express the same genes. MetaNeighbor uses gene expression correlation but aggregates to cell-type level. Recent work (Nat Biotech 2025) showed silhouette-based metrics can mislead.

**This proposal fills the gap**: per-cell marker gene expression correlation with neighbours, stratified by covariate relationships, distinguishing positive vs negative integration failure modes. No published method does this.

**Related work**: scIB (Luecken et al. Nat Methods 2022), LISI/Harmony (Korsunsky et al. 2019), CellMixS (Luecken et al. LSA 2021), kBET (Buttner et al. 2018), MetaNeighbor (Crow et al. Nat Comms 2018), Silhouette shortcomings (Nat Biotech 2025).

**User feedback file**: [sprightly-prancing-cray-user-feedback.md](sprightly-prancing-cray-user-feedback.md) — all user comments with line quotations from original plan.

---

## Three-Level Covariate Hierarchy

Replaces all uses of "batch" and "dataset" with purpose-based terminology.

### Level definitions

| Level | Semantics | Expectation | Required? |
|-------|-----------|-------------|-----------|
| **Library** | Finest technical unit (sequencing run, lane, GEM well). Shares ambient RNA, library size, dispersion. | Cells represented by at least one other library — but NO expectation of matches across all libraries. Major difference from scIB. | Always required |
| **Dataset** | Groups of libraries from same study/experiment/section. Shares protocol and labelling. Libraries nest strictly within datasets. | Many cells unique to one dataset, but at least some overlap expected. When KNN is constructed, cross-dataset neighbours should be genuine matches, not wrong cell type or semi-random. | Optional |
| **Technical** | Broad technical axis (embryo, experiment type, 10x kit). Can intersect non-hierarchically with library/dataset. | Almost all cells expected to find neighbours. Better reference for expected correlations than dataset. Not all-with-all, but all expected to have some correct matches. | Optional, multiple allowed |

### Column mappings

| Level | Immune integration | Embryo |
|-------|-------------------|--------|
| Library | `batch` | `sample_id` |
| Dataset | `dataset` (7 datasets) | `Section` |
| Technical | *(not defined yet, could be `tissue`)* | `Embryo`, `Experiment` |

### Masks (no abbreviations)

| Mask | Definition |
|------|------------|
| `same_library` | Neighbour shares same library value |
| `cross_library` | Different library, same dataset |
| `cross_dataset` | Different dataset |
| `within_{technical_name}` | Same technical covariate value (when provided) |
| `between_{technical_name}` | Different technical covariate value |

**Hierarchy validation**: At setup, verify each library maps to exactly one dataset. `ValueError` if violated.

**Graceful degradation**: When dataset not provided, only `same_library` and `between_libraries` available. When technical not provided, those masks are absent. Missing masks produce no output columns (not NaN columns).

### Function signature

```python
compute_marker_correlation(
    adata, connectivities, marker_genes,
    library_key: str,                           # Required
    dataset_key: str | None = None,             # Optional
    technical_covariate_keys: list[str] | None = None,  # Optional, multiple
)
```

---

## Part 1: Neighbour Distribution Diagnostics

**Input**: `connectivities_euclidean_k50.npz` (sparse CSR, one per model)

1. Per-cell degree (nnz per row) — histogram. Flag cells with degree >> k (incoming edges from symmetrisation).
2. Cross-tabulate high-degree cells with obs columns — compute mean of all numeric obs columns for high-degree vs normal cells.
3. **Per-cell composition** (applied to ALL 3 covariate levels): fraction of neighbours from each library value, each dataset value, and each technical covariate value. Produces multiple columns, one per unique value at each level.
4. **Integration penetration** (applied to ALL 3 covariate levels): fraction of cells with >=10 and >=25 cross-library / cross-dataset / cross-technical neighbours, stratified by the respective covariate. Thresholds 10/25 given k=50.

---

## Part 2: Marker Gene Neighbourhood Correlation

### Normalisation (CORRECTED)

```python
# Keep sparse throughout via element-wise multiply with column vector broadcast
X_normalised = X.multiply(n_vars / total_counts[:, None])  # (n_cells, n_vars) sparse
```

Formula: `normalised = count * (n_vars / total_count)`. Sets per-gene average to 1. Pearson correlation on this normalised data, **NOT log-transformed**.

**Keep sparse throughout**: `compute_cluster_averages` already supports sparse. Subsetting to marker genes keeps the matrix sparse. Only densify when a dense matrix is needed for per-cell correlation (at that point subsetting to ~180 markers gives a dense ~300 MB float32).

### Gene Selection

#### Approach 1: Curated markers
`known_marker_genes.csv` (~192 genes, ~170-180 unique after dedup). Columns: gene, cell_type, lineage, category.

#### Approach 2: Data-driven specificity scores

**Copy `compute_cluster_averages`** from `/nfs/team205/vk7/sanger_projects/BayraktarLab/cell2location/cell2location/cluster_averages/cluster_averages.py` into the new module. Do NOT reimplement.

**Per-dataset gene selection** (applied separately to each dataset due to labelling disagreements):

1. Normalise: `count * (n_vars / total_count)`
2. For each label column (level_1, level_2, level_3, level_4, harmonized_annotation):
   - `compute_cluster_averages(adata_ds, labels=label_col)` → DataFrame (n_genes × n_labels)
3. Concatenate averages across label columns into one DataFrame
4. Specificity: `mean_per_gene_per_label / mean_per_gene_per_label.sum(axis=0, keepdims=True)` (sum across labels per gene)
5. Filter: `absolute_mean > mean_threshold` AND `specificity > specificity_threshold` (hyperparameters, defaults 1.0 and 0.1)
6. Union across datasets. Report gene counts, per-dataset overlap, overlap with curated markers.

**Both curated and data-driven sets used simultaneously** — not sequential.

#### Multiple gene groups (for decision tree Dimension 6)

Split markers by annotation level for multi-resolution analysis:

| Gene group | Source | Purpose |
|------------|--------|---------|
| `all_markers` | Union of curated + data-driven | Primary analysis |
| `broad_lineage_markers` | Specific at level_2/level_3 | Cross-dataset integration (most robust) |
| `cell_type_markers` | Specific at `harmonized_annotation` (informed by most detailed annotations across datasets) | Within-dataset bio preservation, fine-grained |
| `subtype_markers` | Specific at `harmonized_annotation` finest resolution | Fine-grained analysis |
| `per_category_markers` | From curated CSV `category` column | Per-lineage correlation (T cell, B cell, etc.) |

Comparing correlations across gene groups disambiguates: "broadly correct lineage but wrong subtype" vs "completely wrong cell type" vs "noisy cell".

```python
def select_marker_genes(
    adata,
    label_columns: list[str],
    dataset_col: str = "dataset",
    mean_threshold: float = 1.0,
    specificity_threshold: float = 0.1,
    curated_marker_csv: str | Path | None = None,
    per_dataset: bool = True,
    return_per_level: bool = True,
) -> dict[str, pd.Index]:
```

---

### Computation: Two Approaches (BOTH required)

**Approach B (average-then-correlate) — fast, inherently weighted**
- Per cell, per mask: weighted average of neighbour profiles (using connectivity weights) → single Pearson correlation
- Output: `corr_avg_{mask}` — one column per mask
- ~2-5 min runtime

**Approach A (correlate-then-aggregate) — rich, both weighted and unweighted**
- Per cell, per mask: Pearson correlation to each individual neighbour → aggregate statistics
- Unweighted: `corr_mean_{mask}`, `corr_median_{mask}`
- Weighted (by connectivity): `corr_weighted_mean_{mask}`, `corr_weighted_median_{mask}`
- Distribution: `corr_std_{mask}`, `corr_cv_{mask}` (max-gap dropped — std/cv + decision tree + multiple gene groups handle multimodality better)
- ~10-30 min runtime (batched matrix ops)

**Derived metric: A-vs-B discrepancy**
- `corr_discrepancy_{mask} = corr_avg_{mask} - corr_mean_{mask}`
- Large positive: averaging masks heterogeneity — "looks integrated but mixes distinct populations". Could indicate debris/multiplets with excessive n_neighbours.
- Near zero: homogeneous neighbourhood.
- Negative: rare — distinct equally-correlated subtypes cancel in average. Multiple gene groups informative here.

**Also compute base (no mask)**: `corr_avg_all_neighbours`, `corr_weighted_mean_all_neighbours` — for normalisation Setting B.

### Missing neighbours — conceptual model (CORRECTED)

Missing neighbours = zero-weight connections, NOT missing data. The connectivity matrix is sparse representation of a dense relationship — non-neighbours have weight 0.

- **Weighted metrics**: zero neighbours → all weights zero → undefined (NaN as computational consequence)
- **Track `n_neighbours_{mask}`** per cell — this is the key diagnostic
- **Isolation fraction**: computed per mask (not just cross-dataset), normalised by random baseline

### Per-Cell Output Columns

**Per-mask columns** (for each mask in {same_library, cross_library, cross_dataset, within_{tech}, between_{tech}}):

| Column | Source | Description |
|--------|--------|-------------|
| `n_neighbours_{mask}` | graph | Count of neighbours in this mask |
| `frac_neighbours_{mask}` | graph | Fraction of total neighbours |
| `corr_avg_{mask}` | Approach B | Pearson(cell, weighted-mean-neighbours) |
| `corr_mean_{mask}` | Approach A | Unweighted mean of per-neighbour correlations |
| `corr_median_{mask}` | Approach A | Unweighted median |
| `corr_weighted_mean_{mask}` | Approach A | Connectivity-weighted mean |
| `corr_weighted_median_{mask}` | Approach A | Connectivity-weighted median |
| `corr_std_{mask}` | Approach A | Std of per-neighbour correlations |
| `corr_cv_{mask}` | Approach A | Coefficient of variation |
| `corr_discrepancy_{mask}` | A vs B | corr_avg - corr_mean |

**Derived normalised columns** (per mask):

| Column | Formula | Interpretation |
|--------|---------|----------------|
| `corr_norm_by_library_{mask}` | `corr_avg_{mask} / corr_avg_same_library` | Relative to same-library baseline (1.0 = same quality) |
| `corr_norm_by_all_{mask}` | `corr_avg_{mask} / corr_avg_all_neighbours` | Relative to unmasked average |

**Base columns** (mask-independent):

| Column | Description |
|--------|-------------|
| `corr_avg_all_neighbours` | Approach B on all neighbours (no mask) |
| `corr_weighted_mean_all_neighbours` | Approach A weighted mean on all neighbours |
| `n_neighbours_total` | Total non-self neighbours |
| `marker_gene_total_expression` | Sum of normalised expression across markers |

---

## Decision Tree: Failure Mode Classification

Replaces the old 2x2 table. Per-cell classification using 6 dimensions, producing 21 leaves.

### Six dimensions (evaluated in order)

| Dim | Variable | Splits |
|-----|----------|--------|
| 1 | Stratification level | within-library → cross-library → cross-dataset (sequential, conditioned) |
| 2 | Has neighbours? | yes / no — **critical axis separating "no connection" from "wrong connection"** |
| 3 | Correlation level | high / low (relative to within-library reference) |
| 4 | Correlation variability | homogeneous / mixed (from corr_std) |
| 5 | Random KNN baseline | above / at-or-below expected |
| 6 | Marker gene group comparison | broad vs specific markers agree / disagree |

### Random KNN baseline (Dimension 5)

Create random KNN with same N neighbours per cell (variable N to match actual per-cell degree). Correlations across genes can be high — random gene sets are the wrong comparison dimension. Random KNN tests whether the model's neighbourhood is better than chance assignment.

**Analytical shortcut**: For isolation fraction, `P(isolated_cross_dataset | random, k_i) = ((n_same_dataset - 1) / (n_total - 1))^k_i` (self-exclusion: k_i neighbours drawn without replacement from the other n_total-1 cells).

### Complete decision tree

```
WITHIN-LIBRARY LEVEL (establishes reference baseline)
=====================================================
[DIM 2] Has same-library neighbours?
├── NO → LEAF WL-0: "Orphan cell" (isolated even within own library — check QC)
└── YES
    └── [DIM 3] Within-library correlation level?
        ├── HIGH
        │   └── [DIM 4] Variability?
        │       ├── HOMOGENEOUS → LEAF WL-1: "Ideal within-library" (REFERENCE for deeper levels)
        │       └── MIXED → LEAF WL-2: "Merged related types" (broad markers agree, specific disagree)
        └── LOW
            └── [DIM 5] vs Random KNN baseline?
                ├── AT/BELOW random → LEAF WL-3: "Noisy/low-quality cell"
                └── ABOVE random
                    └── [DIM 4] Variability?
                        ├── HOMOGENEOUS → LEAF WL-4: "Bio positive failure (confident false merge)"
                        │   [DIM 6] broad LOW + specific LOW = cross-lineage; broad HIGH + specific LOW = within-lineage
                        └── MIXED → LEAF WL-5: "Bio positive failure (partial)"

CROSS-LIBRARY LEVEL (conditioned on within-library result)
==========================================================
[DIM 2] Has cross-library neighbours?
├── NO
│   ├── Within-library IDEAL → LEAF XL-0a: "Under-integration (cross-library)"
│   └── Within-library FAILED → LEAF XL-0b: "Compounded failure"
└── YES
    └── [DIM 3] Cross-library correlation (relative to within-library)?
        ├── HIGH
        │   ├── HOMOGENEOUS → LEAF XL-1: "Ideal cross-library integration"
        │   └── MIXED → LEAF XL-2: "Partial cross-library integration"
        └── LOW
            ├── Within-library was HIGH
            │   ├── HOMOGENEOUS → LEAF XL-3: "Library positive failure (wrong pairing)"
            │   │   [DIM 6] broad HIGH + specific LOW = same lineage different subtype
            │   └── MIXED → LEAF XL-4: "Bio negative failure (forced distinct populations)"
            │       Interpretation: either forces distinct populations together (informative for covariate design)
            │       or finds irrelevant cells semi-randomly (multiple gene groups disambiguate)
            └── Within-library was LOW → LEAF XL-5: "Poor model (both levels fail)"

CROSS-DATASET LEVEL (conditioned on within-library AND cross-library)
=====================================================================
[DIM 2] Has cross-dataset neighbours?
├── NO
│   ├── Cross-library IDEAL
│   │   └── [MODEL COMPARISON] Do alternative models connect this cell?
│   │       ├── No model connects → LEAF XD-0a: "Dataset-enriched cell type (likely real)"
│   │       │   NOT a failure. Biologically informative. Exclude from cross-dataset benchmark.
│   │       └── Some models connect → LEAF XD-0b: "Under-integration (cross-dataset)"
│   │           True failure — other models prove matching IS possible.
│   ├── Cross-library UNDER-INTEGRATED → LEAF XD-0c: "Cascaded under-integration"
│   └── Cross-library FAILED → LEAF XD-0d: "Compounded failure"
└── YES
    └── [DIM 3] Cross-dataset correlation (relative to within-library)?
        ├── HIGH
        │   ├── [DIM 5] vs Random KNN?
        │   │   ├── ABOVE random
        │   │   │   ├── HOMOGENEOUS → LEAF XD-1: "Ideal cross-dataset integration" ✓
        │   │   │   └── MIXED → LEAF XD-3: "Partial cross-dataset integration"
        │   │   └── AT/BELOW random → LEAF XD-2: "Spurious high correlation"
        └── LOW (= POSITIVE FAILURE, not under-integration)
            ├── Within-library HIGH
            │   ├── HOMOGENEOUS
            │   │   ├── Cross-library was HIGH → LEAF XD-4a: "Dataset positive failure (wrong pairing)"
            │   │   └── Cross-library was LOW → LEAF XD-4b: "Systematic positive failure (all cross-levels)"
            │   └── MIXED
            │       └── [DIM 6]
            │           ├── Broad HIGH, Specific LOW → LEAF XD-5a: "Bio negative failure (forced merge of related types)"
            │           │   Biologically informative outcome — populations need stratified analysis or different covariates
            │           └── Both LOW/MIXED → LEAF XD-5b: "Semi-random cross-dataset pairing"
            └── Within-library LOW → LEAF XD-6: "Poor model (all levels fail)"
```

### Cross-cutting diagnostics (applied to any leaf)

**Discrepancy metric**: Large positive `corr_discrepancy` + excessive `n_neighbours` → suspect debris/multiplets.

**Model comparison** (essential for specific leaves):
- XD-0a vs XD-0b: Only model comparison distinguishes dataset-enriched from under-integration
- XD-4a/5a/5b: Alternative models achieving high correlation prove integration IS possible
- WL-4/WL-5: All models failing → data quality issue, not model problem

### Key distinction: no neighbours ≠ wrong neighbours

| Situation | Interpretation |
|-----------|----------------|
| No neighbours in mask | Under-integration / isolation — model never bridged |
| Neighbours with low correlation | Positive failure — model bridged INCORRECTLY |
| Neighbours with high correlation, high std | Mixed neighbourhood — some correct, some wrong |

This distinction is why `n_neighbours_{mask}` must be a separate axis from correlation values.

---

## Model Comparison for Integration Failure Detection

### The fundamental problem
For a single model, zero cross-dataset neighbours could be dataset-specific biology OR integration failure. Labels cannot resolve this (label-free analysis). **Model comparison can**.

### Methodology

**Step 1: Best-achievable envelope**
```
best_corr_cross_dataset[cell_i] = max over models M: corr_avg_cross_dataset_M[cell_i]
```
If ANY model achieves high cross-dataset correlation, cross-dataset matching IS possible.

**Step 2: 3×3 contingency per cell across model pairs**

| | Model B: high corr | Model B: low corr | Model B: no neighbours |
|---|---|---|---|
| Model A: high corr | Both succeed | A integrates, B mismatches | A integrates, B isolates |
| Model A: low corr | B integrates, A mismatches | Both mismatch | Both fail |
| Model A: no neighbours | B integrates, A isolates | Both fail differently | **Ambiguous** |

**Step 3: Integration failure rate**
```
integration_failure_rate[model] = fraction of cells where:
    best_corr_cross_dataset[cell] > threshold_high
    AND corr_avg_cross_dataset_model[cell] is NaN or < threshold_low
```

**Step 4: Distribution overlap**
OVL(f_A, f_B) = integral of min(f_A(x), f_B(x)) dx. OVL near 0 = "almost complete lack of overlap" = integration failure signal.

### Pure technical covariate as reference (H14)

Technical covariate keys (e.g. `tissue`, `experiment`, `10x_kit`) define groups between which integration is EXPECTED. Headline metric H14 `cross_technical_correlation` = median per-cell Pearson correlation on the `cross_technical` mask (neighbours differing in AT LEAST ONE technical covariate value). High H14 = cells are well-integrated across technical groups; low H14 = technical groups remain unmixed.

When multiple `technical_covariate_keys` are supplied, `cross_technical` is the union of `between_{tech_key}` masks.

For immune: tissue groups (PBMC: pbmc_tea_seq + crohns_pbmc + covid_pbmc; Spleen: lung_spleen + infant_adult_spleen).
For embryo: `Experiment` covariate.

### TEA-seq failure example
scVI baseline: TEA-seq cells have isolation rate ~80% (no cross-dataset PBMC neighbours).
regularizedVI: TEA-seq cells have isolation rate ~15% with high cross-dataset correlation.
Distribution overlap near 0 → definitive integration failure in scVI.

---

## Poor Quality Cell Detection

**Previous metric plans consulted**: `fuzzy-percolating-conway.md` (Latent Space QC, ~60 metrics). Three ideas extracted:

### Idea 1: Ambient RNA fraction (C4)
`ambient_frac = sum(s) / sum(rho + s)` — directly measures what debris IS. High for debris, low for rare cells. Computed from single forward pass (px_scale vs additive_background).

### Idea 2: Reconstruction perplexity (C2)
`recon_perplexity = exp(entropy of px_rate)` — effective number of genes in decoder output. Low for debris (narrow ambient profile), normal/high for rare cells. Depth-invariant.

### Idea 3: KNN-smoothed metrics (G1)
Average any metric over KNN neighbours. Debris near real-cell cluster: positive deviation from neighbours. Rare cell in coherent cluster: matches its neighbours.

### Multi-axis classification (replaces single score)

```
Step 1: corr_deviation = corr_avg_same_library - library_median(corr_avg_same_library)
        If corr_deviation >= 0 → cell matches peers → not flagged

Step 2: best_lineage_corr = max over lineage groups of corr_avg_same_library_lineage_L
        If best_lineage_corr > threshold → RARE CELL TYPE (matches one lineage well)

Step 3: If ambient_frac > threshold OR recon_perplexity < threshold → POOR QUALITY (debris)

Step 4: Otherwise → UNCERTAIN (flag for inspection, do not auto-remove)
```

**Per-cell output columns**:
- `quality_corr_deviation`, `quality_ambient_frac`, `quality_recon_perplexity`
- `quality_best_lineage_corr`, `quality_best_lineage` (which lineage group)
- `quality_local_ambient_deviation` (raw - KNN-smoothed ambient_frac)
- `quality_classification` (good / rare / poor_quality / uncertain)

**Dependency**: ambient_frac and recon_perplexity require `get_latent_qc_metrics()` from fuzzy-percolating-conway plan (Phase 1). Compute before neighbourhood correlation.

---

## Headline Metrics (for benchmarker heatmap)

### Level 1: Within-library (bio preservation)

| # | Metric | Formula | Direction |
|---|--------|---------|-----------|
| H1 | `corr_within_library` | median of corr_avg_same_library | higher = better |
| H2 | `corr_consistency` | median of corr_std_same_library | lower = better |

### Level 2: Cross-library (within-dataset integration)

| # | Metric | Formula | Direction |
|---|--------|---------|-----------|
| H3 | `corr_cross_library` | median of corr_avg_cross_library (cells with >=1 cross-library neighbours) | higher = better |
| H4 | `corr_gap_library` | H1 - H3 | lower = better |
| H5 | `isolation_norm_cross_library` | observed isolation / expected random isolation | lower = better (<1 = good) |
| H6 | `discrepancy_cross_library` | median of corr_discrepancy_cross_library | lower = better |

### Level 3: Cross-dataset (cross-study integration)

| # | Metric | Formula | Direction |
|---|--------|---------|-----------|
| H7 | `corr_cross_dataset` | median of corr_avg_cross_dataset (cells with >=1 cross-dataset neighbours) | higher = better |
| H8 | `corr_gap_dataset` | H1 - H7 | lower = better |
| H9 | `isolation_norm_cross_dataset` | observed / expected random isolation | lower = better |
| H10 | `discrepancy_cross_dataset` | median of corr_discrepancy_cross_dataset | lower = better |

### Distribution-level metrics

| # | Metric | Formula | Direction |
|---|--------|---------|-----------|
| H11 | `distrib_overlap_library` | OVL(corr_avg_same_library, corr_avg_cross_library) | higher = better |
| H12 | `distrib_overlap_dataset` | OVL(corr_avg_same_library, corr_avg_cross_dataset) | higher = better |

### Cross-model comparison (computed post-hoc)

| # | Metric | Formula | Direction |
|---|--------|---------|-----------|
| H13 | `integration_failure_rate` | Fraction of cells that CAN be integrated (per ensemble) but this model fails | lower = better |
| H14 | `cross_technical_correlation` | Median per-cell Pearson correlation on neighbours differing in at least one technical covariate value (union of `between_{tech}` masks) | higher = better |

### Isolation score normalisation

Raw isolation fraction does not reflect integration failure (could be dataset-specific biology). Normalise by random KNN baseline:

```
isolation_norm_{mask} = isolation_frac_{mask}(model) / isolation_frac_{mask}(random)
P(isolated | random, k_i) = ((n_same_group - 1) / (n_total - 1))^{k_i}  # analytical, with self-exclusion
```
Per-mask `n_same_group` varies (see isolation baseline table in implementation notes):
- `same_library` / `within_{tech}`: `n_same - 1` over `n_total - 1`
- `between_libraries` / `cross_technical`: `n_total - n_same` over `n_total - 1`
- `cross_library` (restricted to within-dataset): `n_dataset - n_lib` over `n_total - 1`
- `cross_dataset`: `n_total - n_dataset` over `n_total - 1`

Values <1: model integrates better than random. ~1: no benefit over random. >1: model actively segregates.

### Composite score

```
bio_conservation = corr_within_library

library_integration = 0.4 * corr_cross_library
                    + 0.3 * (1 - clamp(isolation_norm_cross_library, 0, 2))
                    + 0.3 * distrib_overlap_library

dataset_integration = 0.4 * corr_cross_dataset
                    + 0.3 * (1 - clamp(isolation_norm_cross_dataset, 0, 2))
                    + 0.3 * distrib_overlap_dataset

batch_correction = 0.5 * library_integration + 0.5 * dataset_integration

total = 0.6 * bio_conservation + 0.4 * batch_correction
```

When dataset level unavailable: `batch_correction = library_integration`.

### Stratified summaries
- By dataset, by library, by decision tree leaf, by technical group
- Per-library distribution comparison: overlap coefficient between peer libraries
- Percentiles (10th, 25th, 50th, 75th, 90th) of corr_avg for each mask

---

## Visualisations

All scatter comparisons use **hist2d**. Histograms replace violin plots.

| # | Type | Content |
|---|------|---------|
| V1 | UMAP | Colored by each per-cell metric + decision tree leaf |
| V2 | hist2d | corr_avg_same_library vs corr_avg_cross_library |
| V3 | hist2d | corr_avg_same_library vs corr_avg_cross_dataset |
| V4 | hist2d | corr_avg_cross_library vs corr_avg_cross_dataset |
| V5 | hist2d | corr_avg vs corr_mean per mask (shows discrepancy structure) |
| V6 | hist2d | corr_discrepancy_cross_dataset vs corr_avg_cross_dataset |
| V7 | histogram (overlaid) | Distribution overlap: same_library and cross_library per model |
| V8 | histogram (overlaid) | Distribution overlap: same_library and cross_dataset per model |
| V9 | histogram per library | corr_avg_cross_library distribution faceted by dataset |
| V10 | bar chart | Normalised isolation fraction per dataset per model |
| V11 | heatmap | Benchmarker heatmap with H1-H14 columns |
| V12 | UMAP | Decision tree leaf assignment (categorical coloring) |
| V13 | stacked bar | Decision tree leaf distribution per model |
| V14 | hist2d | corr_avg_same_library vs random baseline correlation |

---

## Comparison: Single-Dataset vs Multi-Dataset

Same-library and cross-library metrics directly comparable between:
- Bone marrow alone (single dataset, cross-library only)
- Immune integration (multi-dataset, all masks)

Cross-dataset mask empty for single-dataset → gracefully absent.

**Models to evaluate**: all 16 from z_init_sigma_jobs.tsv (8 BM + 8 immune).

**Embryo application**: sample_id as library, Section as dataset, Embryo/Experiment as technical covariates.

---

## Implementation Workflow

### Step 0: Copy this plan to project `.claude/plans/` reference directory

Before implementation begins, copy this plan and the feedback file to the project's `.claude/plans/` directory so all implementation subagents can reference them:

```bash
cp /nfs/users/nfs_v/vk7/.claude/plans/sprightly-prancing-cray.md \
   /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/.claude/plans/neighbourhood_correlation_plan.md

cp /nfs/users/nfs_v/vk7/.claude/plans/sprightly-prancing-cray-user-feedback.md \
   /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/.claude/plans/neighbourhood_correlation_user_feedback.md
```

This makes the plan available as reference context for implementation subagents so they understand what the metrics mean and why decisions were made.

### Implementation Sub-Plan Structure

Split the implementation into 9 sub-plans. Each sub-plan is assigned to ONE subagent that:
- Reads the ENTIRE main plan and user feedback (for full context)
- Reads any other sub-plan files with shared variables/dependencies (for coordination)
- Implements changes to **ONLY ONE primary file** (minimises merge conflicts)
- May read (not edit) other files needed for shared interfaces

**Sub-plan file location — IMPORTANT for subagent access**:

Implementation subagents spawned via the Agent tool work from the primary working directory `/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/` and may not have access to `/nfs/users/nfs_v/vk7/.claude/plans/`. All sub-plans and the main plan copy therefore live in the PROJECT directory:

```
/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/.claude/plans/
├── neighbourhood_correlation_plan.md                            (main plan copy)
├── neighbourhood_correlation_user_feedback.md                   (feedback with line quotations)
└── neighbourhood_correlation_subplan_NN_<name>.md               (11 sub-plans)
```

User-level plans at `/nfs/users/nfs_v/vk7/.claude/plans/` are maintained as the authoring source and synced to the project directory via `cp`. Implementation subagents should read from `.claude/plans/` (project-relative) only.

Each sub-plan contains:
- Primary file to modify
- Dependencies (shared variables / interfaces with other sub-plans)
- Function signatures with docstrings
- Detailed implementation steps
- Test cases
- Verification criteria

### Sub-Plan List

| # | Name | Primary file | Dependencies (read-only) |
|---|------|--------------|--------------------------|
| 1 | `core_utilities` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (module skeleton + sparse normalisation + `compute_cluster_averages` copy) | cell2location source |
| 2 | `gene_selection` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (add `select_marker_genes()`) | sub-plan 1 |
| 3 | `mask_construction` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (add covariate hierarchy validation + mask construction) | sub-plan 1 |
| 4 | `correlation_computation` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (Approach A + B, weighted/unweighted, derived normalised metrics) | sub-plans 1, 3 |
| 5 | `diagnostics_and_random_baseline` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (Part 1 diagnostics for all 3 covariates + random KNN baseline) | sub-plans 1, 3 |
| 6 | `decision_tree_classification` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (leaf assignment) | sub-plans 4, 5 |
| 7 | `summary_and_composite` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (headline metrics, isolation normalisation, composite score) | sub-plan 4, `_integration_metrics.py` |
| 8 | `model_comparison` | `src/regularizedvi/plt/_neighbourhood_correlation.py` (cross-model DataFrame, OVL, integration failure rate) | sub-plan 7 |
| 9 | `visualisation_and_heatmap` | `src/regularizedvi/plt/_integration_metrics.py` (extend benchmarker heatmap) + `_neighbourhood_correlation.py` (UMAP, hist2d plots) | sub-plans 6, 7, 8 |
| 10 | `evaluation_notebook` | `docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb` | all previous |
| 11 | `claude_md_update` | `CLAUDE.md` — add purpose-based covariate guidance | n/a |

Sub-plans 1-8 all modify the same file `_neighbourhood_correlation.py` sequentially (not in parallel) — each subagent adds new functions to a growing module. Sub-plans 9-11 can run in parallel after the core module is complete.

### Execution order
- Sequential: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 (same file)
- Parallel after 8: 9, 10, 11

## Implementation Architecture

- New module: `src/regularizedvi/plt/_neighbourhood_correlation.py`
- Functions:
  - `compute_cluster_averages()` — copied from cell2location
  - `select_marker_genes()` — data-driven gene selection with hyperparameters
  - `compute_neighbourhood_diagnostics()` — Part 1
  - `compute_marker_correlation()` — Part 2 main (both Approaches A and B)
  - `classify_failure_modes()` — decision tree leaf assignment
  - `compute_random_knn_baseline()` — random KNN for Dimension 5
  - `summarize_marker_correlation()` — headline scores
  - `compare_models_per_cell()` — cross-model comparison
  - `compute_integration_failure_rate()` — headline cross-model metric
  - `plot_marker_correlation_umap()` — UMAP viz
  - `plot_failure_mode_scatter()` — hist2d plots
- Register in `plt/__init__.py`
- Evaluation notebook: `docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb`

## Critical Files
- `src/regularizedvi/plt/_integration_metrics.py` — existing metrics + heatmap to extend
- `src/regularizedvi/plt/__init__.py` — register new functions
- `src/regularizedvi/plt/_dotplot.py` — gene symbol matching logic (lines 69-72)
- `docs/notebooks/known_marker_genes.csv` — curated marker genes with hierarchy
- `docs/notebooks/immune_integration/data_loading_utils.py` — data loading patterns
- `results/*/model/outputs/connectivities_euclidean_k50.npz` — input connectivity matrices
- Cell2location source: `/nfs/team205/vk7/sanger_projects/BayraktarLab/cell2location/cell2location/cluster_averages/cluster_averages.py`
- QC metrics dependency: `fuzzy-percolating-conway.md` (ambient_frac, recon_perplexity)

## CLAUDE.md Update
Add note to CLAUDE.md: purpose-based covariate keys (`library_key`, `dataset_key`, `technical_covariate_keys`) should be used instead of generic `batch_key` in new code. The neighbourhood correlation module uses this terminology; existing code (`_integration_metrics.py`) retains `batch_key` for backward compatibility.

## Verification
- Run on 2+ models with known good/bad integration (compare to existing scIB ranking)
- Verify isolation normalisation against analytical random formula
- Check marker gene CSV genes present in adata.var (use _dotplot.py symbol matching)
- Compare bone marrow vs immune integration (same-library/cross-library metrics should be comparable)
- Test with embryo data (different covariate hierarchy)
- Unit tests for masking, correlation, decision tree classification, random KNN baseline
