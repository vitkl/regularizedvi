# Sub-Plan 10: Evaluation Notebook

**Parent plan**: `neighbourhood_correlation_plan.md`

## Review Corrections (applied)

1. **Template path fix**: reference `docs/notebooks/immune_integration/recompute_integration_metrics.ipynb` (NOT `docs/notebooks/model_comparisons/`).
2. **scvi_baseline registry**: `z_init_sigma_jobs.tsv` does NOT contain scvi_baseline. For TEA-seq case study, add a second parameter:
   ```python
   extra_models_tsv = "docs/notebooks/immune_integration/integration_metrics_experiments.tsv"
   extra_models_filter = ["immune_integration_rna_scvi_baseline", "immune_integration_rna_scvi_baseline_v2"]
   ```
   Load both TSVs and merge registries.
3. **Results folder path pattern**: both `bm_mm_*` and `immune_large_ld_*` share `{name}/model/outputs/` layout. Construct as:
   ```python
   conn_path = Path(results_base) / row["name"] / "model" / "outputs" / "connectivities_euclidean_k50.npz"
   if not conn_path.exists():
       print(f"Skipping {row['name']} — no connectivity matrix"); continue
   ```
   Add `results_base: str = "results/"` as papermill parameter.
4. **Output directory creation**: add to Setup cell:
   ```python
   from pathlib import Path
   Path(output_dir).mkdir(parents=True, exist_ok=True)
   ```
5. **BM run as separate invocation**: bone marrow models require `dataset_key=None`. Do NOT branch inside one notebook; provide a second bsub command in the notebook header comment:
   ```
   # Immune: papermill ... -p dataset_key dataset -p output_dir .../results_neighbourhood_correlation/
   # BM:     papermill ... -p dataset_key None -p output_dir .../results_neighbourhood_correlation_bm/ -p models_filter "bm_mm_*"
   ```
6. **Gene group loop in `compute_marker_correlation`**: `select_marker_genes(return_per_level=True)` returns dict. Iterate gene groups and run `compute_marker_correlation` per group, prefixing output columns with group name (e.g., `corr_avg_cross_dataset_broad_lineage_markers`).
7. **Papermill parameters**: remove any `# papermill parameters` comment above the parameters cell if `_extract_params.py` is strict. Bare assignments only inside the cell.
8. **Embryo section**: keep as commented example with clear path placeholders. Separate full notebook is follow-up work, not blocker for v1.

## Primary file
`docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb` (CREATE)

## Dependencies
- Sub-plans 01–09 complete (full neighbourhood correlation module + heatmap)
- Read-only: `docs/notebooks/model_comparisons/recompute_integration_metrics.ipynb` (template for parameterised notebook structure with papermill)
- Read-only: `docs/notebooks/model_comparisons/z_init_sigma_jobs.tsv` (16 models to evaluate)

## Tasks

### 1. Notebook structure (parameterised with papermill)

**Parameters cell** (bare assignments, no inline comments — papermill constraint):
```python
# papermill parameters
adata_path = "..."
results_folder = "..."
models_tsv = "docs/notebooks/model_comparisons/z_init_sigma_jobs.tsv"
library_key = "batch"
dataset_key = "dataset"
technical_covariate_keys = None
marker_csv = "docs/notebooks/known_marker_genes.csv"
label_columns = ["level_1", "level_2", "level_3", "level_4", "harmonized_annotation"]
output_dir = "docs/notebooks/model_comparisons/results_neighbourhood_correlation/"
k_reference = 50
mean_threshold = 1.0
specificity_threshold = 0.1
```

### 2. Notebook sections

1. **Setup**: imports, load adata, load model registry from TSV
2. **Gene selection**:
   - Run `select_marker_genes()` with curated markers + data-driven
   - Report per-dataset gene counts and overlap
   - Visualise gene set overlaps (Venn-like)
3. **Per-model metrics loop**:
   - For each model in TSV:
     - Load connectivity matrix from `results_folder/.../outputs/connectivities_euclidean_k50.npz`
     - Run `compute_neighbourhood_diagnostics()`
     - Run `compute_marker_correlation()` (all gene groups)
     - Run `compute_random_knn_baseline()`
     - Run `classify_failure_modes()`
     - Save per-cell metrics as parquet: `{output_dir}/{model_name}_metrics.parquet`
     - Save per-cell leaves: `{output_dir}/{model_name}_leaves.parquet`
4. **Per-model summaries**:
   - Load all saved metrics
   - Run `summarise_marker_correlation()` per model → Series → combined DataFrame
   - Run `compute_composite_score()` per model
5. **Cross-model comparison**:
   - `assemble_cross_model_metrics()`
   - `compute_best_achievable()`, `flag_consensus_isolated()`
   - `compute_integration_failure_rate()` per model
   - `compute_model_pair_overlaps()`
6. **Visualisations**:
   - Benchmarker heatmap (V11) with H1-H14 columns
   - Per-model UMAP grids (V1, V12)
   - hist2d plots (V2-V6, V14)
   - Distribution overlap (V7, V8)
   - Per-library (V9)
   - Isolation bars (V10)
   - Leaf distribution (V13)
7. **TEA-seq failure case study**:
   - Restrict to TEA-seq cells
   - Compare scvi_baseline vs regularizedVI distributions (V7/V8 but filtered)
   - Compute integration failure rate restricted to TEA-seq — expect high for scvi_baseline
8. **Summary table**:
   - Export final per-model headline Series as CSV
   - Export composite score ranking

### 3. Single-dataset bone marrow comparison

Run same workflow on 8 BM models (from z_init_sigma_jobs.tsv) with:
- `library_key="batch"`
- `dataset_key=None`
- Only same_library and between_libraries masks available
- Compare BM within-library distributions with immune within-library distributions (should be comparable)

### 4. Embryo extension (optional section, commented)

Cell block showing how to apply to embryo:
```python
# For embryo data:
library_key = "sample_id"
dataset_key = "Section"
technical_covariate_keys = ["Embryo", "Experiment"]
```

### 5. Output files

Saved under `output_dir`:
- `{model_name}_metrics.parquet` — per-cell metrics per model
- `{model_name}_leaves.parquet` — per-cell leaf assignments per model
- `per_model_headline.csv` — H1-H14 per model
- `composite_scores.csv` — composite score per model
- `cross_model_overlap.csv` — pairwise OVL
- `consensus_isolated_cells.csv` — cell IDs flagged as consensus isolated
- PNG figures for each visualisation

### 6. LSF job submission command (in notebook comment)

Provide the bsub command mirroring `recompute_integration_metrics.ipynb`:
```
bsub -q normal -n 8 -M 100000 -R"select[mem>100000] rusage[mem=100000] span[hosts=1]" \
  -e ./%J.err -o ./%J.out -J nbhd_corr_metrics \
  'PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && \
   papermill docs/notebooks/model_comparisons/neighbourhood_correlation_metrics.ipynb \
             docs/notebooks/model_comparisons/neighbourhood_correlation_metrics_out.ipynb'
```

Memory: needs to hold dense markers matrix (~300 MB) + connectivities + per-model metrics → 100 GB safe.

## Verification
- Papermill-parameters parse correctly (use `_extract_params.py`)
- Notebook imports all new functions without error
- End-to-end run on 1-2 models succeeds
- Output files created in `output_dir`
