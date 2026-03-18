# Immune Integration: 7-dataset RNA + ATAC

Multi-dataset integration of human immune cells using `AmbientRegularizedSCVI` (RNA-only) and `RegularizedMultimodalVI` (RNA+ATAC).

## Datasets

| Dataset | Tissue | Batches | Cells | Genes (raw) | Med counts | Med genes | Med MT | Annotated | GEO |
|---------|--------|---------|------:|-------------|----------:|----------:|-------:|----------:|-----|
| Bone marrow (NeurIPS 2021) | bone_marrow | 13 | 69,247 | 25,629 | 1,152 | 810 | 0.006 | 100% | — |
| TEA-seq PBMC | pbmc | 7 | 59,151 | 36,601 | 2,026 | 1,095 | 0.138 | 44.2% | GSE158013 |
| NEAT-seq CD4 T | sorted_cd4t | 2 | 8,457 | 36,717 | 1,998 | 1,143 | 0.011 | 100% | GSE178707 |
| Crohn's PBMC | pbmc | 13 | 106,296 | 36,601 | 2,379 | 1,322 | 0.097 | 65.1% | GSE244831 |
| COVID infant PBMC | pbmc | 43 | 360,624 | 36,601 | 4,112 | 1,915 | 0.064 | 0% | GSE239799 |
| Lung/Spleen (SMO only) | lung, spleen | 9 | 61,821 | 36,601 | 2,525 | 1,209 | 0.074 | 32.8% | GSE319044 |
| Infant/Adult Spleen | spleen | 5 | 40,726 | 36,601 | 2,799 | 1,405 | 0.074 | 0% | GSE311423 |

**After gene intersection**: 25,629 common genes (limited by bone marrow source h5ad).
**After library QC filter** (drop batches with <10% cells above 1500 counts): 7 COB samples from GSE319044 removed (failed GEX libraries, median counts 11-285).
**Total saved**: 706,322 cells x 25,629 genes, 92 batches, 92 donors, 49 cell types.

### NB1 loading times

| Step | Time |
|------|------|
| bone_marrow | 38s |
| tea_seq | 21s |
| neat_seq | 1s |
| crohns | 54s |
| covid | 218s |
| lung_spleen | 46s |
| spleen311 | 17s |
| Gene intersection + concat | 68s |
| QC metrics | 17s |
| Library filter + save | 202s |
| **Total** | **~11.5 min** |

## Notebooks

| # | Notebook | Purpose | Environment |
|---|----------|---------|-------------|
| 1 | `bm_pbmc_data_loading.ipynb` | Load 7 datasets, harmonize metadata, compute RNA QC | regularizedvi |
| 2 | `bm_pbmc_scrublet.ipynb` | Per-batch doublet detection (Scrublet) | regularizedvi |
| 3 | `bm_pbmc_atac_loading.ipynb` | Load ATAC tiles from SnapATAC2 cached h5ad | cell2state_v2026_cuda124_torch25 |
| 5 | `bm_pbmc_rna_training.ipynb` | Train AmbientRegularizedSCVI (RNA-only) | regularizedvi (GPU) |
| 5.5A | `bm_pbmc_annotation_transfer.ipynb` | Hierarchical annotation transfer via Leiden clustering | regularizedvi |
| 5.5B | `bm_pbmc_cre_selection.ipynb` | CRE selection using affinity analysis (draft) | regularizedvi |
| 6 | `bm_pbmc_multimodal_training.ipynb` | Train RegularizedMultimodalVI (RNA+ATAC) | regularizedvi (GPU) |

## Execution order

```
NB1 (RNA loading)
 ├── NB2 (Scrublet) ──────────────────────────────┐
 └── SnapATAC2 jobs (99 samples) ─────────────────┤
                                                   ↓
NB3 (ATAC loading) ───────────────────────────────┐
NB5 (RNA training, GPU) ──────────────────────────┤ parallel
                                                   ↓
NB5.5A (annotation transfer) ─────────────────────┐
NB5.5B (CRE selection) ───────────────────────────┘
                                                   ↓
NB6 (multimodal training, GPU)
```

## Key files

- `data_loading_utils.py` — 7 dataset loaders with validation, annotation harmonization
- `annotation_hierarchy.md` — 4-level cell type hierarchy (68 types)
- `annotation_harmonization.md` — per-dataset label → harmonized name mappings
- `metadata_harmonization.md` — covariate definitions and per-dataset mappings
- `PLAN.md` — detailed plan with progress and next steps

## RNA training template

**Template notebook**: `bm_pbmc_rna_training_v2.ipynb` — parameterized via papermill. All RNA training experiments use this template with different `-r` overrides.

**scVI baseline template**: `bm_pbmc_rna_training_scvi_v2.ipynb` — scVI (128h/2L/30z/zinb) comparison.

### Default model configuration

- **Covariates**: ambient=["batch"], nn_conditioning=["dataset","donor"], feature_scaling=["dataset","donor"]
- **Architecture**: n_hidden=1024, n_layers=1, n_latent=256
- **Training**: max_epochs=2000, ES delta=0.0002/feature, patience=10, batch_size=1024
- **Validation**: stratified by harmonized_annotation+batch
- **Library**: learned (not observed), centering_sensitivity=1.0, log_vars_weight=0.5
- **Background**: additive, Gamma(1,100) prior (mean=0.01), not regularised
- **Feature scaling**: Gamma(200,200) prior (mean≈1.0), regularised
- **Dispersion**: hierarchical variational LogNormal

### Output naming convention

- `*_out.ipynb` — full training + plots (may fail at plot stage, model saved regardless)
- `*_plot_out.ipynb` — plot-only rerun with `skip_training=1` (loads saved model)
- `*_plot_out_lite.ipynb` — plot notebook with marker gene cell outputs stripped (smaller file)

## RNA training experiments

### Round 1: Model size, QC strategy, library variance (2026-03-15)

Testing the effect of model capacity, QC filtering strategy, and library prior variance on integration quality across 706k cells from 7 immune datasets.

**QC strategies**:
- `adaptive` — per-dataset quantile (q20) + lenient global floor
- `compound` — study-specific cutoffs from domain knowledge

| # | Name | Output | QC | n_hidden | n_latent | libvar | bg_prior | Notes |
|---|------|--------|----|----------|----------|--------|----------|-------|
| 1 | xlarge | `*_xlarge_plot_out` | adaptive | 1600 | 400 | 0.5 | Gamma(1,100) | Large model, adaptive QC |
| 2 | libvar1 | `*_libvar1_plot_out` | adaptive | 1024 | 256 | 1.0 | Gamma(1,100) | Wider library prior |
| 3 | studyqc | `*_studyqc_plot_out` | compound | 1024 | 256 | 0.5 | Gamma(1,100) | Study-specific QC baseline |
| 4 | studyqc_xlarge | `*_studyqc_xlarge_plot_out` | compound | 1600 | 400 | 0.5 | Gamma(1,100) | Large model + compound QC |
| 5 | studyqc_xl_lv1 | `*_studyqc_xlarge_libvar1_plot_out` | compound | 1600 | 400 | 1.0 | Gamma(1,100) | Large + compound + wide libvar |
| 6 | default | `*_default_plot_out` | adaptive | 1024 | 256 | 0.5 | Gamma(1,100) | Default settings baseline |
| 7 | small | `*_small_plot_out` | compound | 512 | 128 | 0.5 | Gamma(1,100) | Small model |
| 8 | scvi_baseline | `*_scvi_plot_out` | compound | 128 (2L) | 30 | n/a | n/a | scVI zinb baseline |
| 9 | small_libvar1 | `*_small_libvar1_plot_out` | compound | 512 | 128 | 1.0 | Gamma(1,100) | Small + wide libvar |
| 10 | bg_flat | `*_bg_flat_plot_out` | compound | 1024 | 256 | 0.5 | Gamma(1,1) | Flat bg prior (mean=1.0) |

### Round 2: Decoder regularization and data-informed initialization (2026-03-17)

Testing decoder weight L2 penalty and data-dependent initialization of decoder bias and background parameters. The hypothesis is that informing the decoder about mean expression levels and initializing background proportional to per-gene expression helps the model converge to better solutions, especially for rare cell types that would otherwise be dominated by ambient RNA.

All experiments use compound QC, 1024h/256z, libvar=0.5.

**Hyperparameter axes**:
- `decoder_weight_l2` — L2 penalty on decoder weight matrices (not biases). Acts as a Gaussian prior on weights, preventing individual neurons from dominating. Values: 0, 0.01, 0.1, 1.0.
- `init_decoder_bias` — Initialize decoder output bias from normalized mean expression (`mean`), so softplus(bias) ≈ mean gene expression. Gives the model a reasonable starting point instead of random.
- `bg_init_gene_fraction` — Initialize per-gene background at `fraction × mean_expression` per ambient category. Default 0.2 means background starts at 20% of each batch's mean expression.
- `feature_scaling_prior` — Gamma(α,β) prior on feature scaling factors. Default Gamma(200,200) strongly regularises toward 1.0. Gamma(5,5) is much more permissive, allowing larger per-dataset expression differences.

| # | Name | Output | dwl2 | bias_init | bg_init | fs_prior | Notes |
|---|------|--------|------|-----------|---------|----------|-------|
| 1 | dwl2_001 | `*_dwl2_001_plot_out` | 0.01 | mean | 0.2 | 200/200 | Mild L2 + data-informed init |
| 2 | dwl2_01 | `*_dwl2_01_plot_out` | 0.1 | mean | 0.2 | 200/200 | Moderate L2 + data-informed init |
| 3 | bias_bg | `*_baseline_with_bias_bg_plot_out` | 0 | mean | 0.2 | 200/200 | Data-informed init only, no L2 |
| 4 | no_init | `*_baseline_without_bias_bg_plot_out` | 0 | None | None | 200/200 | No init, no L2 (pure baseline) |
| 5 | dwl2_1 | `*_dwl2_1_plot_out` | 1.0 | mean | 0.2 | 200/200 | Strong L2 + data-informed init |
| 6 | dwl2_01_fs5 | `*_dwl2_01_fsprior5_plot_out` | 0.1 | mean | 0.2 | **5/5** | Moderate L2 + relaxed feature scaling |
| 7 | dwl2_01_ni_fs5 | `*_dwl2_01_no_init_fsprior5_plot_out` | 0.1 | None | None | **5/5** | L2 + relaxed scaling, no data init |

### Round 2b: Per-batch background initialization (2026-03-18)

Rerun of dwl2_01 with per-batch background initialization (bg_init_gene_fraction computes separate mean expression per ambient covariate category, matching the forward-pass one-hot decomposition).

| # | Name | Output | dwl2 | bias_init | bg_init | Notes |
|---|------|--------|------|-----------|---------|-------|
| 8 | dwl2_01_batchbg | `*_dwl2_01_bias_mean_batchbg02_out` | 0.1 | mean | 0.2 (per-batch) | Per-batch bg init |
