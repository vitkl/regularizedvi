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

## Model configuration

- **Covariates**: ambient=["batch"], nn_conditioning=["dataset","donor"], feature_scaling=["dataset","donor"]
- **Architecture**: n_hidden=512, n_layers=1, n_latent=128
- **Training**: max_epochs=2000, ES delta=0.0002, patience=20, batch_size=1024
- **Cell filtering**: 1.5k-80k counts, 800-10k genes, MT<0.20, doublet<0.20, ATAC 2.5k-80k
