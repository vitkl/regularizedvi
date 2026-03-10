# Immune Integration: 7-dataset RNA + ATAC

Multi-dataset integration of human immune cells using `AmbientRegularizedSCVI` (RNA-only) and `RegularizedMultimodalVI` (RNA+ATAC).

## Datasets

| Dataset | Tissue | Batches | Approx. cells | Annotations | GEO |
|---------|--------|---------|---------------|-------------|-----|
| Bone marrow (NeurIPS 2021) | bone_marrow | 13 | 69k | l2_cell_type | — |
| TEA-seq PBMC | pbmc | 7 | 59k | predicted.celltype.l2 (partial) | GSE158013 |
| NEAT-seq CD4 T | sorted_cd4t | 2 | 8.5k | Clusters C1-C7 | GSE178707 |
| Crohn's PBMC | pbmc | 13 | 76k | Celltypes | GSE244831 |
| COVID infant PBMC | pbmc | 43 | ~460k | None | GSE239799 |
| Lung/Spleen | lung, spleen | 16 | 54k | CellType | GSE319044 |
| Infant/Adult Spleen | spleen | 5 | ~50k | None | GSE311423 |

**Total**: ~777k cells, 99 batches, 7 datasets, 5 tissues.

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
- **Cell filtering**: 1k-80k counts, 500-10k genes, MT<0.20, doublet<0.18, ATAC 2k-100k
