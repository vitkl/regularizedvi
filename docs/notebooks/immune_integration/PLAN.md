# Plan: Multi-dataset RNA + ATAC integration (7 datasets)

## Progress (updated 2026-03-09)

| Step | Status | Notes |
|------|--------|-------|
| Notebook 1: RNA loading | **DONE** | 776,742 cells x 25,629 genes, 99 batches, 99 donors, 38 cell types |
| Notebook 2: Scrublet | **RUNNING** | Started 2026-03-09, loading 13 GB h5ad |
| SnapATAC2 jobs | **SUBMITTED** | 99 bsub jobs on `normal` queue (cell2state scripts, `cell2state_v2026_cuda124_torch25` env) |
| Notebook 3: ATAC loading | PENDING | Waiting for SnapATAC2 jobs |
| Notebook 4: QC summary | PENDING | Waiting for notebooks 2+3 |
| Notebook 5: RNA training | PENDING | Waiting for notebook 4 |
| Notebook 6: Multimodal | PENDING | Waiting for notebook 5 |

### Known issues
- TEA-seq PBMC: 0/59k cells annotated (expected ~8k from GSM4949911 only) — annotation mapping needs investigation
- Crohn's PBMC: 69k/106k annotated (some samples lack annotations)
- Lung/Spleen: 20k/132k annotated (only subset has CellType column)

### Bug fix applied (2026-03-09)
- `data_loading_utils.py`: Fixed `.str` accessor crash on all-NaN `harmonized_annotation` columns (COVID, spleen datasets) — added `.astype(object)` in `_apply_hierarchy()`

## Context

Build a combined single-cell dataset from **7 multiome sources** spanning bone marrow, PBMC, lung, and spleen for cell2state training. The pipeline is split into **6 notebooks** + supporting scripts:

1. **Notebook 1** — RNA data loading + concat + save h5ad (**DONE** — 776k cells x 25.6k genes, 2026-03-09)
2. **Notebook 2** — Scrublet doublet scoring on saved RNA h5ad (**RUNNING** — started 2026-03-09)
3. **Notebook 3** — ATAC tile loading from pre-cached snapatac h5ad + save
4. **Notebook 4** — QC summary, annotation comparison, covariate review for model settings
5. **Notebook 5** — RNA-only training (`AmbientRegularizedSCVI`)
6. **Notebook 6** — Multimodal RNA+ATAC training (`RegularizedMultimodalVI`)

Plus a **bash script** to submit parallel bsub jobs for SnapATAC2 preprocessing (run between notebooks 1 and 3).

## Plan file path
`/nfs/users/nfs_v/vk7/.claude/plans/curried-yawning-comet.md`

---

## Dataset Summary

| # | Dataset | GSE | Tissue | Samples | ~Cells | Cell types | Format | Annotations |
|---|---------|-----|--------|---------|--------|------------|--------|-------------|
| 1 | Bone marrow NeurIPS | — | BM | 13 batches | 69,247 | 22 | h5ad | Full (l2_cell_type) |
| 2 | TEA-seq PBMC | GSE158013 | PBMC | 7 wells | 59,151 | 26 | 10x H5 | Partial (GSM4949911 only, needs fix) |
| 3 | NEAT-seq CD4 T | GSE178707 | sorted CD4 T | 2 lanes | 8,457 | 7 | h5ad | Full (C1-C7) |
| 4 | Crohn's PBMC | GSE244831 | PBMC | 13 | 106,296 | 18 | MTX | Partial (69k/106k annotated) |
| 5 | COVID infant PBMC | GSE239799 | PBMC | 43 | 360,624 | 0 | MTX | None |
| 6 | Lung/Spleen immune | GSE319044 | lung+spleen | 16 | 132,241 | 8 | MTX | Partial (20k/132k annotated) |
| 7 | Infant/Adult Spleen | GSE311423 | spleen | 5 | 40,726 | 0 | 10x H5 | None |

---

## Deliverables

### Files to create/modify (all in `docs/notebooks/immune_integration/`)

| # | File | Status | Description |
|---|------|--------|-------------|
| 0 | `PLAN.md` | EXISTS | Copy of this plan |
| 1 | `data_loading_utils.py` | EXISTS | Per-dataset loader functions + harmonization logic |
| 2 | `annotation_harmonization.md` | EXISTS | Cell type name mapping table |
| 3 | `annotation_hierarchy.md` | EXISTS | Unified hierarchy table |
| 4 | `metadata_harmonization.md` | EXISTS | Covariate harmonization table |
| 5 | `bm_pbmc_data_loading.ipynb` | **DONE** | Notebook 1: RNA loading + concat + save (776k cells x 25.6k genes) |
| 6 | `bm_pbmc_scrublet.ipynb` | **RUNNING** | Notebook 2: load RNA h5ad, scrublet, update h5ad |
| 7 | (use cell2state scripts) | **SUBMITTED** | SnapATAC2 jobs (99 bsub jobs submitted 2026-03-09) |
| 8 | `bm_pbmc_atac_loading.ipynb` | CREATE | Notebook 3: load cached snapatac h5ad, tile matrix, save ATAC h5ad |
| 9 | `bm_pbmc_qc_summary.ipynb` | CREATE | Notebook 4: QC summary, annotation comparison, covariate review |
| 10 | `bm_pbmc_rna_training.ipynb` | CREATE | Notebook 5: RNA-only training (AmbientRegularizedSCVI) |
| 11 | `bm_pbmc_multimodal_training.ipynb` | CREATE | Notebook 6: Multimodal training (RegularizedMultimodalVI) |

### Outputs (saved to `results/immune_integration/`)
- `adata_rna.h5ad` — Combined RNA (all cells, harmonized obs, QC metrics, doublet scores)
- `adata_atac_tiles.h5ad` — Combined ATAC tiles (1000bp bins, same cells)
- `obs_metadata.csv` — Full obs table for inspection
- `path_sample_df.csv` — Fragment paths CSV for SnapATAC2 jobs

---

## Annotation Harmonization (separate markdown files)

### File 1: `immune_integration/annotation_harmonization.md`

Maps each dataset's original cell type label to a harmonized name. This is the lookup table used by loader functions.

| original_label | harmonized_name | source_dataset | source_column |
|---|---|---|---|
| **Bone marrow** (l2_cell_type) | | | |
| CD8+ T activated | CD8+ T | bone_marrow | l2_cell_type |
| CD14+ Mono | CD14+ Mono | bone_marrow | l2_cell_type |
| NK | NK | bone_marrow | l2_cell_type |
| CD4+ T activated | CD4+ T activated | bone_marrow | l2_cell_type |
| Naive CD20+ B | Naive B | bone_marrow | l2_cell_type |
| Erythroblast | Erythroblast | bone_marrow | l2_cell_type |
| CD4+ T naive | CD4+ T naive | bone_marrow | l2_cell_type |
| Transitional B | Transitional B | bone_marrow | l2_cell_type |
| Proerythroblast | Proerythroblast | bone_marrow | l2_cell_type |
| CD16+ Mono | CD16+ Mono | bone_marrow | l2_cell_type |
| B1 B | B1 B | bone_marrow | l2_cell_type |
| Normoblast | Normoblast | bone_marrow | l2_cell_type |
| Early Lymphoid | Lymph prog | bone_marrow | l2_cell_type |
| G/M prog | G/M prog | bone_marrow | l2_cell_type |
| pDC | pDC | bone_marrow | l2_cell_type |
| HSC | HSC | bone_marrow | l2_cell_type |
| CD8+ T naive | CD8+ T naive | bone_marrow | l2_cell_type |
| MK/E prog | MK/E prog | bone_marrow | l2_cell_type |
| cDC2 | cDC2 | bone_marrow | l2_cell_type |
| ILC | ILC | bone_marrow | l2_cell_type |
| Plasma | Plasma cell | bone_marrow | l2_cell_type |
| Other Myeloid | ID2-hi myeloid prog | bone_marrow | l2_cell_type |
| **TEA-seq PBMC** (GSM4949911 only) | | | |
| CD4 Naive | CD4+ T naive | pbmc_tea_seq | predicted.celltype.l2 |
| CD4 TCM | CD4+ T central memory | pbmc_tea_seq | predicted.celltype.l2 |
| B naive | Naive B | pbmc_tea_seq | predicted.celltype.l2 |
| CD14 Mono | CD14+ Mono | pbmc_tea_seq | predicted.celltype.l2 |
| CD8 TEM | CD8+ T effector memory | pbmc_tea_seq | predicted.celltype.l2 |
| CD8 Naive | CD8+ T naive | pbmc_tea_seq | predicted.celltype.l2 |
| NK | NK | pbmc_tea_seq | predicted.celltype.l2 |
| CD4 TEM | CD4+ T effector memory | pbmc_tea_seq | predicted.celltype.l2 |
| B intermediate | B intermediate | pbmc_tea_seq | predicted.celltype.l2 |
| MAIT | MAIT | pbmc_tea_seq | predicted.celltype.l2 |
| Treg | Treg | pbmc_tea_seq | predicted.celltype.l2 |
| CD16 Mono | CD16+ Mono | pbmc_tea_seq | predicted.celltype.l2 |
| B memory | B memory | pbmc_tea_seq | predicted.celltype.l2 |
| gdT | gamma-delta T | pbmc_tea_seq | predicted.celltype.l2 |
| CD8 TCM | CD8+ T central memory | pbmc_tea_seq | predicted.celltype.l2 |
| NK_CD56bright | NK CD56bright | pbmc_tea_seq | predicted.celltype.l2 |
| HSPC | HSC | pbmc_tea_seq | predicted.celltype.l2 |
| cDC2 | cDC2 | pbmc_tea_seq | predicted.celltype.l2 |
| NK Proliferating | NK proliferating | pbmc_tea_seq | predicted.celltype.l2 |
| dnT | double-negative T | pbmc_tea_seq | predicted.celltype.l2 |
| Platelet | Platelet | pbmc_tea_seq | predicted.celltype.l2 |
| ILC | ILC | pbmc_tea_seq | predicted.celltype.l2 |
| ASDC | ASDC | pbmc_tea_seq | predicted.celltype.l2 |
| Plasmablast | Plasma cell | pbmc_tea_seq | predicted.celltype.l2 |
| CD4 CTL | CD4+ T CTL | pbmc_tea_seq | predicted.celltype.l2 |
| CD8 Proliferating | CD8+ T proliferating | pbmc_tea_seq | predicted.celltype.l2 |
| **NEAT-seq CD4 T** (Clusters column) | | | |
| C1 | CD4+ T recently activated | neat_seq_cd4t | Clusters |
| C2 | Treg | neat_seq_cd4t | Clusters |
| C3 | Th17 | neat_seq_cd4t | Clusters |
| C4 | CD4+ T central memory | neat_seq_cd4t | Clusters |
| C5 | Th2 | neat_seq_cd4t | Clusters |
| C6 | Th1 | neat_seq_cd4t | Clusters |
| C7 | CD4+ T uncommitted memory | neat_seq_cd4t | Clusters |
| **Crohn's PBMC GSE244831** (Celltypes column) | | | |
| CD14+ Monocytes | CD14+ Mono | crohns_pbmc | Celltypes |
| Tcm | T central memory | crohns_pbmc | Celltypes |
| Naive CD4+ T Cells | CD4+ T naive | crohns_pbmc | Celltypes |
| NK Cells | NK | crohns_pbmc | Celltypes |
| CD8+ Cytotoxic T Cells | CD8+ T | crohns_pbmc | Celltypes |
| Transitional B Cells | Transitional B | crohns_pbmc | Celltypes |
| FCGR3A+ Monocytes | CD16+ Mono | crohns_pbmc | Celltypes |
| Resting Naive B Cells | Naive B | crohns_pbmc | Celltypes |
| MAIT Cells | MAIT | crohns_pbmc | Celltypes |
| Th1/Th17 Cells | Th1/Th17 | crohns_pbmc | Celltypes |
| GdT Cells | gamma-delta T | crohns_pbmc | Celltypes |
| Activated B Cells | Activated B | crohns_pbmc | Celltypes |
| IFN Responding T Cells | IFN-responding T | crohns_pbmc | Celltypes |
| Conventional Dendritic Cells | cDC | crohns_pbmc | Celltypes |
| Proinflammatory Monocytes | Proinflammatory Mono | crohns_pbmc | Celltypes |
| Plasmacytoid Dendritic Cells | pDC | crohns_pbmc | Celltypes |
| Plasma B Cells | Plasma cell | crohns_pbmc | Celltypes |
| TGFB1+ NK Cells | NK TGFB1+ | crohns_pbmc | Celltypes |
| **Lung/Spleen GSE319044** (CellType column) | | | |
| Memory_B | B memory | lung_spleen_gse319044 | CellType |
| CD8_T | CD8+ T | lung_spleen_gse319044 | CellType |
| NK | NK | lung_spleen_gse319044 | CellType |
| CD4_T | CD4+ T | lung_spleen_gse319044 | CellType |
| Naive_B | Naive B | lung_spleen_gse319044 | CellType |
| Th17 | Th17 | lung_spleen_gse319044 | CellType |
| Other | Other | lung_spleen_gse319044 | CellType |
| Treg | Treg | lung_spleen_gse319044 | CellType |
| **COVID PBMC GSE239799** | — | — | — |
| (no annotations deposited) | | covid_pbmc | — |
| **Infant/Adult Spleen GSE311423** | — | — | — |
| (no annotations deposited) | | infant_adult_spleen | — |

### File 2: `immune_integration/annotation_hierarchy.md`

Unified hierarchy for ALL harmonized cell type names across all datasets.

| harmonized_name | level_1 | level_2 | level_3 | level_4 |
|---|---|---|---|---|
| **HSCs & Progenitors** | | | | |
| HSC | HSCs | HSCs | HSCs | Hematopoietic lineage |
| Lymph prog | Lymph prog | Lymph prog | Lymphoid lineage | Hematopoietic lineage |
| G/M prog | G/M prog | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| MK/E prog | Erythroid lineage | Erythroid lineage | Erythroid lineage | Hematopoietic lineage |
| ID2-hi myeloid prog | ID2-hi myeloid prog | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| **Erythroid** | | | | |
| Proerythroblast | Erythroid lineage | Erythroid lineage | Erythroid lineage | Hematopoietic lineage |
| Erythroblast | Erythroid lineage | Erythroid lineage | Erythroid lineage | Hematopoietic lineage |
| Normoblast | Erythroid lineage | Erythroid lineage | Erythroid lineage | Hematopoietic lineage |
| **CD4+ T cells** | | | | |
| CD4+ T naive | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T activated | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T central memory | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T effector memory | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T CTL | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T recently activated | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD4+ T uncommitted memory | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Th1 | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Th2 | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Th17 | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Th1/Th17 | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Treg | CD4+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| **CD8+ T cells** | | | | |
| CD8+ T | CD8+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD8+ T naive | CD8+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD8+ T effector memory | CD8+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD8+ T central memory | CD8+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| CD8+ T proliferating | CD8+ T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| **Other T cells** | | | | |
| T central memory | T-cell lineage | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| MAIT | MAIT | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| gamma-delta T | gamma-delta T | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| double-negative T | double-negative T | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| IFN-responding T | IFN-responding T | T-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| **NK / ILC** | | | | |
| NK | NK-cell lineage | NK-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| NK CD56bright | NK-cell lineage | NK-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| NK proliferating | NK-cell lineage | NK-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| NK TGFB1+ | NK-cell lineage | NK-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| ILC | NK-cell lineage | NK-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| **B cells** | | | | |
| Naive B | Naive B | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Transitional B | Transitional B | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| B intermediate | B intermediate | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| B1 B | B1 B | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| B memory | B memory | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Activated B | Activated B | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| Plasma cell | Plasma cell | B-cell lineage | Lymphoid lineage | Hematopoietic lineage |
| **Myeloid / DC** | | | | |
| CD14+ Mono | Monocyte lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| CD16+ Mono | Monocyte lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| Proinflammatory Mono | Monocyte lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| cDC | DC lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| cDC2 | DC lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| pDC | DC lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| ASDC | DC lineage | Myeloid lineage | Myeloid lineage | Hematopoietic lineage |
| **Other** | | | | |
| Platelet | Platelet / MK lineage | Platelet / MK lineage | Platelet / MK lineage | Hematopoietic lineage |
| Other | Other | Other | Other | Hematopoietic lineage |

### File 3: `immune_integration/metadata_harmonization.md`

Unified obs columns across all datasets.

#### Covariates for combined model

| Covariate | Description | Values per dataset |
|---|---|---|
| `batch` | Per-well/10x reaction (= ambient key) | BM: s1d1..s4d9 (13); TEA: GSM*_tea_seq (7); NEAT: neat_seq_lane1/2; Crohn: Sample_0..Pool_11 (13 wells, pools contain multiple donors); COVID: 30_CC0022..76_CC0829 (43); Lung/Spleen: lungs_1..spleens_3 (16); Spleen311: LI004..PDC149 (5) |
| `site` | Sequencing center | BM: site1-4; TEA: allen_institute; NEAT: stanford; Crohn: emory; COVID: stanford_wimmers; Lung/Spleen: uchicago; Spleen311: columbia |
| `donor` | Individual donor ID | BM: donor1-10; TEA: pbmc_donor1; NEAT: neat_seq_donor1; Crohn: Donors_IDs from demuxlet (e.g. 01_423, 01_785); COVID: CC0022..CC0829; Lung/Spleen: SMO-*/COB-*; Spleen311: LI004/LI018/HDL163/PDC142/PDC149 |
| `dataset` | Dataset identifier | bone_marrow, pbmc_tea_seq, neat_seq_cd4t, crohns_pbmc, covid_pbmc, lung_spleen_gse319044, infant_adult_spleen |
| `tissue` | Tissue of origin | bone_marrow, pbmc, sorted_cd4t, lung, spleen |
| `condition` | Disease/experimental | healthy (default), crohns, covid (for GSE239799), asthmatic/non (for GSE319044) |
| `age_group` | Age category | adult (default), infant (for GSE311423 infant samples), child (for GSE239799) |

#### Standard obs columns (all datasets must have after harmonization)

```
batch, site, donor, dataset, tissue, condition, age_group, sex,
original_annotation, harmonized_annotation,
n_counts_rna, n_genes_rna, mt_frac, doublet_score,
n_counts_atac, n_features_atac,
fragment_file_path, seq_batch
```

---

## Notebook 1: RNA Data Loading (`bm_pbmc_data_loading.ipynb`) — MODIFY EXISTING

**Status**: EXISTS, needs scrublet/ATAC cells removed. Currently has all 7 loaders working.

Modify the existing notebook to **only** load RNA, concat, compute RNA QC metrics, and save. Remove scrublet, ATAC loading, and ATAC QC cells.

### Cells to keep
1. Imports + setup
2. Load all 7 datasets (7 cells)
3. Per-dataset summary table
4. Gene intersection + concat (save SYMBOL before concat, restore after)
5. RNA QC metrics (MT genes, `sc.pp.calculate_qc_metrics`)
6. RNA QC distribution plots per dataset
7. Save `adata_rna.h5ad` + `obs_metadata.csv`
8. **NEW**: Save `path_sample_df.csv` — extract unique `(fragment_file_path, batch)` pairs for SnapATAC2 jobs

### Cells to REMOVE
- Scrublet cell (moved to Notebook 2)
- Doublet score histogram (moved to Notebook 2)
- ATAC tile loading (moved to Notebook 3)
- ATAC QC (moved to Notebook 3)
- Save ATAC (moved to Notebook 3)
- Final summary (moved to Notebook 4)

### Key implementation details
- Crohn's batch fix: use `adata.obs["sample"]` not `adata.obs["Sample"]` (already fixed in data_loading_utils.py)
- SYMBOL preservation: save `symbol_map` before concat, restore after (already in notebook)
- uint16 dtype for counts

---

## Notebook 2: Scrublet (`bm_pbmc_scrublet.ipynb`) — CREATE

**Purpose**: Load saved RNA h5ad, run scrublet per batch, update h5ad with doublet scores.

### Cells
1. Imports + load `adata_rna.h5ad`
2. Per-batch scrublet with small-batch handling:
   ```python
   MIN_CELLS_SCRUBLET = 50
   for batch_name in batch_counts.index:
       mask = adata.obs["batch"] == batch_name
       n_cells = mask.sum()
       if n_cells < MIN_CELLS_SCRUBLET:
           skipped.append((batch_name, n_cells))
           continue
       adata_batch = adata[mask].copy()
       n_comps = min(30, n_cells - 1)
       sc.pp.scrublet(adata_batch, threshold=0.25, n_prin_comps=n_comps)
       # copy scores back
   ```
3. Doublet score distribution plot
4. Overwrite `adata_rna.h5ad` with doublet scores added
5. Print summary (n_doublets, skipped batches)

### Why separate
- Scrublet on ~100 batches x 776k cells takes ~10-30 min
- Isolates a common failure point (small batches, PCA issues)
- Can re-run scrublet without re-loading all 7 datasets

---

## SnapATAC2 Pre-computation — USE CELL2STATE SCRIPTS

**Purpose**: Convert `atac_fragments.tsv.gz` → `atac_fragments.h5ad` in parallel via bsub. SnapATAC2's `import_data` is IO-intensive, so fragment files must be copied to node-local `/var/tmp` before processing and results copied back.

### Scripts (from cell2state package)

Use the generic scripts from the cell2state package — no local copies needed:
- `/nfs/team205/vk7/sanger_projects/my_packages/cell2state/scripts/submit_snapatac_jobs.sh`
- `/nfs/team205/vk7/sanger_projects/my_packages/cell2state/scripts/run_snapatac_one_sample.sh`

See `cell2state/scripts/readme.md` for full documentation.

### `path_sample_df.csv` format
- Column 1: `fragment_file_path` — full path to `atac_fragments.tsv.gz`
- Column 2: `sample_id` — the batch identifier (used as sample name by snapatac2)
- Generated by Notebook 1 from unique `(fragment_file_path, batch)` pairs in `adata.obs`

### Usage

```bash
# Dry run first
bash /nfs/team205/vk7/sanger_projects/my_packages/cell2state/scripts/submit_snapatac_jobs.sh \
  /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/results/immune_integration/path_sample_df.csv \
  /nfs/srpipe_references/downloaded_from_10X/refdata-cellranger-arc-GRCh38-2020-A-2.0.0/ \
  /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/results/immune_integration/ \
  --dry-run

# Submit (remove --dry-run)
bash /nfs/team205/vk7/sanger_projects/my_packages/cell2state/scripts/submit_snapatac_jobs.sh \
  /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/results/immune_integration/path_sample_df.csv \
  /nfs/srpipe_references/downloaded_from_10X/refdata-cellranger-arc-GRCh38-2020-A-2.0.0/ \
  /nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/results/immune_integration/
```

### Features (built into cell2state scripts)
- `--dry-run` flag for testing
- Skip-if-done (checks for existing h5ad) in both submit and worker scripts
- Conda activation built into per-sample worker script
- Configurable `--conda-env`, `--queue`, `--mem`, `--ncores`
- Auto-locates `load_atac_snapatac2.py` relative to script dir
- Error counting and summary stats
- Copies fragments to node-local `/var/tmp` for ~10x IO speedup

### Caching mechanism
- `load_atac_snapatac2` calls `load_atac()` which:
  - Takes `{dir}/atac_fragments.tsv.gz` → creates `{dir}/atac_fragments.h5ad`
  - If h5ad exists and `overwrite=False`, reads cached version
- After jobs complete, Notebook 3's `load_one_sample_tiles()` finds the cached h5ad automatically

---

## Notebook 3: ATAC Tile Loading (`bm_pbmc_atac_loading.ipynb`) — CREATE

**Purpose**: Load cached snapatac h5ad files, create tile matrices, save combined ATAC h5ad.

**Prerequisites**: Notebook 1 done (RNA h5ad exists), SnapATAC2 jobs completed (h5ad files cached).

### Cells
1. Imports + load `adata_rna.h5ad` (for cell barcodes + fragment_file_path)
2. Check that all cached h5ad files exist (fail fast if jobs incomplete)
3. Load ATAC tiles using `concatenate_h5ad` from `cell2state.utils.aggregation_v2`:
   ```python
   from cell2state.utils.aggregation_v2 import concatenate_h5ad
   adatas = concatenate_h5ad(
       adata,
       variable_type="atac_tiles",
       batch_key="batch",
       loading_kwargs=dict(
           path_to_reference=genome_ref,
           path_to_fragment_file_key="fragment_file_path",
           max_frag_size_split=120,
           bin_size=1000,
           counting_strategy="insertion",
           use_complete_path=True,
       ),
   )
   adata_atac = anndata.concat(adatas, join="inner")
   ```
4. Cast to uint16, add counts layer
5. ATAC QC metrics (`sc.pp.calculate_qc_metrics`)
6. ATAC QC distribution plots per dataset
7. Save `adata_atac_tiles.h5ad`

### Why separate
- ATAC tile loading with cached h5ad takes ~30 min (vs hours without cache)
- Decoupled from RNA loading — can iterate on ATAC independently
- Fragment file preprocessing is embarrassingly parallel (bsub jobs)

---

## Annotation: load from markdown files (not hardcoded dicts) — DONE

**Status**: DONE (commit `9b9dcdd`). Markdown files are the single source of truth.

---

## Annotation: validation + hierarchy attachment + `level_1:` convention — TODO

**Status**: TODO. User edited markdown files with new conventions. Code needs updating.

### Context
User edited `annotation_harmonization.md` and `annotation_hierarchy.md` to:
- Rename cell types (e.g., `Plasma cell` → `Plasma B cell`, `B memory` → `Memory B`)
- Add `level_1:{name}` convention for coarse annotations (e.g., `level_1:CD8+ T-cell lineage`)
- Add `NaN` for cells with no meaningful annotation (e.g., lung/spleen `Other`)
- Restructure hierarchy (Th subtypes → `CD4+ T Th1 memory` etc., erythroid/MK grouping)
- Remove `Other` from hierarchy

### What needs to change in `data_loading_utils.py`

**File**: `docs/notebooks/immune_integration/data_loading_utils.py`

#### 1. Add `import warnings` to stdlib imports (line 10 area)

#### 2. Add `_LEVEL_PREFIX = "level_1:"` constant after `STANDARD_OBS_COLS`

#### 3. Add `level_1..level_4` to `STANDARD_OBS_COLS` (after `harmonized_annotation`, before `fragment_file_path`)

#### 4. Update `_parse_harmonization_md()` — handle `NaN` literal
Convert the string `"NaN"` to `np.nan` in the returned dict values:
```python
if harmonized == "NaN":
    maps.setdefault(dataset, {})[original] = np.nan
else:
    maps.setdefault(dataset, {})[original] = harmonized
```
Leave `level_1:*` strings as-is — they flow through `.map()` and get processed later.

#### 5. Add `_validate_annotations(harm_maps, hier_df)` — 6 checks

| # | Check | Severity | Description |
|---|-------|----------|-------------|
| a | Harmonized names exist in hierarchy | ERROR | Every non-NaN, non-`level_1:` harmonized name from harmonization must be in hierarchy |
| b | Hierarchy entries referenced | WARNING | Every harmonized_name in hierarchy should be used by at least one harmonization entry |
| c | `level_1:` values exist | ERROR | Value after `level_1:` prefix must exist as a level_1 value in hierarchy |
| d | No cross-level leaks | WARNING | Higher-level values (level_3/4) should not appear at lower levels (level_1/2) |
| e | No duplicate harmonized_name | ERROR | No duplicate entries in hierarchy |
| f | Consistent parents | ERROR | All rows sharing a level_1 value must have identical level_2/3/4 |

Collect all errors before raising `ValueError` with the full list. Emit warnings separately.

#### 6. Refactor caching — unified `_get_validated_data()`

Replace separate `_HARMONIZATION_MAPS` / `_HIERARCHY_DF` caching with:
```python
_VALIDATED = False

def _get_validated_data():
    global _HARMONIZATION_MAPS, _HIERARCHY_DF, _VALIDATED
    if not _VALIDATED:
        md_dir = Path(__file__).parent
        _HARMONIZATION_MAPS = _parse_harmonization_md(md_dir / "annotation_harmonization.md")
        _HIERARCHY_DF = _parse_hierarchy_md(md_dir / "annotation_hierarchy.md")
        _validate_annotations(_HARMONIZATION_MAPS, _HIERARCHY_DF)
        _VALIDATED = True
    return _HARMONIZATION_MAPS, _HIERARCHY_DF

def _get_harmonization_maps():
    return _get_validated_data()[0]

def _get_hierarchy_df():
    return _get_validated_data()[1]
```

#### 7. Add `_apply_hierarchy(adata, hier_df=None)`

Populate `level_1..level_4` from `harmonized_annotation`. Three cases:

| Case | Input | harmonized_annotation | level_1..level_4 |
|------|-------|-----------------------|------------------|
| Normal | `"CD14+ Mono"` | kept as-is | looked up by harmonized_name |
| `level_1:` prefix | `"level_1:CD8+ T-cell lineage"` | set to NaN | level_1 = value after prefix; level_2/3/4 from hierarchy (any row with matching level_1) |
| NaN | NaN | stays NaN | all NaN |

Uses vectorized pandas: `ha.str.startswith(_LEVEL_PREFIX, na=False)` for detection, `.map()` for lookups.

#### 8. Update `_finalize()` — call `_apply_hierarchy()` before `_standardize_obs()`

```python
def _finalize(adata):
    # ... counts/sparse as before ...
    adata = _apply_hierarchy(adata)  # NEW — before standardize_obs
    adata = _standardize_obs(adata)
    return adata
```

Must come before `_standardize_obs()` because it writes level columns that `_standardize_obs()` then preserves.

#### 9. Update `update_annotations()` — delegate hierarchy to `_apply_hierarchy()`

```python
if add_hierarchy:
    if hierarchy_md_path is not None:
        hier = _parse_hierarchy_md(hierarchy_md_path)
        _apply_hierarchy(adata, hier_df=hier)
    else:
        _apply_hierarchy(adata)
```

### Markdown fixes (before code changes)

1. **Add `CD8+ T activated` to `annotation_hierarchy.md`**: New row under CD8+ T cells section (level_1=CD8+ T-cell lineage, level_2=T-cell lineage, level_3=Lymphoid lineage, level_4=Hematopoietic lineage). Insert after line 33 (`CD8+ T`).
2. **Remove `CD4+ T recently activated` from `annotation_hierarchy.md`**: Delete line 25. No longer referenced by any harmonization entry.
3. **Cross-level collapses** — structural, not errors: e.g., HSC has `HSCs` at level_1/2/3; myeloid progenitors have `Myeloid lineage` at level_2/3. Validation emits WARNING only.

### Verification
1. Run validation: import `data_loading_utils` → should trigger lazy validation on first loader call
2. Check that validation catches `CD8+ T activated` missing from hierarchy (ERROR)
3. Fix the markdown issue, re-import, verify validation passes
4. Load one dataset (e.g., bone_marrow), verify `level_1..level_4` columns are populated
5. Verify `level_1:` entries: Crohn's `CD8+ Cytotoxic T Cells` → harmonized_annotation=NaN, level_1=`CD8+ T-cell lineage`, level_2=`T-cell lineage`
6. Verify NaN entries: lung/spleen `Other` → all NaN

### Workflow after implementation
- **Edit markdown files** → re-import → validation runs automatically
- **Hierarchy auto-populated** during `_finalize()` — no manual `update_annotations()` needed
- **Post-hoc**: `update_annotations(adata, add_hierarchy=True)` still works for existing h5ad files

---

## Notebook 4: QC Summary & Covariate Review (`bm_pbmc_qc_summary.ipynb`) — CREATE

**Purpose**: Comprehensive QC review and covariate analysis to inform model settings.

### Cells
1. Load `adata_rna.h5ad` and `adata_atac_tiles.h5ad`
2. **Combined QC summary table**: per-dataset cell counts, median counts, median genes, median MT frac, median doublet score, median ATAC counts
3. **QC distribution plots**: RNA counts, genes, MT frac, doublet score, ATAC counts — all per dataset
4. **Review and fix annotations** (interactive):
   - Cell type frequency table (harmonized_annotation)
   - Compare against `annotation_harmonization.md` and `annotation_hierarchy.md`
   - Apply rename_map via `update_annotations()` if needed
   - Cells with/without harmonized_annotation per dataset
   - Add hierarchy levels via `update_annotations(adata, hierarchy_df=...)`
5. **Covariate inspection** (for choosing model settings):
   - `batch`: value counts, cells per batch distribution
   - `site`: cross-tabulation with dataset
   - `donor`: n_donors per dataset, cells per donor distribution
   - `dataset`: already summarized above
   - `tissue`: cross-tabulation with dataset
   - `condition`: cross-tabulation with dataset
   - `age_group`: cross-tabulation with dataset
6. **Covariate recommendations for regularizedVI**: based on the distributions, recommend which covariates go in each key:
   - `ambient_covariate_keys` → batch (ambient profile varies per 10x reaction)
   - `nn_conditioning_covariate_keys` → site, donor, dataset (or subset)
   - `feature_scaling_covariate_keys` → site, donor, dataset (or subset)
   - `dispersion_key` → batch
   - `library_size_key` → batch
7. **Fragment file path validation**: check that all fragment_file_path entries are valid files

---

## Notebook 5: RNA-only Training (`bm_pbmc_rna_training.ipynb`) — CREATE

**Reference**: `docs/notebooks/model_comparisons/bone_marrow_gp_es_exp4_out.ipynb`

### Cells
1. Imports (regularizedvi, scanpy, etc.)
2. Load `adata_rna.h5ad`
3. **Cell filtering** (joint RNA+ATAC QC thresholds — TBD after Notebook 4 review):
   - Remove doublets (`predicted_doublet == True`)
   - RNA count thresholds (per-dataset or global)
   - MT fraction threshold
   - Gene count threshold
4. **Gene filtering**: `filter_genes()` from `data_loading_utils.py`
5. Setup model:
   ```python
   AmbientRegularizedSCVI.setup_anndata(
       adata,
       layer="counts",
       ambient_covariate_keys=["batch"],
       nn_conditioning_covariate_keys=["site", "donor"],  # TBD after review
       feature_scaling_covariate_keys=["site", "donor"],  # TBD after review
       dispersion_key="batch",
       library_size_key="batch",
       batch_representation="one-hot",
   )
   ```
6. Create model: `n_hidden=512, n_layers=1, n_latent=128`
7. Train with early stopping:
   ```python
   early_stopping_min_delta_per_feature = 0.0001
   model.train(
       batch_size=1024,
       max_epochs=4000,
       early_stopping=True,
       early_stopping_patience=20,
       early_stopping_min_delta=early_stopping_min_delta_per_feature * adata.n_vars,
       check_val_every_n_epoch=1,
       plan_kwargs=dict(
           regularise_background=False,
           library_log_means_centering_sensitivity=1.0,
           use_additive_background=True,
       ),
   )
   ```
8. Training curves
9. Get latent representation + UMAP
10. Clustering (Leiden)
11. Save model + results

### Training params from reference
- `regularise_background=False`
- `library_log_means_centering_sensitivity=1.0`
- `use_additive_background=True`
- `encoder_covariate_keys=False` (default)
- `use_feature_scaling=True` (default)
- Checkpoint every 200 epochs

---

## Notebook 6: Multimodal Training (`bm_pbmc_multimodal_training.ipynb`) — CREATE

**Reference**: `docs/notebooks/bone_marrow_mm_es_exp8_out.ipynb`

### Cells
1. Imports
2. Load `adata_rna.h5ad` and `adata_atac_tiles.h5ad`
3. Cell filtering (same as Notebook 5)
4. Gene filtering (RNA)
5. CRE selection (ATAC) — TBD, may need pseudobulk + correlation workflow
6. Create MuData
7. Setup model:
   ```python
   RegularizedMultimodalVI.setup_mudata(
       mdata,
       modalities={"rna": "rna", "atac": "atac"},
       layer="counts",
       ambient_covariate_keys=["batch"],
       nn_conditioning_covariate_keys=["site", "donor"],
       feature_scaling_covariate_keys=["site", "donor"],
       dispersion_key="batch",
       library_size_key="batch",
       encoder_covariate_keys=False,
   )
   ```
8. Create model:
   ```python
   n_hidden={"rna": 512, "atac": 256}
   n_latent={"rna": 128, "atac": 64}
   latent_mode="concatenation"
   additive_background_modalities=["rna"]
   feature_scaling_modalities=["rna", "atac"]
   ```
9. Train: same params as Notebook 5, checkpoint every 1000 epochs
10. Training curves, UMAP, clustering
11. Save model + results

---

## Key functions to reuse

| Function | Source | Purpose |
|---|---|---|
| `_read_10x_mtx()` | `data_loading_utils.py` | Load MTX files into AnnData |
| `load_atac_snapatac2` | `cell2state/utils/load_atac_snapatac2.py` | CLI: fragment→h5ad (for bsub jobs) |
| `load_one_sample_tiles()` | `cell2state/utils/load_atac_snapatac2.py` | Load cached h5ad + tile matrix |
| `concatenate_h5ad()` | `cell2state/utils/aggregation_v2.py` | Loop over samples, load ATAC tiles |
| `get_per_batch_anndata()` | `cell2state/utils/aggregation_v2.py` | Per-sample ATAC tile loading |
| `download_bone_marrow_dataset()` | `data_loading_utils.py` | Download BM h5ad |
| `filter_genes()` | `data_loading_utils.py` | Gene selection |

---

## Execution order

```
Notebook 1 (RNA loading)  →  submit_snapatac_jobs.sh  →  Notebook 3 (ATAC loading)
         ↓                                                        ↓
    Notebook 2 (scrublet)                              Notebook 4 (QC review)
                                                              ↓
                                                    Notebook 5 (RNA training)
                                                              ↓
                                                    Notebook 6 (multimodal training)
```

Notebooks 2 and the snapatac jobs can run in parallel after Notebook 1.
Notebook 4 requires both Notebooks 2 and 3 to be done.
Notebooks 5 and 6 require Notebook 4 (covariate review).

### Conda environments

| Task | Environment |
|------|-------------|
| **SnapATAC2 jobs** (`submit_snapatac_jobs.sh` / `run_snapatac_one_sample.sh`) | `conda activate cell2state_v2026_cuda124_torch25` |
| **All notebooks** (1-6) | `conda activate regularizedvi` |

**Important**: Only the SnapATAC2 bsub jobs use `cell2state_v2026_cuda124_torch25` (handled internally by `run_snapatac_one_sample.sh`). All notebook execution via papermill must use the `regularizedvi` environment.

---

## Verification

1. Notebook 1: saves `adata_rna.h5ad` (~776k cells x 25k genes) and `path_sample_df.csv`
2. SnapATAC2 jobs: all h5ad files created next to fragment files (check with `ls`)
3. Notebook 2: `adata_rna.h5ad` updated with `doublet_score` and `predicted_doublet` columns
4. Notebook 3: saves `adata_atac_tiles.h5ad` (same cells as RNA)
5. Notebook 4: produces QC summary and covariate recommendations
6. Notebooks 5/6: model trains with early stopping, UMAP shows cell type separation
