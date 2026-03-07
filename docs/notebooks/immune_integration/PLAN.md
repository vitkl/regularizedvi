# Plan: Multi-dataset RNA + ATAC integration (7 datasets)

## Context

Build a combined single-cell dataset from **7 multiome sources** spanning bone marrow, PBMC, lung, and spleen for cell2state training. This plan covers **Phase 1 only**: loading, QC, harmonization, and producing two h5ad objects (RNA + ATAC tiles). Model training is deferred to Notebook 2.

## Plan file path
`/nfs/users/nfs_v/vk7/.claude/plans/curried-yawning-comet.md`

---

## Dataset Summary

| # | Dataset | GSE | Tissue | Samples | ~Cells | Cell types | Format | Annotations |
|---|---------|-----|--------|---------|--------|------------|--------|-------------|
| 1 | Bone marrow NeurIPS | — | BM | 13 batches | 69k | 22 | h5ad | Full (l2_cell_type) |
| 2 | TEA-seq PBMC | GSE158013 | PBMC | 7 wells | 52k | 26 | 10x H5 | Partial (GSM4949911 only) |
| 3 | NEAT-seq CD4 T | GSE178707 | sorted CD4 T | 2 lanes | 8.5k | 7 | h5ad | Full (C1-C7) |
| 4 | Crohn's PBMC | GSE244831 | PBMC | 13 | 76k | 18 | MTX | Full (Celltypes) |
| 5 | COVID infant PBMC | GSE239799 | PBMC | 43 | ? | 0 | MTX | None |
| 6 | Lung/Spleen immune | GSE319044 | lung+spleen | 16 | 54k | 8 | MTX | Full (CellType) |
| 7 | Infant/Adult Spleen | GSE311423 | spleen | 5 | ? | 0 | 10x H5 | None |

---

## Deliverables

### Files to create (all in `docs/notebooks/immune_integration/`)

0. **`docs/notebooks/immune_integration/PLAN.md`** — Copy of this plan for discoverability
1. **`docs/notebooks/immune_integration/bm_pbmc_data_loading.ipynb`** — Notebook 1: data loading, QC, harmonization
2. **`docs/notebooks/immune_integration/bm_pbmc_model_training.ipynb`** — Notebook 2 (stub/next phase): model training
3. **`docs/notebooks/immune_integration/data_loading_utils.py`** — Per-dataset loader functions + harmonization logic
4. **`docs/notebooks/immune_integration/annotation_harmonization.md`** — Cell type name mapping table (reviewable markdown)
5. **`docs/notebooks/immune_integration/annotation_hierarchy.md`** — Unified hierarchy table (reviewable markdown)
6. **`docs/notebooks/immune_integration/metadata_harmonization.md`** — Covariate harmonization table (reviewable markdown)

### Outputs from Notebook 1 (saved to `results/`)
- `adata_rna.h5ad` — Combined RNA anndata (all datasets, QC-filtered, harmonized obs)
- `adata_atac_tiles.h5ad` — Combined ATAC tile anndata (1000bp bins, same cells as RNA)

### Step 0: Save plan
Copy this plan to `docs/notebooks/immune_integration/PLAN.md` so it is discoverable alongside the code.

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

## Notebook 1: Data Loading & QC (`immune_integration/bm_pbmc_data_loading.ipynb`)

### Architecture: `immune_integration/data_loading_utils.py`

One loader function per dataset, all returning the same standardized anndata:

```python
def load_bone_marrow() -> sc.AnnData:
def load_tea_seq_pbmc() -> sc.AnnData:
def load_neat_seq_cd4t() -> sc.AnnData:
def load_crohns_pbmc_gse244831() -> sc.AnnData:
def load_covid_pbmc_gse239799() -> sc.AnnData:
def load_lung_spleen_gse319044() -> sc.AnnData:
def load_infant_adult_spleen_gse311423() -> sc.AnnData:
```

Each function:
1. Loads raw data (h5ad, 10x H5, or MTX) — using `read_10x_output()` pattern from `multiome_rna_exploratory_embryo_sections_njc.ipynb` for H5/MTX files
2. Extracts GEX features only (filters out ATAC peaks if present in H5)
3. Sets var to ENSEMBL IDs as var_names, SYMBOL in `var["SYMBOL"]`
4. Loads cell annotations from dataset-specific annotation files
5. Applies harmonization mapping (reads `annotation_harmonization.md` or a derived CSV)
6. Sets standardized obs columns: `batch, site, donor, dataset, tissue, condition, age_group, original_annotation, harmonized_annotation, fragment_file_path`
7. Returns anndata with `.X` = raw counts, `.layers["counts"]` = raw counts

### Data loading details per dataset

#### 1. `load_bone_marrow()`
- Source: `download_bone_marrow_dataset()` → h5ad
- Extract GEX: `adata[:, adata.var["feature_types"] == "GEX"]`
- Annotations: `l2_cell_type` column in obs
- Fragment files: `/nfs/team283/vk7/sanger_projects/large_data/bone_marrow/{batch}/atac_fragments.tsv.gz` (one per batch: s1d1..s4d9)
- Map `batch` → fragment path in obs `fragment_file_path` column

#### 2. `load_tea_seq_pbmc()`
- Source: `sample_mapping.csv` → per-sample `sc.read_10x_h5()`, filter to `Gene Expression`
- 7 samples, concatenate with `anndata.concat`
- Annotations: Figure4_SourceData2_TypeLabelsUMAP.csv → `predicted.celltype.l2` for GSM4949911 only
- QC metrics: from annotation CSVs (nCount_RNA, nCount_ATAC, etc.)
- Fragment files: `*_atac_filtered_fragments.tsv.gz` per sample (in sample_mapping.csv)

#### 3. `load_neat_seq_cd4t()`
- Source: `neat_seq_cd4_tcells.h5ad` (pre-built, 8,457 x 36,717)
- Swap var_names: gene_names → var["SYMBOL"], gene_ids → var_names
- Annotations: `Clusters` C1-C7 → map via: C1→CD4+ T recently activated, C2→Treg, C3→Th17, C4→CD4+ T central memory, C5→Th2, C6→Th1, C7→CD4+ T uncommitted memory
- Fragment files: per-lane `*_atac_fragments.tsv.gz`

#### 4. `load_crohns_pbmc_gse244831()`
- Source: `sample_mapping.csv` → per-sample MTX (barcodes + features + matrix)
- Load with adapted `read_10x_output()` — reads MTX instead of H5, filters to Gene Expression features
- 13 samples (3 single-donor + 10 pools of 2-3 donors), ~76k cells
- Annotations: `GSE244831_cell_annotations.csv` → `Celltypes` column, join on barcode (format: `Pool_8#GTTAACGGTGCTTTAC-1`)
- **Covariate mapping**:
  - `batch` = `Sample` column (= 10x well/reaction: Sample_0, Sample_1, Sample_2, Pool_2..Pool_11) — ambient key
  - `donor` = `Donors_IDs` column from demuxlet (individual donor within pool, e.g. 01_423)
  - `condition` = `Status` column (Crohns/Healthy)
  - `seq_batch` = `Batch` column (Batch1/2/3, sequencing batch — keep as additional metadata)
- Fragment files: `*_atac_fragments.tsv.gz` per sample (in sample_mapping.csv)

#### 5. `load_covid_pbmc_gse239799()`
- Source: `sample_mapping.csv` → per-sample MTX
- 43 samples from 18 subjects (longitudinal)
- **No cell annotations** — `harmonized_annotation = NaN`
- Fragment files: `*_atac_fragments.tsv.gz` per sample

#### 6. `load_lung_spleen_gse319044()`
- Source: `sample_mapping.csv` → per-sample MTX
- 16 samples (9 lung + 7 spleen)
- Annotations: `GSE319044_snRNA_cluster_labels.csv.gz` → `CellType` column
- Additional metadata: `tissue.ident` (lungs/spleens), `Age`, `Sex`, `Race`, `Asthmatic.status`
- Fragment files: `*_atac_fragments.tsv.gz` per sample + tabix indices

#### 7. `load_infant_adult_spleen_gse311423()`
- Source: `sample_mapping.csv` → per-sample 10x H5 (`sc.read_10x_h5()`)
- 5 samples (3 infant, 2 adult)
- **No cell annotations** — `harmonized_annotation = NaN`
- Additional metadata: `age_group` (infant/adult)
- Fragment files: `*_atac_fragments.tsv.gz` per sample + tabix indices

### Notebook 1 structure

#### Section 1: Imports & setup
```python
import scanpy as sc, anndata, numpy as np, pandas as pd, ...
from data_loading_utils import (
    load_bone_marrow, load_tea_seq_pbmc, load_neat_seq_cd4t,
    load_crohns_pbmc_gse244831, load_covid_pbmc_gse239799,
    load_lung_spleen_gse319044, load_infant_adult_spleen_gse311423,
)
results_folder = "results/immune_integration/"
```

#### Section 2: Load all datasets (RNA)
- Call each loader function
- Print per-dataset summary (n_cells, n_genes, annotation coverage)

#### Section 3: Combine RNA
- Gene intersection across all datasets (ENSEMBL IDs)
- `adata = anndata.concat([adata_bm, adata_tea, adata_neat, ...], join="inner")`
- Verify standardized obs columns

#### Section 4: QC — RNA metrics
- `sc.pp.calculate_qc_metrics()` on combined object (recompute for consistency)
- MT fraction, total counts, n_genes
- Scrublet per batch: `sc.pp.scrublet(adata, batch_key="batch")`
- QC distribution plots per dataset

#### Section 5: QC — ATAC tile loading
- Use `concatenate_h5ad()` from `cell2state.utils.load_atac_snapatac2` (pattern from `sc_atac_loading_all_tiles_1000bp_120_sections_suspension.ipynb`)
- Input: RNA adata (for cell barcodes + fragment_file_path column)
- Parameters: `bin_size=1000, counting_strategy='insertion', max_frag_size_split=120`
- Reference genome: `/nfs/srpipe_references/downloaded_from_10X/refdata-cellranger-arc-GRCh38-2020-A-2.0.0/`
- Output: `adata_atac` with tiles as vars, cells as obs

#### Section 6: QC — ATAC metrics
- Recompute with scanpy: `sc.pp.calculate_qc_metrics()` on tile matrix
- Total ATAC counts, n_features per cell
- Transfer ATAC QC to RNA obs for joint filtering

#### Section 7: Joint cell filtering
- Apply thresholds using both RNA and ATAC QC metrics
- Per-dataset adaptive thresholds where needed (different technologies)
- Print per-dataset cell counts before/after

#### Section 8: Gene selection (RNA)
- `filter_genes()` on filtered combined RNA (cell_count_cutoff=15, ...)
- Subset RNA adata to selected genes

#### Section 9: Subset ATAC to filtered cells
- Align ATAC adata to same cells as filtered RNA

#### Section 10: Save outputs
- `adata_rna.write_h5ad("results/immune_integration/adata_rna.h5ad")`
- `adata_atac.write_h5ad("results/immune_integration/adata_atac_tiles.h5ad")`
- Save obs table as CSV for inspection

---

## Key functions to reuse

| Function | Source | Purpose |
|---|---|---|
| `read_10x_output()` | `multiome_rna_exploratory_embryo_sections_njc.ipynb` | Load multiple 10x H5/MTX files, concat, QC |
| `concatenate_h5ad()` | `cell2state.utils.load_atac_snapatac2` | Load ATAC fragments into tile h5ad |
| `download_bone_marrow_dataset()` | `regularizedvi/utils/_data.py` | Download BM h5ad |
| `filter_genes()` | `regularizedvi/utils/_utils.py` | Gene selection |
| `sc.pp.scrublet()` | scanpy | Doublet detection |

---

## Data paths summary

| Dataset | Sample mapping | Annotation file | Fragment files |
|---|---|---|---|
| Bone marrow | — (h5ad download) | in h5ad obs | `/nfs/team283/.../bone_marrow/{batch}/atac_fragments.tsv.gz` |
| TEA-seq PBMC | `/nfs/team283/.../tea_seq_pbmc/sample_mapping.csv` | `.../supplementary_data/Figure4_SourceData2_TypeLabelsUMAP.csv` | in sample_mapping |
| NEAT-seq CD4 T | — (h5ad) | in h5ad `Clusters` | `.../cd4_tcells/lane{1,2}/*_atac_fragments.tsv.gz` |
| Crohn's PBMC | `/nfs/team283/.../GSE244831/sample_mapping.csv` | `.../GSE244831/annotations/GSE244831_cell_annotations.csv` | in sample_mapping |
| COVID PBMC | `/nfs/team283/.../GSE239799/sample_mapping.csv` | None | in sample_mapping |
| Lung/Spleen | `/nfs/team283/.../GSE319044/sample_mapping.csv` | `.../series_level/GSE319044_snRNA_cluster_labels.csv.gz` | in sample_mapping |
| Spleen infant | `/nfs/team283/.../GSE311423/sample_mapping.csv` | None | in sample_mapping |

---

## Next steps (Notebook 2 — future phase)

After Notebook 1 produces `adata_rna.h5ad` and `adata_atac_tiles.h5ad`:

1. **RNA-only model training** (`AmbientRegularizedSCVI`)
   - Covariates: `ambient_covariate_keys=["batch"]`, `nn_conditioning_covariate_keys=["site","donor","dataset"]`, `feature_scaling_covariate_keys=["site","donor","dataset"]`, `dispersion_key="batch"`, `library_size_key="batch"`
   - Architecture: n_hidden=512, n_layers=1, n_latent=128
   - Early stopping: patience=20, max_epochs=4000
   - **Important**: NEAT-seq is sorted CD4 T cells — very different ambient profile from whole-tissue samples. Per-batch ambient model handles this.
   - **Important**: ~100 batches total — may need larger batch_size or adjusted learning rate

2. **Clustering & annotation transfer**
   - Leiden clustering on latent space
   - Majority vote annotation from labeled cells to unlabeled (GSE239799, GSE311423, TEA-seq non-GSM4949911)
   - Attach hierarchy levels

3. **CRE selection** (from ATAC tiles)
   - Pseudobulk tiles by sample x cell_cluster
   - Correlation workflow against annotation hierarchy
   - TopN CRE selection

4. **Multimodal model training** (`RegularizedMultimodalVI`)
   - MuData with RNA + selected CREs

---

## Verification

1. Each loader function returns anndata with correct standardized obs columns
2. Gene intersection covers >15k genes
3. ATAC tile loading completes for all datasets (fragment file paths valid)
4. QC metrics computed consistently across datasets
5. Scrublet runs without errors on combined data
6. Output h5ad files loadable and contain all expected metadata
