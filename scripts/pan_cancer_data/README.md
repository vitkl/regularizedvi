# HTAN Pan-Cancer Multiome Pipeline

Download and processing scripts for the Terekhanova et al. 2023 pan-cancer snMultiome dataset
(Nature, DOI: 10.1038/s41586-023-06682-5), used in `immune_integration_v2`.

**Data location**: `/nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome/`

## Contents

| Component | Stats |
|-----------|-------|
| L3 counts + fragments | 136 samples × 4 files = 544 files (~1 TB) |
| L4 ATAC annotations (.rds) | 139 files |
| L4 RNA annotations (.rds) | 124 files |
| ATAC cell annotations CSV | 563,526 cells × 16 cols, 112 MB |
| RNA cell annotations CSV | 485,742 cells × 14 cols, 72 MB |

**Cell type diversity** (across 9 cancer types): Tumor (341k ATAC, 294k RNA), Macrophages (~52k),
T-cells (~37k), Fibroblasts (~27k), Plasma, Endothelial, B-cells, Hepatocytes, Oligodendrocytes, Islets, etc.

## Annotation columns

- **ATAC (16)**: `nucleosome_signal`, `TSS.enrichment`, `blacklist_ratio`, `seurat_clusters`,
  `Piece_ID`, `nCount_pancan`/`peaksMACS2`, `nFeature_pancan`/`peaksMACS2`, `cell_type`,
  `Original_barcode`, `barcode`, `source_file`, `cancer_type`, `UMAP_1`, `UMAP_2`
- **RNA (14)**: `nCount_RNA`, `nFeature_RNA`, `percent.mito`, `nCount_SCT`, `nFeature_SCT`,
  `seurat_clusters`, `Piece_ID`, `cell_type`, `Original_barcode`, `barcode`, `source_file`,
  `cancer_type`, `UMAP_1`, `UMAP_2`

## Scripts

| File | Purpose |
|------|---------|
| `download_htan_multiome.py` | Downloads L3/L4 files via Synapse from manifest |
| `convert_rds_annotations.R` | Extracts cell-level metadata from Seurat `.rds` files |
| `create_sample_mapping.py` | Generates `sample_mapping.csv` with per-sample file paths |
| `submit_htan_download.sh` | bsub wrapper — submits download to long queue |
| `submit_rds_extraction.sh` | bsub wrapper — submits R extraction to normal queue |

## Still missing

**Donor IDs** — not present in cell-level `meta.data`. Next step is to grab
`Sample_ID_Lookup_table.xlsx` from the ding-lab GitHub repo
(https://github.com/ding-lab/PanCan_snATAC_publication) to map `Piece_ID` → donor/case ID.
