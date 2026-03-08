# Annotation Harmonization

Maps each dataset's original cell type label to a harmonized name. Used by loader functions in `data_loading_utils.py`.

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
| **COVID PBMC GSE239799** | | | |
| (no annotations deposited) | | covid_pbmc | — |
| **Infant/Adult Spleen GSE311423** | | | |
| (no annotations deposited) | | infant_adult_spleen | — |
