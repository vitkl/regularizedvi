# Metadata Harmonization

Unified obs columns across all datasets.

## Covariates for combined model

| Covariate | Description | Values per dataset |
|---|---|---|
| `batch` | Per-well/10x reaction (= ambient key) | BM: s1d1..s4d9 (13); TEA: GSM*_tea_seq/multiome (7); NEAT: neat_seq_lane1/2; Crohn: Sample_0..Pool_11 (13); COVID: 30_CC0022..76_CC0829 (43); Lung/Spleen: lung_COB-11..spleen_SMO-1 (16); Spleen311: LI004..PDC149 (5) |
| `site` | Sequencing center | BM: site1-4; TEA: allen_institute; NEAT: stanford; Crohn: emory; COVID: stanford_wimmers; Lung/Spleen: uchicago; Spleen311: columbia |
| `donor` | Individual donor ID | BM: donor1-10; TEA: pbmc_donor1; NEAT: neat_seq_donor1; Crohn: Donors_IDs from demuxlet; COVID: CC0022..CC0829; Lung/Spleen: COB-*/SMO-*; Spleen311: infant_1..adult_2 |
| `dataset` | Dataset identifier | bone_marrow, pbmc_tea_seq, neat_seq_cd4t, crohns_pbmc, covid_pbmc, lung_spleen_gse319044, infant_adult_spleen |
| `tissue` | Tissue of origin | bone_marrow, pbmc, sorted_cd4t, lung, spleen |
| `condition` | Disease/experimental | healthy (default), crohns, covid, asthmatic, non_asthmatic |
| `age_group` | Age category | adult (default), infant, child |

## Standard obs columns

All datasets must have these columns after harmonization:

```
batch, site, donor, dataset, tissue, condition, age_group, sex,
original_annotation, harmonized_annotation,
fragment_file_path
```

QC columns (computed after combining, not per-loader):

```
n_counts_rna, n_genes_rna, mt_frac, doublet_score,
n_counts_atac, n_features_atac
```

## Per-dataset covariate mapping

### 1. Bone marrow (NeurIPS 2021)
| Standard column | Source |
|---|---|
| batch | `batch` in obs (s1d1..s4d9) |
| site | `Site` in obs (site1..site4) |
| donor | `DonorNumber` in obs (donor1..donor10) |
| dataset | `"bone_marrow"` |
| tissue | `"bone_marrow"` |
| condition | `"healthy"` |
| age_group | `"adult"` |
| sex | `"unknown"` |
| original_annotation | `l2_cell_type` |
| fragment_file_path | `/nfs/team283/.../bone_marrow/{batch}/atac_fragments.tsv.gz` |

### 2. TEA-seq PBMC (GSE158013)
| Standard column | Source |
|---|---|
| batch | `sample_id` from sample_mapping.csv (e.g. GSM4949911_tea_seq) |
| site | `"allen_institute"` |
| donor | `"pbmc_donor1"` |
| dataset | `"pbmc_tea_seq"` |
| tissue | `"pbmc"` |
| condition | `"healthy"` |
| age_group | `"adult"` |
| sex | `"unknown"` |
| original_annotation | `predicted.celltype.l2` (GSM4949911 only, NaN for others) |
| fragment_file_path | `fragment_file_path` from sample_mapping.csv |

### 3. NEAT-seq CD4 T (GSE178707)
| Standard column | Source |
|---|---|
| batch | `"neat_seq_lane1"` / `"neat_seq_lane2"` from obs `lane` |
| site | `"stanford"` |
| donor | `"neat_seq_donor1"` |
| dataset | `"neat_seq_cd4t"` |
| tissue | `"sorted_cd4t"` |
| condition | `"healthy"` |
| age_group | `"adult"` |
| sex | `"unknown"` |
| original_annotation | `Clusters` (C1-C7) |
| fragment_file_path | per-lane fragment file |

### 4. Crohn's PBMC (GSE244831)
| Standard column | Source |
|---|---|
| batch | `Sample` column from annotation CSV (Sample_0..Pool_11) |
| site | `"emory"` |
| donor | `Donors_IDs` from annotation CSV (demuxlet) |
| dataset | `"crohns_pbmc"` |
| tissue | `"pbmc"` |
| condition | `Status` column (Crohns/Healthy → crohns/healthy) |
| age_group | `"adult"` |
| sex | `Sex` from annotation CSV |
| original_annotation | `Celltypes` |
| seq_batch | `Batch` column (Batch1/2/3) |
| fragment_file_path | `fragment_file_path` from sample_mapping.csv |

### 5. COVID infant PBMC (GSE239799)
| Standard column | Source |
|---|---|
| batch | `sample_id` from sample_mapping.csv (30_CC0022..76_CC0829) |
| site | `"stanford_wimmers"` |
| donor | `subject_id` from sample_mapping.csv (CC0022..CC0829) |
| dataset | `"covid_pbmc"` |
| tissue | `"pbmc"` |
| condition | `"covid"` |
| age_group | `"child"` |
| sex | `"unknown"` |
| original_annotation | NaN (no annotations) |
| fragment_file_path | `fragment_file_path` from sample_mapping.csv |

### 6. Lung/Spleen (GSE319044)
| Standard column | Source |
|---|---|
| batch | `sample_id` from sample_mapping.csv (lung_COB-11..spleen_SMO-1) |
| site | `"uchicago"` |
| donor | `file_donor_id` from sample_mapping.csv (COB-*/SMO-*) |
| dataset | `"lung_spleen_gse319044"` |
| tissue | `tissue` from sample_mapping.csv (lung/spleen) |
| condition | `Asthmatic.status` from annotation CSV (asthmatic/non_asthmatic) |
| age_group | `"adult"` |
| sex | `Sex` from annotation CSV |
| original_annotation | `CellType` from annotation CSV |
| fragment_file_path | `fragment_file_path` from sample_mapping.csv |

### 7. Infant/Adult Spleen (GSE311423)
| Standard column | Source |
|---|---|
| batch | `library_id` from sample_mapping.csv (LI004..PDC149) |
| site | `"columbia"` |
| donor | `donor_id` from sample_mapping.csv (infant_1..adult_2) |
| dataset | `"infant_adult_spleen"` |
| tissue | `"spleen"` |
| condition | `"healthy"` |
| age_group | `age_group` from sample_mapping.csv (infant/adult) |
| sex | `"unknown"` |
| original_annotation | NaN (no annotations) |
| fragment_file_path | `fragment_file_path` from sample_mapping.csv |
