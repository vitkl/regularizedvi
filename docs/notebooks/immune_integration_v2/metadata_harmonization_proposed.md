
## GBM-Space — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 155 rows × 12 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_dir` | 'cellranger-arc201_count_00d9bb'×1; 'cellranger-arc201_count_03767f'×1; 'cellranger-arc201_count_0bdb76'×1; 'cellranger-arc201_count_0de50c'×1; 'cellranger-arc201_count_0f559e'×1 | 155 | batch |  |
| `irods_path` | '/seq/illumina/cellranger-arc/c'×1; '/seq/illumina/cellranger-arc/c'×1; '/seq/illumina/cellranger-arc/c'×1; '/seq/illumina/cellranger-arc/c'×1; '/seq/illumina/cellranger-arc/c'×1 | 155 | <TODO: not in proposed_mapping> |  |
| `gex_supplier_name` | 'AT11-BRA-6-FO-4b_mG'×1; 'AT15-BRA-4-FO-E1a_mG'×1; 'AT3-BRA-5FO2-S22c_mG'×1; 'AT7-BRA-3-FO-2a_mG'×1; 'AT5-BRA-5-FO-3_3a_mG'×1 | 155 | <auxiliary; drop> |  |
| `atac_supplier_name` | 'AT11-BRA-6-FO-4b_mA'×1; 'AT15-BRA-4-FO-E1a_mA'×1; 'AT3-BRA-5FO2-S22c_mA'×1; 'AT7-BRA-3-FO-2a_mA'×1; 'AT5-BRA-5-FO-3_3a_mA'×1 | 155 | <auxiliary; drop> |  |
| `gex_sanger_id` | 'GBM_RNA13163869'×1; 'GBM_RNA13437545'×1; 'GBM_RNA12936176'×1; 'GBM_RNA13078486'×1; 'GBM_RNA13078519'×1 | 155 | <auxiliary; drop> |  |
| `atac_sanger_id` | 'GBM_RNA13164061'×1; 'GBM_RNA13437641'×1; 'GBM_RNA12930345'×1; 'GBM_RNA13078588'×1; 'GBM_RNA13078621'×1 | 155 | <auxiliary; drop> |  |
| `_obs_donor_id_unique` | '<NaN>'×155 | 1 | <TODO: not in proposed_mapping> |  |
| `_obs_donor_id_n_unique` | '12'×155 | 1 | donor |  |
| `_obs_sample_unique` | '<NaN>'×155 | 1 | <TODO: not in proposed_mapping> |  |
| `_obs_sample_n_unique` | '155'×155 | 1 | batch |  |
| `_obs_site_id_unique` | '<NaN>'×155 | 1 | <TODO: not in proposed_mapping> |  |
| `_obs_site_id_n_unique` | '57'×155 | 1 | site |  |

## hippocampus_aging — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 41 rows × 30 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'GSE278576'×1; 'hc11'×1; 'hc1134'×1; 'hc1153'×1; 'hc12'×1 | 41 | batch |  |
| `gse_id` | 'GSE278576'×41 | 1 | <auxiliary; drop> |  |
| `gsm_gex` | '<NaN>'×40; 'GSE278576'×1 | 2 | <auxiliary; drop> |  |
| `gsm_atac` | '<NaN>'×1; 'GSM8549647'×1; 'GSM8549625'×1; 'GSM8549641'×1; 'GSM8549646'×1 | 41 | <auxiliary; drop> |  |
| `fragment_file_path` | '<NaN>'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 41 | fragment_file_path |  |
| `fragment_index_path` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `gex_h5_path` | '<NaN>'×40; '/nfs/team205/vk7/sanger_projec'×1 | 2 | <TODO: not in proposed_mapping> |  |
| `_meta_bacrode` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_orig.ident` | '<NaN>'×41 | 1 | donor |  |
| `_meta_nCount_RNA` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_nFeature_RNA` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_percent.mt` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_percent.ribo` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_RNA_snn_res.0.5` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_doublet_probability` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_doublet_info` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_Age` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_nCount_SCT` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_nFeature_SCT` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_0.5` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_0.8` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_1` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_1.2` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_1.5` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_1.8` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_seurat_clusters_2` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_Gender` | '<NaN>'×41 | 1 | sex |  |
| `_meta_age_group` | '<NaN>'×41 | 1 | <TODO: not in proposed_mapping> |  |
| `_meta_subclass` | '<NaN>'×41 | 1 | original_annotation |  |

_Proposed columns NOT present in source dataframe (placeholders for nested obs / Zenodo / SDRF joins):_

- `_meta_age` → `age_group`

## lung_smoking — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 16 rows × 7 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'Female,_never-smoker,_subject1'×1; 'Female,_never-smoker,_subject2'×1; 'Female,_never-smoker,_subject3'×1; 'Female,_never-smoker,_subject4'×1; 'Female,_smoker,_subject1'×1 | 16 | batch |  |
| `gse_id` | 'GSE241468'×16 | 1 | <auxiliary; drop> |  |
| `gsm_gex` | 'GSM7729458'×1; 'GSM7729459'×1; 'GSM7729460'×1; 'GSM7729461'×1; 'GSM7729454'×1 | 16 | <TODO: not in proposed_mapping> |  |
| `gsm_atac` | 'GSM7729474'×1; 'GSM7729475'×1; 'GSM7729476'×1; 'GSM7729477'×1; 'GSM7729470'×1 | 16 | <TODO: not in proposed_mapping> |  |
| `fragment_file_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 16 | fragment_file_path |  |
| `fragment_index_path` | '<NaN>'×16 | 1 | <TODO: not in proposed_mapping> |  |
| `gex_h5_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 16 | <TODO: not in proposed_mapping> |  |

_Proposed columns NOT present in source dataframe (placeholders for nested obs / Zenodo / SDRF joins):_

- `_meta_orig.ident` → `donor`
- `_meta_smoker_status` → `condition`
- `_meta_Sex` → `sex`
- `_meta_Age` → `age_group`
- `_meta_seurat_clusters` → `<→ cell_type via SData4 (see annotation review)>`

## intestine_hickey — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 75 rows × 5 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `SampleNameRNA` | 'B001-A-001'×1; 'B001-A-006'×1; 'B001-A-101'×1; 'B001-A-201'×1; 'B001-A-301'×1 | 75 | batch |  |
| `SampleNameOnly` | 'B005-A-501'×2; 'B004-A-404'×2; 'B004-A-408'×2; 'B004-A-504'×2; 'B004-A-104'×2 | 69 | <auxiliary; drop> |  |
| `Donor` | 'B004'×12; 'B006'×9; 'B005'×9; 'B001'×8; 'B008'×8 | 9 | donor |  |
| `Multiome` | 'Yes'×47; 'No'×28 | 2 | <filter: keep only 'Yes'> |  |
| `Location` | 'Mid-jejunum'×10; 'Proximal-jejunum'×10; 'Duodenum'×10; 'Transverse'×10; 'Ileum'×9 | 8 | tissue |  |

## HDMA Spleen/Thymus/Liver — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 16 rows × 10 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'T165_b17_Spleen_PCW19'×1; 'T166_b15_Liver_PCW19'×1; 'T233_b17_Thymus_PCW21'×1; 'T235_b15_Liver_PCW21'×1; 'T23_b17_Thymus_PCW17'×1 | 16 | batch |  |
| `organ` | 'Liver'×7; 'Spleen'×5; 'Thymus'×4 | 3 | tissue |  |
| `donor_id` | '314'×2; '166'×1; '165'×1; '235'×1; '23'×1 | 15 | donor |  |
| `batch` | '17'×8; '15'×7; '16'×1 | 3 | <auxiliary; drop (HDMA's own batch ID, not v1 batch)> |  |
| `PCW` | '21'×5; '18'×5; '19'×3; '17'×2; '15'×1 | 5 | age_group |  |
| `barcodes_path` | '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1 | 16 | <TODO: not in proposed_mapping> |  |
| `features_path` | '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1 | 16 | <TODO: not in proposed_mapping> |  |
| `matrix_path` | '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1 | 16 | <TODO: not in proposed_mapping> |  |
| `fragment_file_path` | '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1 | 16 | fragment_file_path |  |
| `fragment_index_path` | '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1; '/nemo/lab/briscoej/home/users/'×1 | 16 | <TODO: not in proposed_mapping> |  |

## ad_brain_3region — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 21 rows × 7 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'NIH01,_control'×1; 'NIH02,_control'×1; 'NIH03,_control'×1; 'NIH04,_control'×1; 'NIH05,_control'×1 | 21 | batch |  |
| `gse_id` | 'GSE272082'×21 | 1 | <auxiliary; drop> |  |
| `gsm_gex` | 'GSM8392674'×1; 'GSM8392675'×1; 'GSM8392676'×1; 'GSM8392677'×1; 'GSM8392678'×1 | 21 | <TODO: not in proposed_mapping> |  |
| `gsm_atac` | 'GSM8392654'×1; 'GSM8392655'×1; 'GSM8392656'×1; 'GSM8392657'×1; 'GSM8392658'×1 | 21 | <TODO: not in proposed_mapping> |  |
| `fragment_file_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 21 | fragment_file_path |  |
| `fragment_index_path` | '<NaN>'×21 | 1 | <TODO: not in proposed_mapping> |  |
| `gex_h5_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 21 | <TODO: not in proposed_mapping> |  |

## bach2_ap1_gut_tcells — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 78 rows × 9 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'HTO_ADT_of_008_012_015_pooled_'×1; 'HTO_ADT_of_017_023_027_029_poo'×1; 'HTO_ADT_of_035_037_040_pooled_'×1; 'HTO_ADT_of_361_pooled_whole'×1; 'HTO_ADT_of_HD351_A_whole'×1 | 78 | batch |  |
| `gse_id` | 'GSE299348'×78 | 1 | <auxiliary; drop> |  |
| `gsm_gex` | 'GSM9037832'×1; 'GSM9037833'×1; 'GSM9037831'×1; 'GSM9037834'×1; 'GSM9037819'×1 | 78 | <TODO: not in proposed_mapping> |  |
| `gsm_atac` | '<NaN>'×39; 'GSM9037793'×1; 'GSM9037794'×1; 'GSM9037792'×1; 'GSM9037795'×1 | 40 | <TODO: not in proposed_mapping> |  |
| `fragment_file_path` | '<NaN>'×39; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 40 | fragment_file_path |  |
| `fragment_index_path` | '<NaN>'×78 | 1 | <TODO: not in proposed_mapping> |  |
| `barcodes_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 78 | <used by loader> |  |
| `features_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 78 | <used by loader> |  |
| `matrix_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 78 | <used by loader> |  |

## bcg_trained_immunity (→ rename dataset to bcg_bladder_immunotherapy) — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 13 rows × 12 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'GSE295277'×1; 'GSE295308'×1; 'PBMC_Pie_Pre'×1; 'PBMC_Pool_Pre'×1; 'PBMC_Post'×1 | 13 | batch |  |
| `gse_id` | 'GSE295277'×8; 'GSE295308'×5 | 2 | <auxiliary; drop> |  |
| `gsm_gex` | '<NaN>'×11; 'GSE295277'×1; 'GSE295308'×1 | 3 | <TODO: not in proposed_mapping> |  |
| `gsm_atac` | '<NaN>'×2; 'GSM8944727'×1; 'GSM8944729'×1; 'GSM8944731'×1; 'GSM8944733'×1 | 12 | <TODO: not in proposed_mapping> |  |
| `fragment_file_path` | '<NaN>'×2; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 12 | fragment_file_path |  |
| `fragment_index_path` | '<NaN>'×13 | 1 | <TODO: not in proposed_mapping> |  |
| `rna_h5ad_path` | '<NaN>'×11; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 3 | <TODO: not in proposed_mapping> |  |
| `_obs_barcode` | '<NaN>'×13 | 1 | <row id; drop> |  |
| `_obs_experiment` | '<NaN>'×13 | 1 | <auxiliary; record but drop> |  |
| `_obs_orig_barcode` | '<NaN>'×13 | 1 | <10x barcode; drop> |  |
| `_obs_sample` | '<NaN>'×13 | 1 | batch |  |
| `_obs_status` | '<NaN>'×13 | 1 | condition |  |

## rorgt_dc_tonsil — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 12 rows × 9 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | "CD-1813-5,_Chron's_Disease,_Pa"×1; "CD-1813-7,_Chron's_Disease,_Pa"×1; "CD-1818-4,_Chron's_Disease,_Pa"×1; "CD-1818-6,_Chron's_Disease,_Pa"×1; 'CD34_BM_Culture,_IL7/Flt3L/SCF'×1 | 12 | batch |  |
| `gse_id` | 'GSE247692'×12 | 1 | <auxiliary; drop> |  |
| `gsm_gex` | 'GSM7898955'×1; 'GSM7898956'×1; 'GSM7898957'×1; 'GSM7898958'×1; 'GSM7898968'×1 | 12 | <TODO: not in proposed_mapping> |  |
| `gsm_atac` | '<NaN>'×9; 'GSM7898962'×1; 'GSM7898964'×1; 'GSM7898967'×1 | 4 | <TODO: not in proposed_mapping> |  |
| `fragment_file_path` | '<NaN>'×9; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 4 | fragment_file_path |  |
| `fragment_index_path` | '<NaN>'×12 | 1 | <TODO: not in proposed_mapping> |  |
| `barcodes_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 12 | <TODO: not in proposed_mapping> |  |
| `features_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 12 | <TODO: not in proposed_mapping> |  |
| `matrix_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 12 | <TODO: not in proposed_mapping> |  |

## down_fetal_blood — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:43)

Source dataframe shape: 240 rows × 62 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `Source Name` | 'TS21_4_I_GEX'×36; 'TS21_4_H_GEX'×36; 'TS21_4_H_ATAC'×12; 'TS21_4_I_ATAC'×12; 'H_7_C_ATAC'×8 | 24 | batch |  |
| `Comment[ENA_SAMPLE]` | 'ERS15605665'×36; 'ERS15605664'×36; 'ERS15605660'×12; 'ERS15605671'×12; 'ERS15605663'×8 | 24 | <TODO: not in proposed_mapping> |  |
| `Comment[BioSD_SAMPLE]` | 'SAMEA113611486'×36; 'SAMEA113611485'×36; 'SAMEA113611481'×12; 'SAMEA113611492'×12; 'SAMEA113611484'×8 | 24 | <TODO: not in proposed_mapping> |  |
| `Characteristics[original source name]` | 'T21 15582 nuclei A'×48; 'T21 15582 nuclei B'×48; '16216B'×16; '16216A'×16; '16216C'×16 | 12 | <TODO: not in proposed_mapping> |  |
| `Characteristics[individual]` | 'TS21_4'×96; 'H_7'×48; 'TS21_11'×32; 'H_5'×24; 'TS21_16'×24 | 6 | donor |  |
| `Characteristics[organism]` | 'Homo sapiens'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Characteristics[age]` | '13'×120; '14'×80; '12'×40 | 3 | age_group |  |
| `Unit[time unit]` | 'week'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Term Source REF` | 'EFO'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Term Accession Number` | 'UO_0000034'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Characteristics[organism part]` | 'liver'×240 | 1 | tissue |  |
| `Characteristics[developmental stage]` | 'fetus'×240 | 1 | age_group |  |
| `Characteristics[disease]` | 'Down syndrome'×152; 'normal'×88 | 2 | condition |  |
| `Characteristics[protocol]` | 'RNA-Seq'×136; 'ATAC-seq'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Material Type` | 'nucleus'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Protocol REF` | 'P-MTAB-134041'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.1` | 'P-MTAB-134042'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.2` | 'P-MTAB-134043'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Extract Name` | 'TS21_4_I_GEX'×36; 'TS21_4_H_GEX'×36; 'TS21_4_H_ATAC'×12; 'TS21_4_I_ATAC'×12; 'H_7_C_ATAC'×8 | 24 | <TODO: not in proposed_mapping> |  |
| `Material Type.1` | 'RNA'×136; 'DNA'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[LIBRARY_LAYOUT]` | 'PAIRED'×136; 'SINGLE'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[LIBRARY_SELECTION]` | 'Oligo-dT'×136; 'PCR'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[LIBRARY_SOURCE]` | 'TRANSCRIPTOMIC SINGLE CELL'×136; 'GENOMIC SINGLE CELL'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[LIBRARY_STRATEGY]` | 'RNA-Seq'×136; 'ATAC-seq'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[cdna read offset]` | '0'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[cdna read size]` | '91'×136; '50'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[cell barcode offset]` | '0'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[cell barcode read]` | 'read1'×136; 'index1'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[cell barcode size]` | '16'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[end bias]` | '3 prime tag'×136; 'not applicable'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[input molecule]` | 'polyA RNA'×136; 'genomic DNA'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[library construction]` | "10x 3' v3"×136; '10x scATAC-seq'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[sample barcode offset]` | '0'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[sample barcode read]` | 'index1, index2'×136; 'index2'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[sample barcode size]` | '8'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[single cell isolation]` | '10x technology'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[spike in]` | 'none'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[cdna read]` | 'read2'×136; '<NaN>'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[primer]` | 'oligo-dT'×136; '<NaN>'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[umi barcode offset]` | '16.0'×136; '<NaN>'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[umi barcode read]` | 'read1'×136; '<NaN>'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Comment[umi barcode size]` | '12.0'×136; '<NaN>'×104 | 2 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.3` | 'P-MTAB-134044'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Performer` | 'Cancer Research UK Cambridge I'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Assay Name` | 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×4; 'SLX-20673.SINAB6.H2M3FDMXY.s_1'×4; 'SLX-20673.SINAB6.H2M3FDMXY.s_2'×4; 'SLX-21747.SINAF12.H37H3DRX2.s_'×4; 'SLX-21747.SINAF12.H37H3DRX2.s_'×4 | 60 | <TODO: not in proposed_mapping> |  |
| `Comment[technical replicate group]` | 'group 14'×36; 'group 13'×36; 'group 1'×12; 'group 2'×12; 'group 12'×8 | 24 | <TODO: not in proposed_mapping> |  |
| `Technology Type` | 'sequencing assay'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Comment[ENA_EXPERIMENT]` | 'ERX10975667'×36; 'ERX10975666'×36; 'ERX10975662'×12; 'ERX10975673'×12; 'ERX10975665'×8 | 24 | <TODO: not in proposed_mapping> |  |
| `Scan Name` | 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20673.SINAB6.H2M3FDMXY.s_1'×1 | 240 | <TODO: not in proposed_mapping> |  |
| `Comment[SUBMITTED_FILE_NAME]` | 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20183.SINAB6.H3MTJDRXY.s_1'×1; 'SLX-20673.SINAB6.H2M3FDMXY.s_1'×1 | 240 | <TODO: not in proposed_mapping> |  |
| `Comment[ENA_RUN]` | 'ERR11571537'×4; 'ERR11571546'×4; 'ERR11571529'×4; 'ERR11571517'×4; 'ERR11571545'×4 | 60 | <TODO: not in proposed_mapping> |  |
| `Comment[FASTQ_URI]` | 'ftp://ftp.sra.ebi.ac.uk/vol1/r'×1; 'ftp://ftp.sra.ebi.ac.uk/vol1/r'×1; 'ftp://ftp.sra.ebi.ac.uk/vol1/r'×1; 'ftp://ftp.sra.ebi.ac.uk/vol1/r'×1; 'ftp://ftp.sra.ebi.ac.uk/vol1/r'×1 | 240 | <TODO: not in proposed_mapping> |  |
| `Comment[READ_TYPE]` | 'sample_barcode'×94; 'paired'×52; 'cell_barcode, umi_barcode'×34; 'single'×34; 'cell_barcode'×26 | 5 | <TODO: not in proposed_mapping> |  |
| `Comment[READ_INDEX]` | 'index1'×60; 'read1'×60; 'index2'×60; 'read2'×60 | 4 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.4` | 'P-MTAB-134045'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.5` | 'P-MTAB-134046'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Derived Array Data File` | 'TS21_4_H-atac_fragments.tsv.gz'×48; 'TS21_4_I-atac_fragments.tsv.gz'×48; 'H_7_B-atac_fragments.tsv.gz'×16; 'H_7_A-atac_fragments.tsv.gz'×16; 'H_7_C-atac_fragments.tsv.gz'×16 | 12 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.6` | 'P-MTAB-134045'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Protocol REF.7` | 'P-MTAB-134046'×240 | 1 | <TODO: not in proposed_mapping> |  |
| `Derived Array Data File.1` | 'TS21_4_H-filtered_feature_bc_m'×48; 'TS21_4_I-filtered_feature_bc_m'×48; 'H_7_B-filtered_feature_bc_matr'×16; 'H_7_A-filtered_feature_bc_matr'×16; 'H_7_C-filtered_feature_bc_matr'×16 | 12 | <TODO: not in proposed_mapping> |  |
| `Factor Value[disease]` | 'Down syndrome'×152; 'normal'×88 | 2 | <TODO: not in proposed_mapping> |  |
| `Factor Value[protocol]` | 'RNA-Seq'×136; 'ATAC-seq'×104 | 2 | <TODO: not in proposed_mapping> |  |

_Proposed columns NOT present in source dataframe (placeholders for nested obs / Zenodo / SDRF joins):_

- `Characteristics[sex]` → `sex`

## HTAN pan-cancer — column-rename proposal (inspect_dataset_metadata.py 2026-05-20 20:54)

Source dataframe shape: 136 rows × 24 cols

| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |
|---|---|---:|---|---|
| `sample_id` | 'CE336E1-S1'×1; 'CE337E1-S1'×1; 'CE338E1-S1'×1; 'CE339E1-S1'×1; 'CE340E1-S1'×1 | 136 | batch |  |
| `cancer_type` | 'PDAC'×25; 'HNSCC'×23; 'BRCA'×21; 'SKCM'×21; 'CRC'×18 | 9 | cancer_type |  |
| `organ` | 'Pancreas NOS'×25; 'Breast NOS'×15; 'Cervix uteri'×14; 'Colon NOS'×9; 'Ovary'×8 | 32 | tissue |  |
| `diagnosis` | 'Squamous cell carcinoma NOS'×32; 'Adenocarcinoma metastatic NOS'×22; 'Ductal carcinoma NOS'×13; 'Melanoma NOS'×12; 'Malignant melanoma NOS'×9 | 23 | condition |  |
| `fragment_file_path` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 136 | fragment_file_path |  |
| `matrix_dir` | '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 136 | <TODO: not in proposed_mapping> |  |
| `atac_annotation_rds` | '<NaN>'×4; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 133 | <TODO: not in proposed_mapping> |  |
| `rna_annotation_rds` | '<NaN>'×13; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1; '/nfs/team205/vk7/sanger_projec'×1 | 124 | <TODO: not in proposed_mapping> |  |
| `n_atac_annotations` | '1'×125; '2'×7; '0'×4 | 3 | <TODO: not in proposed_mapping> |  |
| `n_rna_annotations` | '1'×122; '0'×13; '2'×1 | 3 | <TODO: not in proposed_mapping> |  |
| `atac_annotation_csv` | '/nfs/team205/vk7/sanger_projec'×132; '<NaN>'×4 | 2 | <TODO: not in proposed_mapping> |  |
| `rna_annotation_csv` | '/nfs/team205/vk7/sanger_projec'×123; '<NaN>'×13 | 2 | <TODO: not in proposed_mapping> |  |
| `annotation_source` | 'atac_l4'×132; 'none'×4 | 2 | <TODO: not in proposed_mapping> |  |
| `piece_id` | '<NaN>'×37; 'CE336E1-S1'×1; 'CE346E1-S1'×1; 'CE347E1-S1K1'×1; 'CE348E1-S1K1'×1 | 100 | <auxiliary; join key only> |  |
| `cancer_type_lookup` | '<NaN>'×37; 'HNSCC'×22; 'SKCM'×21; 'PDAC'×18; 'CRC'×11 | 10 | <TODO: not in proposed_mapping> |  |
| `atac_data_type` | '10x_SC_Multi_ATAC_SEQ'×99; '<NaN>'×37 | 2 | <auxiliary; drop> |  |
| `raw_data_uploaded_to` | 'dbGAP, browse through HTAN DCC'×99; '<NaN>'×37 | 2 | <auxiliary; drop> |  |
| `cds_sample_name` | '<NaN>'×136 | 1 | <auxiliary; drop> |  |
| `processed_data_uploaded_to` | 'Synapse,browse through HTAN DC'×99; '<NaN>'×37 | 2 | <auxiliary; drop> |  |
| `donor_id` | '<NaN>'×37; 'HTA12_102'×2; 'HTA12_112'×2; 'HTA12_148'×2; 'HTA12_28'×2 | 93 | donor |  |
| `biospecimen_id` | '<NaN>'×37; 'HTA12_85_1'×1; 'HTA12_91_1'×1; 'HTA12_92_1'×1; 'HTA12_93_1'×1 | 100 | <auxiliary; HTAN biospecimen ID> |  |
| `geo_sample_name` | '<NaN>'×136 | 1 | <auxiliary; drop> |  |
| `source_sheet` | 'ATAC data'×99; '<NaN>'×37 | 2 | <auxiliary; drop> |  |
| `gdc_bam_file_id` | '<NaN>'×136 | 1 | <auxiliary; drop> |  |
