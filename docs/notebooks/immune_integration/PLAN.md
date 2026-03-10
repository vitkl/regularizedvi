# Plan: Multi-dataset RNA + ATAC integration (7 datasets)

## Progress (updated 2026-03-10)

| Step | Status | Notes |
|------|--------|-------|
| Notebook 1: RNA loading | ✅ DONE | 776,742 cells × 25,629 genes, 99 batches, 7 datasets |
| Notebook 2: Scrublet | ✅ DONE | 22,438 doublets (2.9%), saved as CSV |
| SnapATAC2 jobs | ✅ DONE | All 99/99 completed successfully |
| data_loading_utils.py | ✅ DONE | All 7 loaders, validation, hierarchy, TEA-seq fix |
| TEA-seq annotations | ✅ FIXED | 26,165/59,151 annotated (GSM5123951-54 via Figure4 CSV) |
| Notebook 3: ATAC loading | ✅ CREATED | Needs `cell2state_v2026_cuda124_torch25` env |
| Notebook 4: QC summary | ❌ SKIPPED | QC shown in training notebooks instead |
| Notebook 5: RNA training | ✅ CREATED | 28 cells, all hyperparameters confirmed |
| Notebook 6: MM training | ✅ CREATED | 27 cells, matches reference experiments |
| NB 5.5: Annotation + CRE | 🔲 TODO | After NB3+NB5: label transfer + CRE selection |

### Confirmed decisions (2026-03-10)
- **Covariates**: `dataset` instead of `site` for nn_conditioning and feature_scaling (with `donor`)
- **Stratification**: `harmonized_annotation+batch`
- **Architecture**: n_hidden=512, n_layers=1, n_latent=128
- **Training**: ES delta=0.0002, patience=20, batch_size=1024, **max_epochs=2000**, **checkpoint every 200**
- **MM**: max_epochs=2000, checkpoint every 200; RNA centering=1.0, ATAC centering=0.2
- **MM architecture**: n_hidden={"rna":512,"atac":256}, n_latent={"rna":128,"atac":64}, concatenation mode
- **Cell filtering**: 1k-80k counts, 500-10k genes, MT<0.20, doublet<0.18, ATAC 2k-100k
- **Gene filtering**: `filter_genes(cell_count_cutoff=15, cell_percentage_cutoff2=0.01, nonz_mean_cutoff=1.04)`
- **ATAC max_frag_size_split**: 160 (up from 120, keeps fragments <1 nucleosome)

---

## Completed work (summaries)

### Notebook 1: RNA Data Loading — DONE
- **File**: `bm_pbmc_data_loading.ipynb`
- **Output**: `results/immune_integration/adata_rna.h5ad` (5.7 GB), `obs_metadata.csv`, `path_sample_df.csv`
- 7 datasets loaded, concatenated on 25,629 common genes, uint16 counts

### Notebook 2: Scrublet — DONE
- **File**: `bm_pbmc_scrublet.ipynb`
- **Output**: `results/immune_integration/scrublet_results.csv` (776,742 cells)
- MIN_CELLS=500, try/except for n_components ValueError
- 22,438 predicted doublets (2.9%), no failed batches
- Saved as CSV (avoids h5ad dtype bug)

### SnapATAC2 — DONE
- All 99 samples processed via LSF job array
- Each sample's `atac_fragments.h5ad` cached next to its `atac_fragments.tsv.gz`

### data_loading_utils.py — DONE
- All 7 loaders with validation, hierarchy, and annotation harmonization
- TEA-seq fix: mapped GSM5123951-54 to Seurat suffixes 3-6, annotated 26,165/59,151 cells
- All 26 TEA-seq labels match harmonization table

### Notebook 3: ATAC tile loading — CREATED
- **File**: `bm_pbmc_atac_loading.ipynb` (14 cells)
- Uses `cell2state_v2026_cuda124_torch25` env
- `concatenate_h5ad` with max_frag_size_split=160, bin_size=1000, insertion counting
- Output: `results/immune_integration/adata_atac_tiles.h5ad`

### Notebook 5: RNA-only training — CREATED
- **File**: `bm_pbmc_rna_training.ipynb` (28 cells)
- Loads adata_rna.h5ad + scrublet CSV, QC summary, cell/gene filtering
- setup: ambient=["batch"], nn_conditioning=["dataset","donor"], feature_scaling=["dataset","donor"]
- Model: n_hidden=512, n_layers=1, n_latent=128, regularise_background=False
- Training: max_epochs=2000, ES delta=0.0002, patience=20, checkpoint/200
- UMAP: dataset, batch, harmonized_annotation, level_1

### Notebook 6: Multimodal training — CREATED
- **File**: `bm_pbmc_multimodal_training.ipynb` (27 cells)
- Loads RNA + ATAC + scrublet, intersects cells, creates MuData
- Same covariates as NB5, encoder_covariate_keys=False
- Model: n_hidden={"rna":512,"atac":256}, n_latent={"rna":128,"atac":64}, concatenation
- additive_bg=["rna"], feature_scaling=["rna","atac"], regularise_dispersion=True
- Library centering: RNA=1.0, ATAC=0.2, log_vars_weight=0.2

---

## Next steps

### Step 1: Rerun NB1 (update adata_rna.h5ad with fixed TEA-seq annotations)
- data_loading_utils.py already fixed, just needs rerun via papermill
- Must complete before NB3 and NB5 can run

### Step 2: Submit NB3 ATAC loading + NB5 RNA training (parallel)
- **NB3**: `cell2state_v2026_cuda124_torch25` env, normal queue (high memory)
- **NB5**: `regularizedvi` env, GPU queue

### Step 3: Create NB5.5 — Annotation transfer + CRE selection
**File to create**: `bm_pbmc_annotation_cre.ipynb`
**Prerequisites**: NB3 (ATAC data) + NB5 (trained RNA model with latent space)

**Part A: Label transfer (implement now)**
- Use RNA latent space to propagate `harmonized_annotation` to unannotated cells
- kNN label transfer: k nearest annotated neighbors, majority vote
- Save updated annotations

**Part B: Annotation-associated CRE selection (stub for separate agent)**
- Reference: `cell2state_embryo/notebooks/benchmark/.../affinity_analysis_from_annotation_v2_sections.ipynb`
- Build reference_groups, annotation masks, run affinity analysis, select top-N CREs

### Step 4: Submit NB6 Multimodal training
- Needs NB3 output (adata_atac_tiles.h5ad) + NB5.5 output (CRE selection)
- GPU queue, regularizedvi env

---

## Execution order

```
Rerun NB1 (fix TEA-seq in adata_rna.h5ad)
  ↓
NB3 ATAC loading (cell2state env) ──────────┐
NB5 RNA training (GPU, regularizedvi) ──────┤ parallel
  ↓                                         ↓
NB5.5 Annotation transfer + CRE selection
  ↓
NB6 Multimodal training (GPU)
```

### Conda environments

| Task | Environment |
|------|-------------|
| Notebook 3 (ATAC loading) | `cell2state_v2026_cuda124_torch25` |
| All other notebooks | `regularizedvi` |

### GPU job submission template
```bash
bsub -q gpu-normal -n 8 -M 40000 \
  -R"select[mem>40000] rusage[mem=40000] span[hosts=1]" \
  -gpu "mode=shared:j_exclusive=yes:gmem=80000:num=1" \
  -e ./%J.gpu.err -o ./%J.gpu.out -J <job_name> \
  'PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && papermill <input.ipynb> <output.ipynb>'
```
