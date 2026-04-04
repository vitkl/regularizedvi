# Plan: Dispersion Init — Training Notebook Integration

## Previous Plan Status
Previous plan at `.claude/plans/dispersion-init-plan.md` — completed tasks:
- Task 1: P5 theta extraction — **DONE** (commit `d59db10`)
- Task 2: `_dispersion_init.py` + array px_r_init_mean — **DONE** (commit `d59db10`)
- Task 3: Exploration notebook + MoM computation — **DONE** (4/6 npz, 2 ATAC jobs running)
- Task 4: `dispersion_init="data"` in model constructors — **DONE** (commit `9507b58`)
- bio_frac rename + batch Welford — **DONE** (commit `abbdd8f`)
- Gamma distribution fitting — **DONE** (script created, results show RNA rate≈0.7–1.1 vs current 3)
- **Task 5: Training notebook integration — THIS PLAN**

## Context
MoM analysis shows current `regularise_dispersion_prior=3` (θ mode=9) is too high.
Data supports θ ≈ 1–3 for RNA, θ ≈ 0.3 for ATAC. We now test `dispersion_init="data"` + `prior=1` on 3 representative training runs.

## ATAC Jobs (monitoring)
- `disp_embryo_atac_v4` (job 854121) — running, 200GB, CRE subset via backed mode
- `disp_immune_atac_v4` (job 854122) — running, 200GB, CRE subset via backed mode

---

## 3 Training Notebooks to Modify

### Hyperparameter Table

| Parameter | BM Multimodal | Immune RNA | Embryo 4-mod |
|-----------|--------------|------------|--------------|
| **Template** | `bone_marrow_multimodal_tutorial_early_stopping.ipynb` | `bm_pbmc_rna_training.ipynb` | `embryo_rna_atac_spliced_unspliced.ipynb` |
| **Model class** | RegularizedMultimodalVI | AmbientRegularizedSCVI | RegularizedMultimodalVI |
| **Modalities** | RNA + ATAC | RNA only | RNA + ATAC + Spliced + Unspliced |
| **n_hidden** | {rna: 512, atac: 256} | 512 | {rna: 1600, spl: 400, unspl: 400, atac: 800} |
| **n_latent** | {rna: 128, atac: 64} | 128 | {rna: 400, spl: 100, unspl: 100, atac: 200} |
| **n_layers** | 1 | 1 | 1 |
| **dispersion** | gene-batch | (default) | gene-batch |
| **dispersion_prior_mean** | 3→**1** | (not param yet)→**1** | 3→**1** |
| **dispersion_prior_alpha** | 9 | 9 | 9 |
| **dispersion_init** | (new)→**"data"** | (new)→**"data"** | (new)→**"data"** |
| **dispersion_init_bio_frac** | (new)→**0.9** | (new)→**0.9** | (new)→**0.9** |
| **decoder_weight_l2** | 0.1 | (not param) | 0.1 |
| **residual_library_encoder** | True | (not param)→**True** | True |
| **library_log_vars_weight** | 0.5 | 0.5 | 0.5 |
| **atacsens (ATAC centering)** | 0.2 | N/A | 0.2 |
| **additive_bg_modalities** | [rna] | (single-modal) | [rna, spliced, unspliced] |
| **feature_scaling_modalities** | [rna, atac] | (via covariate keys) | [rna, spl, unspl, atac] |
| **regularise_background** | False | False | False |
| **batch_size** | 1024 | (default) | 1024 |
| **max_epochs** | 5000 | 2000 | 5000 |
| **early_stopping_patience** | 20 | 10 | 20 |
| **ES min_delta** | 0.0002 | 0.0002 | 0.0001 |
| **stratify_key** | l1_cell_type+batch | harmonized_annotation+batch | cell_type_lvl7_reviewed_unique |
| **Queue** | gpu-normal | gpu-normal | gpu-huge |

### New Parameters to Add (papermill)

For each notebook, add these as papermill parameters with default values that preserve backward compatibility:

```python
# Dispersion initialization (new parameters)
dispersion_init = "prior"              # "prior" (default, no change) or "data"
dispersion_init_bio_frac = 0.9         # biological variance fraction
```

When injected via papermill: `dispersion_init = "data"`, `dispersion_prior_mean = "1.0"`.

### Changes per Notebook

#### 1. Bone Marrow Multimodal
**File**: `docs/notebooks/model_comparisons/bone_marrow_multimodal_tutorial_early_stopping.ipynb`
**Reference output**: `bone_marrow_mm_v2_disp1_dwl2_01_atacsens02_out.ipynb`

Changes:
- **Parameters cell**: Add `dispersion_init = "prior"`, `dispersion_init_bio_frac = 0.9`
- **Coercion cell**: Add `dispersion_init=("str_or_none", dispersion_init)`, `dispersion_init_bio_frac=(float, dispersion_init_bio_frac)`
- **Model constructor cell**: Add `dispersion_init=dispersion_init, dispersion_init_bio_frac=dispersion_init_bio_frac`
- **Papermill command**: `-p dispersion_init data -p dispersion_prior_mean 1.0`

#### 2. Immune RNA
**Template**: `docs/notebooks/immune_integration/bm_pbmc_rna_training_v2.ipynb` (commit `7ded4fc`)
**Based on**: `bm_pbmc_rna_training_embryo_es2_no_tea_disp1_out.ipynb` — stripped outputs, rich post-training plots
**Old template** (`bm_pbmc_rna_training.ipynb`): missing most visualization cells, do NOT use

Template features:
- `exclude_datasets = "tea_seq"` (tea-seq removed by default)
- `n_hidden`/`n_latent` as papermill params (default 1024/256)
- `dispersion_init`, `dispersion_init_bio_frac` params added
- All prior/decoder/residual params already present
- Rich plots: per-study UMAPs, marker genes, dotplots, integration metrics

Two runs submitted:
- **Small (512/128)**: `-p n_hidden 512 -p n_latent 128 -p dispersion_init data -p dispersion_prior_mean 1.0`
- **Large (1024/256)**: `-p dispersion_init data -p dispersion_prior_mean 1.0` (uses defaults)

#### 3. Embryo 4-Modality
**File**: `/nfs/team205/vk7/sanger_projects/cell2state_embryo/notebooks/benchmark/regularizedvi/embryo_rna_atac_spliced_unspliced.ipynb`

Changes:
- **Parameters cell (1)**: Add `dispersion_init = "prior"`, `dispersion_init_bio_frac = 0.9`
- **Coercion cell (2)**: Add coercion for `dispersion_init` (str_or_none) and `dispersion_init_bio_frac` (float)
- **Model constructor cell (21)**: Add `dispersion_init=dispersion_init, dispersion_init_bio_frac=dispersion_init_bio_frac`
- **Papermill command**: `-p dispersion_init data -p dispersion_prior_mean 1.0`

---

## Job Submission

### Papermill Commands

```bash
# 1. Bone marrow multimodal (gpu-normal, 40GB RAM, 80GB GPU)
TEMPLATE="docs/notebooks/model_comparisons/bone_marrow_multimodal_tutorial_early_stopping.ipynb"
OUTPUT="docs/notebooks/model_comparisons/bone_marrow_mm_v2_disp_data_prior1_atacsens02_out.ipynb"
bsub -q gpu-normal -n 8 -M 40000 -R"select[mem>40000] rusage[mem=40000] span[hosts=1]" \
  -gpu "mode=shared:j_exclusive=yes:gmem=80000:num=1" \
  -e ./%J.gpu.err -o ./%J.gpu.out -J bm_mm_disp_data \
  "PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && \
   papermill $TEMPLATE $OUTPUT \
   -p dispersion_init data -p dispersion_prior_mean 1.0 \
   -p decoder_weight_l2 0.1 -p library_log_means_centering_sensitivity_atac 0.2 \
   -p results_folder results/mm_disp_data_prior1_atacsens02/ \
   -p wandb_name mm_disp_data_prior1_atacsens02"

# 2. Immune RNA (gpu-normal, 40GB RAM, 80GB GPU)
TEMPLATE="docs/notebooks/immune_integration/bm_pbmc_rna_training.ipynb"
OUTPUT="docs/notebooks/immune_integration/bm_pbmc_rna_training_disp_data_prior1_out.ipynb"
bsub -q gpu-normal -n 8 -M 40000 -R"select[mem>40000] rusage[mem=40000] span[hosts=1]" \
  -gpu "mode=shared:j_exclusive=yes:gmem=80000:num=1" \
  -e ./%J.gpu.err -o ./%J.gpu.out -J immune_rna_disp_data \
  "PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && \
   papermill $TEMPLATE $OUTPUT \
   -p dispersion_init data -p dispersion_prior_mean 1.0 \
   -p results_folder results/immune_integration_rna_disp_data_prior1/ \
   -p wandb_name immune_rna_disp_data_prior1"

# 3. Embryo 4-modality (gpu-huge, 100GB RAM, 80GB GPU)
TEMPLATE="/nfs/team205/vk7/sanger_projects/cell2state_embryo/notebooks/benchmark/regularizedvi/embryo_rna_atac_spliced_unspliced.ipynb"
OUTPUT="/nfs/team205/vk7/sanger_projects/cell2state_embryo/notebooks/benchmark/regularizedvi/results/embryo_disp_data_prior1_out.ipynb"
bsub -q gpu-huge -n 8 -M 100000 -R"select[mem>100000] rusage[mem=100000] span[hosts=1]" \
  -gpu "mode=shared:j_exclusive=yes:gmem=80000:num=1" \
  -e ./%J.gpu.err -o ./%J.gpu.out -J embryo_disp_data \
  "PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && cd /nfs/team205/vk7/sanger_projects/cell2state_embryo && \
   papermill $TEMPLATE $OUTPUT \
   -p dispersion_init data -p dispersion_prior_mean 1.0 \
   -p run_name embryo_disp_data_prior1"
```

---

## Verification Steps

1. **Before submission**: Launch subagent to verify each modified notebook:
   - Parameter names match constructor signatures
   - Coercion types are correct
   - No broken cell references or undefined variables
   - New params have backward-compatible defaults ("prior" not "data")
2. **After training**: Compare training curves, final loss, integration metrics vs baselines
3. **Check**: Does `dispersion_init="data"` produce reasonable log messages during init?

## Critical Files
| File | Change |
|------|--------|
| `docs/notebooks/model_comparisons/bone_marrow_multimodal_tutorial_early_stopping.ipynb` | Add dispersion_init, bio_frac params |
| `docs/notebooks/immune_integration/bm_pbmc_rna_training.ipynb` | Add dispersion_init, prior_mean, residual_lib, decoder_l2 params |
| `cell2state_embryo/.../embryo_rna_atac_spliced_unspliced.ipynb` | Add dispersion_init, bio_frac params |
