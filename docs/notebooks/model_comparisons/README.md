# Bone Marrow Model Comparisons

Systematic experiments on the NeurIPS 2021 adult bone marrow multiome dataset (~35k cells, RNA+ATAC, 5 batches, 4 sites, 8 donors). All regularizedvi models use n_hidden=512, n_latent=128, GammaPoisson likelihood with variational LogNormal dispersion. Experiments progress from basic feature toggles (Era 1) through training refinements (Era 2) to decoder initialization sweeps (Era 3) and a full factorial design (Era 4).

## Templates

- `bone_marrow_multimodal_tutorial_early_stopping.ipynb` — multimodal (RNA+ATAC) template
- `bone_marrow_gamma_poisson_early_stopping.ipynb` — RNA-only template

## Era 1: Early Experiments

Initial exploration of background correction, library centering, library variance, and learnable modality scaling. All use default early stopping.

| Name | Type | Key Changes | Notebook |
|------|------|-------------|----------|
| exp1_rna_bg_noctr | RNA | bg=T, centering=OFF | `bone_marrow_gp_es_exp1_out.ipynb` |
| exp2_rna_nobg_noctr | RNA | bg=F, centering=OFF | `bone_marrow_gp_es_exp2_out.ipynb` |
| exp3_rna_bg_ctr | RNA | bg=T, centering=ON (1.0) | `bone_marrow_gp_es_exp3_out.ipynb` |
| exp4_rna_nobg_ctr | RNA | bg=F, centering=ON (1.0) | `bone_marrow_gp_es_exp4_out.ipynb` |
| exp5_mm_noctr | MM | centering=OFF | `bone_marrow_mm_es_exp5_out.ipynb` |
| exp6_mm_ctr_rna | MM | centering=RNA only (1.0) | `bone_marrow_mm_es_exp6_out.ipynb` |
| exp7_mm_ctr_both | MM | centering=RNA(1.0)+ATAC(0.2) | `bone_marrow_mm_es_exp7_out.ipynb` |
| exp8_mm_ctr_both_lowes | MM | centering=both, ES=3e-5 | `bone_marrow_mm_es_exp8_out.ipynb` |
| exp9_rna_nobg_ctr_lowes | RNA | centering=ON, ES=3e-5 | `bone_marrow_gp_es_exp9_out.ipynb` |
| exp10a_mm_libvar_0.1 | MM | centering=both, libvar=0.1 | `bone_marrow_mm_es_exp10a_out.ipynb` |
| exp10b_mm_libvar_0.2 | MM | centering=both, libvar=0.2 | `bone_marrow_mm_es_exp10b_out.ipynb` |
| exp11_mm_learnable_scale | MM | centering=both, learnable modality scaling | `bone_marrow_mm_es_exp11_out.ipynb` |
| exp12_rna_libvar_0.2 | RNA | centering=ON, libvar=0.2 | `bone_marrow_gp_es_exp12_out.ipynb` |

## Era 2: Stratified + Filtered

Added stratified validation splits (`l1_cell_type+batch`), ATAC feature filtering (>2k), and swept library variance (0.2, 0.5, 1.0), early stopping thresholds, per-modality library variance, and background prior strength.

| Name | Type | Key Changes | Notebook |
|------|------|-------------|----------|
| exp8stratified | MM | centering=both, stratified | `bone_marrow_mm_es_exp8stratified_out.ipynb` |
| exp9stratified | RNA | centering=ON, stratified | `bone_marrow_gp_es_exp9stratified_out.ipynb` |
| exp10bstratified | MM | libvar=0.2, stratified | `bone_marrow_mm_es_exp10bstratified_out.ipynb` |
| exp11stratified | MM | learnable scale, stratified | `bone_marrow_mm_es_exp11stratified_out.ipynb` |
| exp12stratified | RNA | libvar=0.2, stratified | `bone_marrow_gp_es_exp12stratified_out.ipynb` |
| exp10bstratified_filtered | MM | libvar=0.2, stratified, ATAC>2k | `bone_marrow_mm_es_exp10bstratified_filtered_out.ipynb` |
| exp10bstratified_lowes | MM | libvar=0.2, stratified, ES=1e-4 | `bone_marrow_mm_es_exp10bstratified_lowes_out.ipynb` |
| exp11stratified_lowes | MM | learnable scale, stratified, ES=1e-4 | `bone_marrow_mm_es_exp11stratified_lowes_out.ipynb` |
| exp10bstratified_filtered_lowes | MM | libvar=0.2, ATAC>2k, ES=1e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes_out.ipynb` |
| exp10bstratified_filtered_lowes_libvar05 | MM | libvar=0.5, ATAC>2k, ES=1e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes_libvar05_out.ipynb` |
| exp10bstratified_filtered_lowes2_libvar05 | MM | libvar=0.5, ATAC>2k, ES=2e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes2_libvar05_out.ipynb` |
| exp10bstratified_filtered_lowes_ataclr2 | MM | ATAC LR 2x, ES=1e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes_ataclr2_out.ipynb` |
| exp10bstratified_filtered_lowes2_libvar10 | MM | libvar=1.0, ES=2e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes2_libvar10_out.ipynb` |
| exp12stratified_filtered_lowes2_libvar10 | RNA | libvar=1.0, ES=2e-4 | `bone_marrow_gp_es_exp12stratified_filtered_lowes2_libvar10_out.ipynb` |
| exp10bstratified_filtered_lowes2_libvar05_repeat2 | MM | libvar=0.5, ES=2e-4, repeat run | `bone_marrow_mm_es_exp10bstratified_filtered_lowes2_libvar05_repeat2_out.ipynb` |
| exp10bstratified_filtered_lowes2_libvar02_ataclibvar15 | MM | RNA-lv=0.2, ATAC-lv=1.5, ES=2e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes2_libvar02_ataclibvar15_out.ipynb` |
| exp10bstratified_filtered_lowes2_libvar05_bgprior1 | MM | libvar=0.5, bg prior G(1,1), ES=2e-4 | `bone_marrow_mm_es_exp10bstratified_filtered_lowes2_libvar05_bgprior1_out.ipynb` |
| exp12stratified_filtered_lowes2_libvar05 | RNA | libvar=0.5, ES=2e-4 | `bone_marrow_gp_es_exp12stratified_filtered_lowes2_libvar05_out.ipynb` |
| exp12stratified_filtered_lowes2_libvar05_bgprior1 | RNA | libvar=0.5, bg G(1,1) — FAILED (wandb) | `bone_marrow_gp_es_exp12stratified_filtered_lowes2_libvar05_bgprior1_out.ipynb` |
| exp12stratified_filtered_lowes2_libvar02_bgprior1 | RNA | libvar=0.2, bg G(1,1) | `bone_marrow_gp_es_exp12stratified_filtered_lowes2_libvar02_bgprior1_out.ipynb` |
| exp12stratified_filtered_lowes2_libvar02_bgprior033 | RNA | libvar=0.2, bg G(1,3) | `bone_marrow_gp_es_exp12stratified_filtered_lowes2_libvar02_bgprior033_out.ipynb` |
| exp12stratified_filtered_lowes2_libvar05_bgprior033 | RNA | libvar=0.5, bg G(1,3) | `bone_marrow_gp_es_exp12stratified_filtered_lowes2_libvar05_bgprior033_out.ipynb` |

## Era 3: Decoder Init Sweep (libvar05 era)

All experiments share: `regularise_background=0`, `centering_rna=1.0`, `centering_atac=0.2`, `libvar=0.5`, stratified validation, `ES=2e-4`.

| Name | decoder_weight_l2 | init_decoder_bias | bg_init_gene_fraction | Notebook |
|------|--------------------|-------------------|-----------------------|----------|
| mm_es_libvar05_dwl2_00001 | 1e-4 | — | — | `bone_marrow_mm_es_libvar05_dwl2_00001_out.ipynb` |
| mm_es_libvar05_dwl2_0001 | 1e-3 | — | — | `bone_marrow_mm_es_libvar05_dwl2_0001_out.ipynb` |
| mm_es_libvar05_dwl2_001 | 0.01 | — | — | `bone_marrow_mm_es_libvar05_dwl2_001_out.ipynb` |
| mm_es_libvar05_dwl2_01 | 0.1 | — | — | `bone_marrow_mm_es_libvar05_dwl2_01_out.ipynb` |
| mm_es_libvar05_dwl2_1 | 1.0 | — | — | `bone_marrow_mm_es_libvar05_dwl2_1_out.ipynb` |
| mm_es_libvar05_bias_mean | 0.0 | mean | — | `bone_marrow_mm_es_libvar05_bias_mean_out.ipynb` |
| mm_es_libvar05_bias_mean_bg02 | 0.0 | mean | 0.2 | `bone_marrow_mm_es_libvar05_bias_mean_bg02_out.ipynb` |
| mm_es_libvar05_bg_gene02 | 0.0 | — | 0.2 | `bone_marrow_mm_es_libvar05_bg_gene02_out.ipynb` |
| mm_es_libvar05_bias_topN_bg02 | 0.0 | topN | 0.2 | `bone_marrow_mm_es_libvar05_bias_topN_bg02_out.ipynb` |
| mm_es_libvar05_dwl2_001_bias_mean_bg02 | 0.01 | mean | 0.2 | `bone_marrow_mm_es_libvar05_dwl2_001_bias_mean_bg02_out.ipynb` |
| mm_es_libvar05_dwl2_01_bias_mean_bg02 | 0.1 | mean | 0.2 | `bone_marrow_mm_es_libvar05_dwl2_01_bias_mean_bg02_out.ipynb` |
| mm_es_libvar05_dwl2_001_bias_mean_bg02_repeat2 | 0.01 | mean | 0.2 | `bone_marrow_mm_es_libvar05_dwl2_001_bias_mean_bg02_repeat2_out.ipynb` |
| mm_es_libvar05_dwl2_01_bias_mean_bg02_repeat2 | 0.1 | mean | 0.2 | `bone_marrow_mm_es_libvar05_dwl2_01_bias_mean_bg02_repeat2_out.ipynb` |
| **mm_es_libvar05_dwl2_01_bias_mean_batchbg02** | **0.1** | **mean** | **0.2** | `bone_marrow_mm_es_libvar05_dwl2_01_bias_mean_batchbg02_out.ipynb` |

Best experiment: **mm_es_libvar05_dwl2_01_bias_mean_batchbg02** (dwl2=0.1, bias=mean, bg_init=0.2 per batch).

## Era 4: V2 Factorial Sweep

Built on the Era 3 best config. New features: dispersion prior mean sweep (1.0/2.0/3.0), ATAC centering sensitivity (0.2/1.0), decoder L2 (0.1/1.0), residual library encoder (on/off). All pending.

| Name | Type | disp_mean | dwl2 | atac_sens | residual_lib | Notebook |
|------|------|-----------|------|-----------|--------------|----------|
| mm_v2_disp3_dwl2_01_atacsens02 | MM | 3.0 | 0.1 | 0.2 | True | `bone_marrow_mm_v2_disp3_dwl2_01_atacsens02_out.ipynb` |
| mm_v2_disp3_dwl2_01_atacsens10 | MM | 3.0 | 0.1 | 1.0 | True | `bone_marrow_mm_v2_disp3_dwl2_01_atacsens10_out.ipynb` |
| mm_v2_disp3_dwl2_1_atacsens02 | MM | 3.0 | 1.0 | 0.2 | True | `bone_marrow_mm_v2_disp3_dwl2_1_atacsens02_out.ipynb` |
| mm_v2_disp3_dwl2_1_atacsens10 | MM | 3.0 | 1.0 | 1.0 | True | `bone_marrow_mm_v2_disp3_dwl2_1_atacsens10_out.ipynb` |
| mm_v2_disp2_dwl2_01_atacsens02 | MM | 2.0 | 0.1 | 0.2 | True | `bone_marrow_mm_v2_disp2_dwl2_01_atacsens02_out.ipynb` |
| mm_v2_disp2_dwl2_01_atacsens10 | MM | 2.0 | 0.1 | 1.0 | True | `bone_marrow_mm_v2_disp2_dwl2_01_atacsens10_out.ipynb` |
| mm_v2_disp2_dwl2_1_atacsens02 | MM | 2.0 | 1.0 | 0.2 | True | `bone_marrow_mm_v2_disp2_dwl2_1_atacsens02_out.ipynb` |
| mm_v2_disp2_dwl2_1_atacsens10 | MM | 2.0 | 1.0 | 1.0 | True | `bone_marrow_mm_v2_disp2_dwl2_1_atacsens10_out.ipynb` |
| mm_v2_disp1_dwl2_01_atacsens02 | MM | 1.0 | 0.1 | 0.2 | True | `bone_marrow_mm_v2_disp1_dwl2_01_atacsens02_out.ipynb` |
| mm_v2_disp1_dwl2_01_atacsens10 | MM | 1.0 | 0.1 | 1.0 | True | `bone_marrow_mm_v2_disp1_dwl2_01_atacsens10_out.ipynb` |
| mm_v2_disp1_dwl2_1_atacsens02 | MM | 1.0 | 1.0 | 0.2 | True | `bone_marrow_mm_v2_disp1_dwl2_1_atacsens02_out.ipynb` |
| mm_v2_disp1_dwl2_1_atacsens10 | MM | 1.0 | 1.0 | 1.0 | True | `bone_marrow_mm_v2_disp1_dwl2_1_atacsens10_out.ipynb` |
| mm_v2_disp3_dwl2_01_atacsens02_noreslib | MM | 3.0 | 0.1 | 0.2 | False | `bone_marrow_mm_v2_disp3_dwl2_01_atacsens02_noreslib_out.ipynb` |
| rna_v2_disp3_dwl2_01 | RNA | 3.0 | 0.1 | — | True | `bone_marrow_rna_v2_disp3_dwl2_01_out.ipynb` |
| rna_v2_disp3_dwl2_1 | RNA | 3.0 | 1.0 | — | True | `bone_marrow_rna_v2_disp3_dwl2_1_out.ipynb` |
| rna_v2_disp2_dwl2_01 | RNA | 2.0 | 0.1 | — | True | `bone_marrow_rna_v2_disp2_dwl2_01_out.ipynb` |
| rna_v2_disp2_dwl2_1 | RNA | 2.0 | 1.0 | — | True | `bone_marrow_rna_v2_disp2_dwl2_1_out.ipynb` |
| rna_v2_disp1_dwl2_01 | RNA | 1.0 | 0.1 | — | True | `bone_marrow_rna_v2_disp1_dwl2_01_out.ipynb` |
| rna_v2_disp1_dwl2_1 | RNA | 1.0 | 1.0 | — | True | `bone_marrow_rna_v2_disp1_dwl2_1_out.ipynb` |
| rna_v2_disp3_dwl2_01_noreslib | RNA | 3.0 | 0.1 | — | False | `bone_marrow_rna_v2_disp3_dwl2_01_noreslib_out.ipynb` |

## Links

- [`experiments.tsv`](experiments.tsv) — complete experiment tracking table with all hyperparameters
- [`experiment_report.html`](experiment_report.html) — visual report with plots
- [`parameter_diagnostics.ipynb`](parameter_diagnostics.ipynb) — interactive comparison notebook
