# Model comparisons (bone marrow dataset)

Systematic comparison of regularizedvi vs standard scVI on the NeurIPS 2021 bone marrow multiome dataset (46,534 cells, 5 batches, 4 sites, 8 donors).

## Key findings

- **Large scVI overfits at 2000 epochs**: Standard scVI with large architecture (512/128/1) shows double descent and overfitting when trained for 2000 epochs. The same model with default training settings (~400 epochs, batch_size=128) performs better.
- **regularizedvi works across model sizes**: GammaPoisson mode produces good integration at all tested architectures (128/10, 256/30, 512/128) without hyperparameter tuning.
- **All regularizedvi models still improving at 2000 epochs**: Training curves suggest regularizedvi benefits from longer training, motivating the early stopping experiment.
- **Dispersion (theta)**: Unconstrained scVI learns theta ~1-2 (high overdispersion), while regularizedvi GammaPoisson learns theta ~53 (closer to Poisson). The prior prevents NB collapse during training.
- **Batch-free decoder**: Separating constrained additive (ambient RNA per batch) from flexible multiplicative (categorical covariates for donor/protocol) correction is the key architectural benefit.

## Notebook inventory

| Notebook | Model | Architecture | Epochs | Likelihood | Notes |
|----------|-------|-------------|--------|------------|-------|
| `bone_marrow_regularizedvi_nb.ipynb` | regularizedvi | 512/128/1 | 2000 | NB | Original NB default |
| `bone_marrow_gamma_poisson.ipynb` | regularizedvi | 512/128/1 | 2000 | GammaPoisson | Now the default |
| `bone_marrow_gamma_poisson_1375ep.ipynb` | regularizedvi | 512/128/1 | 1375 | GammaPoisson | Lueken-adjusted epochs |
| `bone_marrow_gamma_poisson_early_stopping.ipynb` | regularizedvi | 512/128/1 | ≤4000 | GammaPoisson | Early stopping + checkpoints |
| `bone_marrow_gamma_poisson_small.ipynb` | regularizedvi | 128/10/1 | 2000 | GammaPoisson | Small architecture |
| `bone_marrow_gamma_poisson_medium.ipynb` | regularizedvi | 256/30/1 | 2000 | GammaPoisson | Medium architecture |
| `bone_marrow_gamma_poisson_small_default_train.ipynb` | regularizedvi | 128/10/1 | default | GammaPoisson | Small + default scVI training |
| `bone_marrow_scvi_default.ipynb` | scVI | 128/30/1 | default | ZINB | scVI defaults |
| `bone_marrow_scvi_custom.ipynb` | scVI | 512/128/1 | 2000 | NB | Large scVI, 2000 epochs |
| `bone_marrow_scvi_custom_zinb.ipynb` | scVI | 512/128/1 | 2000 | ZINB | Large scVI, ZINB |
| `bone_marrow_scvi_custom_default_train.ipynb` | scVI | 512/128/1 | default | NB | Large scVI, default training |
| `bone_marrow_scvi_lueken.ipynb` | scVI | 128/30/2 | 1375 | ZINB | Lueken lung atlas settings |
| `model_comparison_theta.ipynb` | — | — | — | — | Theta distribution analysis |
| `distribution_comparison.ipynb` | — | — | — | — | NB vs GammaPoisson comparison |

Architecture notation: `n_hidden/n_latent/n_layers`. "default" training = scVI defaults (~400 epochs, batch_size=128).

Epochs for Lueken-adjusted training: `round(400 * (20000/N) * (batch_size/128))` = 1375 for N=46534, batch_size=1024.

## Ablation experiments (exp1–12+)

Systematic ablation study investigating dispersion drift with longer training. All experiments use GammaPoisson likelihood with variational LogNormal dispersion posterior and early stopping. Executed via `_gpu_jobs.yaml` + papermill. Diagnostics in `parameter_diagnostics.ipynb`.

### Single-modal RNA (exp1–4): regularise_background x centering

| Exp | bg prior | centering | ES | theta_rna | Key finding |
|-----|----------|-----------|-----|-----------|-------------|
| exp1 | True | OFF | default | 70.7 | Uncentered: theta explosion |
| exp2 | False | OFF | default | 67.5 | bg prior has no effect without centering |
| exp3 | True | ON (1.0) | default | 11.5 | Centering fixes theta |
| exp4 | False | ON (1.0) | default | 11.7 | Best RNA baseline |

**Conclusion**: Library centering (`library_log_means_centering_sensitivity=1.0`) is the key fix. Background prior (`regularise_background`) has minimal effect.

### Multimodal RNA+ATAC (exp5–7): centering variations

| Exp | centering | ES | theta_rna | Key finding |
|-----|-----------|-----|-----------|-------------|
| exp5 | OFF | default | 19.1 | MM uncentered baseline |
| exp6 | RNA only (1.0) | default | 16.7 | RNA centering helps |
| exp7 | RNA (1.0) + ATAC (0.2) | default | 11.3 | Best MM baseline |

**Conclusion**: Centering both modalities (with lower ATAC sensitivity) gives best results.

### Lower early stopping sensitivity (exp8–9): longer training

| Exp | Model | centering | ES | theta_rna | Key finding |
|-----|-------|-----------|-----|-----------|-------------|
| exp8 | MM | RNA+ATAC | lowES (0.00003) | 20.4 | Longer training worsens theta |
| exp9 | RNA | ON (1.0) | lowES (0.00003) | 17.9 | Same — theta drifts up |

**Conclusion**: With 10x lower ES sensitivity, models train longer but dispersion drifts upward. Default ES catches this at the right time.

### Wider library prior variance (exp10a/b, exp12): hypothesis E

Tests whether relaxing the tight library prior (`library_log_vars_weight=0.05` → 0.1 or 0.2) allows the encoder to capture more per-cell variation, reducing pressure on dispersion.

| Exp | Model | lib var weight | ES | Purpose |
|-----|-------|---------------|-----|---------|
| exp10a | MM | 0.1 (2x wider) | default | Moderate relaxation |
| exp10b | MM | 0.2 (4x wider) | default | Aggressive relaxation |
| exp12 | RNA | 0.2 (4x wider) | default | Same test for single-modal |

### Learnable modality scaling (exp11): hypothesis D

| Exp | Model | Feature | ES | Purpose |
|-----|-------|---------|-----|---------|
| exp11 | MM | learnable per-modality scaling | default | Per-modality scalar on px_rate, initialized from centering sensitivity (RNA=1.0, ATAC=0.2), Gamma(5, 5/init) prior |

Tests whether a learnable per-modality scale factor prevents dispersion drift by allowing the model to adjust per-modality expected counts.

### Stratified validation split (hypothesis C): default ES

All stratified experiments use default ES (0.0003), making them directly comparable to exp7 (MM baseline) and exp4 (RNA baseline). The stratified split ensures proportional representation of cell types and batches in the validation set.

| Exp | Base hyperparams | ES | Purpose |
|-----|-----------------|-----|---------|
| exp8stratified | exp8 (MM, ctr=both) | default (0.0003) | Stratified split, comparable to exp7 |
| exp9stratified | exp9 (RNA, ctr=ON) | default (0.0003) | Stratified split, comparable to exp4 |
| exp10bstratified | exp10b (MM, lib var 0.2) | default (0.0003) | Stratified + wider lib var |
| exp11stratified | exp11 (MM, learnable scale) | default (0.0003) | Stratified + learnable scaling |
| exp12stratified | exp12 (RNA, lib var 0.2) | default (0.0003) | Stratified + wider lib var |

Tests whether stratified validation split (by `l1_cell_type+batch`) produces more stable metrics than random splitting.

### Hypotheses for dispersion drift

- **(A)** Pearson correlation is misleading; reconstruction loss is the correct metric
- **(B)** Both RNA and ATAC overfit by early stopping — default ES catches this just in time
- **(C)** Random validation cell selection gives variable metrics → stratified experiments test this
- **(D)** Modalities need learnable scaling factors → exp11 tests this
- **(E)** Library prior variance too tight → exp10a/b, exp12 test this
