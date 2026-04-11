# regularizedvi

Bayesian extension of scVI for single-cell/nucleus RNA-seq and multiome (RNA+ATAC) data integration. Built on cell2location/cell2fate modelling principles (Kleshchevnikov et al. 2022, Aivazidis et al. 2025). Adds structural inductive biases — ambient RNA correction, hierarchical dispersion prior, batch-free decoder, learned library size — that make high-capacity models (n_hidden=512+, n_latent=128+) well-behaved without substantial per-dataset hyperparameter tuning. Designed for complex datasets with hundreds of cell types (whole-embryo atlases, cross-atlas integration).

## Two Models

| Model class | Module | File | Purpose |
|-------------|--------|------|---------|
| AmbientRegularizedSCVI | RegularizedVAE | `_module.py` (1372L) | Single-modal RNA |
| RegularizedMultimodalVI | RegularizedMultimodalVAE | `_multimodule.py` (1843L) | Multi-modal RNA+ATAC |

Supporting: `_model.py` (942L), `_multimodel.py` (1571L), `_components.py` (472L), `_constants.py` (78L).

## 5-Key Covariate Design (Core Innovation)
- `ambient_covariate_keys` — additive background per batch
- `nn_conditioning_covariate_keys` — decoder categorical injection
- `feature_scaling_covariate_keys` — multiplicative scaling
- `dispersion_key` — overdispersion groups
- `library_size_key` — library size prior groups
- `encoder_covariate_keys` — encoder injection (default False, matching scVI/MultiVI/PeakVI)
- `batch_key` alone fans out to `ambient_covariate_keys` `library_size_key` `dispersion_key` (backward compatibility)

## Mathematical Components
- **Ambient RNA**: Additive background s_e,g = exp(beta) with Gamma(1,100) prior — per gene/batch (by default not regularised with prior)
- **Feature scaling**: y_t,g = softplus(gamma)/0.7 with Gamma(200,200) prior — multiplicative bias per covariate group (by default regularised with prior)
- **Hierarchical dispersion**: Variational LogNormal posterior, containment prior 1/sqrt(theta) ~ Exp(lambda), lambda ~ Gamma(9,3)
- **Learned library size**: With 0.5 variance scaling on the prior
- **Observation model**: GammaPoisson (=NB), softplus activation (not softmax) since rho + s need not sum to 1

## Multi-Modal Architecture
Per-modality encoders, concatenated latent space [z_atac; z_rna] (alphabetical sort of modality_names), all decoders see full z. Decoders are symmetrical with options to switch off purpose-based covariates (unlike MultiVI).

### Modality Attribution (`get_modality_attribution()`)
- Computes Jacobian of decoder output w.r.t. each modality's latent dimensions using finite differences
- Per-cell attribution scores: ||J_rna||_F vs ||J_atac||_F (Frobenius norm of per-modality Jacobian blocks)
- `plot_attribution_scatter()`: convenience method for UMAP-colored attribution visualization

## Downstream Methods
- `get_latent_representation()`: returns posterior mean z (or sampled z) per cell
- `get_normalized_expression()`: returns denoised expression (decoder output, library-size normalized)
- `get_modality_attribution()`: Modality attribution via Jacobian analysis
- Both support `batch_size` for memory-efficient inference on large datasets

## Model Internals (RegularizedVAE)

### Inference Flow (`inference()`)
1. Input x (log-library-normalized) → z_encoder → qz (mean, var) → z sample
2. x → l_encoder → ql (library mean, var) → library sample
3. Returns dict: `z`, `qz`, `ql`, `library`
4. Continuous covariates always concatenated to encoder input
5. Categorical covariates to encoder only if `encoder_covariate_keys` explicitly set (default False)

### Generative Flow (`generative()`)
1. z → decoder → px_rate (rho, unnormalized gene expression)
2. Dispersion: sample theta from LogNormal(px_r_mu[group], px_r_log_sigma[group]) per cell
3. Feature scaling: y_t,g = softplus(gamma[group])/0.7, applied multiplicatively
4. Ambient RNA: s_e,g = exp(beta[batch]), added to rate
5. Final rate: `px_rate = (rho + s) * y * library` (softplus, not softmax — no sum-to-1 constraint)
6. Likelihood: GammaPoisson(theta, px_rate) — equivalent to NegativeBinomial

### Loss (`loss()`)
1. **Reconstruction**: -log p(x|z) via GammaPoisson log-prob
2. **KL(q(z)||p(z))**: standard VAE latent KL
3. **KL(q(l)||p(l))**: library size KL with 0.5 variance scaling
4. **Hierarchical dispersion penalty**: KL between variational LogNormal and containment prior
5. **Background penalty**: Gamma(1,100) prior on exp(beta) — keeps ambient small
6. **Feature scaling penalty**: Gamma(200,200) prior on softplus(gamma)/0.7 — keeps scaling near 1
7. All penalties logged via `extra_metrics` → `compute_and_log_metrics()` → `model.history_`

### Loss normalization convention (scvi-tools minibatching)
- `loss()` takes an explicit `n_obs` argument = **full training-set size**, injected automatically by `TrainingPlan.n_obs_training` setter via signature introspection (`signature(module.loss).parameters`). Validation also uses `n_obs_training` (not `n_val`) so train/val losses are on the same scale — scvi-tools convention (`_trainingplans.py:356-358`).
- **Local (cell-plate) terms** — reconstruction loss, `KL(qz‖pz)`, `KL(ql‖pl)`, z-sparsity, horseshoe KL, hidden-activation sparsity — are summed over non-batch dims and **meaned over the batch axis** inside the main `torch.mean(...)`.
- **Global (gene-plate / batch-plate / covariate-plate / plate-less) priors** — dispersion variational KL + λ hyperprior, ambient RNA β, feature scaling γ, decoder L1/L2, ARD on z, modality scaling, residual library `w` KL — are added to the loss as `penalty / n_obs` where **`n_obs` is the `loss()` argument (= N_train), NEVER `recon_loss.shape[0]` or `x.shape[0]` (minibatch size)**.
- `loss()` asserts `n_obs >= batch_size` (overridable via `skip_n_obs_check=True`) to catch missing injection at train time. Unit tests that call `module.loss(...)` directly must pass `skip_n_obs_check=True`.
- **Historical bug** (fixed 2026-04-11): prior to this fix, all global priors used `n_obs = recon_loss.shape[0]` and `kl_w_total` was added raw, over-weighting every prior by ~B² per epoch (B = n_minibatches). **Sweep results from before this fix cannot be directly compared to post-fix results.**

### Neural Network Components (`_components.py`)
- **RegularizedFCLayers**: dropout applied to INPUT (not output), LayerNorm default (not BatchNorm), configurable activation
- **RegularizedEncoder**: FCLayers → (mean_encoder, var_encoder) linear heads → Normal distribution
- **RegularizedDecoderSCVI**: FCLayers → px_scale_decoder linear head → softplus activation

## Build & Test
- `bash run_tests.sh tests/test_model.py -x -q` (114 tests)
- Pre-commit: pyproject-fmt, ruff check, ruff format (auto-fix)
- Python >=3.11

## Key Architecture Gotchas
- `_model.py` passes kwargs BOTH via `_module_kwargs` dict AND explicit constructor — add new params to BOTH
- `_multimodel.py` uses `**kwargs` from `_module_kwargs` — adding to dict is sufficient
- `extra_metrics` in `loss()` -> `compute_and_log_metrics()` -> `model.history_` as `{key}_{train|validation}`
- Papermill cannot parse parameter lines with inline comments — use bare assignments only
- `batch_representation="one-hot"` required (embedding incompatible with per-batch ambient RNA)
- `use_feature_scaling=True` (default) creates (1,n_genes) fallback param even without covariates
- `loss()` requires `n_obs` kwarg (= N_train from TrainingPlan). Direct test calls must pass `skip_n_obs_check=True`; NEVER use `recon_loss.shape[0]` / `x.shape[0]` to normalize global priors.

## Active Experiments
GPU experiment specs in `_gpu_jobs.yaml` (~20+ experiments on NeurIPS 2021 adult bone marrow multiome). Testing: library centering, library prior variance, early stopping sensitivity, stratified validation, learnable modality scaling, ATAC filtering thresholds, per-modality learning rates.

## Immune Integration Pipeline
`docs/notebooks/immune_integration/` — 7-dataset multi-site study (706k cells after QC): bone marrow, TEA-seq PBMC, NEAT-seq CD4, Crohn's PBMC, COVID infant PBMC, lung/spleen, infant/adult spleen. 7 notebooks: data loading -> scrublet -> ATAC loading -> RNA training -> annotation -> CRE selection -> multimodal training.

## Plan Completion Verification — MANDATORY
When a plan reaches its final step, launch a subagent (Agent tool) to independently verify that ALL steps in the plan have been completed. The subagent must check each step against the actual codebase/output state and report any incomplete or missing items before the plan is marked as done.

## Subagent File Creation — MANDATORY
Subagents (Plan, Explore, etc.) that lack Write/Edit tools must NEVER use Bash heredocs (`cat > file << EOF`, `echo >`) to create files. Instead, they must return the file content in their response text, and the parent agent must use Write/Edit to create the file. When launching subagents that need to produce files (plans, scripts, configs), use `subagent_type: "general-purpose"` which has Write/Edit access, or handle file creation in the parent after the subagent returns.

## GPU Job Submission
bsub -q gpu-normal -n 8 -M 40000 -R"select[mem>40000] rusage[mem=40000] span[hosts=1]" -gpu "mode=shared:j_exclusive=yes:gmem=80000:num=1" -e ./%J.gpu.err -o ./%J.gpu.out -J <job_name> 'PYTHONNOUSERSITE=TRUE module load ISG/conda && conda activate regularizedvi && papermill <input.ipynb> <output.ipynb>'

### GPU Memory Requirements (measured 2026-04-03)
| Dataset | Cells | Features | Modalities | MAX MEM | Request | Queue |
|---------|-------|----------|------------|---------|---------|-------|
| Immune RNA | 416k | 20k genes | 1 | 30 GB | 60 GB | gpu-normal |
| Bone marrow | 35k | 13k + 116k | 2 (RNA+ATAC) | ~25 GB | 40 GB | gpu-normal |
| Embryo | 424k | 28k + 20k + 23k + 342k | 4 (RNA+spliced+unspliced+ATAC) | 187 GB | 300 GB | gpu-huge, -sp 99 |
