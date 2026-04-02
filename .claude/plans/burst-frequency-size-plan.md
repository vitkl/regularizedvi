# Plan: Bursting Model Decoder (`burst_frequency_size`) for regularizedvi

## Step 0: Copy this plan
Copy to `/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/.claude/plans/` for permanent reference.

## Context

Realistic per-cell sensitivity is ~5-10% for RNA and ~0.1-3% for ATAC. The current model has theta (NB concentration) as a purely per-gene parameter that conflates biological and technical overdispersion. With a flexible decoder, the containment prior pushes theta toward Poisson AND pushes the decoder to explain ALL variance above Poisson — leading to theta growing during training and the decoder potentially overfitting noisy genes (especially high burst_size / low burst_frequency genes).

Previous experience: the accidentally inverted prior (`sqrt(theta) ~ Exp(lambda)` instead of `1/sqrt(theta)`) performed OK — it prevented extreme theta growth without forcing the decoder to explain everything. This suggests the current containment prior may be too aggressive at pushing theta up.

**Key insight from the [cell2state bursting model](https://github.com/vitkl/cell2state/blob/main/cell2state/models/bursting_programmes/_module_stochastic.py)**: decompose NB concentration into biological (burst_frequency) and technical (stochastic_v) components. The biological part is coupled to the mean via `mean = burst_frequency * (burst_size + burst_size_intercept)` (intercept default 1.0, hyperparameter), providing built-in regularization: the decoder can't increase mean predictions for noisy genes without also increasing concentration (expecting less noise).

This plan implements `decoder_type="burst_frequency_size"` as a new per-modality decoder option, following the switching architecture defined in the [imperative-riding-phoenix plan](imperative-riding-phoenix.md). It is NOT a replacement of the current model — it is one of 4 decoder types switchable per modality.

---

## Mathematical Foundation

### 1. Binomial Thinning Invariance (Exact)

For x ~ NB(mu, theta) with y|x ~ Bin(x, p): y ~ NB(p*mu, theta). Theta is invariant.

**Gamma additivity applies to pooling cells (pseudo-bulk), NOT to per-cell library size variation.** Confirmed: applying Gamma scaling to single-cell models (cell2location regression, Visium) reduced accuracy.

### 2. The Real Problem: Identifiability + Decoder Overfitting

At low sensitivity (p=0.05 for RNA), mu_obs << 1 for most genes. The NB excess (mu_obs^2/theta) is small relative to Poisson noise (mu_obs) — though not negligible because the decoder operates in rate space. Three consequences:
1. **MoM theta init is noisy** — the estimator reports negative excess (apparent sub-Poisson) for many genes due to estimation noise, not because genes are truly sub-Poisson
2. **Training signal for theta is weak** — the model struggles to distinguish NB from Poisson
3. **Decoder overfitting** — the containment prior pushes theta up (toward Poisson), incentivizing the decoder to explain all variance. High burst_size / low burst_frequency genes are worst affected.

### 3. DESeq2's Approach

```
K_ij ~ NB(s_j * q_ij, alpha_i)        where alpha = 1/theta
alpha_tr(mu_bar) = a_1/mu_bar + alpha_0    [dispersion-mean trend]
log(alpha_i) ~ N(log(alpha_tr(mu_bar_i)), sigma_d^2)   [shrinkage]
```

Key: DESeq2 does NOT make alpha depend on library size — it depends on normalized mean. The trend + shrinkage regularizes low-expression genes more (two mechanisms: data is less informative so any prior dominates, AND the trend target is mean-dependent). However, DESeq2 has fixed categorical covariates, whereas our VAE has a flexible decoder — so regularizing technical theta alone does not prevent decoder overfitting. The DESeq2 dispersion-mean trend likely exists because of the biological bursting relationship (burst_freq ∝ mean when burst_size is roughly constant), not because of technical noise. Once biological and technical variance are separated via the bursting decomposition, the technical component likely does NOT need a mean-dependent prior — a flat containment prior on stochastic_v is sufficient.

### 4. Bursting Model: Biological + Technical Decomposition

Source: [cell2state bursting module](https://github.com/vitkl/cell2state/blob/main/cell2state/models/bursting_programmes/_module_stochastic.py) (main branch, cloned to /tmp/cell2state_main/)

**Generative model:**
```
G ~ Gamma(burst_freq, 1/burst_size)         [biological bursting rate]
   E[G] = burst_freq * burst_size
   Var[G] = burst_freq * burst_size^2 = var_biol

D ~ Poisson(sensitivity * (G + s))          [observed count]
```
Plus count-space stochastic technical variance `stochastic_v`.

**Variance decomposition** (s treated as fixed for moment-matching):
```
Var[D] = E[Var[D|G]] + Var[E[D|G]]
       = sens*(E[G]+s) + sens^2 * Var[G]
       = mu + sens^2 * var_biol              [Poisson + biological bursting]
       + stochastic_v                         [technical, count-space]
```
**Why sens^2 * var_biol:** sensitivity scales burst_size — each burst produces `sens * burst_size` observed counts instead of `burst_size` true molecules. Since var_biol = burst_freq * burst_size^2, the count-space biological variance is burst_freq * (sens * burst_size)^2 = sens^2 * var_biol.

**Why library has a dual role (same as feature_scaling):** library enters through sensitivity = exp(library) * feature_scaling. In `alpha = mu^2 / var`, larger sensitivity increases alpha (less overdispersion) because technical variance `stochastic_v` becomes relatively smaller vs `sens^2 * var_biol`. Currently, library only scales the mean.

Note: contamination s contributes ONLY to mu (Poisson term), NOT to the variance terms — because s is constant (not random), so Var[G+s] = Var[G].

**Total NB concentration — two forms:**

**Form 1 (cell2state code, approximate):**
```python
# From cell2state _module_stochastic.py lines 554-556
alpha = (alpha_biol + contaminating.pow(2) / var_biol) / (
    self.ones + (stochastic_v_cg / sensitivity.pow(2)) / var_biol
)
```
Obtained from exact form by dividing both sides by var_biol and dropping `2*contaminating/burst_size` cross-term. Breaks when var_biol → 0. Used in cell2state because NMF factors ensure alpha_biol > 0 always. Interpretable: `(bio_concentration + contamination_correction) / (1 + technical/biological_ratio)`.

**Form 2 (for regularizedvi, exact count-space, `alpha = mu^2 / var`):**
```python
sensitivity = torch.exp(library) * feature_scaling     # cell*gene specific
mu = (burst_freq * burst_size + contaminating) * sensitivity
var_biol = burst_freq * burst_size.pow(2)
var = sensitivity.pow(2) * var_biol + stochastic_v_cg  # count-space excess variance
alpha = mu.pow(2) / var
```

| Aspect | Form 1 (cell2state) | Form 2 (proposed) |
|--------|---------------------|-------------------|
| Exactness | Approximate (drops 2s/burst_size) | EXACT |
| Division by zero | Fails when var_biol=0 | Handles gracefully |
| Compute cost | 3 divisions | 1 division (mu already computed) |
| Empty droplets | Breaks (0/0) | Works: alpha = s^2 * sens^2 / stochastic_v |

**Verified against**: cell2state main branch (lines 536-556)

**Special cases (Form 2):**
- No technical variance: alpha → burst_freq (+ contamination terms)
- High sensitivity: stochastic_v negligible, alpha → burst_freq
- Low sensitivity: alpha → 0 (maximum overdispersion)
- **Empty droplet** (burst_freq=0, var_biol=0): alpha = s^2 * sens^2 / stochastic_v

### 4b. Covariate Mapping: regularizedvi → bursting model

| regularizedvi covariate | Bursting model role | Enters alpha? |
|------------------------|---------------------|---------------|
| **library** (learned per cell) | Part of sensitivity | YES — dual role: scales mean AND reduces effective technical overdispersion |
| **feature_scaling_covariate_keys** (y_tg) | Part of sensitivity | YES — same dual role as library |
| **ambient_covariate_keys** (s_eg) | Contamination | YES — in mu = (bio + s) * sens |
| **nn_conditioning_covariate_keys** | Decoder injection | YES indirectly — affects burst_freq/burst_size |
| **dispersion_key** | Indexes stochastic_v groups (was: total theta) | YES — selects v_tech per group |
| **library_size_key** | Library prior groups | Indirectly — affects learned library |
| **encoder_covariate_keys** | Encoder injection | Indirectly — affects z → decoder |

### 5. Key Structural Difference: regularizedvi vs cell2state

| Aspect | cell2state bursting | regularizedvi (current) |
|--------|-------------------|------------------------|
| Alpha/theta cell-specific? | YES | NO (gene or gene-batch) |
| Depends on sensitivity? | YES (denominator) | NO |
| Biological alpha modeled? | YES (NMF burst_freq) | NO (decoder outputs mean only) |
| Technical variance separate? | YES (stochastic_v) | NO (theta absorbs both) |
| Exp-Gamma prior targets | Technical variance only | Total dispersion |
| Mean coupled to concentration? | YES (mean = burst_freq * burst_size) | NO (independent) |

### 6. Doublets/Multiplets

Homotypic doublet: theta_eff = 2*theta (Gamma additivity). Heterotypic: theta_eff >= theta.

In the bursting model, burst_freq naturally doubles for doublets (additive concentration), handling them correctly. In the current model, doublets contribute to theta growing during training (secondary effect vs decoder flexibility).

---

## Implementation: `decoder_type="burst_frequency_size"`

This is one of 4 decoder types defined in [imperative-riding-phoenix.md](imperative-riding-phoenix.md):
```python
DECODER_TYPES = ("expected_RNA", "Kon_Koff", "burst_frequency_size", "probability")
```

Switchable per modality via: `decoder_type={"rna": "burst_frequency_size", "atac": "probability"}`. The multimodal model should also support N=1 modalities to avoid maintaining complex features in a separate single-modality model.

### New Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `burst_size_intercept` | `float \| dict[str, float]` | `1.0` | Added to softplus(decoder_size) output. Prevents low burst_size + low burst_freq explaining zero detection. Settable per modality. |
| `burst_size_n_hidden` | `int \| dict[str, int] \| None` | `None` | n_hidden for burst_size decoder. If None, uses n_hidden//2 of main decoder. Same n_latent as main decoder. Same signature as main decoder (modality-specific params). |

Existing hyperparameters that change behavior under `burst_frequency_size`:
| Parameter | Role under burst_frequency_size |
|-----------|-------------------------------|
| `dispersion_key` | Indexes stochastic_v groups (technical variance, not total theta) |
| `regularise_dispersion_prior` | Containment prior rate for stochastic_v (flat, not mean-dependent) |
| `dispersion_hyper_prior_alpha/beta` | Gamma hyper-prior for stochastic_v Exp rate |
| `dispersion_init` | `"prior"` (default) or `"data"` or `"variance"` or `"variance_burst_size"`. `"prior"`: use prior values. `"data"`: per-gene theta via MoM (current). `"variance"`: per-gene stochastic_v via MoM. `"variance_burst_size"`: stochastic_v + burst_freq/burst_size init |
| `dispersion_init_bio_frac` | Fraction attributed to biology. Used for BOTH: burst_freq init (bio_frac of variance) and stochastic_v init (1-bio_frac of variance) |

### Architecture Changes

**1. Decoder class** (`_components.py`):

Add secondary FCLayers for burst_size, same decoder class signature:
```python
if decoder_type == "burst_frequency_size":
    self.secondary_decoder = RegularizedFCLayers(
        n_in=n_latent, n_out=burst_size_n_hidden,  # default n_hidden//2
        n_layers=max(1, n_layers - 1), ...)
    self.secondary_head = nn.Linear(burst_size_n_hidden, n_output)
```

Primary head → softplus → burst_frequency
Secondary head → softplus + burst_size_intercept → burst_size

**2. Generative model** (`_module.py` / `_multimodule.py`) — when `decoder_type="burst_frequency_size"` (default remains `"expected_RNA"`):

```python
# Decoders (z includes nn_conditioning covariates via decoder injection)
burst_freq = decoder_freq(z, *categorical_input)                           # softplus
burst_size = decoder_size(z, *categorical_input) + burst_size_intercept    # softplus + intercept

# Purpose-based covariates
sensitivity   = torch.exp(library) * feature_scaling   # library * y_tg
contaminating = additive_background                     # s_eg

# Indexed by dispersion_key (technical variance groups)
stochastic_v_cg = ...                                   # per-gene technical variance

# Biological variance (rate-space)
var_biol = burst_freq * burst_size.pow(2)

# Mean and excess variance (count-space)
mu  = (burst_freq * burst_size + contaminating) * sensitivity
var = sensitivity.pow(2) * var_biol + stochastic_v_cg

# NB concentration (Form 2, exact)
alpha = mu.pow(2) / var

# Likelihood
D ~ GammaPoisson(concentration=alpha, rate=alpha / mu)
```

**3. Technical variance** (`_module.py`):

Replace current theta path (for this decoder type) with per-gene stochastic_v:
- Same Exp-Gamma hierarchical prior structure (flat, NOT mean-dependent — technical noise does not scale with expression once biology is separated, though gene-specific factors like length, GC content, PCR bias may matter but are not mean-dependent)
- Prior: `stochastic_v_ag ~ Exponential(stochastic_v_ag_hyp)`, `stochastic_v_ag_hyp ~ Gamma(2, 0.04)`
- Then `stochastic_v_cg = stochastic_v_ag.pow(2)` indexed by dispersion_key
- `Exp(lambda)` produces a **scale/std** parameter, `.pow(2)` converts to variance. This matches cell2state: `stochastic_v_ag.pow(2)` (line 551). No `1/sqrt` needed (unlike theta where `1/sqrt(theta) ~ Exp(lambda)` and theta is a concentration)
- Hyper-prior `Gamma(2, 0.04)` gives `E[lambda]=50`, marginal median(v_std)=0.017, matching empirical `median(sqrt(excess_technical))=0.012` from MoM analysis across bone marrow/immune/embryo datasets. Wider than the theta prior `Gamma(9,3)` because technical variance spans ~3 orders of magnitude

**4. Initialization:**

- `stochastic_v`: 10% of per-gene MoM excess variance (technical component)
- `burst_freq` decoder bias: theta estimated from 90% of variance (biological component)
- `burst_size` decoder bias (numerically stable, eps=1e-4):
  ```python
  eps = 1e-4
  valid = burst_freq_init > eps
  burst_size_valid = gene_mean[valid] / burst_freq_init[valid]
  default_burst_size = burst_size_valid.min()  # smallest valid value, not median
  raw_burst_size = torch.where(valid, gene_mean / burst_freq_init, default_burst_size) - burst_size_intercept
  burst_size_bias = softplus_inv(torch.clamp(raw_burst_size, min=eps))
  ```
- **Theta MoM weighting applies to theta/burst_freq init but NOT to count-space variance init**: For theta, MoM estimator precision scales as mu^2, so `w = mu^2/(mu^2 + sigma^2)` is correct. For count-space variance (stochastic_v) with n >> 1000 cells, the sample variance estimator is already precise (precision ~ n/2, approximately constant across genes), so minimal shrinkage is needed — direct MoM estimate suffices.
- Other init values unchanged

**5. Attributions:** For `get_modality_attribution()`, compute Jacobian of `mean = burst_freq * burst_size` (the full decoder output), not separate burst components

**6. Per-modality toggle:** `decoder_type` per modality (default `"expected_RNA"`). When `"burst_frequency_size"`: use bursting architecture. When `"expected_RNA"`: current architecture unchanged.

---

## Optional: Weighted MoM Init (Option B)

Gene-level confidence weighting for MoM theta/burst_freq initialization:
```
theta_init_g = w_g * theta_MoM_g + (1 - w_g) * theta_trend_g
w_g = mu_obs_g^2 / (mu_obs_g^2 + sigma^2)
```

**How it works:** The MoM estimator `theta = mu^2 / excess` has precision scaling as mu^2 (the NB signal mu^2/theta lives at scale mu^2; for low mu, this signal is tiny vs Poisson noise). The weight w_g is a precision-weighted Bayesian average — the optimal combination given `precision_MoM ∝ mu^2` and `precision_trend = 1/sigma^2`:
- **High mu (mu >> sigma):** w → 1, trust gene-specific MoM
- **Low mu (mu << sigma):** w → 0, fall back to trend (MoM too noisy)
- **sigma** controls BOTH the crossover position (w=0.5 at mu=sigma) AND the steepness of the transition (they cannot be decoupled in this formula — larger sigma shifts right AND makes transition shallower)

**Estimating sigma from data** (following DESeq2 approach):
```python
# 1. Fit trend: theta_tr(mu) = a1/mu + a0 via Gamma-family GLM or lowess
# 2. Compute residuals: r_g = log(theta_MoM_g) - log(theta_tr(mu_g))
# 3. Robust sigma estimate:
sigma = MAD(residuals) / 0.6745    # MAD scaled to normal SD
sigma = max(sigma, 0.25)            # floor to prevent over-shrinkage (DESeq2 value)
```

**Trend line fitting:**
- Parametric: `theta_tr(mu) = a1/mu + a0` via iterative Gamma GLM with outlier exclusion (DESeq2 style)
- Non-parametric: `lowess(log_theta, log_mu, frac=0.3)`
- The trend captures the mean-dispersion relationship driven by bursting biology (burst_freq ∝ mean when burst_size ~ constant)
- Implement both and compare via hist2d against uncorrected values in `docs/notebooks/model_comparisons/dispersion_init_analysis.ipynb`

**Applies to theta/burst_freq init only.** For count-space variance (stochastic_v), with n >> 1000 cells the sample variance estimator already has precision ~ n/2 (approximately constant across genes), so shrinkage is not needed.

Compatible with all decoder types.

**Difficulty:** Low (~30-40 lines in `_dispersion_init.py`)

---

## Training Metrics for Alpha/Theta Monitoring

Add to `extra_metrics_payload` in `loss()` (around line 1053 of `_module.py`):

| Metric | Source | Notes |
|--------|--------|-------|
| `theta_mean` | `generative_outputs["px_r_sampled"].mean()` | Mean sampled theta across batch |
| `theta_median` | `torch.quantile(px_r_sampled, 0.5)` | Median sampled theta |
| `px_r_mu_mean` | `generative_outputs["px_r_mu"].mean()` | Log-space posterior mean |
| `px_r_log_sigma_mean` | `generative_outputs["px_r_log_sigma"].mean()` | Posterior variance (log scale) |
| `dispersion_kl` | Already computed (line 985), not logged | Add `(dispersion_kl / n_obs).detach()` |

For `burst_frequency_size` decoder, additionally log:
| `alpha_biol_mean` | `burst_freq.mean()` | Mean burst frequency |
| `burst_size_mean` | `burst_size.mean()` | Mean burst size |
| `stochastic_v_mean` | `stochastic_v_cg.mean()` | Mean technical variance |
| `alpha_total_mean` | `alpha.mean()` | Mean total NB concentration |

All values from `generative_outputs`, `.detach()` before adding.

---

## N=1 Multimodal Support

RegularizedMultimodalVI does NOT break on N=1 — latent concat, encoder/decoder, loss all handle it. No core code changes needed.

**Required changes:**
1. **Test fixture** in `conftest.py`: `mdata_single_rna()` → `mu.MuData({"rna": adata_rna})`
2. **N=1 integration tests** in `test_model.py`:
   - `test_setup_mudata_single_modality` — verify data registration
   - `test_model_init_single_modality` — verify `module.modality_names == ["rna"]`
   - `test_train_single_modality` — verify training completes, latent shape = (n_obs, n_latent)
   - Test with `decoder_type="burst_frequency_size"` on N=1
3. **Optional**: Add `setup_anndata()` convenience that wraps `AnnData → MuData({"rna": adata})` internally
4. **Long-term**: Consider deprecating AmbientRegularizedSCVI (barriers: missing ArchesMixin, EmbeddingMixin)

---

## Open Questions

1. **ATAC decoder:** Separate [imperative-riding-phoenix.md](imperative-riding-phoenix.md) plan addresses ATAC via `"probability"` and `"Kon_Koff"` decoder types.
2. **Per-modality toggle granularity:** Per modality (confirmed).
3. **Technical variance factors:** Gene length, GC content, PCR bias may cause gene-specific technical variance not captured by the flat prior. Could add gene-level covariates to stochastic_v in future.

---

## Critical Files

| File | Changes |
|------|---------|
| `src/regularizedvi/_components.py` | Add secondary decoder FCLayers for `burst_frequency_size` |
| `src/regularizedvi/_module.py` | Bursting generative path, alpha formula, stochastic_v path, extra_metrics |
| `src/regularizedvi/_multimodule.py` | Per-modality decoder switching, bursting path |
| `src/regularizedvi/_model.py` | `decoder_type`, `burst_size_intercept`, `burst_size_n_hidden` params |
| `src/regularizedvi/_multimodel.py` | Per-modality config, init routing, `setup_anndata()` wrapper |
| `src/regularizedvi/_dispersion_init.py` | Weighted MoM shrinkage (Option B); stochastic_v init |
| `src/regularizedvi/_constants.py` | `DECODER_TYPES`, `DEFAULT_BURST_SIZE_INTERCEPT` |

## Reference Code
- cell2state bursting module: https://github.com/vitkl/cell2state/blob/main/cell2state/models/bursting_programmes/_module_stochastic.py (main branch)
- Decoder switching architecture: [imperative-riding-phoenix.md](imperative-riding-phoenix.md)
- Previous dispersion init plan: `.claude/plans/dispersion-init-plan.md`

## Verification & Testing

### Integration tests
- Run existing tests with `decoder_type="expected_RNA"` — verify backward compatibility
- Add new test for `decoder_type="burst_frequency_size"` — verify training loop completes, alpha > 0, no NaN

### Next steps (post-implementation, post-training)
1. Synthetic Gamma-Poisson data: verify alpha formula produces correct concentration at varying sensitivity (p = 1.0, 0.1, 0.05, 0.01)
2. Decoder overfitting check: compare high burst_size / low burst_freq genes between `"burst_frequency_size"` and `"expected_RNA"` decoder types
3. Training dynamics: monitor alpha/theta growth over epochs using new extra_metrics — do we still see theta growing indefinitely?
4. Empty droplet edge case: verify Form 2 handles burst_freq→0 gracefully on real data
5. Trend fitting comparison: parametric vs lowess vs uncorrected in `dispersion_init_analysis.ipynb`

### Application to 3 datasets (following dispersion-init-plan.md structure)

| Dataset | Template | Model | decoder_type | Queue |
|---------|----------|-------|-------------|-------|
| **Bone marrow multimodal** | `bone_marrow_multimodal_tutorial_early_stopping.ipynb` | RegularizedMultimodalVI | `{"rna": "burst_frequency_size", "atac": "expected_RNA"}` | gpu-normal |
| **Immune RNA** | `bm_pbmc_rna_training_v2.ipynb` | AmbientRegularizedSCVI | `"burst_frequency_size"` | gpu-normal |
| **Embryo 4-modality** | `embryo_rna_atac_spliced_unspliced.ipynb` | RegularizedMultimodalVI | `{"rna": "burst_frequency_size", "spliced": "burst_frequency_size", "unspliced": "burst_frequency_size", "atac": "expected_RNA"}` | gpu-huge |

**6 experiments total** (2 per dataset): `burst_size_intercept=1.0` vs `burst_size_intercept=0.01`.
All submitted to gpu-normal except embryo (gpu-huge). Compare against baseline experiments from dispersion-init-plan.md. Add `burst_size_intercept` and `decoder_type` as papermill parameters.

### Plan completion verification (MANDATORY — before committing and before running notebooks)
Launch subagent to independently verify ALL implementation steps completed against actual codebase state. Must pass before committing changes or submitting notebook jobs.

---

## Future Option: DESeq2-Style Mean-Dependent Prior (Option A)

Preserved for future consideration. With the bursting decoder, the DESeq2 trend is largely explained by biology (burst_freq ∝ mean). Technical variance (stochastic_v) likely does NOT need a mean-dependent prior. However, if empirical evidence shows stochastic_v has mean-dependence, this option can be revisited.

**Concept:** Replace/modulate the containment prior with a mean-dependent target:
```
log(v_tech_g) ~ N(log(v_tr(mu_bar_g)), sigma_d^2)
v_tr(mu_bar) = a_1/mu_bar + v_0
```

**Why mean-dependent prior pushes low-expression genes harder** (two mechanisms):
1. **Data informativeness**: For low-mu genes, the likelihood is flat (near-zero Fisher information for dispersion). ANY prior dominates the posterior. Even a uniform prior has more influence on low-expression genes.
2. **Trend target**: DESeq2's trend has higher alpha (more overdispersed) for low-expression genes, so the prior centers on a different target.

**Trend tuning:** Push bias to Poisson (alpha_0 → 0) and weight to 0 for small gene-specific deviations.

**Limitation:** Only regularizes v_tech, not the decoder. The bursting decoder's mean-concentration coupling is more powerful for preventing decoder overfitting.
