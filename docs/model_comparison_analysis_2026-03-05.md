# Single-Modal vs Multi-Modal Model Comparison Analysis

**Date**: 2026-03-05
**Context**: RNA single-modal model results noticeably worse than before; multiome results good and consistent across hyperparameters.

## Summary

A systematic comparison of `AmbientRegularizedSCVI` (single-modal) and `RegularizedMultimodalVI` (multi-modal) found that **all purpose-specific parameters are initialized identically** and **all covariate data flows are equivalent**. The primary behavioral difference is in **encoder categorical injection**: the single-modal model diverges from standard scVI by always injecting batch information into the encoder.

## What Is IDENTICAL Between Models

| Component | Single-modal file:line | Multi-modal file:line | Details |
|-----------|----------------------|---------------------|---------|
| **px_r init** | `_module.py:301-332` | `_multimodule.py:413-445` | `log(rate^2) + 0.1*randn`, noise=0.1 |
| **dispersion_prior_rate_raw** | `_module.py:334-347` | `_multimodule.py:447-460` | `inverse_softplus(3.0)`, Gamma(9,3) hyper-prior |
| **additive_background init** | `_module.py:352-365` | `_multimodule.py:465-479` | `log(alpha/beta) + 0.01*randn`, noise=0.01 |
| **feature_scaling init** | `_module.py:367-381` | `_multimodule.py:481-490` | `torch.zeros()`, no noise |
| **library_log_means/vars** | `_module.py:290-299` | `_multimodule.py:401-411` | Same computation, same `library_log_vars_weight` scaling |
| **scale_activation** | `_model.py:207` | `_multimodel.py:147` | Both use `DEFAULT_SCALE_ACTIVATION="softplus"` |
| **Feature scaling priors** | `_constants.py:42-43` | Same constants | Both `Gamma(200, 200)` |
| **Dispersion priors** | `_constants.py:29-30` | Same constants | Both `Gamma(9, 3)` hyper-prior |
| **Additive bg priors** | `_constants.py:35-36` | Same constants | Both `Gamma(1, 100)` |
| **Decoder class** | `_components.py:341-472` | Same class | `RegularizedDecoderSCVI`, no custom weight init |
| **bg application** | `_components.py:466-469` | Same | `rate = exp(library) * (px_scale + bg)` — AFTER softplus |
| **Covariate data flow** | `_module.py:476-523` | `_multimodule.py:492-525` | All 5 covariates present in both `_get_inference/generative_input` |

## What DIFFERS Between Models

### 1. Encoder Categorical Gating (CRITICAL - Root Cause of Regression)

**Standard scVI** (`scvi-tools/src/scvi/module/_vae.py:237`):
```python
encoder_cat_list = cat_list if encode_covariates else None  # default encode_covariates=False
```
With default `encode_covariates=False`, **NO categorical inputs** go to encoder. Same for MultiVI, PeakVI.

**Multi-modal regularizedvi** (`_multimodule.py:291`):
```python
encoder_cat_list = encoder_cat_list if encode_covariates else None  # matches scVI
```

**Single-modal regularizedvi** (`_module.py:407`):
```python
encoder_cat_list = _ambient_cats + (_cat_cats if encode_covariates else []) + _lib_cats
# ALWAYS includes ambient_covs + library_size — DIVERGES from scVI!
```

This means single-modal regularizedvi is the ONLY model that injects batch information into the encoder by default. This was introduced during the purpose-key refactor. Consequences:
- Leaks batch info into the latent space
- Conflicts with batch-free decoder (`use_batch_in_decoder=False`) + additive background
- Adds unnecessary categorical dimensions increasing encoder complexity

### 2. Feature Scaling Always-On Fallback (MINOR)

- **Single-modal** (`_module.py:378`): `(1, n_genes)` parameter ALWAYS created, even with empty covariate keys. Always penalized with Gamma(200,200).
- **Multi-modal** (`_multimodule.py:486-490`): Only created for modalities in `feature_scaling_modalities`. Empty list = no parameter.
- **Impact**: Minor — user already uses `feature_scaling_covariate_keys=["site","donor"]` so the (1,n_genes) fallback isn't active.

### 3. Labels `y` Passed to Decoder (CONFIRMED NO-OP)

- **Single-modal**: Passes `y` (labels) as extra positional arg to decoder in `generative()` (lines 705, 715, 725)
- **Multi-modal**: Does NOT pass labels
- **Why it's a no-op**: `n_labels` is NOT in decoder's `n_cat_list`. `RegularizedFCLayers.forward()` uses `zip(n_cat_list, cat_list, strict=False)` → extra positional args are silently ignored. Zero parameters created, zero computation affected.

### 4. Structural Differences (By Design)

| Aspect | Single-modal | Multi-modal |
|--------|-------------|-------------|
| Parameter storage | Single tensors | Per-modality `ParameterDict` |
| Library priors | Global buffers | Per-modality buffers |
| Encoders | 1 z_encoder + 1 l_encoder | Per-modality encoders + optional joint |
| Decoders | 1 decoder | Per-modality decoders |
| KL(z) | Single N(0,I) prior | Per-modality priors (concatenation mode) |
| Masking | N/A | Missing-modality masks |

## Implemented Fixes (2026-03-05)

All fixes implemented and tested (114 tests pass):

1. **✅ `encoder_covariate_keys` registry key** — default `False` (no encoder categoricals) for both models, matching scVI. Warning emitted if user sets to `list[str]` or `None`. Continuous covariates always included in encoder when registered.
2. **✅ `use_feature_scaling` flag** — gates parameter creation, application in `generative()`, and penalty in `loss()` for single-modal. Default `True` (backward compatible).
3. **✅ Removed `labels_key`** — from both models entirely; `n_labels`, `gene-label` dispersion, `y` in generative all removed.
4. **✅ `plot_attribution_scatter()` method** — per-cell attribution values on UMAP, with log2 ratio panel for 2-modality case.
5. **✅ Multimodal notebooks updated** — explicit `encoder_covariate_keys=False` in all 3 input notebooks.

## Key File References

- Single-modal model: `src/regularizedvi/_model.py`
- Single-modal module: `src/regularizedvi/_module.py`
- Multi-modal model: `src/regularizedvi/_multimodel.py`
- Multi-modal module: `src/regularizedvi/_multimodule.py`
- Shared components: `src/regularizedvi/_components.py`
- Constants: `src/regularizedvi/_constants.py`
- Standard scVI reference: `other_packages/scvi-tools/src/scvi/module/_vae.py`
