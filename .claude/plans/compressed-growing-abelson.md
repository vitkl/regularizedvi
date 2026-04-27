# Latent Z Sparsity + Hidden Activation Sparsity + Decoder L1

## Context

regularizedvi has no sparsity mechanism on Z dimensions or hidden activations. Prior on Z is `Normal(0,1)` equally on all dims (`_module.py:927`). Decoder uses ReLU activations (non-negative) with only L2 weight penalty (0.1).

**Template:** `bm_pbmc_rna_training_v3.ipynb` — has `n_hidden`, `n_latent`, `decoder_type_rna`, `dispersion_init`, `burst_size_intercept` as papermill params.

### KL Warmup Discovery

scvi-tools defaults to `n_epochs_kl_warmup=400`, linearly ramping `kl_weight` from 0→1. Both single-modal and multimodal use this default. Training typically stops at epoch 40-150 via early stopping → kl_weight only reaches 0.1-0.37. The z_sparsity_penalty (added to `kl_local_for_warmup`) is similarly suppressed.

**Action:** All sparsity experiments must use `n_epochs_kl_warmup=5`. Need KL warmup baseline experiments first.

**How kl_weight is applied:**
- Single-modal (`_module.py:972`): `weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup`. The `kl_local_for_warmup` contains kl_z (+ z_sparsity_penalty when enabled). Library KL is in `kl_local_no_warmup` → NOT weighted.
- Multimodal (`_multimodule.py:1241`): `loss = torch.mean(recon_loss + kl_weight * kl_z + kl_l) + kl_w_total`. Adding z_sparsity_penalty to `kl_z` before this line means it IS weighted by `kl_weight`. Library KL (`kl_l`) is NOT weighted.
- Both use scvi-tools `TrainingPlan` which computes `kl_weight` from the warmup schedule and passes it to `module.loss(kl_weight=...)`. The multimodal `MultimodalTrainingPlan` only overrides `configure_optimizers()` (per-modality LR), not `training_step()`.

---

## Step 1: Z Diagnostics in Single-Modal Module

**`_module.py` loss(), after line 959:**
```python
z = inference_outputs[MODULE_KEYS.Z_KEY]
extra_metrics_payload["z_var"] = z.var(dim=0).mean().detach()
```

---

## Step 2: Gamma Prior Penalty on |z|

### `_module.py`

**`__init__`:** Add `z_sparsity_prior: str | None = None`, `n_active_latent_per_cell: float = 20.0`

**`loss()` before `kl_local_for_warmup = kl_divergence_z`:**
```python
if self.z_sparsity_prior == "gamma":
    z = inference_outputs[MODULE_KEYS.Z_KEY]
    shape = torch.tensor(self.n_active_latent_per_cell / self.n_latent, device=z.device, dtype=z.dtype)
    gamma_dist = Gamma(concentration=shape, rate=torch.tensor(1.0, device=z.device, dtype=z.dtype))
    z_sparsity_penalty = -gamma_dist.log_prob(z.abs() + 1e-8).sum(dim=-1)
    extra_metrics_payload["z_sparsity_penalty"] = z_sparsity_penalty.mean().detach()
else:
    z_sparsity_penalty = 0.0
kl_local_for_warmup = kl_divergence_z + z_sparsity_penalty
```

### `_multimodule.py`

**`__init__`:** Add same 2 params (explicit, no **kwargs).

**`loss()` before line 1241:**
```python
if self.z_sparsity_prior == "gamma":
    z = inference_outputs["z"]
    shape = torch.tensor(self.n_active_latent_per_cell / self.total_latent_dim, device=z.device, dtype=z.dtype)
    gamma_dist = Gamma(concentration=shape, rate=torch.tensor(1.0, device=z.device, dtype=z.dtype))
    z_sparsity_penalty = -gamma_dist.log_prob(z.abs() + 1e-8).sum(dim=-1)
    extra_metrics["z_sparsity_penalty"] = z_sparsity_penalty.mean().detach()
    kl_z = kl_z + z_sparsity_penalty
```

### Model API
- `_model.py`: `__init__` + `_module_kwargs` + explicit constructor
- `_multimodel.py`: `__init__` + `_module_kwargs`

---

## Step 3: Decoder Weight L1 Penalty

Starting value: **0.01**. Rationale: L1 and L2 are NOT directly comparable at the same coefficient — L1 sums |w|, L2 sums w². For typical weight magnitudes (±0.1-1.0), L1 ≈ 0.1-0.3× of L2 gives similar total penalty. With L2=0.1, this suggests L1 ≈ 0.01-0.03. Literature recommends 0.001-0.1 range for L1 in deep learning; 0.01 achieves ~20-40% weight sparsity in dense layers.

### `_module.py`

**`__init__`:** Add `decoder_hidden_l1: float = 0.0`

**New method:**
```python
def _decoder_hidden_l1_penalty(self) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(self.decoder.parameters()).device)
    for layer_seq in self.decoder.px_decoder.fc_layers:
        for sublayer in layer_seq:
            if isinstance(sublayer, torch.nn.Linear):
                penalty = penalty + sublayer.weight.abs().sum()
    return penalty
```

**`loss()`:** After existing L2 penalty.

### `_multimodule.py`

Same method, iterating over `self.decoders[name]` per modality.

### Model API: Add to both `_model.py` and `_multimodel.py`.

---

## Step 4: Gamma Prior on Decoder Hidden Activations

### Return `px` from decoder in all paths

**`_components.py` RegularizedDecoderSCVI.forward():**

Insert `px` at position 4 (after px_dropout, before burst elements). This preserves index-based access at `_dec_out[2]` = px_rate used in attribution code.

```python
# Path 1 — burst_frequency_size (line 501):
# Before: return px_scale, None, px_rate, px_dropout, burst_freq, burst_size  (6)
return px_scale, None, px_rate, px_dropout, px, burst_freq, burst_size        # (7)

# Path 2 — expected_RNA (line 509):
# Before: return px_scale, None, px_rate, px_dropout  (4)
return px_scale, None, px_rate, px_dropout, px                                # (5)
```

### ALL callers to update (7 unpacking sites):

**`_module.py` generative() — 3 decoder calls, shared unpacking at lines 855-857:**
```python
# Before:
if self.decoder_type == "burst_frequency_size":
    px_scale, px_r, px_rate, px_dropout, _burst_freq, _burst_size = _dec_out
else:
    px_scale, px_r, px_rate, px_dropout = _dec_out

# After:
if self.decoder_type == "burst_frequency_size":
    px_scale, px_r, px_rate, px_dropout, hidden_act, _burst_freq, _burst_size = _dec_out
else:
    px_scale, px_r, px_rate, px_dropout, hidden_act = _dec_out
```
Store: add `"hidden_activations": hidden_act` to generative_outputs dict.

**`_multimodule.py` generative() — 2 decoder calls per modality, shared unpacking at lines 1008-1011:**
```python
# Before:
if _mod_decoder_type == "burst_frequency_size":
    px_scale, px_r_cell, px_rate, px_dropout, _burst_freq, _burst_size = _dec_out
else:
    px_scale, px_r_cell, px_rate, px_dropout = _dec_out

# After:
if _mod_decoder_type == "burst_frequency_size":
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act, _burst_freq, _burst_size = _dec_out
else:
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act = _dec_out
```
Store per-modality in generative_outputs.

**`_multimodel.py` get_modality_attribution() — 2 decoder calls, index-based access at line 1338:**
```python
px_rate = _dec_out[2]  # Index 2 = px_rate — UNCHANGED, px inserted at index 4
```
**NO CHANGE NEEDED** — index 2 still points to px_rate.

### Loss computation

**`_module.py`:**
```python
if self.hidden_activation_sparsity:
    hidden_act = generative_outputs["hidden_activations"]
    shape = torch.tensor(self.n_active_hidden_per_cell / hidden_act.shape[-1], device=hidden_act.device, dtype=hidden_act.dtype)
    gamma_dist = Gamma(concentration=shape, rate=torch.tensor(1.0, device=hidden_act.device, dtype=hidden_act.dtype))
    hidden_sparsity_penalty = -gamma_dist.log_prob(hidden_act + 1e-8).sum(dim=-1)
    loss = loss + hidden_sparsity_penalty.mean() / n_obs
    extra_metrics_payload["hidden_sparsity_penalty"] = hidden_sparsity_penalty.mean().detach()
```

**`_multimodule.py`:** Same, iterating over modalities.

### New params: `hidden_activation_sparsity: bool = False`, `n_active_hidden_per_cell: float = 40.0`

### Warning for n_layers>1 at model creation.

---

## Step 5: Notebook Template Updates

### Templates and their datasets:

| Template | Dataset | Location |
|----------|---------|----------|
| **Immune RNA** | `bm_pbmc_rna_training_v3.ipynb` | `docs/notebooks/immune_integration/` |
| **BM Multimodal** | `bone_marrow_multimodal_tutorial_early_stopping.ipynb` | `docs/notebooks/model_comparisons/` |
| **Embryo** | `embryo_rna_atac_spliced_unspliced.ipynb` | `/nfs/team205/vk7/sanger_projects/cell2state_embryo/notebooks/benchmark/regularizedvi/` |

### Params to add to each template:
```python
z_sparsity_prior = None
n_active_latent_per_cell = 20
decoder_hidden_l1 = 0.0
hidden_activation_sparsity = 0
n_active_hidden_per_cell = 40
n_epochs_kl_warmup = 400
```

In each training cell, pass `plan_kwargs={"n_epochs_kl_warmup": n_epochs_kl_warmup}`.
Add these to `coerce_papermill_params()` call in each template.

### Key parameter differences between burst/baseline:

| Parameter | Baseline (expected_RNA) | Burst (burst_frequency_size) |
|-----------|------------------------|------------------------------|
| decoder_type_rna | expected_RNA | burst_frequency_size |
| burst_size_intercept | - (not set) | 1.0 |
| dispersion_init | data | variance_burst_size |
| dispersion_prior_mean | 1.0 | 1.0 |

---

## Step 6: Experiments TSV

Create `docs/notebooks/model_comparisons/z_sparsity_jobs.tsv` with same structure as `burst_jobs.tsv` + new columns: `z_sparsity_prior`, `n_active_latent_per_cell`, `decoder_hidden_l1`, `hidden_activation_sparsity`, `n_active_hidden_per_cell`, `n_epochs_kl_warmup`.

### A. KL Warmup Baselines (6) — n_epochs_kl_warmup=5, no sparsity

| Name | Template | Queue/Mem/Pri | Base overrides |
|------|----------|---------------|----------------|
| immune_baseline_small_klw5 | immune v3 | gpu-normal/60G/50 | decoder_type_rna=expected_RNA, dispersion_init=data, dispersion_prior_mean=1.0, n_hidden=512, n_latent=128 |
| immune_baseline_large_klw5 | immune v3 | gpu-normal/60G/50 | decoder_type_rna=expected_RNA, dispersion_init=data, dispersion_prior_mean=1.0, n_hidden=1024, n_latent=256 |
| immune_burst_vbs2_int1_small_klw5 | immune v3 | gpu-normal/60G/50 | decoder_type_rna=burst_frequency_size, burst_size_intercept=1.0, dispersion_init=variance_burst_size, dispersion_prior_mean=1.0, n_hidden=512, n_latent=128 |
| immune_burst_vbs2_int1_large_klw5 | immune v3 | gpu-normal/60G/50 | Same as above but n_hidden=1024, n_latent=256 |
| bm_mm_burst_vbs2_int1_klw5 | BM multimodal | gpu-normal/40G/50 | decoder_type_rna=burst_frequency_size, decoder_type_atac=expected_RNA, burst_size_intercept=1.0, dispersion_init=variance_burst_size, dispersion_prior_mean=1.0 |
| embryo_burst_vbs2_int1_n_klw5 | embryo | gpu-normal/300G/99 | decoder_type_rna=burst_frequency_size, decoder_type_atac=expected_RNA, decoder_type_spliced=burst_frequency_size, decoder_type_unspliced=burst_frequency_size, burst_size_intercept=1.0, dispersion_init=variance_burst_size, dispersion_prior_mean=1.0 |

### B. Architecture Diagnostics (4) — default n_epochs_kl_warmup, no sparsity

All use immune v3 template, gpu-normal/60G/50.

| Name | decoder_type_rna | dispersion_init | n_hidden | n_latent |
|------|-----------------|-----------------|----------|----------|
| immune_diag_baseline_hilatent | expected_RNA | data | 512 | 256 |
| immune_diag_baseline_hihidden | expected_RNA | data | 1024 | 128 |
| immune_diag_burst_hilatent | burst_frequency_size | variance_burst_size | 512 | 256 |
| immune_diag_burst_hihidden | burst_frequency_size | variance_burst_size | 1024 | 128 |

### C. Z Sparsity — Immune (4) — Gamma on Z, n_epochs_kl_warmup=5

All use immune v3 template, gpu-normal/60G/50. z_sparsity_prior=gamma, n_active_latent_per_cell=20.

| Name | decoder_type_rna | dispersion_init | n_hidden | n_latent |
|------|-----------------|-----------------|----------|----------|
| immune_zsparse_baseline_small | expected_RNA | data | 512 | 128 |
| immune_zsparse_baseline_large | expected_RNA | data | 1024 | 256 |
| immune_zsparse_burst_small | burst_frequency_size | variance_burst_size | 512 | 128 |
| immune_zsparse_burst_large | burst_frequency_size | variance_burst_size | 1024 | 256 |

### D. Z + Hidden Gamma — Immune (4) — both Gamma priors, n_epochs_kl_warmup=5

Same as C but + hidden_activation_sparsity=1, n_active_hidden_per_cell=40.

| Name | decoder_type_rna | dispersion_init | n_hidden | n_latent |
|------|-----------------|-----------------|----------|----------|
| immune_zsparse_hsparse_baseline_small | expected_RNA | data | 512 | 128 |
| immune_zsparse_hsparse_baseline_large | expected_RNA | data | 1024 | 256 |
| immune_zsparse_hsparse_burst_small | burst_frequency_size | variance_burst_size | 512 | 128 |
| immune_zsparse_hsparse_burst_large | burst_frequency_size | variance_burst_size | 1024 | 256 |

### E. Z + L1 — Immune (4) — Z Gamma + decoder L1=0.01, n_epochs_kl_warmup=5

Same as C but + decoder_hidden_l1=0.01.

| Name | decoder_type_rna | dispersion_init | n_hidden | n_latent |
|------|-----------------|-----------------|----------|----------|
| immune_zsparse_l1_baseline_small | expected_RNA | data | 512 | 128 |
| immune_zsparse_l1_baseline_large | expected_RNA | data | 1024 | 256 |
| immune_zsparse_l1_burst_small | burst_frequency_size | variance_burst_size | 512 | 128 |
| immune_zsparse_l1_burst_large | burst_frequency_size | variance_burst_size | 1024 | 256 |

### F. BM Multimodal Sparsity (3) — n_epochs_kl_warmup=5

All use BM multimodal template, gpu-normal/40G/50.

| Name | Base config | z_sparsity | hidden_sparsity | decoder_hidden_l1 |
|------|-------------|------------|-----------------|-------------------|
| bm_mm_zsparse_burst_vbs2 | bm_mm_burst_vbs2_int1 overrides (decoder_type_rna=burst_frequency_size, decoder_type_atac=expected_RNA, dispersion_init=variance_burst_size, dispersion_prior_mean=1.0) | gamma | - | - |
| bm_mm_zsparse_hsparse_burst_vbs2 | Same | gamma | True | - |
| bm_mm_zsparse_l1_burst_vbs2 | Same | gamma | - | 0.01 |

### G. Embryo Sparsity (1) — both Gamma priors, n_epochs_kl_warmup=5

Uses embryo template, gpu-normal/300G/99. Base overrides from embryo_burst_vbs2_int1_n.

| Name | z_sparsity | hidden_sparsity | n_active_latent | n_active_hidden |
|------|------------|-----------------|-----------------|-----------------|
| embryo_zsparse_hsparse_burst_vbs2 | gamma | True | 20 | 40 |

### **Total: 26 experiments** (6 KL warmup baselines + 4 diagnostics + 4 z_sparsity + 4 z+hidden_gamma + 4 z+L1 + 3 BM multimodal + 1 embryo)

---

## Files to Modify

| File | Changes |
|------|---------|
| `_components.py` | `RegularizedDecoderSCVI.forward()`: return `px` at index 4 in all paths |
| `_module.py` | `__init__`: 5 new params; `generative()`: unpack hidden_act; `loss()`: z_var, z_sparsity, L1, hidden sparsity; `_decoder_hidden_l1_penalty()` |
| `_multimodule.py` | Same 5 params in `__init__`; `generative()`: unpack per-modality; `loss()`: 4 penalties; L1 per modality |
| `_model.py` | `__init__` + `_module_kwargs` + constructor |
| `_multimodel.py` | `__init__` + `_module_kwargs`; attribution at line 1338 UNCHANGED (index 2 = px_rate still) |
| Immune v3 template | 6 new papermill params + plan_kwargs in training cell |
| BM multimodal template | Same 6 new papermill params |
| Embryo template | Same 6 new papermill params |
| `z_sparsity_jobs.tsv` | New file, 26 experiments |

---

## Verification

1. **Tests:** `bash scripts/helper_scripts/run_tests.sh tests/test_model.py -x -q` — all pass
2. **New tests:** Each flag trains + penalty logged
3. **/verify-implementation** before job submission
4. **GPU runs:** Submit 26 experiments
