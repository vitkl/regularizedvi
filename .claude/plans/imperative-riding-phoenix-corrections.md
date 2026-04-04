# Corrected ATAC Observation Model Plan — Critical Fixes

**Based on code review and user feedback.** This document provides the corrected versions of Steps 3-6 and Step 4 that replace the problematic sections in `imperative-riding-phoenix.md`.

---

## CORRECTION 1: Step 3b — Decoder `forward()` Methods

### ORIGINAL (INCORRECT)
```python
# Lines 173-179: probability path
if additive_background is not None:
    px_rate = torch.exp(library) * (p + additive_background)
else:
    px_rate = torch.exp(library) * p
px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
px_dropout = self.px_dropout_decoder(px)
return p, px_r, px_rate, px_dropout  # MISSING px!

# Lines 190-196: Kon_Koff path
if additive_background is not None:
    px_rate = torch.exp(library) * (p + additive_background)
else:
    px_rate = torch.exp(library) * p
px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
px_dropout = self.px_dropout_decoder(px)
return p, px_r, px_rate, px_dropout, kon, koff
```

### CORRECTED (`_components.py`)
```python
elif self.decoder_type == "probability":
    """Probability-based decoder for ATAC or other 0-1 bounded modalities.

    Library is passed as sensitivity ∈ (0,1), already sigmoid-mapped in inference.
    No additive_background (incompatible with probability space).
    """
    px = self.px_decoder(z, *cat_list)
    p = torch.sigmoid(self.px_p_decoder(px))  # accessibility probability ∈ (0,1)

    # px_rate = library * p (both in probability space)
    px_rate = library * p

    px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
    px_dropout = self.px_dropout_decoder(px)
    return p, px_r, px_rate, px_dropout, px

elif self.decoder_type == "Kon_Koff":
    """Kinetic 2-state decoder (ON/OFF rates) for ATAC or other 0-1 bounded modalities.

    Computes p = Kon / (Kon + Koff) then multiplies by library sensitivity.
    No additive_background (incompatible with probability space).
    """
    px = self.px_decoder(z, *cat_list)
    kon = self.px_scale_decoder(px)  # softplus output, positivity ensured
    koff = self.burst_size_head(self.burst_size_decoder(z, *cat_list)) + self.burst_size_intercept

    # p ∈ (0,1): accessibility probability from kinetic rates
    p = kon / (kon + koff)

    # px_rate = library * p (both in probability space)
    px_rate = library * p

    px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
    px_dropout = self.px_dropout_decoder(px)
    return p, px_r, px_rate, px_dropout, px, kon, koff
```

### Validation in `__init__`
Add after decoder initialization (around line 350-360):
```python
# Validate compatibility: additive_background incompatible with probability/Kon_Koff
if self.decoder_type in ("probability", "Kon_Koff"):
    # These decoders operate in probability space (0-1).
    # additive_background (absolute counts) is mathematically incompatible.
    # Validation happens at model level (RegularizedMultimodalVAE.__init__),
    # but we can add a warning here for clarity.
    pass  # Validation will be enforced in _multimodel.py
```

And add to `RegularizedMultimodalVAE.__init__` (around line 250-280):
```python
# Validate that additive_background is only used with count-based decoders
for name, dec_type in self.decoder_type_dict.items():
    if dec_type in ("probability", "Kon_Koff"):
        if name in additive_background_modalities:
            raise ValueError(
                f"Modality '{name}' uses decoder_type='{dec_type}' which operates in probability "
                f"space ∈ (0,1). additive_background (absolute counts) is mathematically incompatible. "
                f"Remove '{name}' from 'additive_background_modalities' or switch to 'expected_RNA' decoder."
            )
```

---

## CORRECTION 2: Step 4c — Simplified Logit-Space Library

### ORIGINAL (OVER-COMPLICATED)
```python
# Lines 311-312
lib_logit = obs_contribution + lib_enc
lib = -torch.nn.functional.softplus(-lib_logit)  # Confusing -softplus(-x) trick
```

### CORRECTED (`_multimodule.py`, lines 846-867 region)
```python
decoder_type_for_name = self.decoder_type_dict.get(name, "expected_RNA")

if decoder_type_for_name in ("probability", "Kon_Koff"):
    # Logit-space library for probability/Kon_Koff decoders
    global_max_robust = getattr(self, f"library_global_max_robust_{name}")
    raw_sum = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    p_obs = (raw_sum / global_max_robust).clamp(1e-6, 1 - 1e-6)
    logit_obs = torch.log(p_obs / (1 - p_obs))

    glm = getattr(self, f"library_global_log_mean_{name}", None)
    ls = getattr(self, f"library_log_sensitivity_{name}", None)
    if glm is not None and ls is not None:
        logit_obs_centered = logit_obs - glm + ls
    else:
        logit_obs_centered = logit_obs
    log_sens = ls if ls is not None else torch.tensor(0.0, device=x.device)

    # Shrinkage + residual encoder (same structure as log-space)
    w_mu = self.library_obs_w_mu[name]
    w_sigma = torch.exp(self.library_obs_w_log_sigma[name])
    w = LogNormal(w_mu, w_sigma).rsample()
    obs_contribution = log_sens + w * (logit_obs_centered - log_sens)
    lib_logit = obs_contribution + lib_enc

    # SIMPLIFIED: Apply sigmoid directly to get probability ∈ (0,1)
    library[name] = torch.sigmoid(lib_logit)

    # Posterior and prior both in logit space (Normal distributions)
    # KL divergence is: KL(Normal(loc_q, scale_q) || Normal(loc_p, scale_p))
    ql = Normal(obs_contribution + ql_enc.loc, ql_enc.scale)
    ql_per_modality[name] = ql
else:
    # Existing log-space path (unchanged)
    x_ = torch.log1p(x) if self.log_variational else x
    if cont_covs is not None:
        x_ = torch.cat([x_, cont_covs], dim=-1)
    ql_enc, lib_enc = self.l_encoders[name](x_, *encoder_categorical_input)

    if self.residual_library_encoder:
        log_obs = torch.log(x.sum(dim=-1, keepdim=True).clamp(min=1.0))
        glm = getattr(self, f"library_global_log_mean_{name}", None)
        ls = getattr(self, f"library_log_sensitivity_{name}", None)
        if glm is not None and ls is not None:
            log_obs_centered = log_obs - glm + ls
        else:
            log_obs_centered = log_obs
        log_sens = ls if ls is not None else torch.tensor(0.0, device=x.device)

        w_mu = self.library_obs_w_mu[name]
        w_sigma = torch.exp(self.library_obs_w_log_sigma[name])
        w = LogNormal(w_mu, w_sigma).rsample()
        obs_contribution = log_sens + w * (log_obs_centered - log_sens)
        lib = obs_contribution + lib_enc
        ql = Normal(obs_contribution + ql_enc.loc, ql_enc.scale)
        library[name] = lib
        ql_per_modality[name] = ql
    else:
        library[name] = lib_enc
        ql_per_modality[name] = ql_enc
```

### Why This is Better
1. **No `-softplus(-x)` magic**: Direct `sigmoid()` is clear and standard
2. **Library[name] is directly usable**: Contains probabilities ∈ (0,1) for probability/Kon_Koff decoders
3. **KL divergence unchanged**: Both posterior and prior remain Normal distributions in logit space
4. **Decoder contract clear**: Receives `library ∈ (0,1)` and multiplies by `p`, both in probability space
5. **Matches mathematical notation**: Library is literally "sensitivity" for these decoder types

---

## CORRECTION 3: Step 4a — Library Stats Computation

### SIMPLIFIED (NO CHANGES NEEDED)
The existing code in `_multimodel.py` lines 377-411 is **fine as-is**. The key addition is to detect decoder type:

```python
# Around line 388-411, in the library stats computation loop:
for name in modality_names:
    reg_key = f"X_{name}"
    data = self.adata_manager.get_from_registry(reg_key)
    log_means = np.zeros(n_library_cats)
    log_vars = np.ones(n_library_cats)

    # NEW: Check decoder type to decide logit vs log space
    from regularizedvi._constants import DEFAULT_DECODER_TYPE
    _decoder_type = self._module_kwargs.get("decoder_type", DEFAULT_DECODER_TYPE)
    _decoder_type_resolved = (
        _decoder_type if isinstance(_decoder_type, dict)
        else dict.fromkeys(modality_names, _decoder_type)
    )
    decoder_type_for_mod = _decoder_type_resolved.get(name, "expected_RNA")

    for i_group in np.unique(lib_indices):
        idx_group = np.squeeze(lib_indices == i_group)
        group_data = data[idx_group.nonzero()[0]]
        sum_counts = group_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)

        if decoder_type_for_mod in ("probability", "Kon_Koff"):
            # ATAC-style: compute pseudo-sensitivity in logit space
            K = min(100, max(1, len(sum_counts) // 100))
            global_max_robust = np.sort(sum_counts)[-K:].mean()

            # Store global_max_robust for use in inference (Step 4c)
            if i_group == 0:  # Store once per modality
                setattr(self, f"_global_max_robust_{name}", global_max_robust)

            p_obs = np.clip(sum_counts / global_max_robust, 1e-6, 1 - 1e-6)
            logit_obs = np.log(p_obs / (1 - p_obs))
            log_means[i_group] = np.mean(logit_obs).astype(np.float32)
            log_vars[i_group] = np.var(logit_obs).astype(np.float32)
        else:
            # RNA-style: log space (existing code)
            if np.ma.is_masked(masked_log_sum):
                logger.warning(
                    "Modality '%s' has cells with zero total counts in library group %d. "
                    "Consider filtering with scanpy.pp.filter_cells().",
                    name, i_group,
                )
            log_counts = masked_log_sum.filled(0)
            log_means[i_group] = np.mean(log_counts).astype(np.float32)
            log_vars[i_group] = np.var(log_counts).astype(np.float32)

    library_log_means[name] = log_means.reshape(1, -1)
    library_log_vars[name] = log_vars.reshape(1, -1)
```

Then in `_multimodule.py.__init__()` (around line 512-535), register the buffer:
```python
# Around line 530-560, in the library buffer registration loop:
for name in modality_names:
    # ... existing registration ...

    # NEW: Register global_max_robust for logit-mode decoders
    decoder_type_for_name = self.decoder_type_dict.get(name, "expected_RNA")
    if decoder_type_for_name in ("probability", "Kon_Koff"):
        if hasattr(self._adata_manager_instance, f"_global_max_robust_{name}"):
            gmr = getattr(self._adata_manager_instance, f"_global_max_robust_{name}")
            self.register_buffer(f"library_global_max_robust_{name}", torch.tensor(gmr, dtype=torch.float32))
        else:
            # Fallback: reasonable default
            self.register_buffer(f"library_global_max_robust_{name}", torch.tensor(1000.0, dtype=torch.float32))
```

---

## CORRECTION 4: Step 5a — BetaBinomial Generative (SIMPLIFIED)

### ORIGINAL (PROBLEMATIC)
```python
# Lines 336-365: Used px_rate = exp(lib) * p, which is wrong for probability decoders
```

### CORRECTED (`_multimodule.py`, generative method)
```python
# Around line 1030-1095, update unpacking and generative logic:

# Unpack decoder outputs (pattern fixes)
_mod_decoder_type = self.decoder_type_dict[name]
_burst_freq = _burst_size = _kon = _koff = None

if self.use_batch_in_decoder:
    _dec_out = self.decoders[name](
        disp,
        decoder_input,
        lib,
        batch_index,
        *categorical_input,
        additive_background=bg,
    )
else:
    _dec_out = self.decoders[name](
        disp,
        decoder_input,
        lib,
        *categorical_input,
        additive_background=bg,
    )

# Explicit unpacking prevents index errors when return arity changes
if _mod_decoder_type == "burst_frequency_size":
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act, _burst_freq, _burst_size = _dec_out
elif _mod_decoder_type == "Kon_Koff":
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act, _kon, _koff = _dec_out
else:  # expected_RNA and probability both return 5 values
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act = _dec_out

# Feature scaling and other downstream processing...
# (lines 1035-1072 unchanged)

# Build likelihood: CORRECTED for decoder types
if _mod_decoder_type == "probability":
    from scvi.distributions import BetaBinomial as BetaBinomialDist

    p = px_scale  # sigmoid output from decoder
    d = px_r      # concentration parameter from hierarchical dispersion
    n = getattr(self, f"total_count_{name}")  # Buffer registered with n value

    # px_rate is already = library * p (both in probability space)
    # Clamp to valid range for BetaBinomial
    sp = px_rate.clamp(1e-8, 1 - 1e-8)
    alpha = d * sp
    beta = d * (1 - sp)
    px = BetaBinomialDist(total_count=n, alpha=alpha, beta=beta)

elif _mod_decoder_type == "Kon_Koff":
    from scvi.distributions import BetaBinomial as BetaBinomialDist

    kon = _kon  # Kinetic ON rate from decoder
    koff = _koff  # Kinetic OFF rate from decoder (burst_size_head)
    d = px_r    # Concentration parameter
    n = getattr(self, f"total_count_{name}")

    # px_rate is already = library * kon/(kon+koff) (both in probability space)
    sp = px_rate.clamp(1e-8, 1 - 1e-8)
    alpha = d * sp
    beta = d * (1 - sp)
    px = BetaBinomialDist(total_count=n, alpha=alpha, beta=beta)

elif _mod_decoder_type == "burst_frequency_size":
    # px_r (from LogNormal posterior) reused as stochastic_v (technical variance)
    stochastic_v_cg = px_r

    # sensitivity = exp(library) * feature_scaling
    _sensitivity = torch.exp(lib)
    if _feature_scaling_factor is not None:
        _sensitivity = _sensitivity * _feature_scaling_factor

    # Form 2: alpha = mu^2 / var
    _var_biol = _burst_freq * _burst_size.pow(2)
    _var = _sensitivity.pow(2) * _var_biol + stochastic_v_cg
    _alpha = px_rate.pow(2) / (_var + 1e-8)

    # scale = burst_freq * burst_size (biological rate, for get_normalized_expression)
    _bio_rate = _burst_freq * _burst_size
    px = GammaPoissonWithScale(concentration=_alpha, rate=_alpha / px_rate, scale=_bio_rate)

else:  # expected_RNA
    # Standard expected_RNA: concentration=theta, rate=theta/mu
    px = GammaPoissonWithScale(concentration=px_r, rate=px_r / px_rate, scale=px_scale)

px_dict[name] = px
```

### Loss Clamping (Step 5c)
In the `loss()` method, reconstruction loop:
```python
# Around line 1130-1150 in loss() method:
for name in self.modality_names:
    if name not in px_dict:
        continue

    x = modality_data.get(f"x_{name}")
    px = px_dict[name]
    _mod_decoder_type = self.decoder_type_dict[name]

    if _mod_decoder_type in ("probability", "Kon_Koff"):
        # Clamp observations to total_count (BetaBinomial support is 0:n)
        n = getattr(self, f"total_count_{name}")
        x_for_loss = torch.clamp(x, max=n)
        rl = -px.log_prob(x_for_loss).sum(-1)
        extra_metrics[f"frac_clipped_{name}"] = (x > n).float().mean().detach()
    else:
        # No clamping for count-based decoders
        rl = -px.log_prob(x).sum(-1)

    # Standard KL and other losses...
```

---

## CORRECTION 5: Step 6 — Dispersion Init Routing

### ORIGINAL (PROBLEMATIC)
Lines 412-450 tried to add routing inside the "data" block but were unclear about interaction with "variance_burst_size" block.

### CORRECTED (`_multimodel.py`, around lines 515-545)

```python
# Data-driven dispersion initialization (per-modality, decoder-type aware)
px_r_init_mean_dict = None
if self._dispersion_init == "data":
    from regularizedvi._dispersion_init import compute_dispersion_init, compute_bb_dispersion_init

    px_r_init_mean_dict = {}

    # Resolve decoder types once
    _decoder_type = self._module_kwargs.get("decoder_type", DEFAULT_DECODER_TYPE)
    _decoder_type_resolved = (
        _decoder_type if isinstance(_decoder_type, dict)
        else dict.fromkeys(modality_names, _decoder_type)
    )

    # Resolve dispersion_init_bio_frac (per-modality if dict, else scalar)
    _dispersion_init_bio_frac = getattr(self, "_dispersion_init_bio_frac", 0.9)
    _dispersion_init_bio_frac_resolved = (
        _dispersion_init_bio_frac if isinstance(_dispersion_init_bio_frac, dict)
        else dict.fromkeys(modality_names, _dispersion_init_bio_frac)
    )

    # Resolve modality_genomic_width (per-modality if dict, else scalar)
    _modality_genomic_width = self._module_kwargs.get("modality_genomic_width", None)
    _modality_genomic_width_resolved = (
        _modality_genomic_width if isinstance(_modality_genomic_width, dict)
        else dict.fromkeys(modality_names, _modality_genomic_width) if _modality_genomic_width is not None
        else dict.fromkeys(modality_names, DEFAULT_MODALITY_GENOMIC_WIDTH)
    )

    for mod_name in modality_names:
        decoder_type_for_mod = _decoder_type_resolved.get(mod_name, "expected_RNA")

        if decoder_type_for_mod in ("probability", "Kon_Koff"):
            # BetaBinomial dispersion init
            n_val = _modality_genomic_width_resolved.get(mod_name, DEFAULT_MODALITY_GENOMIC_WIDTH)
            bf_val = _dispersion_init_bio_frac_resolved.get(mod_name, 0.9)

            logger.info(f"Computing BetaBinomial dispersion init for modality '{mod_name}'...")
            log_theta_init, _diag = compute_bb_dispersion_init(
                self.adata.mod[mod_name],
                modality_genomic_width=n_val,
                biological_variance_fraction=bf_val,
                theta_min=self._dispersion_init_theta_min,
                theta_max=self._dispersion_init_theta_max,
                verbose=False,
            )
            px_r_init_mean_dict[mod_name] = log_theta_init
            logger.info(
                f"  {mod_name}: median d={np.exp(np.median(log_theta_init)):.3f}, "
                f"modality_genomic_width={n_val}"
            )
        else:
            # Gamma-Poisson NB dispersion init (expected_RNA, or any other default)
            bf_val = _dispersion_init_bio_frac_resolved.get(mod_name, 0.9)

            logger.info(f"Computing Gamma-Poisson dispersion init for modality '{mod_name}'...")
            log_theta_init, _diag = compute_dispersion_init(
                self.adata.mod[mod_name],
                biological_variance_fraction=bf_val,
                theta_min=self._dispersion_init_theta_min,
                theta_max=self._dispersion_init_theta_max,
                verbose=False,
            )
            px_r_init_mean_dict[mod_name] = log_theta_init
            logger.info(
                f"  {mod_name}: median theta={np.exp(np.median(log_theta_init)):.3f}, "
                f"CV²(L)={_diag['cv2_L']:.3f}, sub-Poisson={_diag['n_sub_poisson']}/{len(log_theta_init)}"
            )

# Bursting model init (per-modality, ONLY for burst_frequency_size decoder)
_bursting_init_per_modality = {}
if self._dispersion_init == "variance_burst_size":
    from regularizedvi._dispersion_init import compute_dispersion_init, compute_bursting_init

    _decoder_type = self._module_kwargs.get("decoder_type", DEFAULT_DECODER_TYPE)
    _decoder_type_resolved = (
        _decoder_type if isinstance(_decoder_type, dict)
        else dict.fromkeys(modality_names, _decoder_type)
    )

    px_r_init_mean_dict = px_r_init_mean_dict or {}
    for mod_name in modality_names:
        if _decoder_type_resolved.get(mod_name) == "burst_frequency_size":
            # Use bursting init for burst_frequency_size
            # (existing code, unchanged)
            ...
        else:
            # For non-burst modalities, fall back to standard init
            logger.warning(
                f"dispersion_init='variance_burst_size' but modality '{mod_name}' has "
                f"decoder_type='{_decoder_type_resolved.get(mod_name, 'expected_RNA')}'. "
                f"Using standard dispersion init for this modality."
            )
            # Optionally call compute_dispersion_init for non-burst modalities
            # Or raise error to enforce homogeneous initialization
```

### Key Points
1. **"data" path now decoder-aware**: Routes to `compute_bb_dispersion_init()` for probability/Kon_Koff, `compute_dispersion_init()` for RNA
2. **"variance_burst_size" path unchanged**: Still calls `compute_bursting_init()` only for burst_frequency_size
3. **No new parameter**: Detection based on existing `decoder_type_dict`
4. **Backward compatible**: Single-decoder-type models work exactly as before
5. **Mixed-type models supported**: Can use "data" for heterogeneous decoders, "variance_burst_size" for homogeneous bursting

---

## Summary of Key Changes

| Section | Change | Why |
|---------|--------|-----|
| Step 3b Decoder | Remove `exp(library)`, remove `additive_background` | probability/Kon_Koff work in probability space, not count space |
| Step 3b Return | Add `px` to probability return | Consistency with other decoder types |
| Step 3b Validation | Add check for additive_background incompatibility | Prevent silent errors from mixing spaces |
| Step 4c Library | Replace `-softplus(-x)` with `sigmoid()` | Simpler, clearer, directly usable |
| Step 5a Generative | Use `px_rate = library * p` directly | No exp() needed when library is probability |
| Step 5a Clamping | Clamp observations to `total_count` for BB | BetaBinomial support is finite |
| Step 6 Routing | Add decoder-type check in "data" block | Support mixed-decoder models |
