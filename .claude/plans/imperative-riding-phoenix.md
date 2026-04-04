# Plan: ATAC Observation Model — BetaBinomial + Kon_Koff Decoder Types

## Context

ATAC has massive overdispersion (median theta ~0.1-0.3, 10-30x lower than RNA). The parent plan (`sunny-painting-gosling.md`) proposed a bursting model decoder for RNA — **now fully implemented** (`burst_frequency_size` code path). This plan adds the remaining two decoder types (`probability`, `Kon_Koff`) for ATAC, and the within-cell-type dispersion analysis.

**Research findings:** `imperative-riding-phoenix-research.md`

---

## What Already Exists (Verified)

| Feature | Status | Location |
|---|---|---|
| `DECODER_TYPES` tuple: `("expected_RNA", "Kon_Koff", "burst_frequency_size", "probability")` | ✓ | `_constants.py:66` |
| `DECODER_TYPE_DEFAULTS` for expected_RNA + burst_frequency_size | ✓ | `_constants.py:72-85` |
| `decoder_type` param + secondary head in `RegularizedDecoderSCVI` | ✓ | `_components.py:388-437` |
| `burst_frequency_size` complete path (decoder, generative, loss, init, metrics) | ✓ 100% | Multiple files |
| Per-modality `decoder_type_dict` via `_resolve_per_modality()` | ✓ | `_multimodule.py:268` |
| Per-modality prior hyperparams (regularise_dispersion_prior, hyper_alpha/beta) | ✓ | `_multimodule.py:277-309` |
| `library_log_means_centering_sensitivity` (per-modality dict) | ✓ | `_multimodule.py:187`, `_multimodel.py:152` |
| Residual library encoder + shrinkage | ✓ | `_multimodule.py:846-867` |
| `compute_bursting_init()` | ✓ | `_dispersion_init.py:473+` |
| `compute_atac_theta_subset.py` helper | ✓ | `scripts/claude_helper_scripts/` |
| **`probability` decoder path** | **TODO** | |
| **`Kon_Koff` decoder path** | **TODO** | |
| **`DECODER_TYPE_DEFAULTS` for probability/Kon_Koff** | **TODO** | |
| **BetaBinomial in generative** | **TODO** | |
| **Logit-space library for sensitivity** | **TODO** | |
| **`compute_bb_dispersion_init()`** | **TODO** | |
| **Per-modality `dispersion_init_bio_frac`** | **TODO** (currently scalar) | |
| **`label_key` in `_dispersion_init.py`** | **TODO** | |
| **`modality_genomic_width`** | **TODO** | |

---

## Step 1: Within-Cell-Type Theta Analysis

### 1a: Extend `_dispersion_init.py` with `label_key`

Add to `compute_dispersion_init()`:
- `label_key: str | None = None`
- `halve_counts: bool = False`
- `min_cells_per_group: int = 50`

Single-pass Welford with per-group accumulators. Store per-label_key `mean_g_k`, `var_g_k` as arrays. Return individual per-cell-type arrays in diagnostics:
```python
"mean_g_per_group": {group_name: array},   # per-gene mean within this cell type
"var_g_per_group": {group_name: array},     # per-gene variance within this cell type
"theta_per_group": {group_name: array},     # per-gene theta within this cell type
"group_sizes": {group_name: n_cells},
```
Aggregation (mean across cell types, quantiles) done outside — weight by `mean_g_k > 0` to exclude uninformative peaks.

### 1b: `compute_bb_dispersion_init()` — BetaBinomial MoM

New function alongside existing `_compute_theta()`:

```python
def compute_bb_dispersion_init(mean_g, var_g, n, cv2_L, biological_variance_fraction, d_min, d_max):
    """MoM estimator for BetaBinomial concentration parameter d.

    BetaBinomial(n, d*p, d*(1-p)) has:
      Var = n * p * (1-p) * (n+d) / (1+d)

    Define R = Var / [n * p * (1-p)] = (n+d) / (1+d)   [overdispersion ratio]
    Solving for d:  d = (n - R) / (R - 1)

    Library correction (law of total variance with p_cell = s_cell * pi_gene):
      Total Var = E_s[Var(x|s)] + Var_s[E(x|s)]
      Library component = n^2 * p^2 * CV^2(s)
      R_corrected = (Var - n^2*p^2*CV^2) / (n * p * (1 - p*(1+CV^2)))

    Edge cases:
      R <= 1 (sub-Binomial): d → d_max (near-Binomial, analogous to sub-Poisson)
      R >= n (exceeds BB max): d → d_min (maximum overdispersion)
    """
    p_hat = np.clip(mean_g / n, 1e-8, 1 - 1e-8)
    V_bin = n * p_hat * (1 - p_hat)
    V_lib = n**2 * p_hat**2 * cv2_L
    denom_factor = 1 - p_hat * (1 + cv2_L)
    valid = denom_factor > 0.01
    R = np.where(valid, (var_g - V_lib) / (n * p_hat * denom_factor), var_g / V_bin)
    R_excess = R - 1
    R_technical = 1 + R_excess * (1 - biological_variance_fraction)
    R_technical = np.clip(R_technical, 1 + 1e-6, n - 1e-6)
    d = (n - R_technical) / (R_technical - 1)
    d = np.clip(d, d_min, d_max)
    return np.log(d).astype(np.float32)
```

### 1c: Count > n Frequency Check

For both datasets, compute per-peak fraction of cells where count > n (n = 2 × round(1000/147) = 14).

### 1d: Extend `compute_atac_theta_subset.py`

Add `--label-key`, `--halve-counts` CLI args. Run for:
- **Bone marrow**: feature_type='ATAC', layer='counts', label_key='l2_cell_type'
- **Embryo**: layer='counts_120', label_key='cell_type_lvl5', var_names from model.pt

Submit both as jobs.

---

## Step 2: Constants (`_constants.py`)

Add:
```python
NUCLEOSOME_UNIT_BP = 147
DEFAULT_MODALITY_GENOMIC_WIDTH = 1000

# Defaults for probability and Kon_Koff decoder types
# d is BetaBinomial concentration — weaker containment than NB theta
DECODER_TYPE_DEFAULTS["probability"] = {
    "dispersion_hyper_prior_alpha": 2.0,
    "dispersion_hyper_prior_beta": 1.0,
    "regularise_dispersion_prior": 0.5,
}
DECODER_TYPE_DEFAULTS["Kon_Koff"] = {
    "dispersion_hyper_prior_alpha": 2.0,
    "dispersion_hyper_prior_beta": 1.0,
    "regularise_dispersion_prior": 0.5,
}
```

(Exact values will be refined from Step 1 MoM results.)

---

## Step 3: Decoder Changes (`_components.py`)

### 3a: `__init__` — New heads

The secondary decoder (`burst_size_decoder`, `burst_size_head`, `burst_size_intercept`, `burst_size_n_hidden`) is reused for `Kon_Koff`:

| `burst_size_*` param | Role in `burst_frequency_size` | Role in `Kon_Koff` |
|---|---|---|
| `burst_size_decoder` (FCLayers) | Computes burst_size hidden | Computes Koff hidden |
| `burst_size_head` (Linear+Softplus) | Outputs burst_size | Outputs Koff |
| `burst_size_intercept` (float, default 1.0) | `softplus(head) + intercept` → burst_size min | `softplus(head) + intercept` → Koff min |
| `burst_size_n_hidden` (int, default n_hidden//2) | Hidden dim of secondary FCLayers | Same |

All four are already per-modality dicts in `_multimodel.py`.

**Note:** While named "burst_size_*", these parameters serve double duty:
- For `burst_frequency_size`: compute burst_size (multiplicative scaling of mean)
- For `Kon_Koff`: compute Koff (kinetic rate, OFF rate)

Mathematically compatible: both use Softplus activation to ensure positivity. Consider future refactoring to `secondary_decoder` / `secondary_head` if additional decoder types are added.

```python
# Existing condition expanded:
if decoder_type in ("burst_frequency_size", "Kon_Koff"):
    # Secondary decoder (same code as now, reuse burst_size_decoder/burst_size_head)
    ...

if decoder_type == "probability":
    # Separate linear head for sigmoid (no softplus activation)
    self.px_p_decoder = nn.Linear(n_hidden, n_output)
```

### 3c: Validation — Additive Background Incompatibility

Add to `RegularizedMultimodalVAE.__init__()` (around line 250-280):

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

**Reason:** probability/Kon_Koff decoders output px_rate in probability space ∈ (0,1). additive_background is absolute counts. Mixing the two spaces produces invalid results. This validation prevents silent errors.

### 3b: `forward()` — New Branches

**`probability` decoder:**
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
```

**`Kon_Koff` decoder:**
```python
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

**Important:** Both `library` and `p` are in probability space ∈ (0,1). Library is prepared as sigmoid(logit) in inference (Step 4c). The decoder receives probability and multiplies by `p`, both in same space. **No additive_background is compatible** with probability decoders.

---

## Step 4: Logit-Space Library for Sensitivity

**Principle:** For decoder types `probability` and `Kon_Koff`, `library` lives in **logit space** instead of log space. The decoder still does `exp(library)`, but `library` is set up so that `exp(library)` gives sensitivity ∈ (0,1). This is achieved by storing `library = log(sigmoid(logit_value)) = -softplus(-logit_value)` — so `exp(library) = sigmoid(logit_value)`.

**Detection of mode:** Based on `decoder_type_dict[name]` — if `"probability"` or `"Kon_Koff"`, use logit-space library; otherwise, use log-space. No new parameter needed.

### 4a: `_multimodel.py` — Compute Library Stats

New parameter:
```python
modality_genomic_width: int | dict[str, int] | None = None,
# CRITICAL: Also make per-modality dict (affects both BetaBinomial AND burst_frequency_size):
dispersion_init_bio_frac: float | dict[str, float] = 0.9,
```

**Why per-modality dict for dispersion_init_bio_frac:**
- Different modalities (RNA vs ATAC) have different overdispersion signatures
- RNA: typically driven by biological variation (burst kinetics, biology) → higher bio_frac
- ATAC: often driven by technical factors (detection, batch) → lower bio_frac
- Example: `{"rna": 0.9, "atac": 0.7}` reflects tissue-specific priors

This change affects BOTH `compute_dispersion_init()` (for expected_RNA) AND `compute_bb_dispersion_init()` (for probability/Kon_Koff) AND `compute_bursting_init()` (for burst_frequency_size).

In the library statistics computation block (around line 377-400), for logit-mode modalities:

```python
decoder_type_for_mod = _resolve_per_modality(decoder_type, name, "expected_RNA")
if decoder_type_for_mod in ("probability", "Kon_Koff"):
    # Compute pseudo-sensitivity: p_obs = sum(x) / global_max_robust
    sum_counts = group_data.sum(axis=1)
    # Top-K mean for robust max (K=100 or fraction of cells)
    K = min(100, max(1, len(sum_counts) // 100))
    global_max_robust = np.sort(sum_counts)[-K:].mean()
    p_obs = np.clip(sum_counts / global_max_robust, 1e-6, 1 - 1e-6)
    # Logit transform
    logit_obs = np.log(p_obs / (1 - p_obs))
    log_means[i_group] = np.mean(logit_obs)
    log_vars[i_group] = np.var(logit_obs)
else:
    # Existing: log(sum_counts)
    log_means[i_group] = np.mean(np.log(sum_counts))
    log_vars[i_group] = np.var(np.log(sum_counts))
```

Store `global_max_robust` as a buffer for use in inference.

### 4b: `_multimodule.py` `__init__` — Library Prior Centering

In the centering block (around line 512-535), for logit-mode modalities:

```python
decoder_type_for_name = self.decoder_type_dict.get(name, "expected_RNA")
if decoder_type_for_name in ("probability", "Kon_Koff"):
    # means are already in logit space (computed in _multimodel.py)
    _sens = _sensitivity.get(name, 0.2)  # library_log_means_centering_sensitivity
    logit_target = math.log(_sens / (1 - _sens))  # logit(0.2) = -1.386
    global_logit_mean = means.mean()
    means = means - global_logit_mean + logit_target

    self.register_buffer(f"library_global_log_mean_{name}", torch.tensor(global_logit_mean.item()))
    self.register_buffer(f"library_log_sensitivity_{name}", torch.tensor(logit_target))
else:
    # Existing log-space centering (unchanged)
    ...
```

Also register `global_max_robust` buffer:
```python
self.register_buffer(f"library_global_max_robust_{name}", torch.tensor(global_max_robust))
```

And register `total_count` for genomic width:
```python
width = _resolve_per_modality(modality_genomic_width, name, None)
if width is not None:
    n = 2 * round(width / NUCLEOSOME_UNIT_BP)
    self.register_buffer(f"total_count_{name}", torch.tensor(n, dtype=torch.long))
```

### 4c: `_multimodule.py` `inference()` — Logit-Space Library (SIMPLIFIED)

In the residual library encoder block (lines 846-867), add logit-mode branch:

```python
decoder_type_for_name = self.decoder_type_dict.get(name, "expected_RNA")
if decoder_type_for_name in ("probability", "Kon_Koff"):
    # Logit-space library: compute per-cell sensitivity as probability
    global_max_robust = getattr(self, f"library_global_max_robust_{name}")
    raw_sum = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    p_obs = (raw_sum / global_max_robust).clamp(1e-6, 1 - 1e-6)
    logit_obs = torch.log(p_obs / (1 - p_obs))

    global_log_mean = getattr(self, f"library_global_log_mean_{name}", None)
    log_sensitivity = getattr(self, f"library_log_sensitivity_{name}", None)
    if global_log_mean is not None and log_sensitivity is not None:
        logit_obs_centered = logit_obs - global_log_mean + log_sensitivity
    else:
        logit_obs_centered = logit_obs
    log_sens = log_sensitivity if log_sensitivity is not None else torch.tensor(0.0, device=x.device)

    # Shrinkage + residual encoder (same structure as log-space)
    w_mu = self.library_obs_w_mu[name]
    w_sigma = torch.exp(self.library_obs_w_log_sigma[name])
    w = LogNormal(w_mu, w_sigma).rsample()
    obs_contribution = log_sens + w * (logit_obs_centered - log_sens)
    lib_logit = obs_contribution + lib_enc

    # SIMPLIFIED: Apply sigmoid directly to get probability ∈ (0,1)
    # Decoder receives library as a probability, not log-transformed
    library[name] = torch.sigmoid(lib_logit)

    # Posterior and prior both in logit space (Normal distributions)
    # KL divergence is computed in logit space, unchanged
    ql = Normal(obs_contribution + ql_enc.loc, ql_enc.scale)
    ql_per_modality[name] = ql
else:
    # Existing log-space path (unchanged)
    ...
```

**Why this is better:** Direct `sigmoid()` is clear and standard. Library[name] is directly usable as probability ∈ (0,1). Decoder contract is simple: receives probability and multiplies by `p`. No confusing `-softplus` magic.

### 4d: Library Prior in `generative()`

The prior `pl_dict[name]` uses `library_log_means_{name}` and `library_log_vars_{name}`, which are in logit space for probability/Kon_Koff modalities. The KL is `KL(ql || pl)` where both are Normal in logit space. **No changes needed in generative for the prior.**

---

## Step 5: BetaBinomial Generative Path (`_multimodule.py`)

### 5a: Generative Branching (CORRECTED)

After dispersion resolution (px_r = exp(sampled)), add branches:

```python
if _mod_decoder_type == "probability":
    from scvi.distributions import BetaBinomial as BetaBinomialDist

    p = px_scale  # sigmoid output from decoder = accessibility probability
    d = px_r      # concentration parameter from hierarchical dispersion
    n = getattr(self, f"total_count_{name}")

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
    ...  # existing code (unchanged)

else:  # expected_RNA
    ...  # existing code (unchanged)
```

### 5b: Decoder Output Unpacking (CORRECTED)

The decoder returns different numbers of values based on `decoder_type`. **Explicit unpacking is essential to prevent index errors:**

```python
# Initialize accumulators
_burst_freq = _burst_size = _kon = _koff = None

# Explicit unpacking prevents index errors when return arity changes
if _mod_decoder_type == "burst_frequency_size":
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act, _burst_freq, _burst_size = _dec_out
elif _mod_decoder_type == "Kon_Koff":
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act, _kon, _koff = _dec_out
else:  # expected_RNA and probability both return 5 values
    px_scale, px_r_cell, px_rate, px_dropout, hidden_act = _dec_out
```

All decoders return `px` (the distribution object). This unpacking pattern ensures correct handling of variable arity.

### 5c: Loss Clamping

In `loss()`, reconstruction loop:
```python
if _mod_decoder_type in ("probability", "Kon_Koff"):
    n = getattr(self, f"total_count_{name}")
    x_for_loss = torch.clamp(x, max=n)
    rl = -px.log_prob(x_for_loss).sum(-1)
    extra_metrics[f"frac_clipped_{name}"] = (x > n).float().mean().detach()
else:
    rl = -px.log_prob(x).sum(-1)
```

---

## Step 6: Model Constructor (`_multimodel.py`)

New parameters:
```python
modality_genomic_width: int | dict[str, int] | None = None,
dispersion_init_bio_frac: float | dict[str, float] = 0.9,  # make per-modality
```

Thread `modality_genomic_width` through `_module_kwargs`.

When `dispersion_init="data"`:

**New dispersion init routing** (in `_multimodel.py._setup_module()` around lines 515-545):

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

    # Resolve per-modality parameters
    _dispersion_init_bio_frac = getattr(self, "_dispersion_init_bio_frac", 0.9)
    _dispersion_init_bio_frac_resolved = (
        _dispersion_init_bio_frac if isinstance(_dispersion_init_bio_frac, dict)
        else dict.fromkeys(modality_names, _dispersion_init_bio_frac)
    )

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

            log_theta_init, _diag = compute_bb_dispersion_init(
                self.adata.mod[mod_name],
                modality_genomic_width=n_val,
                biological_variance_fraction=bf_val,
                theta_min=self._dispersion_init_theta_min,
                theta_max=self._dispersion_init_theta_max,
                verbose=False,
            )
            px_r_init_mean_dict[mod_name] = log_theta_init
        else:
            # Gamma-Poisson NB dispersion init (expected_RNA, or any other default)
            bf_val = _dispersion_init_bio_frac_resolved.get(mod_name, 0.9)

            log_theta_init, _diag = compute_dispersion_init(
                self.adata.mod[mod_name],
                biological_variance_fraction=bf_val,
                theta_min=self._dispersion_init_theta_min,
                theta_max=self._dispersion_init_theta_max,
                verbose=False,
            )
            px_r_init_mean_dict[mod_name] = log_theta_init

# Bursting model init UNCHANGED (per-modality, ONLY for burst_frequency_size decoder)
if self._dispersion_init == "variance_burst_size":
    # Existing code: calls compute_bursting_init() for burst_frequency_size modalities
    # (lines 537-570 unchanged)
    ...
```

**Key differences from current code:**
1. Current `dispersion_init="data"` block (lines 515-533) calls `compute_dispersion_init()` for ALL modalities
2. NEW routing checks `decoder_type_for_mod` to decide which init function to call per modality
3. `variance_burst_size` path UNCHANGED (still calls `compute_bursting_init()` only for burst_frequency_size)
4. Backward compatible: single-decoder-type models work exactly as before
5. Mixed-type models supported: can use "data" for heterogeneous decoders (expected_RNA, probability, Kon_Koff together)

---

## Step 7: Notebook Updates

Add decoder_type options to bone marrow and embryo multimodal training notebooks:
```python
decoder_type = {"rna": "burst_frequency_size", "atac": "probability"}
modality_genomic_width = {"atac": 1000}
library_log_means_centering_sensitivity = {"rna": 1.0, "atac": 0.2}
```

---

## Step 8: Verification

1. Existing tests pass (backward compat)
2. New tests for `probability` and `Kon_Koff` decoder paths
3. Synthetic BetaBinomial data: generate, train, recover d
4. Bone marrow: train with mixed decoder types, compare to all-NB baseline
5. **Plan completion verification subagent** (mandatory)

---

## Implementation Order

1. **Step 1** — `_dispersion_init.py` extensions + analysis jobs (independent)
2. **Step 2** — Constants (trivial)
3. **Step 3** — Decoder changes (depends on Step 2)
4. **Step 4** — Logit-space library (independent of Step 3)
5. **Step 5** — BetaBinomial generative (depends on Steps 3+4)
6. **Step 6** — Model constructor (depends on Step 5)
7. **Step 7** — Notebooks
8. **Step 8** — Verification

Steps 1, 2, 3, 4 can proceed in parallel.

---

## Critical Files

| File | Changes |
|---|---|
| `src/regularizedvi/_dispersion_init.py` | `label_key`, `halve_counts`; `compute_bb_dispersion_init()` |
| `src/regularizedvi/_constants.py` | `NUCLEOSOME_UNIT_BP`, `DEFAULT_MODALITY_GENOMIC_WIDTH`; `DECODER_TYPE_DEFAULTS` for probability/Kon_Koff |
| `src/regularizedvi/_components.py` | `probability` forward (sigmoid head), `Kon_Koff` forward (Kon/(Kon+Koff)) |
| `src/regularizedvi/_multimodule.py` | Logit-space library in inference (decoder_type conditional); BetaBinomial generative; loss clamping; `total_count` buffer |
| `src/regularizedvi/_multimodel.py` | `modality_genomic_width`; per-modality `dispersion_init_bio_frac`; logit-space library stats; BB dispersion init routing |
| `scripts/claude_helper_scripts/compute_atac_theta_subset.py` | `--label-key`, `--halve-counts`, count>n check |

## Reference
- Research: `imperative-riding-phoenix-research.md`
- Parent plan: `sunny-painting-gosling.md`
