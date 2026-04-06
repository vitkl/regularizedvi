# Plan: ATAC Observation Model — BetaBinomial + Kon_Koff Decoder Types (REVISED)

## Context

ATAC has massive overdispersion (median theta ~0.1-0.3, 10-30x lower than RNA). The parent plan (`sunny-painting-gosling.md`) proposed a bursting model decoder for RNA — **now fully implemented** (`burst_frequency_size` code path). This plan adds the remaining two decoder types (`probability`, `Kon_Koff`) for ATAC, and the within-cell-type dispersion analysis.

**Research findings:** `imperative-riding-phoenix-research.md`
**Previous version:** `imperative-riding-phoenix.md` (reviewed in `-review.md`)

---

## What Already Exists (Verified 2026-04-05)

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
| Feature scaling: `softplus(param)/0.7`, Gamma(200,200) prior | ✓ | `_multimodule.py:654,1035,1331` |
| Attribution: `_dec_out[2]` = px_rate, called on decoder directly | ✓ | `_multimodel.py:1306-1410` |
| `scvi.distributions.BetaBinomial(total_count, alpha, beta)` | ✓ | scvi-tools 1.4.1 |
| **`probability` decoder path** | **TODO** | |
| **`Kon_Koff` decoder path** | **TODO** | |
| **`DECODER_TYPE_DEFAULTS` for probability/Kon_Koff** | **TODO** | |
| **BetaBinomial in generative** | **TODO** | |
| **Logit-space library for sensitivity** | **TODO** | |
| **`compute_bb_dispersion_init()`** | **TODO** | |
| **Per-modality `dispersion_init_bio_frac`** | **TODO** (currently scalar) | |
| **`label_key` in `_dispersion_init.py`** | **TODO** | |
| **`modality_genomic_width`** | **TODO** | |
| **Probability-space feature_scaling** | **TODO** | |

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

**NaN handling:** When all values are 0 for a peak in a cell type → var_g_k = 0, mean_g_k = 0. Use pandas mean weighted by non-zero mean across cell types. If ALL groups have mean=0, theta is undefined → clamp to theta_max.

### 1b: `compute_bb_dispersion_init()` — BetaBinomial MoM

New function alongside existing `_compute_theta()`. **Must be a full AnnData wrapper** (like `compute_dispersion_init`): accepts AnnData, computes mean_g/var_g via chunked Welford, computes library CV², then applies the BB MoM formula.

Core formula:
```python
def _compute_bb_concentration(mean_g, var_g, n, cv2_L, biological_variance_fraction, d_min, d_max):
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
    "regularise_dispersion_prior": 1.0,  # default 1.0, data-driven after Step 1
}
DECODER_TYPE_DEFAULTS["Kon_Koff"] = {
    "dispersion_hyper_prior_alpha": 2.0,
    "dispersion_hyper_prior_beta": 1.0,
    "regularise_dispersion_prior": 1.0,  # default 1.0, data-driven after Step 1
}
```

**Note on `regularise_dispersion_prior`**: This scales the containment prior KL weight (`1/sqrt(d) ~ Exp(lambda)`). Value of 1.0 means standard-strength containment. ATAC may need weaker containment since overdispersion is legitimately high — exact value to be refined from Step 1 MoM results. Must not create subexponential tails: the Exp(lambda) prior is exponential-tailed by construction; the weight only scales the KL penalty magnitude.

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

```python
# Existing condition expanded:
if decoder_type in ("burst_frequency_size", "Kon_Koff"):
    # Secondary decoder (same code as now, reuse burst_size_decoder/burst_size_head)
    ...

# NEW: Probability decoder head (sigmoid activation, not softplus)
if decoder_type == "probability":
    self.px_p_decoder = nn.Linear(n_hidden, n_output)
```

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
    # library = sigmoid(logit) = cell-level sensitivity (no feature_scaling)
    # feature_scaling applied by generative() AFTER decoder (Step 4e)
    # Attribution reads _dec_out[2] and applies feature_scaling in its closure
    px_rate = library * p

    px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
    px_dropout = self.px_dropout_decoder(px)
    return p, None, px_rate, px_dropout, px
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
    # library = sigmoid(logit) = cell-level sensitivity (no feature_scaling)
    # feature_scaling applied by generative() AFTER decoder (Step 4e)
    # Attribution reads _dec_out[2] and applies feature_scaling in its closure
    px_rate = library * p

    px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
    px_dropout = self.px_dropout_decoder(px)
    return p, None, px_rate, px_dropout, px, kon, koff
```

**Return signatures (all decoder types):**
```
expected_RNA:          5 values: (px_scale, None, px_rate, px_dropout, px)
burst_frequency_size:  7 values: (px_scale, None, px_rate, px_dropout, px, burst_freq, burst_size)
probability:           5 values: (p, None, px_rate, px_dropout, px)
Kon_Koff:              7 values: (p, None, px_rate, px_dropout, px, kon, koff)
```
**Index 2 is ALWAYS px_rate** (the full observable mean) — attribution reads this directly.

**Important:** For probability/Kon_Koff, `library` passed into the decoder is `sigmoid(logit)` — cell-level library sensitivity only, NO feature_scaling. Feature scaling is applied by generative() AFTER the decoder returns (Step 4e), matching the pattern for expected_RNA and burst_frequency_size. Attribution manually applies feature_scaling in its closure, also matching the existing pattern.

### 3c: Validation — Incompatible Covariates

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

**Note:** Feature scaling is NOT blocked — it is converted to probability space (Step 4e).

---

## Step 4: Logit-Space Library & Probability-Space Sensitivity

**Principle:** For decoder types `probability` and `Kon_Koff`, `library` lives in **logit space** during inference and is converted to **probability** via sigmoid before being passed to the decoder. Feature scaling is also converted to probability space and multiplied into library (so that `library = sigmoid(logit) * feature_scaling_prob` is the full sensitivity `s`).

**Detection of mode:** Based on `decoder_type_dict[name]` — if `"probability"` or `"Kon_Koff"`, use logit-space library; otherwise, use log-space. No new parameter needed.

### 4a: `_multimodel.py` — Compute Library Stats

New parameters:
```python
modality_genomic_width: int | dict[str, int] | None = None,
# Make per-modality dict (affects NB, BB, and bursting init):
dispersion_init_bio_frac: float | dict[str, float] = 0.9,
```

In the library statistics computation block (around line 377-400), for logit-mode modalities:

```python
from regularizedvi._constants import DEFAULT_DECODER_TYPE
_decoder_type = self._module_kwargs.get("decoder_type", DEFAULT_DECODER_TYPE)
_decoder_type_resolved = (
    _decoder_type if isinstance(_decoder_type, dict)
    else dict.fromkeys(modality_names, _decoder_type)
)
decoder_type_for_mod = _decoder_type_resolved.get(name, "expected_RNA")

if decoder_type_for_mod in ("probability", "Kon_Koff"):
    # Compute pseudo-sensitivity: p_obs = sum(x) / global_max_robust
    sum_counts = group_data.sum(axis=1)
    # Top-K mean for robust max (K=100 or fraction of cells)
    K = min(100, max(1, len(sum_counts) // 100))
    global_max_robust = np.sort(sum_counts)[-K:].mean()
    p_obs = np.clip(sum_counts / global_max_robust, 1e-6, 1 - 1e-6)
    # Logit transform
    logit_obs = np.log(p_obs / (1 - p_obs))
    log_means[i_group] = np.mean(logit_obs).astype(np.float32)
    log_vars[i_group] = np.var(logit_obs).astype(np.float32)
else:
    # Existing: log(sum_counts)
    log_counts = masked_log_sum.filled(0)
    log_means[i_group] = np.mean(log_counts).astype(np.float32)
    log_vars[i_group] = np.var(log_counts).astype(np.float32)
```

Store `global_max_robust` per modality in `_module_kwargs` dict (NOT via fragile setattr):
```python
self._module_kwargs[f"_global_max_robust_{name}"] = global_max_robust
```

### 4b: `_multimodule.py` `__init__` — Library Prior Centering

In the centering block (around line 512-535), for logit-mode modalities:

```python
decoder_type_for_name = self.decoder_type_dict.get(name, "expected_RNA")
if decoder_type_for_name in ("probability", "Kon_Koff"):
    # means are already in logit space (computed in _multimodel.py)
    _sens = _sensitivity.get(name, 0.2)  # library_log_means_centering_sensitivity
    logit_target = np.log(_sens / (1 - _sens))  # logit(0.2) = -1.386
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
global_max_robust = self._init_params.get(f"_global_max_robust_{name}", None)
if global_max_robust is not None:
    self.register_buffer(f"library_global_max_robust_{name}", torch.tensor(global_max_robust, dtype=torch.float32))
```

And register `total_count` for genomic width:
```python
from regularizedvi._constants import NUCLEOSOME_UNIT_BP
width = _resolve_per_modality(modality_genomic_width, name, None)
if width is not None:
    n = 2 * round(width / NUCLEOSOME_UNIT_BP)
    self.register_buffer(f"total_count_{name}", torch.tensor(n, dtype=torch.long))
```

### 4c: `_multimodule.py` `inference()` — Logit-Space Library

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

    # Apply sigmoid to get probability ∈ (0,1)
    lib_prob = torch.sigmoid(lib_logit)

    # Posterior and prior both in logit space (Normal distributions)
    ql = Normal(obs_contribution + ql_enc.loc, ql_enc.scale)
    ql_per_modality[name] = ql

    # Decoder receives lib_prob directly (NO feature_scaling — applied in generative() Step 4e)
    library[name] = lib_prob
else:
    # Existing log-space path (unchanged)
    ...
    library[name] = lib  # log-space
    ql_per_modality[name] = ql
```

### 4d: Library Prior in `generative()`

The prior `pl_dict[name]` uses `library_log_means_{name}` and `library_log_vars_{name}`, which are in logit space for probability/Kon_Koff modalities. The KL is `KL(ql || pl)` where both are Normal in logit space. **No changes needed in generative for the prior.**

### 4e: Probability-Space Feature Scaling (Applied AFTER Decoder — Same Pattern as Burst Model)

**Established pattern (verified):** The decoder NEVER sees feature_scaling. It receives only `lib` and `z`. Feature scaling is applied by generative() AFTER the decoder returns px_rate. Attribution (`get_modality_attribution`) manually replicates this by applying feature_scaling in its closure. **This pattern must be preserved for probability/Kon_Koff.**

For probability/Kon_Koff decoders:
- Decoder receives `lib_prob = sigmoid(logit)` — library probability only, NO feature_scaling
- Decoder computes `px_rate = lib_prob * p` (attribution reads this at `_dec_out[2]`)
- generative() applies feature_scaling to px_rate AFTER decoder (sigmoid transform for prob decoders)
- generative() constructs `sensitivity = lib_prob * feature_scaling_prob` for biophysical alpha/beta

**In `__init__`**: For probability/Kon_Koff modalities, initialize feature_scaling params to `torch.full(..., 4.6)` instead of `torch.zeros(...)` (since `sigmoid(4.6) ≈ 0.99`):
```python
for name in feature_scaling_modalities:
    n_feat = n_input_per_modality[name]
    if self.decoder_type_dict[name] in ("probability", "Kon_Koff"):
        # Initialize so sigmoid(4.6) ≈ 0.99 (near-unity scaling)
        self.feature_scaling[name] = nn.Parameter(torch.full((n_feature_scaling_rows, n_feat), 4.6))
    else:
        self.feature_scaling[name] = nn.Parameter(torch.zeros(n_feature_scaling_rows, n_feat))
```

**In `generative()`** — feature_scaling applied AFTER decoder returns (lines 1035-1049):
```python
_feature_scaling_factor = None
_is_prob_decoder = self.decoder_type_dict[name] in ("probability", "Kon_Koff")

if name in self.feature_scaling:
    if _is_prob_decoder:
        # Probability space: sigmoid transform, output ∈ (0,1)
        rf_transformed = torch.sigmoid(self.feature_scaling[name])
    else:
        # Count space: softplus/0.7 transform (existing, unchanged)
        rf_transformed = torch.nn.functional.softplus(self.feature_scaling[name]) / 0.7

    if feature_scaling_indicator is not None:
        _feature_scaling_factor = torch.matmul(feature_scaling_indicator, rf_transformed)
    else:
        _feature_scaling_factor = rf_transformed

    # Apply to px_rate (same pattern for ALL decoder types)
    px_rate = px_rate * _feature_scaling_factor
```

For burst_frequency_size and Kon_Koff, generative() also constructs full sensitivity for variance/alpha-beta formulas:
```python
if _mod_decoder_type == "burst_frequency_size":
    # Existing code (unchanged):
    _sensitivity = torch.exp(lib)
    if _feature_scaling_factor is not None:
        _sensitivity = _sensitivity * _feature_scaling_factor

elif _mod_decoder_type == "Kon_Koff":
    # Same pattern, probability space:
    _sensitivity = lib  # already sigmoid-mapped probability from Step 4c
    if _feature_scaling_factor is not None:
        _sensitivity = _sensitivity * _feature_scaling_factor
    # _sensitivity is now full s = lib_prob * feature_scaling_prob
    # Used for: alpha = d * _sensitivity * _kon, beta = d * _koff * (1 - _sensitivity)

elif _mod_decoder_type == "probability":
    # px_rate already has feature_scaling applied above
    # For BetaBinomial: sp = px_rate (already = lib_prob * feature_scaling_prob * p)
    pass
```

**In `loss()`**: For probability/Kon_Koff modalities, use Beta(100, 1) prior instead of Gamma(200, 200):
```python
from torch.distributions import Beta

if _is_prob_decoder:
    # Beta(100, 1) prior: mode ≈ 0.99, strongly peaked near 1
    neg_log_prior = -Beta(
        self.feature_scaling_prob_prior_alpha,   # default 100.0
        self.feature_scaling_prob_prior_beta,     # default 1.0
    ).log_prob(rf_transformed.clamp(1e-6, 1 - 1e-6)).sum()
else:
    # Existing Gamma(200, 200) prior for count-space scaling
    neg_log_prior = -Gamma(
        self.feature_scaling_prior_alpha,
        self.feature_scaling_prior_beta,
    ).log_prob(rf_transformed).sum()
```

**Attribution** (`_multimodel.py:1349-1357`): The existing closure already applies feature_scaling to `_dec_out[2]`. For probability/Kon_Koff, it must use `sigmoid()` instead of `softplus/0.7`:
```python
if name in module.feature_scaling:
    if module.decoder_type_dict[name] in ("probability", "Kon_Koff"):
        rf_transformed = torch.sigmoid(module.feature_scaling[name])
    else:
        rf_transformed = softplus(module.feature_scaling[name]) / 0.7
    scaling = matmul(feature_scaling_indicator, rf_transformed)
    px_rate = px_rate * scaling
```

New constructor parameters (with defaults):
```python
feature_scaling_prob_prior_alpha: float = 100.0,
feature_scaling_prob_prior_beta: float = 1.0,
```

---

## Step 5: BetaBinomial Generative Path (`_multimodule.py`)

### 5a: Generative Branching

Import at top of file:
```python
from scvi.distributions import BetaBinomial as BetaBinomialDist
```

After dispersion resolution (px_r = exp(sampled)), add branches:

```python
if _mod_decoder_type == "probability":
    p = px_scale  # sigmoid output from decoder = accessibility probability
    d = px_r      # concentration parameter from hierarchical dispersion
    n = getattr(self, f"total_count_{name}")

    # px_rate = s * p (already computed in decoder as library * p)
    # where library = lib_prob * feature_scaling_prob = full sensitivity
    sp = px_rate.clamp(1e-8, 1 - 1e-8)
    alpha = d * sp
    beta_param = d * (1 - sp)
    px = BetaBinomialDist(total_count=n, alpha=alpha, beta=beta_param)

elif _mod_decoder_type == "Kon_Koff":
    # BIOPHYSICAL parameterization: sensitivity interacts asymmetrically with Kon/Koff
    d = px_r    # Concentration parameter
    n = getattr(self, f"total_count_{name}")

    # _sensitivity constructed in Step 4e: lib_prob * feature_scaling_prob
    # decoder returned px_rate = lib_prob * kon/(kon+koff) for attribution (no feature_scaling)
    # generative() already applied feature_scaling to px_rate above (line 1044)
    # For BetaBinomial, we use the biophysical form with full sensitivity:
    alpha = d * _sensitivity * _kon
    beta_param = d * _koff * (1 - _sensitivity)
    px = BetaBinomialDist(total_count=n, alpha=alpha, beta=beta_param)

elif _mod_decoder_type == "burst_frequency_size":
    ...  # existing code (unchanged)

else:  # expected_RNA
    ...  # existing code (unchanged)
```

### 5b: Decoder Output Unpacking

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

Store kon/koff outputs (like burst_outputs):
```python
if _mod_decoder_type == "Kon_Koff" and _kon is not None:
    kon_koff_outputs[name] = {"kon": _kon, "koff": _koff, "p": px_scale}
```

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
feature_scaling_prob_prior_alpha: float = 100.0,
feature_scaling_prob_prior_beta: float = 1.0,
```

Thread `modality_genomic_width` and `feature_scaling_prob_prior_*` through `_module_kwargs`.

**Dispersion init routing** — decoder-type aware for BOTH `"data"` AND `"variance_burst_size"`:

```python
# Data-driven dispersion initialization (per-modality, decoder-type aware)
px_r_init_mean_dict = None
if self._dispersion_init == "data":
    from regularizedvi._dispersion_init import compute_dispersion_init, compute_bb_dispersion_init

    px_r_init_mean_dict = {}
    _decoder_type_resolved = (
        _decoder_type if isinstance(_decoder_type, dict)
        else dict.fromkeys(modality_names, _decoder_type)
    )
    _bio_frac_resolved = (
        _dispersion_init_bio_frac if isinstance(_dispersion_init_bio_frac, dict)
        else dict.fromkeys(modality_names, _dispersion_init_bio_frac)
    )
    _width_resolved = (
        _modality_genomic_width if isinstance(_modality_genomic_width, dict)
        else dict.fromkeys(modality_names, _modality_genomic_width) if _modality_genomic_width is not None
        else dict.fromkeys(modality_names, DEFAULT_MODALITY_GENOMIC_WIDTH)
    )

    for mod_name in modality_names:
        dt = _decoder_type_resolved.get(mod_name, "expected_RNA")
        bf = _bio_frac_resolved.get(mod_name, 0.9)

        if dt in ("probability", "Kon_Koff"):
            n_val = _width_resolved.get(mod_name, DEFAULT_MODALITY_GENOMIC_WIDTH)
            log_d_init, _diag = compute_bb_dispersion_init(
                self.adata.mod[mod_name], modality_genomic_width=n_val,
                biological_variance_fraction=bf, ...)
            px_r_init_mean_dict[mod_name] = log_d_init
        else:
            log_theta_init, _diag = compute_dispersion_init(
                self.adata.mod[mod_name], biological_variance_fraction=bf, ...)
            px_r_init_mean_dict[mod_name] = log_theta_init

# Bursting model init — also decoder-type aware for fallback
if self._dispersion_init == "variance_burst_size":
    ...  # existing burst_frequency_size code unchanged
    # FIXED: fallback for non-burst modalities is also decoder-type aware
    for mod_name in modality_names:
        if _decoder_type_resolved.get(mod_name) != "burst_frequency_size":
            dt = _decoder_type_resolved.get(mod_name, "expected_RNA")
            if dt in ("probability", "Kon_Koff"):
                log_d_init, _ = compute_bb_dispersion_init(...)
                px_r_init_mean_dict[mod_name] = log_d_init
            else:
                log_theta_init, _ = compute_dispersion_init(...)
                px_r_init_mean_dict[mod_name] = log_theta_init
```

---

## Step 7: Attribution & Notebook Updates

### 7a: Verify Attribution

`get_modality_attribution()` (`_multimodel.py:1306-1410`) reads `_dec_out[2]` (px_rate) directly from the decoder. Since all decoder types return px_rate at index 2 (the full observable mean), attribution works with **zero changes**. Verify this with a test.

### 7b: Notebook Updates

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
4. **Step 4** — Logit-space library + probability-space feature scaling (independent of Step 3)
5. **Step 5** — BetaBinomial generative (depends on Steps 3+4)
6. **Step 6** — Model constructor (depends on Step 5)
7. **Step 7** — Attribution verification + notebooks
8. **Step 8** — Verification

Steps 1, 2, 3, 4 can proceed in parallel.

---

## Critical Files

| File | Changes |
|---|---|
| `src/regularizedvi/_dispersion_init.py` | `label_key`, `halve_counts`; `compute_bb_dispersion_init()` |
| `src/regularizedvi/_constants.py` | `NUCLEOSOME_UNIT_BP`, `DEFAULT_MODALITY_GENOMIC_WIDTH`; `DECODER_TYPE_DEFAULTS` for probability/Kon_Koff |
| `src/regularizedvi/_components.py` | `px_p_decoder` init; `probability` forward (sigmoid head); `Kon_Koff` forward (Kon/(Kon+Koff)) |
| `src/regularizedvi/_multimodule.py` | Logit-space library in inference; probability-space feature_scaling (sigmoid + Beta prior); BetaBinomial generative; loss clamping; `total_count` + `global_max_robust` buffers; kon_koff_outputs dict |
| `src/regularizedvi/_multimodel.py` | `modality_genomic_width`; per-modality `dispersion_init_bio_frac`; logit-space library stats; BB dispersion init routing; `feature_scaling_prob_prior_*` params |
| `scripts/claude_helper_scripts/compute_atac_theta_subset.py` | `--label-key`, `--halve-counts`, count>n check |

## Reference
- Research: `imperative-riding-phoenix-research.md`
- Parent plan: `sunny-painting-gosling.md`
- Previous version review: `imperative-riding-phoenix-review.md`
