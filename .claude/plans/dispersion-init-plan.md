# Plan: Data-Driven Dispersion Initialization

## Context
The model's NB dispersion theta is initialized at `log(rate²)` in log-space. P5 experiments test `disp_mean=1` (theta_init=1) and `disp_mean=2` (theta_init=4). The user wants to:
1. Extract learned theta from P5 experiments (+ check sigma for median vs mean question)
2. Compute principled data-driven initialization via variance decomposition
3. Put the computation into the package for use during model init
4. Compare across 3 datasets to decide: global scalar default (option A) vs per-gene per-dataset (option B)

---

## Potential Bug: Inference uses median, not mean

At `_module.py:846-851`, comment says "use mean at inference" but `exp(px_r_mu)` = LogNormal **median**. The model uses `exp()`, NOT `softplus()` — confirmed by reading `_module.py:851`. True LogNormal mean = `exp(px_r_mu + sigma²/2)`. Will check actual sigma values from P5 models (Task 1) before deciding whether to fix code or just fix comment.

---

## Literature Context

Literature search completed (via web search subagent). Key findings:
- **scVI**: `torch.randn(n_genes)` in log-space → median theta = exp(0) = **1** (effectively initializes at theta=1 with spread, no prior, no data-driven init)
- **DESeq2**: Method-of-moments initial estimate → gene-wise MLE → empirical Bayes shrinkage toward mean-dispersion trend
- **sctransform**: Per-gene ML theta from Poisson GLM residuals → kernel smoothing of theta vs gene mean
- **edgeR**: Common dispersion → trended → tagwise via empirical Bayes shrinkage
- **BASiCS**: Only method decomposing Poisson + technical + biological (requires spike-ins)
- **No single-cell VAE** (scVI, totalVI, cell2location) uses data-driven theta init — this would be novel

---

## Pre- vs Post-Normalization: Which Counts to Use?

The computation should use **raw counts** (before normalization). Rationale:
- The NB likelihood operates on raw counts: `x ~ NB(theta, rate = theta / (rho * library))`
- Library size is modeled explicitly as a latent variable, not removed by normalization
- The variance decomposition separates Poisson + library + gene-specific on raw counts

**What about the 50% shrunk library?** The model uses `library_log_vars_weight=0.5` (`_constants.py:14`), which shrinks the library size prior variance by 50%. This means the model's effective library is partially shrunk toward the batch mean. For the MoM computation on raw counts, we use observed library sizes (not shrunk), because:
- The shrinkage happens inside the model during training
- Our goal is to estimate what theta should be *before* the model starts learning
- The (1+CV²) correction already accounts for library variability; if anything, the model's shrinkage will reduce the effective CV², making our raw-count estimate conservative

---

## Variance Decomposition — Full Derivation (verified by subagent)

### Setup
- Gene g, cell i. **Raw counts** x_{g,i}
- L_i = total count (library size), L̄ = E[L]
- Model: x_{g,i} | L_i ~ NB(μ_g · L_i/L̄, θ_g)
- NB: E[X] = μ, Var[X] = μ + μ²/θ

### Step 1: Conditional expectation
```
E[x_g | L] = μ_g · L/L̄
```

### Step 2: Between-library variance
```
Var_L[E(x_g | L)] = Var[μ_g · L/L̄]
                   = (μ_g/L̄)² · Var(L)        # Var(aX) = a² · Var(X)
                   = μ_g² · Var(L)/L̄²
                   = μ_g² · CV²(L)
```
**Why μ²?** Expression scales linearly with L (x_g ∝ L). Variance of a linearly scaled variable scales quadratically with the scaling factor. A gene with 10× higher mean has 100× more variance from library size fluctuations.

**Why CV²?** CV²(L) = Var(L)/E(L)² is the squared coefficient of variation. It normalizes the library size variability to be independent of the absolute scale of L.

### Step 3: Within-library NB variance (conditional)
```
Var(x_g | L) = μ_g·L/L̄  +  (μ_g·L/L̄)² / θ_g
```
This is just the NB variance formula applied with conditional mean μ_g·L/L̄.

### Step 4: Average within-library variance across cells
```
E_L[Var(x_g | L)] = E[μ_g·L/L̄]  +  E[(μ_g·L/L̄)²] / θ_g
                   = μ_g  +  (μ_g²/θ_g) · E[L²]/L̄²
```
Using E[L²] = Var(L) + L̄², so E[L²]/L̄² = 1 + CV²(L):
```
                   = μ_g  +  (μ_g²/θ_g) · (1 + CV²(L))
```
**Why (1+CV²) here?** Cells with larger L have larger conditional mean, hence larger μ²/θ contribution. Averaging this over the L distribution inflates the NB overdispersion by (1+CV²).

### Step 5: Total variance = between + within (law of total variance)
```
Var(x_g) = μ_g                                    [A: Poisson noise]
         + μ_g² · CV²(L)                          [B+C: library size variability]
         + (μ_g²/θ_g) · (1 + CV²(L))             [D: gene-specific NB overdispersion]
```

**Note on B vs C**: The user asked to separate within-batch library variance (B) from between-batch library mean variance (C). When `batch_key` is provided, the computation will:
- Compute per-batch: E_b(L), Var_b(L), CV²_b(L) — within-batch library variability [B]
- Compute between-batch: Var across batch means of L — between-batch variability [C]
- Report both separately, though for the theta formula they combine into one library term

### Step 6: Solve for θ — absorbing everything into excess_technical
```
excess_raw = Var(x_g) - μ_g - μ_g² · CV²(L)           # remove Poisson [A] + library [B+C]
excess_adjusted = excess_raw / (1 + CV²(L))             # see note below
excess_technical = excess_adjusted / biological_fraction # assume 1/N is technical (default N=10)
θ_g = μ_g² / max(excess_technical, eps)                 # clean formula: θ = μ²/excess_technical
```

**Why divide excess_raw by (1+CV²)?** From Step 5: `excess_raw = (μ²/θ) · (1+CV²)`. This is because the NB overdispersion term μ²/θ gets inflated by the library size distribution when we observe total (marginal) variance. To recover the true per-cell θ, we need to undo that inflation. Algebraically: `θ = μ² · (1+CV²) / excess_raw = μ² / (excess_raw / (1+CV²)) = μ² / excess_adjusted`.

### Code (transparent, step-by-step):
```python
# Per-gene: mean_g, var_g from Welford on raw counts
# Global or per-batch: cv2_L from library sizes

# Step A: Poisson variance
poisson_var = mean_g

# Step B+C: Library size variance
library_var = (mean_g ** 2) * cv2_L

# Step D: Excess (gene-specific, beyond Poisson + library)
excess_raw = var_g - poisson_var - library_var

# Correction: NB overdispersion inflated by library variability
nb_inflation = 1 + cv2_L
excess_adjusted = excess_raw / nb_inflation

# Shrinkage: assume 1/biological_fraction is technical
excess_technical = excess_adjusted / biological_fraction   # biological_fraction=10 default

# Theta (gene-specific overdispersion)
# Note: sub-Poisson genes have excess_raw < 0. After np.maximum(..., eps),
# these become eps → theta becomes very large (≈ near-Poisson), which is correct.
theta_option1 = (mean_g ** 2) / np.maximum(excess_technical, eps)

# Option 2: simple (no library correction)
excess_simple = var_g - poisson_var
excess_technical_simple = excess_simple / biological_fraction
theta_option2 = (mean_g ** 2) / np.maximum(excess_technical_simple, eps)

# For model init: clamp and log (clip bounds are hyperparameters)
theta_clamped = np.clip(theta, theta_min, theta_max)     # theta_min=0.01, theta_max=10.0
log_theta = np.log(theta_clamped)                         # for px_r_init
```

### Edge cases
- excess_raw ≤ 0 for some genes (sub-Poisson after removing library): θ → very large → clipped to theta_max
- Very lowly expressed genes: mean_g ≈ 0 → θ → clipped
- Typical CV²(L) ≈ 0.3–0.5 for QC'd 10x data → (1+CV²) ≈ 1.3–1.5

### How init interacts with the hierarchical prior
Data-driven init sets `px_r_mu` (starting point for optimization). The hierarchical prior `1/sqrt(θ) ~ Exp(rate), rate ~ Gamma(α,β)` remains unchanged and continues to regularize during training. The init just gives the optimizer a better starting point — closer to the data-supported region — so training converges faster and more reliably. The prior still pulls theta toward its preferred range.

---

## Task 1: Extract Learned Theta from P5 Experiments

### Script: `scripts/claude_helper_scripts/extract_learned_dispersion.py`
Run via: `bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/extract_learned_dispersion.py`

```python
For each model.pt:
  state_dict = torch.load(model.pt, map_location='cpu')
  px_r_mu = state_dict['px_r_mu']
  px_r_log_sigma = state_dict['px_r_log_sigma']
  px_r_sigma = exp(px_r_log_sigma)

  theta_median = exp(px_r_mu)                              # what model uses at inference
  theta_mean = exp(px_r_mu + (px_r_sigma ** 2) / 2)       # true LogNormal mean
  learned_rate = softplus(state_dict['dispersion_prior_rate_raw'])

  Report:
  - Quantiles of theta_median across genes: min, 5%, 25%, 50%, 75%, 95%, max
  - Quantiles of theta_mean across genes (same)
  - Quantiles of px_r_sigma across genes (how far from init 0.1?)
  - Ratio theta_mean/theta_median — quantiles (measures median-mean gap)
  - Learned rate value
  - For gene-batch: per-batch and pooled statistics
```

**8 model paths** (all `results/immune_integration_rna_embryo_es2_{name}/model/model.pt`):
- no_tea_small_disp1, no_tea_disp1 (disp_mean=1, dwl2=0.1)
- no_tea_small_disp2, no_tea_disp2 (disp_mean=2, dwl2=0.1)
- no_tea_small_disp1_dwl2_1, no_tea_disp1_dwl2_1 (disp_mean=1, dwl2=1.0)
- no_tea_small_disp2_dwl2_1, no_tea_disp2_dwl2_1 (disp_mean=2, dwl2=1.0)

---

## Task 2: Package Utility for Data-Driven Init

### New file: `src/regularizedvi/_dispersion_init.py`

```python
def compute_dispersion_init(
    adata_or_path: AnnData | str | Path,
    layer: str | None = None,
    dispersion_key: str | None = None,          # same key used for dispersion grouping in model
    biological_variance_fraction: float = 10.0,   # assume bio = 10× technical
    theta_min: float = 0.01,                      # clip floor (linear scale)
    theta_max: float = 10.0,                      # clip ceiling (linear scale)
    chunk_size: int = 5000,
) -> tuple[np.ndarray, dict]:
    """Compute per-gene NB dispersion init from method of moments on raw counts.

    Returns:
        log_theta: array of shape (n_genes,) — log(clipped theta), for px_r_init_mean
        diagnostics: dict with unclipped theta arrays, CV²(L), per-gene breakdown,
                     within/between-batch library stats (if batch_key provided)

    Uses chunked h5py reading via Welford's online algorithm for memory efficiency.
    Welford = numerically stable single-pass streaming computation of mean + variance.
    """
```

`theta_min` and `theta_max` are **hyperparameters** exposed to the user, not hardcoded.

The `diagnostics` dict returns **unclipped** values for inspection (needed for Task 3 exploration):
```python
diagnostics = {
    "theta_option1": theta_option1,           # unclipped, per gene
    "theta_option2": theta_option2,           # unclipped, per gene
    "mean_g": mean_g,                         # per-gene mean (raw counts)
    "var_g": var_g,                           # per-gene variance (raw counts)
    "cv2_L": cv2_L,                           # global CV²(L)
    "cv2_L_within_batch": cv2_L_within,       # within-batch CV²(L) (if batch_key)
    "cv2_L_between_batch": cv2_L_between,     # between-batch CV²(L) (if batch_key)
    "excess_raw": excess_raw,                 # per gene
    "excess_adjusted": excess_adjusted,       # per gene
    "n_sub_poisson": (excess_raw <= 0).sum(), # genes where var ≤ Poisson + library
}
```

### Two-pass algorithm (Welford's online):
- **Pass 1**: Read sparse chunks → per-cell library sizes L_i (and batch codes if batch_key). Compute E(L), Var(L), CV²(L) globally. If batch_key: also per-batch E_b(L), Var_b(L) → pooled within-batch CV² and between-batch CV².
- **Pass 2**: Read sparse chunks → per-gene Welford accumulators for mean_g, var_g on raw counts.

Welford accumulator (confirmed correct by subagent against existing `compute_gene_variance.py`):
```python
count += 1
delta = x - mean
mean += delta / count
delta2 = x - mean
m2 += delta * delta2
# After all data: variance = m2 / count
```

### Integration into both models

**`_model.py`** (single-modal) and **`_multimodel.py`** (multi-modal):
```python
# In __init__:
if dispersion_init == "data":
    from ._dispersion_init import compute_dispersion_init
    log_theta_per_gene, diagnostics = compute_dispersion_init(
        adata, layer=layer,
        biological_variance_fraction=biological_variance_fraction,
        theta_min=theta_min, theta_max=theta_max,
    )
    # Use as px_r_init_mean (array of log(theta) per gene)
```

**`_module.py:340-352`** — generalize `px_r_init_mean` from `float | None` to `float | np.ndarray | None`:
```python
if isinstance(_px_r_init, np.ndarray):
    init_tensor = torch.tensor(_px_r_init, dtype=torch.float32)
    self.px_r_mu = nn.Parameter(init_tensor + _px_r_std * torch.randn(n_input))
elif _px_r_init is not None:
    self.px_r_mu = nn.Parameter(torch.full((n_input,), _px_r_init) + _px_r_std * torch.randn(n_input))
```

For gene-batch branch (`_module.py:358-363`): when `px_r_init_mean` is an array of shape `(n_genes,)`, broadcast to `(n_genes, n_dispersion_cats)` by repeating across the dispersion_key dimension. The dispersion_key groups share the same data-driven init per gene but diverge during training. Same pattern in `_multimodule.py` per-modality `ParameterDict`.

---

## Task 3: Exploration Notebook (Unclipped Values + Visualizations)

### Notebook: `docs/notebooks/model_comparisons/dispersion_init_analysis.ipynb`

Calls `compute_dispersion_init()` with `theta_min=0, theta_max=inf` (no clipping) to get raw estimates. Includes histograms and hist2d plots for all computed quantities.

**Per dataset**:
- CV²(L) global, within-batch, between-batch
- Quantiles of **unclipped** theta (option 1 and 2): min, 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, max
- Quantiles of log(theta) (same percentiles)
- Number of sub-Poisson genes (excess_raw ≤ 0)
- **Histograms**: log(theta) distribution, theta distribution, excess_raw distribution
- **hist2d (linear + log scale)**: theta vs gene mean (mean-dispersion relationship)

**Cross-dataset comparison**: side-by-side quantile table

**Datasets and modalities** (separate numbers per modality):

| # | Dataset | Modality | Path | Notes |
|---|---------|----------|------|-------|
| 1 | Immune integration | RNA | `results/immune_integration/adata_rna.h5ad` | ~706k cells |
| 2 | Immune integration | ATAC | `results/immune_integration/adata_atac_tiles.h5ad` | tiles |
| 3 | Bone marrow | RNA | `data/bmmc_multiome_multivi_neurips21_curated.h5ad` | Split by `var['feature_types']=='GEX'` |
| 4 | Bone marrow | ATAC | same file | Split by `var['feature_types']=='Peaks'`, 142k features total |
| 5 | Embryo | RNA | `/nfs/team283/vk7/.../scvi_customV1_.../outputs/transferred_rna.h5ad` | |
| 6 | Embryo | ATAC | `/nfs/team283/vk7/.../scvi_customV1_.../outputs/topn539781_bin1000_atac_05_05_2024.h5ad` | Use `layers['counts_120']` as X (training counts layer). 539k cells × 3M features before CRE selection |

Full embryo ATAC path: `/nfs/team283/vk7/sanger_projects/cell2state_embryo/results/scvi_customV1_sn_reseq2_njc_genes28k_batch1024_nn3000latent700layer1_1000epochs_NucleiTech_embryo_Nhid16Var02_add_NoBatchDecoder_dropIN_DispMAP/outputs/topn539781_bin1000_atac_05_05_2024.h5ad`

Reference notebook: `/nfs/team205/vk7/sanger_projects/cell2state_embryo/notebooks/benchmark/regularizedvi/results/embryo_rna_atac_spliced_unspliced_filtered_decl2_biasM_bgf02_resid_phaseA_es1e4_data_v0_out_full.ipynb`

ATAC modality is critical to include — dispersion behavior may be very different for binary/sparse tile counts vs RNA UMI counts.

---

## Task 4: Synthesis & Decision

After Tasks 1–3, compare (in the same notebook):
1. **hist2d: P5 learned theta vs MoM theta** (per gene) — are they correlated?
2. **hist2d: init=1 vs init=2 vs init=3 (default) learned theta** — do they converge to similar values? Include default prior=3 models if available.
3. **Cross-dataset MoM theta** — are quantiles consistent enough for a global default?
4. **Cross-modality MoM theta** — how different is RNA vs ATAC?
5. **Sigma values** — decide whether median vs mean bug matters.

**Decision**: Choose between:
- **Option A**: Update global default `regularise_dispersion_prior` to a better scalar (e.g., if MoM median ≈ 1 across datasets, make default=1)
- **Option B**: Per-gene init via `dispersion_init="data"` for datasets where it matters
- **Both**: New global default + optional per-gene override

---

## Execution order
1. Task 1: Extract learned theta from P5 models (quick: just load .pt files)
2. Task 2: Create `_dispersion_init.py` + generalize `px_r_init_mean` in module/multimodule
3. Task 3: Create exploration notebook, run MoM on all datasets/modalities with unclipped values + histograms
4. Task 4: Synthesis in same notebook — hist2d comparisons, cross-dataset/modality tables, decide on init strategy
5. Fix median/mean comment (or code) based on Task 1 sigma findings

## Critical files to modify
| File | Change |
|------|--------|
| `src/regularizedvi/_dispersion_init.py` | NEW: variance decomposition utility |
| `src/regularizedvi/_module.py:340-368` | Generalize px_r_init_mean to accept array |
| `src/regularizedvi/_module.py:846` | Fix comment (median not mean) or fix code |
| `src/regularizedvi/_multimodule.py` | Same array init generalization |
| `src/regularizedvi/_model.py` | Add `dispersion_init="data"` option |
| `src/regularizedvi/_multimodel.py` | Same dispersion_init option |
| `scripts/claude_helper_scripts/extract_learned_dispersion.py` | NEW: P5 theta extraction |
| `docs/notebooks/model_comparisons/dispersion_init_analysis.ipynb` | NEW: exploration notebook with hist2d plots |

## Verification
- Extract theta from P5 models and compare distributions across init values
- Check sigma values from P5: assess median vs mean gap magnitude
- Run MoM on 3 datasets: check cross-dataset consistency
- Compare MoM log(theta) quantiles to learned values
- Run package tests after code changes: `bash run_tests.sh tests/test_model.py -x -q`
