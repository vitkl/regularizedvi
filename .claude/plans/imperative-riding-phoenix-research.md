# ATAC Theta Modelling: BetaBinomial vs NegativeBinomial Investigation

## Context

The plan at `sunny-painting-gosling.md` proposes a bursting model decoder (Option E) for RNA but leaves ATAC theta modelling as an open question (Q2). This investigation examines whether the base-pair-level BetaBinomial model for ATAC accessibility changes the recommendation for the ATAC observation model in regularizedvi.

**The base-pair level model:**
```
data_{cells, bp} = BetaBinomial(Kon_{bp} * d * s_cell, Koff_{bp} * d * (1 - s_cell), n=2)
s_cell ~ Beta(1, 1)        # detection probability
1/sqrt(d) ~ Exp(1)         # technical variance
```
where n=2 reflects diploidy. regularizedvi aggregates ATAC into 500-1000bp bins, summing ~K base pairs per feature.

---

## Finding 1: Sum of K BetaBinomials is NOT a BetaBinomial

The BetaBinomial family is **not closed under convolution**. S = sum(X_i) for K independent BetaBinomial(a_i, b_i, 2) does not belong to any named parametric family.

However, since each X_i takes values {0, 1, 2}, the exact PMF of S can be computed via iterative polynomial convolution in O(K^2) — trivial for K=1000 (~4M multiply-adds). This is also differentiable via autograd.

**Key references:** Teerapabolarn (2014, 2015) provides Stein-method bounds for Binomial and Poisson approximations to sums of BetaBinomials.

---

## Finding 2: Independent-p vs Shared-p is a Critical Distinction

**Independent per-base-pair p (biologically correct for uncorrelated sites):** Each bp draws p_i ~ Beta(a_i, b_i) independently, then X_i ~ Bin(2, p_i). Variance of S grows **linearly** in K.

**Shared p = BetaBinomial(a, b, 2K) (the naive aggregation):** One p draw for the entire peak. Variance grows **quadratically** in K.

Variance ratio (homogeneous case):
```
Var[shared-p] / Var[independent-p] = (a+b+2K) / (a+b+2)
```

| K    | a+b=1  | a+b=10 | a+b=100 |
|------|--------|--------|---------|
| 500  | 334x   | 84x    | 10x     |
| 1000 | 667x   | 167x   | 20x     |

**However — base pairs within a peak are NOT independent** (see Finding 6). The effective number of independent units is much smaller than K.

---

## Finding 3: ~~At Low Sensitivity (~1-3%), Aggregated Counts are Essentially Poisson~~ REVISED

### Original theoretical prediction (WRONG for real data)

Typical parameters assumed: K=1000, s=0.02, Kon~1, Koff~10, d=1.
- Predicted overdispersion ratio: ~1.09 (near-Poisson)

### Empirical reality (from `dispersion_init_analysis_out.ipynb`)

**ATAC has massive overdispersion — theta is 10-30x LOWER than RNA:**

| Dataset | Median theta (bf=0.9, Opt1) | Median theta (bf=0.9, Opt2) |
|---|---|---|
| **bm_atac** (116k peaks) | **0.317** | **0.207** |
| **embryo_atac** (342k tiles) | **0.112** | **0.051** |
| bm_rna (26k genes) | 3.452 | 1.817 |
| embryo_rna (37k genes) | 1.462 | 0.779 |
| immune_rna (26k genes) | 1.455 | 0.738 |

Key observations:
- **Zero sub-Poisson features in ATAC** (vs 4500+ in RNA)
- ATAC 25th percentile theta: 0.06-0.18 — extremely overdispersed
- Even the 99th percentile ATAC theta (5-20) is well below Poisson

### Why the theoretical model underestimated overdispersion

The simple base-pair model assumed independent sites, but real ATAC overdispersion comes from multiple correlated sources:
1. **Within-peak correlation** — accessibility is determined at nucleosome scale (~147bp), not per-bp. A 500bp peak has ~3 effective independent units, not 500.
2. **Library size variation** — CV²(L) = 0.5-1.2 across datasets, creating substantial count inflation variance
3. **Batch effects** — systematic accessibility differences across experiments
4. **Cell type mixing within clusters** — the decoder can't perfectly separate all cell states
5. **Doublets/multiplets** — inflate both mean and variance

**Conclusion: ATAC is NOT Poisson-like. NB dispersion is essential, not a safety valve. The PoissonVI claim (Martens et al. 2023) likely reflects limited datasets or insufficient sensitivity analysis.**

Note on PoissonVI: Their fragment-vs-read insight (A) is valid — counting fragments instead of cut sites removes the even-count enrichment from paired-end reads. But their Poisson claim (B) does not hold across diverse datasets with the overdispersion levels seen here.

---

## Finding 4: Literature Landscape for ATAC Observation Models

| Distribution | Used by | Notes |
|---|---|---|
| **Bernoulli** | PeakVI, MultiVI, SCALE, cisTopic | Binarize counts; "biology is binary" |
| **Poisson** | PoissonVI (2023) | Fragment counts; outperforms Bernoulli but Poisson claim is dataset-dependent |
| **NB/ZINB** | scaDA (2024), BAVARIA (2022) | For DA testing; adds overdispersion |
| **BetaBinomial** | scDALI (2022) — allelic counts only | Overdispersion theta~2-5; Binomial miscalibrated |
| **Neg. Multinomial** | BAVARIA (2022) | Joint profile modeling, handles overdispersion |
| **LSI/NMF** | ArchR, Signac, SnapATAC, scOpen | No explicit distribution |

**No VAE/generative model uses BetaBinomial for total (non-allelic) scATAC peak counts.**

### SCALE (Xiong et al. 2019) — Deeper Investigation

- **Key innovation is GMM prior** (VaDE-based), not the Bernoulli likelihood
- Shallow decoder (single linear layer + sigmoid) — GMM does the heavy lifting
- **SCALEX (2022 successor) dropped GMM** in favor of domain-specific batch normalization
- **CASTLE (2024) showed VQ-VAE outperforms GMM-VAE** for scATAC
- **scMVP (2021)**: multimodal VAE (RNA+ATAC) with GMM prior — closest to our use case
- regularizedvi's existing inductive biases (ambient correction, hierarchical dispersion, feature scaling, batch-free decoder) may serve the same role as GMM (preventing mode collapse in sparse data)

---

## Finding 5: Within-Peak Base Pairs are NOT Independent

This is critical for evaluating BetaBinomial(a, b, n=2K) as an aggregate model.

**Why base pairs within a peak are correlated:**
1. **Nucleosome positioning** — a ~147bp nucleosome either occupies a position or doesn't; this creates ~150bp blocks of co-accessible DNA
2. A 500bp peak spans **~3 nucleosome-free regions**, a 1000bp peak spans **~6-7**
3. Within a nucleosome-free region, all base pairs are co-accessible
4. Tn5 has sequence preferences but these operate within accessible regions

**Effective independent units per peak: ~3-7 nucleosome-sized blocks, NOT 500-1000 base pairs**

This means:
- BetaBinomial(a, b, n=2×3) to BetaBinomial(a, b, n=2×7) per nucleosome block, then summed
- The effective n per peak is ~6-14, not ~1000-2000
- With small effective n, the BetaBinomial overdispersion is much more relevant (not averaged away)
- This is consistent with the empirical low theta: few effective independent units → high variance → low theta

### Pseudobulk BetaBinomial — Valid Decomposition

For pseudobulk with M cells:
- **Per base pair position**: BetaBinomial(a, b, n=2M) is **valid** — cells are truly independently sampled
- **Per nucleosome unit**: BetaBinomial(a, b, n=2M) is valid — each cell's nucleosome configuration is independent
- **Extending n to include base pairs**: **NOT valid** — positions within a nucleosome-free region are correlated, not independent trials
- Correct pseudobulk model: BetaBinomial(a_j, b_j, n=2M) per nucleosome unit j, then sum across ~3-7 units per peak

---

## Finding 6: Multiplets — Both a Problem and an Opportunity for BetaBinomial

### The Problem (original analysis)
For Binomial/BetaBinomial, n is part of the distribution:
- Singlet: n = 2 × effective_units (~6-14)
- Doublet: n = 4 × effective_units (~12-28)

We don't know which cells are multiplets, so n is misspecified for doublets.

### The Opportunity (user insight)
**The likelihood penalty IS informative for doublet detection.** If singlets are modeled as BetaBinomial(a, b, n_singlet), doublets would have **systematically lower per-cell log-likelihood** because:
- Their counts are generated from a larger n (more trials)
- The variance structure differs (sum of two independent BB draws ≠ one BB with larger n)
- This is the likelihood-space analogue of AMULET's hard-count threshold (>2 fragments per locus)

**Per-cell log-likelihood under the singlet model could serve as a continuous doublet score.** NB obscures this signal by absorbing doublet effects into the flexible mean and library size.

### Practical consideration
Whether this doublet-detection benefit outweighs the n-misspecification cost depends on:
- What fraction of cells are doublets (typically 5-15% after QC)
- Whether the model can still learn good latent representations despite misspecified n for doublets
- Whether external doublet removal (scrublet, AMULET) is sufficient

---

## Analysis A-D: Model Comparisons (Updated with Empirical Theta)

### A: BetaBinomial(Kon*d*s, Koff*d*(1-s), n=2K) — Two Biological Parameters

- Uses shared-p assumption → overestimates variance
- At low s: only ratio Kon/Koff is identifiable
- **But**: if n is reinterpreted as n=2×effective_nucleosome_units (~6-14), the shared-p model becomes more reasonable (accessibility IS correlated at nucleosome scale)

**Revised verdict**: More reasonable than initially thought if n reflects nucleosome units, not base pairs. Two parameters still have identifiability issues at low sensitivity.

### B: BetaBinomial(d*s*p, d*(1-s*p), n=2×effective_units) — One Biological Parameter

- Mean = n × s × p
- Overdispersion controlled by d
- n = 2 × effective_nucleosome_units per peak (could be precomputed from peak width or set as a fixed ratio)

**Revised verdict**: This is actually a viable alternative to NB. With small effective n (~6-14), overdispersion from the Beta mixing is substantial (not averaged away), consistent with empirical low theta.

### C: Binomial(s*p, n=2×effective_units) — No Overdispersion

**Verdict**: Insufficient — empirical data shows massive overdispersion in ATAC.

### D: NB (current model)

- No n parameter — handles multiplets naturally
- Can represent any level of overdispersion via theta
- Same architecture as RNA
- **Empirically adequate** — the current model works

---

## Current Status: Open Questions for Decision

### Keep NB (conservative) vs Explore BetaBinomial (principled)

**Arguments for keeping NB:**
1. Works adequately for current experiments
2. Handles multiplets without n specification
3. Same architecture as RNA — shared code, shared priors
4. No need to estimate effective_units per peak

**Arguments for exploring BetaBinomial:**
1. More principled generative story for chromatin accessibility
2. The n parameter provides structural information (peak width → effective trials)
3. Could enable doublet detection via per-cell likelihood
4. scDALI shows BB is necessary for allelic counts; similar logic applies to peak counts
5. Natural connection to the Kon/Koff biophysical model

**If BetaBinomial is pursued, it should be a separate plan** (per user instruction). Key design questions:
- How to estimate effective_nucleosome_units per peak (from peak width? from fragment size distribution? fixed ratio?)
- Whether n should be per-peak or fixed across all peaks
- How to handle the n-misspecification for doublets (feature or bug?)
- Parameterization: (d*s*p, d*(1-s*p), n) or (Kon*d*s, Koff*d*(1-s), n)?
- scvi-tools already has `scvi.distributions.BetaBinomial` (wrapping Pyro)

---

## Key References

| Paper | Key Finding |
|---|---|
| Martens et al. 2023 (PoissonVI) | Fragment (not read) counting is correct; Poisson claim dataset-dependent |
| Heinen et al. 2022 (scDALI) | BetaBinomial needed for allelic ATAC; theta~2-5; Binomial miscalibrated |
| Ashuach et al. 2022 (PeakVI) | Bernoulli — binary model baseline |
| Ashuach et al. 2023 (MultiVI) | Bernoulli for ATAC + NB for RNA in joint model |
| Xiong et al. 2019 (SCALE) | GMM prior is the key innovation, not Bernoulli likelihood |
| CASTLE 2024 | VQ-VAE outperforms GMM-VAE for scATAC |
| scMVP 2021 | Multimodal VAE (RNA+ATAC) with GMM prior |
| Teerapabolarn 2014, 2015 | Binomial/Poisson bounds for sums of BetaBinomials |
| Thibodeau et al. 2021 (AMULET) | Doublets detected via >2 fragments per locus |
| Larsson et al. 2019 | Burst kinetics; chromatin state ↔ transcription state |
| Kwok et al. 2025 (Genome Biology) | Hierarchical Poisson-Binomial model for scATAC |
| scaDA Zhao et al. 2024 | ZINB for differential ATAC accessibility |
| BAVARIA 2022 | Negative multinomial for ATAC profile modeling |

---

## Implementation Plan

### Step 1: Extend `_dispersion_init.py` with `groupby_key` + `halve_counts`

**File:** `src/regularizedvi/_dispersion_init.py` (471 lines)

#### New parameters for `compute_dispersion_init()`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `groupby_key` | `str \| None` | `None` | obs column for cell type grouping (e.g. `l2_cell_type`) |
| `halve_counts` | `bool` | `False` | Apply `floor(x/2)` to approximate fragment counts from cut sites |
| `min_cells_per_group` | `int` | `50` | Skip groups smaller than this |

#### `halve_counts` implementation:
- Applied in the chunk loading step, before Welford accumulation
- In both `_compute_from_h5ad` and `_compute_from_anndata`, after `chunk = ...toarray()...`:
  ```python
  if halve_counts:
      chunk = chunk // 2
  ```
- Library sizes recomputed from (halved) counts
- Thread `halve_counts` through `compute_dispersion_init` → `_compute_from_h5ad` / `_compute_from_anndata`

#### `groupby_key` implementation:
New internal function `_compute_within_group_stats()`:

```python
def _compute_within_group_stats(mat, group_codes, unique_groups, min_cells, chunk_size, halve_counts, verbose):
    """Compute pooled within-group mean, variance, and library stats.

    For each group k with n_k >= min_cells:
      1. Compute mean_g_k, var_g_k, lib_sizes_k via Welford (chunked)
      2. Accumulate: pooled_m2_g += n_k * var_g_k
                     pooled_mean_sum += n_k * mean_g_k
                     pooled_n += n_k

    Returns pooled_mean_g, pooled_var_g, pooled_cv2_L, group_info_dict
    """
```

The key change: instead of one pass over all cells computing global mean/var, we do **one pass per group** (or smarter: one pass over all cells, accumulating per-group Welford stats simultaneously).

**Efficient approach (single pass):**
- Read group_codes upfront
- During the chunk loop, for each chunk:
  - Determine which groups are present in this chunk
  - For each group in the chunk, extract the subset of rows
  - Update per-group Welford accumulators (mean_g_k, m2_g_k, count_k, lib_sum_k, lib_m2_k)
- After all chunks: compute per-group var_g_k = m2_g_k / count_k
- Pool: `var_within_g = sum(count_k * var_g_k) / sum(count_k)`

This is a single pass over the data, same as current, with per-group bookkeeping.

**Memory for per-group accumulators:** For G groups and N features: `2 * G * N * 8 bytes` (mean + m2 per group). For embryo (139 groups × 3.1M features): ~3.5 GB. May need to stream groups separately for very large data.

**Fallback for large data:** If `n_groups * n_features > threshold` (e.g. 500M), fall back to iterating groups sequentially (one pass per group). Slower (139 passes) but constant memory.

#### Output changes:
When `groupby_key` is provided, diagnostics dict gets additional keys:
```python
"theta_within_celltype": theta_within,          # per-gene, from pooled within-group variance
"var_within_g": pooled_within_var,              # pooled within-group variance
"mean_within_g": pooled_within_mean,            # pooled within-group mean
"cv2_L_within_celltype": cv2_L_pooled,          # pooled within-group library CV²
"n_groups_used": n_valid_groups,                # groups with >= min_cells
"group_sizes": group_size_dict,                 # {group_name: n_cells}
```

The primary return `log_theta` remains computed from the global statistics (backward compatible). The within-cell-type theta is in diagnostics for comparison.

### Step 2: Run Analysis on Bone Marrow and Embryo ATAC

**Script:** New helper script `scripts/claude_helper_scripts/compute_within_celltype_theta.py`

Runs `compute_dispersion_init` with different configurations and saves results as npz files.

#### Bone Marrow ATAC (quick, ~5 min)
```
Path: /nfs/team283/vk7/sanger_projects/large_data/bone_marrow/bmmc_multiome_multivi_neurips21_curated.h5ad
layer='counts', feature_type='ATAC', groupby_key='l2_cell_type'
```

Configurations to run:
1. Global, cut sites: `groupby_key=None, halve_counts=False`
2. Global, fragments: `groupby_key=None, halve_counts=True`
3. Within-cell-type, cut sites: `groupby_key='l2_cell_type', halve_counts=False`
4. Within-cell-type, fragments: `groupby_key='l2_cell_type', halve_counts=True`

For configs 1-2, use `biological_variance_fraction` in [0.0, 0.8, 0.9, 0.95].
For configs 3-4, use `biological_variance_fraction=0.0` (within-type variance is already technical+residual).

#### Embryo ATAC (larger, ~30-60 min)
```
Path: /nfs/team283/vk7/sanger_projects/cell2state_embryo/results/scvi_customV1_sn_reseq2_njc_genes28k_batch1024_nn3000latent700layer1_1000epochs_NucleiTech_embryo_Nhid16Var02_add_NoBatchDecoder_dropIN_DispMAP/outputs/topn539781_bin1000_atac_05_05_2024.h5ad
layer='counts_120', groupby_key='cell_type_lvl5'
```

**Important: subset var_names first** using existing `compute_atac_theta_subset.py` approach — load var_names from trained model.pt, subset to those CREs before computing. Full 3.1M tiles is too large.
```
--model-pt /nfs/team283/vk7/sanger_projects/cell2state_embryo/results/regularizedvi_embryo_v1/embryo_rna_atac_spliced_unspliced_filtered_decl2_biasM_bgf02_resid_phaseA/model/model.pt
--modality atac --layer counts_120
```

Same 4 configurations. After var_names subsetting, the feature count will be manageable. The existing script (`compute_atac_theta_subset.py`) already does this subsetting + MoM computation — it needs extending with the new `groupby_key` and `halve_counts` parameters.

#### Output: comparison tables
For each dataset, print:
```
                      median_theta    25%    75%    n_sub_poisson
global_cut_bf0.0         0.033     0.018  0.070        0
global_cut_bf0.9         0.317     0.184  0.696        0
global_frag_bf0.0        0.021     ...    ...          ...
global_frag_bf0.9        0.180     ...    ...          ...
within_ct_cut_bf0.0      ???       ...    ...          ???
within_ct_frag_bf0.0     ???       ...    ...          ???
```

Save as npz to `results/dispersion_init_analysis/{dataset}_within_celltype.npz`.

### Step 3: Interpret Results and Update ATAC Strategy

**Expected outcome (user's prediction):** Within-cell-type theta will still be very low — overdispersion is real, not an artifact of between-cell-type variance.

**What the comparison tells us:**
- If within-cell-type theta ≈ global theta at bf=0.0: ALL excess variance is technical → bf should be 0 (or near 0)
- If within-cell-type theta ≈ global theta at bf=0.9: most excess variance was biological → bf=0.9 was correct
- If within-cell-type theta is between: calibrates the right bf value for ATAC

**In all cases, ATAC theta will be low (0.01-1.0 range).** This confirms:
1. ATAC needs overdispersion modeling (not Poisson)
2. The containment prior (pushing theta up) is actively harmful for ATAC
3. The decoder overfitting problem (Q2) is real for ATAC

### Step 4: ATAC-Specific Dispersion Strategy (No Code Changes Yet — Decision Point)

Based on Step 3 results, choose between:

**Option A: Per-modality containment prior hyperparameters**
- ATAC: weaker containment (or inverted — push theta DOWN toward empirical values)
- RNA: current containment prior (or bursting model from parent plan)
- Minimal code change: add `containment_prior_strength` per modality in `_multimodule.py`

**Option B: Fixed ATAC theta from data**
- Initialize ATAC theta from within-cell-type MoM estimate
- Don't learn theta for ATAC (freeze parameter)
- Risk: may be too rigid if theta should vary across peaks

**Option C: BetaBinomial for ATAC** (separate plan if pursued)
- Replace NB with BetaBinomial(d×s×p, d×(1-s×p), n=2×round(width/147))
- Decoder outputs p ∈ (0,1) via sigmoid
- d is per-peak dispersion with hierarchical prior
- n is fixed per peak from peak width
- Counts > n: clipped
- Requires: new plan, new implementation, tests

**Option D: Mean-dependent prior for ATAC theta**
- DESeq2-style log(theta) ~ N(log(theta_trend(mu)), sigma²)
- theta_trend is mean-dependent: more shrinkage for low-count peaks
- Compatible with NB architecture, moderate code change

### Critical Files to Modify

| Step | File | Changes |
|---|---|---|
| 1 | `src/regularizedvi/_dispersion_init.py` | Add `groupby_key`, `halve_counts`, `min_cells_per_group` params; add `_compute_within_group_stats()`; update both h5ad and anndata paths |
| 2 | `scripts/claude_helper_scripts/compute_within_celltype_theta.py` | New analysis script |
| 2 | Results npz files | New output files |

### Verification
1. Run on bone marrow first (smaller, faster) — check output is sensible
2. Compare within-cell-type theta to global theta at various bf values
3. Verify halve_counts produces similar results to the earlier PMF-based computation
4. Run on embryo — check memory doesn't blow up with 139 groups × 3.1M features
5. Produce comparison table and save results

---

## Research Findings (Reference)

### Fragment vs Cut Site Theta (Empirical)

**Converting cut sites to fragments does NOT reduce overdispersion — it increases it.**

| Dataset | bf | theta_cut (median) | theta_frag (median) | ratio |
|---|---|---|---|---|
| bm_atac | 0.9 | 0.327 | 0.180 | 0.55 |
| embryo_atac | 0.9 | 0.114 | 0.062 | 0.54 |
| bm_atac | 0.0 | 0.033 | 0.021 | 0.63 |
| embryo_atac | 0.0 | 0.011 | 0.007 | 0.60 |

Method: for NB(mu, theta), compute Y = floor(X/2) exact moments via PMF summation. theta_frag = E[Y]^2 / (Var[Y] - E[Y]).

Fragment counting is correct practice (avoids paired-end artifact) but does not change the modeling decision.

### SCALE Investigation

SCALE's key innovation is the GMM prior, not the Bernoulli likelihood. regularizedvi's existing inductive biases already address mode collapse. Not pursued further.

### ATAC Decoder Overfitting (Q2 from User)

This is the ATAC analogue of the RNA problem in `sunny-painting-gosling.md`. With NB for ATAC:
- Containment prior pushes theta up → decoder explains all variance
- No bursting model coupling for ATAC (unlike RNA Option E)

The bounded sigmoid output of BetaBinomial does NOT directly regularize the decoder — a flexible decoder can still fit noise via sharp sigmoid transitions. The regularization would come from the bounded mean (n×s×p ≤ n×s) preventing the decoder from predicting arbitrarily high means for noisy features, but this is a weak constraint since n×s is small anyway at low sensitivity.

The real solution for ATAC decoder overfitting likely needs one of:
- Per-modality containment prior (weaker for ATAC)
- Mean-dependent prior
- Fixed theta from data
- Architectural constraint (smaller decoder for ATAC)

### Detailed Agent Reports
- `imperative-riding-phoenix-agent-aecdf9d51ad3bf4c8.md` — Sum of BetaBinomials mathematics
- `imperative-riding-phoenix-agent-a2f64399a7691856c.md` — Literature landscape
- `imperative-riding-phoenix-agent-a8e8135b40032c386.md` — NB vs BB comparison analysis

### Key References

| Paper | Key Finding |
|---|---|
| Martens et al. 2023 (PoissonVI) | Fragment (not read) counting is correct; Poisson claim dataset-dependent |
| Heinen et al. 2022 (scDALI) | BetaBinomial needed for allelic ATAC; theta~2-5; Binomial miscalibrated |
| Ashuach et al. 2022 (PeakVI) | Bernoulli — binary model baseline |
| Ashuach et al. 2023 (MultiVI) | Bernoulli for ATAC + NB for RNA in joint model |
| Teerapabolarn 2014, 2015 | Binomial/Poisson bounds for sums of BetaBinomials |
| Thibodeau et al. 2021 (AMULET) | Doublets detected via >2 fragments per locus |
| Larsson et al. 2019 | Burst kinetics; chromatin state ↔ transcription state |
| Kwok et al. 2025 (Genome Biology) | Hierarchical Poisson-Binomial model for scATAC |
| scaDA Zhao et al. 2024 | ZINB for differential ATAC accessibility |
| BAVARIA 2022 | Negative multinomial for ATAC profile modeling |
