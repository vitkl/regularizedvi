# regularizedvi

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/vitkl/regularizedvi/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/regularizedvi

Regularized scVI with ambient RNA correction and overdispersion regularisation, based on [cell2location](https://www.nature.com/articles/s41587-021-01139-4)/[cell2fate](https://www.nature.com/articles/s41592-025-02608-3) modelling principles (Kleshchevnikov et al. 2022, Aivazidis et al. 2025, Simpson et al. 2017).

The modifications (ambient RNA correction, dispersion prior, batch-free decoder, learned library size) act as structural inductive biases that make a high-capacity model (`n_hidden=512+`, `n_latent=128+`) well-behaved by default, removing the need for careful per-dataset hyperparameter tuning. This is particularly important for complex datasets with hundreds of cell types (e.g. whole-embryo atlases, cross-atlas integration) where large latent spaces and wide hidden layers are needed to avoid competition between cell types for representational capacity.

## Motivation

### Standard scVI generative model

Standard scVI (Lopez et al. 2018) models observed UMI counts $x\_{ng}$ for cell $n$ and gene $g$ as:

$$z_n \sim \text{Normal}(0, I)$$
$$\ell_n \sim \text{LogNormal}(\ell_\mu^\top s_n,\; \ell_{\sigma^2}^\top s_n)$$
$$\rho_{ng} = f_w(z_n, s_n, c_n) \in \Delta^{G-1}$$
$$x_{ng} \sim \text{NB}(\mu = \ell_n \rho_{ng},\; \theta_{g,s_n})$$

where:
- $z\_n \in \mathbb{R}^d$ — low-dimensional latent cell state
- $\ell\_n \in (0, \infty)$ — library size (by default fixed to total UMI count per cell), with log-normal prior parameterised per batch $s\_n$
- $\rho\_n \in \Delta^{G-1}$ — decoder output on the probability simplex (via softmax) as a fraction of total $\ell\_n$ RNA per cell, representing denoised normalised gene expression
- $f\_w(z\_n, s\_n): \mathbb{R}^d \times \{0,1\}^K \to \Delta^{G-1}$ — decoder neural network, conditioned on batch $s\_n$
- $\theta\_{g,s\_n} \in (0, \infty)$ — per-gene, per-batch inverse dispersion (code: `px_r`, stored as unconstrained $\phi\_{g,s\_n}$ where $\theta\_{g,s\_n} = \exp(\phi\_{g,s\_n})$ )
- $s\_n \in \{0,1\}^K$ — one-hot batch indicator for cell $n$
- $c\_n \in \{0,1\}^K$ — one-hot categorical covariate indicator for cell $n$

The inference model uses amortised variational inference to fit all cell specific variables (encoder NNs): $q\_\eta(z\_n, \ell\_n \mid x\_n, s\_n, c\_n) = q\_\eta(z\_n \mid x\_n, s\_n, c\_n) \, q\_\eta(\ell\_n \mid x\_n, s\_n, c\_n)$. Note that both batch $s\_n$ and $c\_n$ categorical covariates are used in both decoders (model) and encoders (amortised variational inference of  $z\_n, \ell\_n$).

### regularizedvi generative model

**regularizedvi** adapts [cell2location](https://doi.org/10.1038/s41587-021-01139-4)/[cell2fate](https://doi.org/10.1038/s41592-025-02608-3) modelling principles to scVI. All learnable parameters are initialised at their prior means to improve training stability.

**Latent variable and library size** — standard scVI structure with a constrained library prior. Library prior parameters $\ell\_p^{\mu}$, $\ell\_p^{\sigma^2}$ are computed per `library_size_key` group $p$ (one key, [`_module.py:270–275`](src/regularizedvi/_module.py#L270-L275)):

$$z_n \sim \text{Normal}(0, I)$$
$$\ell_n \sim \text{LogNormal}(\ell_p^{\mu},\; 0.05 \cdot \ell_p^{\sigma^2})$$

**Decoder output** — batch-free decoder maps $z\_n$ and categorical covariates $c\_{k,n}$ (not ambient/library covariates) to non-negative gene expression via softplus. Categorical covariates are selected by `categorical_covariate_keys` (many keys, [`_module.py:654–685`](src/regularizedvi/_module.py#L654-L685), [`_components.py:461–462`](src/regularizedvi/_components.py#L461-L462)):

$$\rho_{ng} = \text{softplus}\big(f_w(z_n, c_{k,n})\big) \in \mathbb{R}_{\geq 0}^G$$

**Additive background** — per-gene ambient RNA with Gamma prior pushing $s\_{e,g}$ toward 0.01. Background parameters are indexed by `ambient_covariate_keys` (many keys, concatenated one-hot, [`_module.py:336–341`](src/regularizedvi/_module.py#L336-L341) init, [`_module.py:641–651`](src/regularizedvi/_module.py#L641-L651) one-hot selection, [`_module.py:790–798`](src/regularizedvi/_module.py#L790-L798) prior penalty):

$$s_{e,g} = \exp(\beta_{e,g}), \qquad s_{e,g} \sim \text{Gamma}(1,\, 100)$$

**Hierarchical dispersion prior** — two-level prior on inverse dispersion $\theta\_{g,d} = \exp(\phi\_{g,d})$ where `px_r` stores $\phi$. Dispersion groups $d$ are selected by `dispersion_key` (one key). A learned rate $\lambda\_d$ adapts regularisation strength per group ([`_module.py:750–786`](src/regularizedvi/_module.py#L750-L786) full block, [`_module.py:762`](src/regularizedvi/_module.py#L762) Level 1 softplus, [`_module.py:784–785`](src/regularizedvi/_module.py#L784-L785) Level 2 transform):

$$\lambda_d \sim \text{Gamma}(9,\, 3), \qquad 1/\sqrt{\theta_{g,d}} \sim \text{Exponential}(\lambda_d)$$

**Observation model** — GammaPoisson (= negative binomial) with mean $\mu\_{ng} = \ell\_n \cdot (\rho\_{ng} + s\_{e\_n,g})$ ([`_components.py:467`](src/regularizedvi/_components.py#L467) rate computation):

$$x_{ng} \sim \text{GammaPoisson}\Big(\text{concentration} = \theta_{g,d_n},\;\; \text{rate} = \frac{\theta_{g,d_n}}{\ell_n \cdot \big(\rho_{ng} + s_{e_n,g}\big)}\Big)$$

**Notation:**
- $s\_{e,g} = \exp(\beta\_{e,g})$ — per-gene ambient background indexed by `ambient_covariate_keys` (many keys); $\beta$ is the unconstrained parameter (code: `additive_background`). When `batch_key` is used alone, $e$ = batch group.
- $c\_{k,n}$ — categorical covariates (site, donor, etc.), selected by `categorical_covariate_keys` (many keys). Injected into encoder and decoder.
- $\rho\_{ng} \in \mathbb{R}\_{\geq 0}^G$ — decoder output via softplus (not on the simplex), since $\rho\_{ng} + s\_{e,g}$ need not sum to 1
- $\theta\_{g,d} = \exp(\phi\_{g,d})$ — inverse dispersion indexed by `dispersion_key` (one key); initialised at $\log(\lambda^2) \approx 2.2$ so $\theta \approx 9$ (equilibrium: $1/\sqrt{\theta} = 1/\lambda$) ([`_module.py:287–295`](src/regularizedvi/_module.py#L287-L295))
- $\lambda\_d$ — learned Exponential rate, one per dispersion group; $\text{Gamma}(9, 3)$ hyper-prior has mean 3 ([`_module.py:314–323`](src/regularizedvi/_module.py#L314-L323))
- $\ell\_p^{\mu}$, $\ell\_p^{\sigma^2}$ — library prior mean and variance per `library_size_key` group $p$ (one key). $0.05$ scaling factor prevents library size from absorbing biological signal.
- **Backward compat**: When `batch_key` is used alone, $e = d = p$ (all index the same batch groups) and $t = k$ (categorical and scaling covariates share groups).

The NB variance is $\text{Var}(x) = \mu + \mu^2/\theta$. Large $\theta$ → less overdispersion (Poisson limit). The Exponential prior on $1/\sqrt{\theta}$ is a containment prior (Simpson et al. 2017) that penalises large $1/\sqrt{\theta}$ (= small $\theta$, excessive overdispersion), regularising the NB toward the Poisson baseline. The data likelihood provides the opposing force, pulling $\theta$ toward values needed to explain observed count variance. At equilibrium $\theta \approx \lambda^2 = 9$, giving moderate overdispersion. This forces the decoder to capture biological signal through its mean structure rather than absorbing residuals via high variance.

### RegularizedMultimodalVI generative model

**RegularizedMultimodalVI** extends regularizedvi to $M$ paired modalities (e.g., RNA + ATAC from 10x Multiome). Each modality has its own dedicated encoder and decoder, but all decoders share a single joint latent space formed by concatenating per-modality codes ("symmetric concatenation"). The generative model for every modality follows the same structure — only which optional correction terms are active differs between modalities.

#### Latent space

Each modality $m$ contributes a private slice of the joint latent space. These slices are independently drawn from a standard normal prior and concatenated to form the full cell representation $z\_n$ that is fed to all decoders:

$$z^{(m)}_n \sim \text{Normal}(0, I_{d_m})$$
$$z_n = [\,z^{(1)}_n;\; z^{(2)}_n;\; \ldots;\; z^{(M)}_n\,] \in \mathbb{R}^{\sum_m d_m}$$

where $d\_m$ is the latent dimensionality assigned to modality $m$ (e.g. `n_latent={"rna": 96, "atac": 32}`). Because every decoder receives the full $z\_n$, signals across modalities can interact through the decoders even though each encoder sees only its own modality.

#### Generative model (per modality $m$)

The following equations describe how observed counts $x^{(m)}\_{nf}$ — UMIs for RNA, fragment counts for ATAC — are generated for cell $n$ and feature $f$ (gene or chromatin peak). All modalities share this structure; the optional terms in the mean $\mu^{(m)}\_{nf}$ are selectively activated per modality.

**Library size** — always learned (observed totals include ambient contamination). A low-capacity encoder infers library size per cell, regularised by a tight LogNormal prior estimated per `library_size_key` group $p$ (one key, [`_multimodule.py:396–397`](src/regularizedvi/_multimodule.py#L396-L397) prior buffers, [`_multimodule.py:925–937`](src/regularizedvi/_multimodule.py#L925-L937) loss):

$$\ell^{(m)}_n \sim \text{LogNormal}\big(\ell^{(m),\mu}_p,\; 0.05 \cdot \ell^{(m),\sigma^2}_p\big)$$

**Decoder output** — maps joint latent code $z\_n$ and categorical covariates $c\_{k,n}$ (selected by `categorical_covariate_keys`, many keys) to non-negative feature signal via softplus ([`_multimodule.py:760–775`](src/regularizedvi/_multimodule.py#L760-L775)):

$$\rho^{(m)}_{nf} = \text{softplus}\big(f^{(m)}_w(z_n,\, c_{k,n})\big) \in \mathbb{R}_{\geq 0}$$

**Additive background** — per-feature ambient contamination with Gamma prior, indexed by `ambient_covariate_keys` (many keys, concatenated one-hot, [`_multimodule.py:458–465`](src/regularizedvi/_multimodule.py#L458-L465) init, [`_multimodule.py:756`](src/regularizedvi/_multimodule.py#L756) one-hot selection, [`_multimodule.py:991–1003`](src/regularizedvi/_multimodule.py#L991-L1003) prior penalty):

$$s^{(m)}_{e,f} = \exp(\beta^{(m)}_{e,f}), \qquad s^{(m)}_{e,f} \sim \text{Gamma}(1,\, 100)$$

**Region factors** — per-feature, per-covariate multiplicative scaling capturing systematic biases (GC content, mappability, peak caller sensitivity). Parameterised as $\text{softplus}(\gamma)/0.7$ with a tight Gamma prior centered at 1. Scaling covariates $t$ are selected by `modality_scaling_covariate_keys` (many keys); each covariate category gets its own factor ([`_multimodule.py:472–476`](src/regularizedvi/_multimodule.py#L472-L476) init, [`_multimodule.py:779–787`](src/regularizedvi/_multimodule.py#L779-L787) activation and selection, [`_multimodule.py:972–989`](src/regularizedvi/_multimodule.py#L972-L989) prior penalty):

$$y^{(m)}_{t,f} = \text{softplus}(\gamma^{(m)}_{t,f})\,/\,0.7, \qquad y^{(m)}_{t,f} \sim \text{Gamma}(200,\, 200)$$

**Expected mean counts** — decoder output plus optional background, scaled by library size and region factor ([`_components.py:467`](src/regularizedvi/_components.py#L467) base rate, [`_multimodule.py:789`](src/regularizedvi/_multimodule.py#L789) region factor scaling):

$$\mu^{(m)}_{nf} = \ell^{(m)}_n \cdot \big(\rho^{(m)}_{nf} + s^{(m)}_{e_n,f}\big) \cdot y^{(m)}_{t_n,f}$$

**Hierarchical dispersion prior** — same two-level structure as single-modality, per modality and `dispersion_key` group $d$ (one key, [`_multimodule.py:940–970`](src/regularizedvi/_multimodule.py#L940-L970)):

$$\lambda^{(m)}_d \sim \text{Gamma}(9,\, 3), \qquad 1/\sqrt{\theta^{(m)}_{f,d}} \sim \text{Exponential}(\lambda^{(m)}_d)$$

**Observation model** — GammaPoisson (= negative binomial) with mean $\mu^{(m)}\_{nf}$ and inverse dispersion $\theta^{(m)}\_{f,d}$:

$$x^{(m)}_{nf} \sim \text{GammaPoisson}\!\Big(\text{concentration} = \theta^{(m)}_{f,d_n},\;\; \text{rate} = \frac{\theta^{(m)}_{f,d_n}}{\mu^{(m)}_{nf}}\Big)$$

#### Optional per-modality correction terms

| Term | Symbol | Prior | What it captures | RNA default | ATAC default |
|------|--------|-------|-----------------|-------------|--------------|
| Additive background | $s^{(m)}\_{e,f} = \exp(\beta^{(m)}\_{e,f})$ | $\text{Gamma}(1, 100)$, mean 0.01 | Per-feature ambient contamination; `ambient_covariate_keys` (many keys) | **ON** | off |
| Region factor | $y^{(m)}\_{t,f} = \text{softplus}(\gamma^{(m)}\_{t,f})/0.7$ | $\text{Gamma}(200, 200)$, mean 1.0 | Per-feature multiplicative bias; `modality_scaling_covariate_keys` (many keys) | off | **ON** |
| Learned library size | $\ell^{(m)}\_n$ | $\text{LogNormal}$, 0.05 var scaling | Low-capacity encoder; `library_size_key` (one key) | **always ON** | **always ON** |
| Dispersion regularisation | $1/\sqrt{\theta^{(m)}\_{f,d}}$ | $\text{Exp}(\lambda)$, $\lambda \sim \text{Gamma}(9,3)$ | Containment prior; `dispersion_key` (one key) | ON | ON |
| Batch-free decoder | — | — | Decoder conditioned only on $c\_{k,n}$; `categorical_covariate_keys` (many keys) | ON | ON |

Setting $s^{(m)}\_{e,f} = 0$ (no ambient) and $y^{(m)}\_{t,f} = 1$ (no region factor) recovers the standard regularizedvi single-modality model for that modality. The defaults reflect domain knowledge for snRNA+ATAC multiome: ambient RNA contamination is substantial in single-nucleus RNA-seq and well-captured by an additive term, while ATAC peaks have systematic per-peak biases from GC content, mappability and peak caller thresholds. See the [bone marrow multiome tutorial](docs/notebooks/bone_marrow_multimodal_tutorial.ipynb) for a worked RNA+ATAC example.

#### Inference: per-modality encoders and posterior concatenation

**Per-modality encoder** — each modality's encoder takes its own observed counts as input and independently constructs a Gaussian posterior over its private latent slice. The RNA encoder sees only RNA counts; the ATAC encoder sees only ATAC counts. This forces the model to build a dedicated representation for each modality before combining them:

$$q_\eta(z^{(m)}_n \mid x^{(m)}_n, e_n, c_{k,n}, p_n) = \text{Normal}\!\big(\mu^{(m)}_\eta(x^{(m)}_n),\; (\sigma^{(m)}_\eta)^2(x^{(m)}_n)\big)$$

**Posterior concatenation** — samples from the per-modality posteriors are concatenated to form the joint representation fed to all decoders. Because every decoder $f^{(m)}\_w$ receives the full $z\_n$, cross-modal coupling can emerge through the decoders during training. The training objective (ELBO) penalises each encoder's KL divergence independently:

$$z_n = [z^{(1)}_n;\; \ldots;\; z^{(M)}_n], \quad z^{(m)}_n \sim q_\eta(z^{(m)}_n \mid x^{(m)}_n)$$
$$\text{KL} = \sum_m \text{KL}\big[q_\eta(z^{(m)}_n \mid x^{(m)}_n)\;\|\;\mathcal{N}(0, I_{d_m})\big]$$

**Alternative latent strategies** (selectable via `latent_mode`):
- `"concatenation"` (default) — per-modality encoders, posteriors concatenated; total latent dim $= \sum_m d_m$
- `"weighted_mean"` — per-modality encoders, posteriors mixed into a single shared latent by learned scalar weights (MultiVI-style); requires equal $d\_m$ across modalities
- `"single_encoder"` — one joint encoder on all concatenated inputs, producing a single shared latent; simplest but loses per-modality interpretability

#### Latent-to-modality mapping via decoder attribution

With a concatenated latent space it is useful to know which latent dimensions each decoder actually uses. `get_modality_attribution()` computes the mean absolute Jacobian of each decoder's predicted mean $\mu^{(m)}\_{nf}$ with respect to each latent dimension $j$, using forward finite differences over the full cell population:

$$\text{attribution}^{(m)}_j = \frac{1}{N \cdot F_m} \sum_{n,f} \left| \frac{\partial \mu^{(m)}_{nf}}{\partial z_j} \right|$$

This reveals the empirical partition of the latent space: even though concatenation assigns each slice to a modality by construction, decoders can learn to cross-use other modalities' slices. The weighted representation `weighted_z` $= z\_n \times \text{attribution}^{(m)}$ provides a modality-specific view of cell state for downstream analysis (e.g. a separate UMAP per modality), as demonstrated in the [tutorial notebook](docs/notebooks/bone_marrow_multimodal_tutorial.ipynb).

### Key modifications

1. **Ambient RNA correction with Gamma prior**: Per-gene, per-ambient-category additive background $s\_{e,g} = \exp(\beta\_{e,g})$ captures ambient RNA contamination, mirroring cell2location's $s\_g \cdot g\_{e,g}$ structure. A $\text{Gamma}(1, 100)$ prior pushes $s\_{e,g}$ toward 0.01, keeping background small relative to biological signal. Initialised at `log(0.01)` (prior mean) with per-category selection via concatenated one-hot encoding across all `ambient_covariate_keys`.

2. **Hierarchical dispersion regularisation**: Prior $1/\sqrt{\theta\_{g,d}} \sim \text{Exponential}(\lambda\_d)$ is a containment prior (Simpson et al. 2017) that penalises small $\theta$ (excessive overdispersion), regularising the NB toward the Poisson baseline during gradient-based training. The data likelihood provides the opposing force, pulling $\theta$ toward values that explain observed count variance. The rate $\lambda\_d$ is learned per dispersion group (selected by `dispersion_key`) with a $\text{Gamma}(9, 3)$ hyper-prior (mean 3). Dispersion $\theta = \exp(\phi)$ is initialised at $\lambda^2 = 9$ (equilibrium). As used in cell2location/cell2fate.

3. **Batch-free decoder with separated correction paths**: The decoder $f\_w(z\_n, c\_{k,n})$ receives only categorical covariates $c\_{k,n}$ (site, donor, protocol via `categorical_covariate_keys`), **not** the ambient or dispersion covariates. This separates batch correction into structurally different paths: (a) a **constrained additive** path ($s\_{e,g}$ with Gamma prior, selected by `ambient_covariate_keys`) for per-sample ambient RNA, (b) a **flexible multiplicative** path through categorical covariates in the decoder for systematic differences between donors, protocols, or sites (e.g. PCR bias, RT efficiency, 10x chemistry versions), and (c) per-group dispersion $\theta\_{g,d}$ (selected by `dispersion_key`) for variance differences. In standard scVI, the decoder handles all batch effects through a single flexible path, which can absorb biological variation. The separation is most beneficial when batches have high within-batch cell type diversity (e.g. whole-embryo samples), because the additive background can be cleanly identified as the baseline signal shared across all cells in a batch.

4. **Softplus activation**: Because $\rho\_{ng} + s\_{e\_n,g}$ must be non-negative but need not sum to 1 across genes, softmax is replaced with softplus. The library size $\ell\_n$ acts as a true normalisation factor.

5. **Learned library size with constrained prior**: The observed total counts include ambient RNA, so library size must be learned (not observed). Prior variance is scaled by 0.05 to prevent the library size from absorbing biological signal. Library encoder has low capacity (`n_hidden=16`).

6. **LayerNorm and dropout-on-input**: LayerNorm replaces BatchNorm (independent of batch composition). Dropout is applied before the linear layer (feature-level masking).

### Practical notes and caveats

- **Best suited for single-nucleus RNA-seq** (independent modality and multiome), which typically has substantial ambient RNA contamination. The ambient correction is less necessary for single-cell RNA-seq where ambient levels are lower.

- **Study design matters**: The structured assumptions (additive ambient + multiplicative categorical covariates) depend on the experimental design. With some study designs, every batch has both additive effects (ambient RNA) and multiplicative effects (PCR bias, RT differences, 10x 3' v1 vs v2 vs v3, 3' vs 5'). These assumptions may not hold for Smart-seq type data where every cell can have PCR bias and RT differences.

- **Using as standard scVI with ambient correction**: If you provide the batch covariate to both `batch_key` and `categorical_covariate_keys`, the model effectively operates as standard scVI with ambient RNA correction (batch effects handled through both additive and multiplicative paths).

- **Not a strict ambient correction model**: Unlike CellBender (Fleming et al. 2023), this model is not constrained by the ambient count distribution from empty droplets. However, because it does not require empty droplets data, it can be more easily applied to integration of published datasets where empty droplet profiles are unavailable.

- **Additivity in non-negative space**: The additive background operates in non-negative space ($s\_{e,g} = \exp(\beta\_{e,g})$), reflecting the ambient RNA correction mechanism. Without empty droplets data, the additive component can learn the minimal expression of each gene across cells — for many genes this reflects ambient levels, but for ubiquitously expressed genes it captures genuine baseline expression. The additive mechanism therefore works best when individual batches are composed of diverse cell types.

- **Regularised overdispersion alone likely helps**: The containment prior on overdispersion regularises the NB toward the Poisson baseline, preventing the model from absorbing residuals through excessive variance (small $\theta$). This forces the decoder to capture genuine biological signal through its mean structure rather than relying on high overdispersion to explain noise. This likely contributes to improved sensitivity, but needs more systematic testing.

## Covariate design

### Why decouple batch_key?

In standard scVI (and early regularizedvi), a single `batch_key` controls all batch-dependent model components: additive ambient background, dispersion, and library size prior. This works well for 10x Chromium experiments where one sample = one GEM well = one set of technical biases.

However, in complex experimental designs — particularly combinatorial indexing protocols like sci-RNA-seq3 — different technical effects arise at different experimental granularities:

| Technical effect | 10x Chromium source | sci-RNA-seq3 source | Model component |
|-----------------|--------------------|--------------------|-----------------|
| Ambient RNA contamination | GEM well | Embryo (lysis batch) | `ambient_covariate_keys` |
| Library size distribution | GEM well | PCR well (amplification batch) | `library_size_key` |
| Overdispersion profile | GEM well | Embryo or experiment | `dispersion_key` |
| Multiplicative biases (RT, PCR) | Protocol version | Experiment batch | `categorical_covariate_keys` |
| Per-feature scaling (region factors) | — | Experiment batch | `modality_scaling_covariate_keys` |

Using a single `batch_key` forces all components to share the same granularity, which either under-corrects (too few groups) or over-parameterises (too many groups for components that don't need that resolution).

### Purpose-driven covariate keys

| Parameter | Model component | Encoder | Decoder | Shape per modality |
|-----------|----------------|---------|---------|-------------------|
| `ambient_covariate_keys` | Additive background $s\_{e,g} = \exp(\beta\_{e,g})$: single parameter with concatenated one-hot across all ambient keys | Yes (concat one-hot) | No | `(n_feat, sum(n_cats))` |
| `categorical_covariate_keys` | Standard scVI-style injection $c\_{k,n}$ into encoder/decoder | Yes (one-hot) | Yes (one-hot) | Standard |
| `modality_scaling_covariate_keys` | Region factors $y\_{t,g}$: multiplicative per-feature scaling | No | Multiplicative on rate | `(sum(n_cats), n_feat)` |
| `dispersion_key` | Inverse dispersion $\theta\_{g,d}$: per-gene, per-group | No | Indexing | `(n_feat, n_disp_cats)` |
| `library_size_key` | Library prior $\ell\_n \sim \text{LogNormal}(\ell\_p^{\mu}, 0.05 \cdot \ell\_p^{\sigma^2})$: per-group mean/var | Yes (one-hot) | No | `(1, n_lib_cats)` |

### Default behaviour and backward compatibility

- **`batch_key` alone** (backward compatible): Automatically fans out to `ambient_covariate_keys=[batch_key]`, `dispersion_key=batch_key`, `library_size_key=batch_key`. Equivalent to the original single-batch design — in the notation above, $e = d = p$ (same batch groups) and $t = k$ (same categorical groups).

- **`batch_key` + purpose-specific keys**: Raises `ValueError`. These are mutually exclusive — use one approach or the other.

- **`modality_scaling_covariate_keys`**: If not specified but `categorical_covariate_keys` is provided, defaults to `categorical_covariate_keys` (multi-modal only).

### Encoder/decoder composition

The encoder and decoder receive different subsets of covariates:

- **Encoder** receives: `[ambient_covs...] + [cat_covs (if encode_covariates)] + [library_size]` — ambient and library covariates are always injected, even when `encode_covariates=False`, because the encoder needs to know the expected background level and library size group.

- **Decoder** receives: `[cat_covs...]` only (batch-free by default). When `use_batch_in_decoder=True`, batch is additionally included.

### Example: 10x Chromium (simple)

```python
# batch_key fans out to all components — equivalent to original API
regularizedvi.AmbientRegularizedSCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="sample",
    categorical_covariate_keys=["donor", "site"],
)
```

### Example: sci-RNA-seq3 (purpose-driven)

```python
regularizedvi.AmbientRegularizedSCVI.setup_anndata(
    adata,
    layer="counts",
    # Ambient RNA comes from lysis: each embryo has its own ambient profile
    ambient_covariate_keys=["embryo_id", "pcr_well"],
    # Multiplicative effects from experiment-level protocol differences
    categorical_covariate_keys=["experiment"],
    # Dispersion varies by embryo (tissue composition → variance structure)
    dispersion_key="embryo_id",
    # Library size determined by PCR well (amplification batch)
    library_size_key="pcr_well",
)
```

## Installation

### GPU environment (recommended)

```bash
export PYTHONNOUSERSITE="1"
conda create -y -n regularizedvi python=3.11
conda activate regularizedvi

# Install PyTorch with CUDA 12.4 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install JAX (optional, for some scvi-tools features)
pip install jax

# Install scvi-tools and regularizedvi
pip install scvi-tools
pip install git+https://github.com/vitkl/regularizedvi.git@main

# Install additional analysis packages
pip install scanpy igraph matplotlib ipykernel jupyter

# Register Jupyter kernel
python -m ipykernel install --user --name=regularizedvi --display-name='Environment (regularizedvi)'
```

### Development installation

```bash
git clone https://github.com/vitkl/regularizedvi.git
cd regularizedvi
pip install -e ".[dev,test]"
```

## Quick start

### Simple setup (batch_key)

When a single batch variable can describe several technical effects — ambient RNA contamination, library size distribution, and overdispersion profile (typical for 10x Chromium where one sample = one GEM well):

```python
import regularizedvi

regularizedvi.AmbientRegularizedSCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    categorical_covariate_keys=["site", "donor"],
)
model = regularizedvi.AmbientRegularizedSCVI(
    adata,
    n_hidden=512,
    n_layers=1,
    n_latent=128,
)
model.train(
    train_size=1.0,
    max_epochs=2000,
    batch_size=1024,
)
latent = model.get_latent_representation()
```

### Purpose-driven covariate keys

When different technical effects arise at different experimental granularities (e.g. sci-RNA-seq3, combinatorial indexing), you can assign each model component its own covariate:

```python
regularizedvi.AmbientRegularizedSCVI.setup_anndata(
    adata,
    layer="counts",
    ambient_covariate_keys=["embryo_id", "pcr_well"],   # additive background
    categorical_covariate_keys=["experiment"],           # encoder/decoder injection
    dispersion_key="embryo_id",                          # per-group overdispersion
    library_size_key="pcr_well",                         # library size prior groups
)
```

See [Covariate design](#covariate-design) below for full details.

### Default configuration

The model now uses **GammaPoisson likelihood** (cell2location-style, mathematically equivalent to NB) by default with a containment prior on overdispersion to regularise the model. The default dispersion is `"gene-batch"`, providing per-gene, per-batch inverse dispersion parameters.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

## References

- Lopez, R., Regier, J., Cole, M.B. et al. Deep generative modeling for single-cell transcriptomics. *Nat Methods* 15, 1053–1058 (2018). [doi:10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2)
- Kleshchevnikov, V., Shmatko, A., Dann, E. et al. Cell2location maps fine-grained cell types in spatial transcriptomics. *Nat Biotechnol* 40, 661–671 (2022). [doi:10.1038/s41587-021-01139-4](https://doi.org/10.1038/s41587-021-01139-4)
- Aivazidis, A., Memi, F., Kleshchevnikov, V. et al. Cell2fate infers RNA velocity modules to improve cell fate prediction. *Nat Methods* 22, 698–707 (2025). [doi:10.1038/s41592-025-02608-3](https://doi.org/10.1038/s41592-025-02608-3)
- Simpson, D., Rue, H., Riebler, A. et al. Penalising Model Component Complexity: A Principled, Practical Approach to Constructing Priors. *Statist. Sci.* 32(1), 1-28 (2017). [doi:10.1214/16-STS576](https://doi.org/10.1214/16-STS576)

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/vitkl/regularizedvi/issues
[tests]: https://github.com/vitkl/regularizedvi/actions/workflows/test.yaml
[documentation]: https://regularizedvi.readthedocs.io
[changelog]: https://regularizedvi.readthedocs.io/en/latest/changelog.html
[api documentation]: https://regularizedvi.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/regularizedvi
