# regularizedvi

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/vitkl/regularizedvi/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/regularizedvi

Regularized scVI with ambient RNA correction and overdispersion regularisation, based on [cell2location](https://www.nature.com/articles/s41587-021-01139-4)/[cell2fate](https://www.nature.com/articles/s41592-025-02608-3) modelling principles (Kleshchevnikov et al. 2022, Aivazidis et al. 2025, Simpson et al. 2017).

The modifications (ambient RNA correction, dispersion prior, batch-free decoder, learned library size) act as structural inductive biases that make a high-capacity model (`n_hidden=512+`, `n_latent=128+`) well-behaved by default, removing the need for careful per-dataset hyperparameter tuning. This is particularly important for complex datasets with hundreds of cell types (e.g. whole-embryo atlases, cross-atlas integration) where large latent spaces and wide hidden layers are needed to avoid competition between cell types for representational capacity.

**[Quick start](#quick-start)** | [Installation](#installation) | [Model description](#motivation) | [Covariate design](#covariate-design)

> Jump to: [Quick start](#quick-start) to get running, [Installation](#installation) for setup, or read on for the model description and math.

---

## Motivation

### Standard scVI generative model

Standard scVI (Lopez et al. 2018) models observed UMI counts $`x_{ng}`$ for cell $`n`$ and gene $`g`$ as:

$$
z_n \sim \text{Normal}(0, I)
$$

$$
\ell_n \sim \text{LogNormal}(\ell_\mu^\top s_n,\; \ell_{\sigma^2}^\top s_n)
$$

$$
\rho_{ng} = f_w(z_n, s_n, c_n) \in \Delta^{G-1}
$$

$$
x_{ng} \sim \text{NB}(\mu = \ell_n \rho_{ng},\; \theta_{g,s_n})
$$

where:
- $`z_n \in \mathbb{R}^d`$ — low-dimensional latent cell state
- $`\ell_n \in (0, \infty)`$ — library size (by default fixed to total UMI count per cell), with log-normal prior parameterised per batch $`s_n`$
- $`\rho_n \in \Delta^{G-1}`$ — decoder output on the probability simplex (via softmax) as a fraction of total $`\ell_n`$ RNA per cell, representing denoised normalised gene expression
- $`f_w(z_n, s_n): \mathbb{R}^d \times \{0,1\}^K \to \Delta^{G-1}`$ — decoder neural network, conditioned on batch $`s_n`$
- $`\theta_{g,s_n} \in (0, \infty)`$ — per-gene, per-batch inverse dispersion (code: `px_r`, stored as unconstrained $`\phi_{g,s_n}`$ where $`\theta_{g,s_n} = \exp(\phi_{g,s_n})`$ )
- $`s_n \in \{0,1\}^K`$ — one-hot batch indicator for cell $`n`$
- $`c_n \in \{0,1\}^K`$ — one-hot categorical covariate indicator for cell $`n`$

The inference model uses amortised variational inference to fit all cell specific variables (encoder NNs): $`q_\eta(z_n, \ell_n \mid x_n, s_n, c_n) = q_\eta(z_n \mid x_n, s_n, c_n) \, q_\eta(\ell_n \mid x_n, s_n, c_n)`$. Note that both batch $`s_n`$ and $`c_n`$ categorical covariates are used in both decoders (model) and encoders (amortised variational inference of  $`z_n, \ell_n`$).

### regularizedvi generative model

**regularizedvi** adapts [cell2location](https://doi.org/10.1038/s41587-021-01139-4)/[cell2fate](https://doi.org/10.1038/s41592-025-02608-3) modelling principles to scVI. All learnable parameters are initialised at their prior means to improve training stability.

**Latent variable and library size** — standard scVI structure with a constrained library prior. Library prior parameters $`\ell_p^{\mu}`$, $`\ell_p^{\sigma^2}`$ are computed per `library_size_key` group $`p`$ (one key, [`_module.py:307–333`](src/regularizedvi/_module.py#L307-L333)):

$$
z_n \sim \text{Normal}(0, I)
$$

$$
\ell_n \sim \text{LogNormal}(\ell_p^{\mu},\; 0.5 \cdot \ell_p^{\sigma^2})
$$

**Decoder output** — batch-free decoder maps $`z_n`$ and categorical covariates $`c_{k,n}`$ (not ambient/library covariates) to non-negative gene expression via softplus. Categorical covariates are selected by `nn_conditioning_covariate_keys` (many keys, [`_module.py:791–835`](src/regularizedvi/_module.py#L791-L835), [`_components.py:461–462`](src/regularizedvi/_components.py#L461-L462)):

$$
\rho_{ng} = \text{softplus}\big(f_w(z_n, c_{k,n})\big) \in \mathbb{R}_{\geq 0}^G
$$

**Additive background** — per-gene ambient RNA with Gamma prior pushing $`s_{e,g}`$ toward 0.01. Background parameters are indexed by `ambient_covariate_keys` (many keys, concatenated one-hot, [`_module.py:390–418`](src/regularizedvi/_module.py#L390-L418) init, [`_module.py:794–804`](src/regularizedvi/_module.py#L794-L804) one-hot selection, [`_module.py:981–991`](src/regularizedvi/_module.py#L981-L991) prior penalty). The background parameter is always initialized at the prior mean (or at `bg_init_gene_fraction` of per-gene, per-batch mean expression when data-dependent init is active), but the Gamma prior penalty in the loss is **off by default** (`regularise_background=False`); enable it explicitly if needed:

$$
s_{e,g} = \exp(\beta_{e,g}), \qquad s_{e,g} \sim \text{Gamma}(1,\, 100)
$$

**Feature scaling** — per-gene, per-covariate multiplicative scaling capturing systematic biases (e.g. PCR amplification, RT efficiency differences between protocols). Parameterised as $`\text{softplus}(\gamma)/0.7`$ with a tight Gamma prior centered at 1.0. Scaling covariates $`t`$ are selected by `feature_scaling_covariate_keys` (many keys); each covariate category gets its own scaling factor. When no scaling covariates are provided, a single $`(1, G)`$ fallback parameter is created ([`_module.py:420–432`](src/regularizedvi/_module.py#L420-L432) init, [`_module.py:853–867`](src/regularizedvi/_module.py#L853-L867) application, [`_module.py:993–1002`](src/regularizedvi/_module.py#L993-L1002) prior penalty):

$$
y_{t,g} = \text{softplus}(\gamma_{t,g})\,/\,0.7, \qquad y_{t,g} \sim \text{Gamma}(200,\, 200)
$$

**Hierarchical dispersion prior with variational posterior** — two-level prior on inverse dispersion $`\theta_{g,d}`$ with a variational LogNormal posterior $`q(\log \theta_{g,d}) = \text{Normal}(\mu_{g,d}, \sigma_{g,d})`$ parameterised by learnable `px_r_mu` and `px_r_log_sigma`. During training, $`\theta`$ is sampled via reparameterisation: $`\theta = \exp(\mu + \sigma \varepsilon)`$, $`\varepsilon \sim \text{Normal}(0,1)`$; at inference the posterior mean $`\theta = \exp(\mu)`$ is used ([`_module.py:846–851`](src/regularizedvi/_module.py#L846-L851) sampling). Dispersion groups $`d`$ are selected by `dispersion_key` (one key). A learned rate $`\lambda_d`$ adapts regularisation strength per group. The KL divergence is computed as $`-\text{entropy}(q) - \mathbb{E}_q[\log p(1/\sqrt{\theta})]`$ ([`_module.py:933–979`](src/regularizedvi/_module.py#L933-L979) full block, [`_module.py:950`](src/regularizedvi/_module.py#L950) Level 1 softplus, [`_module.py:973–975`](src/regularizedvi/_module.py#L973-L975) Level 2 transform):

$$
\lambda_d \sim \text{Gamma}(9,\, 3), \qquad 1/\sqrt{\theta_{g,d}} \sim \text{Exponential}(\lambda_d)
$$

**Data-driven dispersion initialisation** (`dispersion_init="data"`) — the variational posterior mean `px_r_mu` can be initialised from data using method-of-moments variance decomposition via the law of total variance ([`_dispersion_init.py`](src/regularizedvi/_dispersion_init.py)). The total variance of raw counts $`x_g`$ for gene $`g`$ is decomposed into four components, and the equation is rearranged to solve for the technical $`\theta_g^{\text{tech}}`$:

$$
\text{Var}(x_g) = \underbrace{\mu_g}_{\text{Poisson}} + \underbrace{\mu_g^2 \cdot \text{CV}^2(L)}_{\text{library size}} + \underbrace{\frac{\mu_g^2}{\theta_g^{\text{bio}}} \cdot (1 + \text{CV}^2(L)) \cdot 0.9}_{\text{biological overdispersion}} + \underbrace{\frac{\mu_g^2}{\theta_g^{\text{tech}}} \cdot (1 + \text{CV}^2(L)) \cdot 0.1}_{\text{technical overdispersion}}
$$

where $`\text{CV}^2(L) = \text{Var}(L)/\mathbb{E}[L]^2`$ is the squared coefficient of variation of library sizes, and the 0.9/0.1 split (controlled by `biological_variance_fraction`) assumes 90% of gene-specific excess variance is biological (captured by the latent space) and 10% is technical (captured by $`\theta`$). Per-gene mean and variance are computed via chunked batch Welford's algorithm for memory efficiency on large datasets.

**Expected mean counts** — decoder output plus optional background, scaled by library size and feature scaling ([`_components.py:467`](src/regularizedvi/_components.py#L467) base rate, [`_module.py:853–867`](src/regularizedvi/_module.py#L853-L867) feature scaling):

$$
\mu_{ng} = \ell_n \cdot \big(\rho_{ng} + s_{e_n,g}\big) \cdot y_{t_n,g}
$$

**Observation model** — GammaPoisson (= negative binomial) with mean $`\mu_{ng}`$ and inverse dispersion $`\theta_{g,d}`$:

$$
x_{ng} \sim \text{GammaPoisson}\Big(\text{concentration} = \theta_{g,d_n},\;\; \text{rate} = \frac{\theta_{g,d_n}}{\mu_{ng}}\Big)
$$

**Notation:**
- $`s_{e,g} = \exp(\beta_{e,g})`$ — per-gene ambient background indexed by `ambient_covariate_keys` (many keys); $`\beta`$ is the unconstrained parameter (code: `additive_background`). When `batch_key` is used alone, $`e`$ = batch group.
- $`c_{k,n}`$ — categorical covariates (site, donor, etc.), selected by `nn_conditioning_covariate_keys` (many keys). Injected into the decoder; optionally into the encoder via `encoder_covariate_keys`.
- $`y_{t,g} = \text{softplus}(\gamma_{t,g})/0.7`$ — per-gene feature scaling indexed by `feature_scaling_covariate_keys` (many keys); tight $`\text{Gamma}(200, 200)`$ prior centered at 1.0
- $`\rho_{ng} \in \mathbb{R}_{\geq 0}^G`$ — decoder output via softplus (not on the simplex), since $`\rho_{ng} + s_{e,g}`$ need not sum to 1
- $`\theta_{g,d}`$ — inverse dispersion indexed by `dispersion_key` (one key) with variational LogNormal posterior: `px_r_mu` ($`\mu_{g,d}`$) and `px_r_log_sigma` ($`\log \sigma_{g,d}`$) are learnable parameters; $`\theta = \exp(\mu)`$ at inference. Initialised at $`\mu = \log(\lambda^2) \approx 2.2`$ so $`\theta \approx 9`$ (equilibrium: $`1/\sqrt{\theta} = 1/\lambda`$) ([`_module.py:335–368`](src/regularizedvi/_module.py#L335-L368))
- $`\lambda_d`$ — learned Exponential rate, one per dispersion group; $`\text{Gamma}(9, 3)`$ hyper-prior has mean 3 ([`_module.py:370–382`](src/regularizedvi/_module.py#L370-L382))
- $`\ell_p^{\mu}`$, $`\ell_p^{\sigma^2}`$ — library prior mean and variance per `library_size_key` group $`p`$ (one key). $`0.5`$ scaling factor prevents library size from absorbing biological signal.
- **Backward compat**: When `batch_key` is used alone, $`e = d = p`$ (all index the same batch groups) and $`t = k`$ (categorical and scaling covariates share groups).

The NB variance is $`\text{Var}(x) = \mu + \mu^2/\theta`$. Large $`\theta`$ → less overdispersion (Poisson limit). The Exponential prior on $`1/\sqrt{\theta}`$ is a containment prior (Simpson et al. 2017) that penalises large $`1/\sqrt{\theta}`$ (= small $`\theta`$, excessive overdispersion), regularising the NB toward the Poisson baseline. The data likelihood provides the opposing force, pulling $`\theta`$ toward values needed to explain observed count variance. At equilibrium $`\theta \approx \lambda^2 = 9`$, giving moderate overdispersion. This forces the decoder to capture biological signal through its mean structure rather than absorbing residuals via high variance.

### RegularizedMultimodalVI generative model

**RegularizedMultimodalVI** extends regularizedvi to $`M`$ paired modalities (e.g., RNA + ATAC from 10x Multiome). Each modality has its own dedicated encoder and decoder, but all decoders share a single joint latent space formed by concatenating per-modality codes ("symmetric concatenation"). The generative model for every modality follows the same structure — only which optional correction terms are active differs between modalities.

#### Latent space

Each modality $`m`$ contributes a private slice of the joint latent space. These slices are independently drawn from a standard normal prior and concatenated to form the full cell representation $`z_n`$ that is fed to all decoders:

$$
z^{(m)}_n \sim \text{Normal}(0, I_{d_m})
$$

$$
z_n = [\,z^{(1)}_n;\; z^{(2)}_n;\; \ldots;\; z^{(M)}_n\,] \in \mathbb{R}^{\sum_m d_m}
$$

where $`d_m`$ is the latent dimensionality assigned to modality $`m`$ (e.g. `n_latent={"rna": 96, "atac": 32}`). Because every decoder receives the full $`z_n`$, signals across modalities can interact through the decoders even though each encoder sees only its own modality.

#### Generative model (per modality $`m`$)

The following equations describe how observed counts $`x^{(m)}_{nf}`$ — UMIs for RNA, fragment counts for ATAC — are generated for cell $`n`$ and feature $`f`$ (gene or chromatin peak). All modalities share this structure; the optional terms in the mean $`\mu^{(m)}_{nf}`$ are selectively activated per modality.

**Library size** — always learned (observed totals include ambient contamination). A low-capacity encoder infers library size per cell, regularised by a tight LogNormal prior estimated per `library_size_key` group $`p`$ (one key, [`_multimodule.py:443–479`](src/regularizedvi/_multimodule.py#L443-L479) prior buffers, [`_multimodule.py:1101–1110`](src/regularizedvi/_multimodule.py#L1101-L1110) loss):

$$
\ell^{(m)}_n \sim \text{LogNormal}\big(\ell^{(m),\mu}_p,\; 0.5 \cdot \ell^{(m),\sigma^2}_p\big)
$$

**Decoder output** — maps joint latent code $`z_n`$ and categorical covariates $`c_{k,n}`$ (selected by `nn_conditioning_covariate_keys`, many keys) to non-negative feature signal via softplus ([`_multimodule.py:911–928`](src/regularizedvi/_multimodule.py#L911-L928)):

$$
\rho^{(m)}_{nf} = \text{softplus}\big(f^{(m)}_w(z_n,\, c_{k,n})\big) \in \mathbb{R}_{\geq 0}
$$

**Additive background** — per-feature ambient contamination with Gamma prior, indexed by `ambient_covariate_keys` (many keys, concatenated one-hot, [`_multimodule.py:535–560`](src/regularizedvi/_multimodule.py#L535-L560) init, [`_multimodule.py:899–909`](src/regularizedvi/_multimodule.py#L899-L909) one-hot selection, [`_multimodule.py:1194–1210`](src/regularizedvi/_multimodule.py#L1194-L1210) prior penalty):

$$
s^{(m)}_{e,f} = \exp(\beta^{(m)}_{e,f}), \qquad s^{(m)}_{e,f} \sim \text{Gamma}(1,\, 100)
$$

**Feature scaling** — per-feature, per-covariate multiplicative scaling capturing systematic biases (GC content, mappability, peak caller sensitivity). Parameterised as $`\text{softplus}(\gamma)/0.7`$ with a tight Gamma prior centered at 1. Scaling covariates $`t`$ are selected by `feature_scaling_covariate_keys` (many keys); each covariate category gets its own factor ([`_multimodule.py:562–571`](src/regularizedvi/_multimodule.py#L562-L571) init, [`_multimodule.py:930–940`](src/regularizedvi/_multimodule.py#L930-L940) activation and selection, [`_multimodule.py:1175–1192`](src/regularizedvi/_multimodule.py#L1175-L1192) prior penalty):

$$
y^{(m)}_{t,f} = \text{softplus}(\gamma^{(m)}_{t,f})\,/\,0.7, \qquad y^{(m)}_{t,f} \sim \text{Gamma}(200,\, 200)
$$

**Expected mean counts** — decoder output plus optional background, scaled by library size and feature scaling ([`_components.py:467`](src/regularizedvi/_components.py#L467) base rate, [`_multimodule.py:940`](src/regularizedvi/_multimodule.py#L940) feature scaling):

$$
\mu^{(m)}_{nf} = \ell^{(m)}_n \cdot \big(\rho^{(m)}_{nf} + s^{(m)}_{e_n,f}\big) \cdot y^{(m)}_{t_n,f}
$$

**Hierarchical dispersion prior** — same two-level structure as single-modality with variational LogNormal posterior, per modality and `dispersion_key` group $`d`$ (one key, [`_multimodule.py:1131–1173`](src/regularizedvi/_multimodule.py#L1131-L1173)):

$$
\lambda^{(m)}_d \sim \text{Gamma}(9,\, 3), \qquad 1/\sqrt{\theta^{(m)}_{f,d}} \sim \text{Exponential}(\lambda^{(m)}_d)
$$

**Observation model** — GammaPoisson (= negative binomial) with mean $`\mu^{(m)}_{nf}`$ and inverse dispersion $`\theta^{(m)}_{f,d}`$:

$$
x^{(m)}_{nf} \sim \text{GammaPoisson}\!\Big(\text{concentration} = \theta^{(m)}_{f,d_n},\;\; \text{rate} = \frac{\theta^{(m)}_{f,d_n}}{\mu^{(m)}_{nf}}\Big)
$$

#### Optional per-modality correction terms

| Term | Symbol | Prior | What it captures | RNA default | ATAC default |
|------|--------|-------|-----------------|-------------|--------------|
| Additive background | $`s^{(m)}_{e,f} = \exp(\beta^{(m)}_{e,f})`$ | $`\text{Gamma}(1, 100)`$, mean 0.01 | Per-feature ambient contamination; `ambient_covariate_keys` (many keys) | **ON** | off |
| Feature scaling | $`y^{(m)}_{t,f} = \text{softplus}(\gamma^{(m)}_{t,f})/0.7`$ | $`\text{Gamma}(200, 200)`$, mean 1.0 | Per-feature multiplicative bias; `feature_scaling_covariate_keys` (many keys) | off | **ON** |
| Learned library size | $`\ell^{(m)}_n`$ | $`\text{LogNormal}`$, 0.5 var scaling | Residual low-capacity encoder with LogNormal shrinkage weight; `library_size_key` (one key) | **always ON** | **always ON** |
| Dispersion regularisation | $`1/\sqrt{\theta^{(m)}_{f,d}}`$ | $`\text{Exp}(\lambda)`$, $`\lambda \sim \text{Gamma}(9,3)`$ | Containment prior; `dispersion_key` (one key) | ON | ON |
| Batch-free decoder | — | — | Decoder conditioned only on $`c_{k,n}`$; `nn_conditioning_covariate_keys` (many keys) | ON | ON |

Setting $`s^{(m)}_{e,f} = 0`$ (no ambient) and $`y^{(m)}_{t,f} = 1`$ (no feature scaling) recovers the standard regularizedvi single-modality model for that modality. The defaults reflect domain knowledge for snRNA+ATAC multiome: ambient RNA contamination is substantial in single-nucleus RNA-seq and well-captured by an additive term, while ATAC peaks have systematic per-peak biases from GC content, mappability and peak caller thresholds. See the [bone marrow multiome tutorial](docs/notebooks/bone_marrow_multimodal_tutorial.ipynb) for a worked RNA+ATAC example.

#### Inference: per-modality encoders and posterior concatenation

**Per-modality encoder** — each modality's encoder takes its own observed counts as input and independently constructs a Gaussian posterior over its private latent slice. The RNA encoder sees only RNA counts; the ATAC encoder sees only ATAC counts. This forces the model to build a dedicated representation for each modality before combining them:

$$
q_\eta(z^{(m)}_n \mid x^{(m)}_n, e_n, c_{k,n}, p_n) = \text{Normal}\!\big(\mu^{(m)}_\eta(x^{(m)}_n),\; (\sigma^{(m)}_\eta)^2(x^{(m)}_n)\big)
$$

**Posterior concatenation** — samples from the per-modality posteriors are concatenated to form the joint representation fed to all decoders. Because every decoder $`f^{(m)}_w`$ receives the full $`z_n`$, cross-modal coupling can emerge through the decoders during training. The training objective (ELBO) penalises each encoder's KL divergence independently:

$$
z_n = [z^{(1)}_n;\; \ldots;\; z^{(M)}_n], \quad z^{(m)}_n \sim q_\eta(z^{(m)}_n \mid x^{(m)}_n)
$$

$$
\text{KL} = \sum_m \text{KL}\big[q_\eta(z^{(m)}_n \mid x^{(m)}_n)\;\|\;\mathcal{N}(0, I_{d_m})\big]
$$

**Alternative latent strategies** (selectable via `latent_mode`):
- `"concatenation"` (default) — per-modality encoders, posteriors concatenated; total latent dim $`= \sum_m d_m`$
- `"weighted_mean"` — per-modality encoders, posteriors mixed into a single shared latent by learned scalar weights (MultiVI-style); requires equal $`d_m`$ across modalities
- `"single_encoder"` — one joint encoder on all concatenated inputs, producing a single shared latent; simplest but loses per-modality interpretability

#### Latent-to-modality mapping via decoder attribution

With a concatenated latent space it is useful to know which latent dimensions each decoder actually uses. `get_modality_attribution()` computes the mean absolute Jacobian of each decoder's predicted mean $`\mu^{(m)}_{nf}`$ with respect to each latent dimension $`j`$, using forward finite differences over the full cell population:

$$
\text{attribution}^{(m)}_j = \frac{1}{N \cdot F_m} \sum_{n,f} \left| \frac{\partial \mu^{(m)}_{nf}}{\partial z_j} \right|
$$

This reveals the empirical partition of the latent space: even though concatenation assigns each slice to a modality by construction, decoders can learn to cross-use other modalities' slices. The weighted representation `weighted_z` $`= z_n \times \text{attribution}^{(m)}`$ provides a modality-specific view of cell state for downstream analysis (e.g. a separate UMAP per modality), as demonstrated in the [tutorial notebook](docs/notebooks/bone_marrow_multimodal_tutorial.ipynb).

### Key modifications

1. **Ambient RNA correction with Gamma prior**: Per-gene, per-ambient-category additive background $`s_{e,g} = \exp(\beta_{e,g})`$ captures ambient RNA contamination, mirroring cell2location's $`s_g \cdot g_{e,g}`$ structure. A $`\text{Gamma}(1, 100)`$ prior pushes $`s_{e,g}`$ toward 0.01, keeping background small relative to biological signal. Initialised at `log(0.01)` (prior mean) with per-category selection via concatenated one-hot encoding across all `ambient_covariate_keys`.

2. **Hierarchical dispersion regularisation**: Prior $`1/\sqrt{\theta_{g,d}} \sim \text{Exponential}(\lambda_d)`$ is a containment prior (Simpson et al. 2017) that penalises small $`\theta`$ (excessive overdispersion), regularising the NB toward the Poisson baseline during gradient-based training. The data likelihood provides the opposing force, pulling $`\theta`$ toward values that explain observed count variance. The rate $`\lambda_d`$ is learned per dispersion group (selected by `dispersion_key`) with a $`\text{Gamma}(9, 3)`$ hyper-prior (mean 3). Dispersion $`\theta = \exp(\phi)`$ is initialised at $`\lambda^2 = 9`$ (equilibrium). As used in cell2location/cell2fate.

3. **Batch-free decoder with separated correction paths**: The decoder $`f_w(z_n, c_{k,n})`$ receives only categorical covariates $`c_{k,n}`$ (site, donor, protocol via `nn_conditioning_covariate_keys`), **not** the ambient or dispersion covariates. This separates batch correction into structurally different paths: (a) a **constrained additive** path ($`s_{e,g}`$ with Gamma prior, selected by `ambient_covariate_keys`) for per-sample ambient RNA, (b) a **flexible multiplicative** path through categorical covariates in the decoder for systematic differences between donors, protocols, or sites (e.g. PCR bias, RT efficiency, 10x chemistry versions), and (c) per-group dispersion $`\theta_{g,d}`$ (selected by `dispersion_key`) for variance differences. In standard scVI, the decoder handles all batch effects through a single flexible path, which can absorb biological variation. The separation is most beneficial when batches have high within-batch cell type diversity (e.g. whole-embryo samples), because the additive background can be cleanly identified as the baseline signal shared across all cells in a batch.

4. **Softplus activation**: Because $`\rho_{ng} + s_{e_n,g}`$ must be non-negative but need not sum to 1 across genes, softmax is replaced with softplus. The library size $`\ell_n`$ acts as a true normalisation factor.

5. **Learned library size with residual encoder and constrained prior**: The observed total counts include ambient RNA, so library size must be learned (not observed). Prior variance is scaled by `library_log_vars_weight=0.5` to prevent the library size from absorbing biological signal. Library encoder has low capacity (`n_hidden=16`). Library prior means are always centered (global mean subtracted). When `residual_library_encoder=True` (default), the library is computed as `library = log(sens) + w * (centered_obs - log(sens)) + encoder_output`, where `centered_obs = log(total_counts) - global_log_mean + log(sensitivity)` and `w ~ LogNormal` with an `Exponential(1)` shrinkage prior. The library encoder bias is initialized to `log(sensitivity)` so `exp(library)` starts at the correct scale.

6. **LayerNorm and dropout-on-input**: LayerNorm replaces BatchNorm (independent of batch composition). Dropout is applied before the linear layer (feature-level masking).

7. **Auto-scaled early stopping**: The `early_stopping_min_delta_per_feature` parameter (default: 0.0002) auto-scales the early stopping threshold as $`\text{min\_delta} = n_\text{features} \times \text{early\_stopping\_min\_delta\_per\_feature}`$. This adapts the stopping criterion to dataset size: datasets with more features produce larger expected ELBO values and need a proportionally larger delta to distinguish meaningful improvement from noise.

### Default configuration

The following defaults were validated in production training runs on whole-embryo data (460k+ cells, 28k genes, 88 multiome samples). All parameters listed below are set as code defaults in the model constructors (`AmbientRegularizedSCVI` and `RegularizedMultimodalVI`).

| Parameter | Default | Effect |
|---|---|---|
| `residual_library_encoder` | `True` | Blend observed log-library with encoder via LogNormal shrinkage weight |
| `init_decoder_bias` | `"mean"` | Initialize decoder bias from mean normalized expression |
| `bg_init_gene_fraction` | `0.2` | Initialize additive background at 20% of per-gene, per-batch mean expression |
| `decoder_weight_l2` | `0.1` | L2 penalty on decoder weights |
| `library_log_vars_weight` | `0.5` | Scale library prior variance by 0.5 |
| `regularise_dispersion` | `True` | Containment prior on overdispersion |
| `regularise_background` | `False` | Gamma prior penalty on background (off by default) |

**Residual library encoder** (`residual_library_encoder=True`, default): Instead of inferring library size purely from the encoder, the residual formulation blends the observed log-library with the encoder output via a learnable shrinkage weight:

$$
\text{library} = \log(\text{sens}) + w \cdot (\text{centered\_obs} - \log(\text{sens})) + \text{encoder\_output}
$$

where $`\text{centered\_obs} = \log(\text{total\_counts}) - \text{global\_log\_mean} + \log(\text{sensitivity})`$ and $`w \sim \text{LogNormal}(\mu_w, \sigma_w)`$ with an $`\text{Exponential}(1)`$ shrinkage prior. At initialization $`w \approx 1`$, so library $`\approx`$ observed. The centering (subtracting `global_log_mean` from library prior means) is always on, regardless of the residual encoder setting.

**Library encoder bias init** (always on): The library encoder's mean bias is initialized to `log(sensitivity)` so that `exp(library)` starts at the correct absolute scale from the first training step.

**Data-dependent decoder and background init** (`init_decoder_bias="mean"`, `bg_init_gene_fraction=0.2`): The decoder output layer bias is initialized from the mean normalized expression across all cells, via `softplus_inv(mean_expr)`. The additive background parameter is initialized at `log(0.2 * mean_expr_per_batch)`, i.e. 20% of the per-gene, per-batch mean expression. Both initializations ensure the model starts near the data manifold, reducing the number of epochs needed for convergence.

The model uses **GammaPoisson likelihood** (cell2location-style, mathematically equivalent to NB) by default with the containment prior on overdispersion. The default dispersion is `"gene-batch"`, providing per-gene, per-batch inverse dispersion parameters.

### Practical notes and caveats

- **Best suited for single-nucleus RNA-seq** (independent modality and multiome), which typically has substantial ambient RNA contamination. The ambient correction is less necessary for single-cell RNA-seq where ambient levels are lower.

- **Study design matters**: The structured assumptions (additive ambient + multiplicative categorical covariates) depend on the experimental design. With some study designs, every batch has both additive effects (ambient RNA) and multiplicative effects (PCR bias, RT differences, 10x 3' v1 vs v2 vs v3, 3' vs 5'). These assumptions may not hold for Smart-seq type data where every cell can have PCR bias and RT differences.

- **Using as standard scVI with ambient correction**: If you provide the batch covariate to both `batch_key` and `nn_conditioning_covariate_keys`, the model effectively operates as standard scVI with ambient RNA correction (batch effects handled through both additive and multiplicative paths).

- **Not a strict ambient correction model**: Unlike CellBender (Fleming et al. 2023), this model is not constrained by the ambient count distribution from empty droplets. However, because it does not require empty droplets data, it can be more easily applied to integration of published datasets where empty droplet profiles are unavailable.

- **Additivity in non-negative space**: The additive background operates in non-negative space ($`s_{e,g} = \exp(\beta_{e,g})`$), reflecting the ambient RNA correction mechanism. Without empty droplets data, the additive component can learn the minimal expression of each gene across cells — for many genes this reflects ambient levels, but for ubiquitously expressed genes it captures genuine baseline expression. The additive mechanism therefore works best when individual batches are composed of diverse cell types.

- **Regularised overdispersion alone likely helps**: The containment prior on overdispersion regularises the NB toward the Poisson baseline, preventing the model from absorbing residuals through excessive variance (small $`\theta`$). This forces the decoder to capture genuine biological signal through its mean structure rather than relying on high overdispersion to explain noise. This likely contributes to improved sensitivity, but needs more systematic testing.

## Covariate design

### Why decouple batch_key?

In standard scVI (and early regularizedvi), a single `batch_key` controls all batch-dependent model components: additive ambient background, dispersion, and library size prior. This works well for 10x Chromium experiments where one sample = one GEM well = one set of technical biases.

However, in complex experimental designs — particularly combinatorial indexing protocols like sci-RNA-seq3 — different technical effects arise at different experimental granularities:

| Technical effect | 10x Chromium source | sci-RNA-seq3 source | Model component |
|-----------------|--------------------|--------------------|-----------------|
| Ambient RNA contamination | GEM well | Embryo (lysis batch) | `ambient_covariate_keys` |
| Library size distribution | GEM well | PCR well (amplification batch) | `library_size_key` |
| Overdispersion profile | GEM well | Embryo or experiment | `dispersion_key` |
| Multiplicative biases (RT, PCR) | Protocol version | Experiment batch | `nn_conditioning_covariate_keys` |
| Per-feature scaling (feature scaling) | — | Experiment batch | `feature_scaling_covariate_keys` |

Using a single `batch_key` forces all components to share the same granularity, which either under-corrects (too few groups) or over-parameterises (too many groups for components that don't need that resolution).

### Purpose-driven covariate keys

| Parameter | Model component | Encoder | Decoder | Shape per modality |
|-----------|----------------|---------|---------|-------------------|
| `ambient_covariate_keys` | Additive background $`s_{e,g} = \exp(\beta_{e,g})`$: single parameter with concatenated one-hot across all ambient keys | No | No (additive on rate) | `(n_feat, sum(n_cats))` |
| `nn_conditioning_covariate_keys` | Standard scVI-style injection $`c_{k,n}`$ into decoder | No | Yes (one-hot) | Standard |
| `feature_scaling_covariate_keys` | Feature scaling $`y_{t,g}`$: multiplicative per-feature scaling | No | Multiplicative on rate | `(sum(n_cats), n_feat)` |
| `dispersion_key` | Inverse dispersion $`\theta_{g,d}`$: per-gene, per-group | No | Indexing | `(n_feat, n_disp_cats)` |
| `library_size_key` | Library prior $`\ell_n \sim \text{LogNormal}(\ell_p^{\mu}, 0.5 \cdot \ell_p^{\sigma^2})`$: per-group mean/var | No | No | `(1, n_lib_cats)` |
| `encoder_covariate_keys` | Categorical covariates for encoder only | Yes (one-hot) | No | Standard |

### Default behaviour and backward compatibility

- **`batch_key` alone** (backward compatible): Automatically fans out to `ambient_covariate_keys=[batch_key]`, `dispersion_key=batch_key`, `library_size_key=batch_key`. Equivalent to the original single-batch design — in the notation above, $`e = d = p`$ (same batch groups) and $`t = k`$ (same categorical groups).

- **`batch_key` + purpose-specific keys**: Raises `ValueError`. These are mutually exclusive — use one approach or the other.

- **`feature_scaling_covariate_keys`**: If not specified but `nn_conditioning_covariate_keys` is provided, defaults to `nn_conditioning_covariate_keys` (multi-modal only).

### Encoder/decoder composition

The encoder and decoder receive different subsets of covariates:

- **Encoder** receives: gene expression $`x_n`$ + continuous covariates (if any) + `encoder_covariate_keys` categoricals (if any). By default `encoder_covariate_keys=False`, so the encoder sees **only expression and continuous covariates** — matching the scVI/MultiVI/PeakVI default. This keeps the latent space free of batch information. Setting `encoder_covariate_keys` to a list of keys (e.g. `["batch", "site"]`) injects those categoricals into the encoder; a warning is emitted for non-default values.

- **Decoder** receives: `[cat_covs...]` from `nn_conditioning_covariate_keys` only (batch-free by default). When `use_batch_in_decoder=True`, batch is additionally included.

### Example: 10x Chromium (simple)

```python
# batch_key fans out to all components — equivalent to original API
regularizedvi.AmbientRegularizedSCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="sample",
    nn_conditioning_covariate_keys=["donor", "site"],
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
    nn_conditioning_covariate_keys=["experiment"],
    # Dispersion varies by embryo (tissue composition → variance structure)
    dispersion_key="embryo_id",
    # Library size determined by PCR well (amplification batch)
    library_size_key="pcr_well",
)
```

---

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
    nn_conditioning_covariate_keys=["site", "donor"],
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
    nn_conditioning_covariate_keys=["experiment"],           # encoder/decoder injection
    dispersion_key="embryo_id",                          # per-group overdispersion
    library_size_key="pcr_well",                         # library size prior groups
)
```

See [Covariate design](#covariate-design) below for full details.

See [Default configuration](#default-configuration) above for all current defaults (residual library encoder, data-dependent init, decoder L2, etc.).

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
