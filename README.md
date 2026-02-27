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

**Latent variable and library size** — standard scVI structure with a constrained library prior ([`_module.py:267–272`](src/regularizedvi/_module.py#L267-L272)):

$$z_n \sim \text{Normal}(0, I)$$
$$\ell_n \sim \text{LogNormal}(\ell_\mu^\top s_n,\; 0.05 \cdot \ell_{\sigma^2}^\top s_n)$$

**Decoder output** — batch-free decoder maps $z\_n$ and categorical covariates $c\_n$ (not batch $s\_n$) to non-negative gene expression via softplus ([`_module.py:638–645`](src/regularizedvi/_module.py#L638-L645), [`_components.py:461–462`](src/regularizedvi/_components.py#L461-L462)):

$$\rho_{ng} = \text{softplus}\big(f_w(z_n, c_n)\big) \in \mathbb{R}_{\geq 0}^G$$

**Additive background** — per-gene, per-batch ambient RNA with Gamma prior pushing $b\_{g,s\_n}$ toward 0.01 ([`_module.py:327–331`](src/regularizedvi/_module.py#L327-L331) init, [`_module.py:608–613`](src/regularizedvi/_module.py#L608-L613) one-hot selection, [`_module.py:767–777`](src/regularizedvi/_module.py#L767-L777) prior penalty):

$$b_{g,s_n} = \exp(\beta_{g,s_n}), \qquad b_{g,s_n} \sim \text{Gamma}(1,\, 100)$$

**Hierarchical dispersion prior** — two-level prior on inverse dispersion $\theta\_{g,s\_n} = \exp(\phi\_{g,s\_n})$ where `px_r` stores $\phi$. A learned rate $\lambda\_{s\_n}$ adapts regularisation strength per batch ([`_module.py:731–765`](src/regularizedvi/_module.py#L731-L765) full block, [`_module.py:737`](src/regularizedvi/_module.py#L737) Level 1 softplus, [`_module.py:758–764`](src/regularizedvi/_module.py#L758-L764) Level 2 transform):

$$\lambda_{s_n} \sim \text{Gamma}(9,\, 3), \qquad 1/\sqrt{\theta_{g,s_n}} \sim \text{Exponential}(\lambda_{s_n})$$

**Observation model** — GammaPoisson (= negative binomial) with mean $\mu\_{ng} = \ell\_n \cdot (\rho\_{ng} + b\_{g,s\_n})$ ([`_components.py:467`](src/regularizedvi/_components.py#L467) rate computation):

$$x_{ng} \sim \text{GammaPoisson}\Big(\text{concentration} = \theta_{g,s_n},\;\; \text{rate} = \frac{\theta_{g,s_n}}{\ell_n \cdot \big(\rho_{ng} + b_{g,s_n}\big)}\Big)$$

**Notation:**
- $b\_{g,s\_n} = \exp(\beta\_{g,s\_n})$ — per-gene, per-batch ambient background; $\beta$ is the unconstrained parameter (code: `additive_background`)
- $c\_n$ — categorical covariates only (site, donor, etc.), **not** batch $s\_n$ (batch-free decoder)
- $\rho\_{ng} \in \mathbb{R}\_{\geq 0}^G$ — decoder output via softplus (not on the simplex), since $\rho\_{ng} + b\_{g,s\_n}$ need not sum to 1
- $\theta\_{g,s\_n} = \exp(\phi\_{g,s\_n})$ — inverse dispersion in log-space (code: `px_r` stores $\phi$); initialised at $\log(\lambda^2) \approx 2.2$ so $\theta \approx 9$ (equilibrium: $1/\sqrt{\theta} = 1/\lambda$) ([`_module.py:277–284`](src/regularizedvi/_module.py#L277-L284))
- $\lambda\_{s\_n}$ — learned Exponential rate, one per batch; $\text{Gamma}(9, 3)$ hyper-prior has mean 3 ([`_module.py:314–322`](src/regularizedvi/_module.py#L314-L322))
- $0.05$ — library prior variance scaling factor, preventing library size from absorbing biological signal

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

**Library size** — always learned (observed totals include ambient contamination). A low-capacity encoder infers library size per cell, regularised by a tight LogNormal prior estimated per batch ([`_multimodule.py:375–384`](src/regularizedvi/_multimodule.py#L375-L384) prior buffers, [`_multimodule.py:772–773`](src/regularizedvi/_multimodule.py#L772-L773) loss):

$$\ell^{(m)}_n \sim \text{LogNormal}\big(\ell^{(m)}_{\mu}{}^\top s_n,\; 0.05 \cdot \ell^{(m)}_{\sigma^2}{}^\top s_n\big)$$

**Decoder output** — maps joint latent code $z\_n$ and categorical covariates $c\_n$ (not batch $s\_n$) to non-negative feature signal via softplus ([`_multimodule.py:717–724`](src/regularizedvi/_multimodule.py#L717-L724)):

$$\rho^{(m)}_{nf} = \text{softplus}\big(f^{(m)}_w(z_n,\, c_n)\big) \in \mathbb{R}_{\geq 0}$$

**Additive background** — per-feature, per-batch ambient contamination with Gamma prior ([`_multimodule.py:439–445`](src/regularizedvi/_multimodule.py#L439-L445) init, [`_multimodule.py:701–705`](src/regularizedvi/_multimodule.py#L701-L705) one-hot selection, [`_multimodule.py:943–955`](src/regularizedvi/_multimodule.py#L943-L955) prior penalty):

$$b^{(m)}_{f,s_n} = \exp(\beta^{(m)}_{f,s_n}), \qquad b^{(m)}_{f,s_n} \sim \text{Gamma}(1,\, 100)$$

**Region factors** — per-feature, per-covariate multiplicative scaling capturing systematic biases (GC content, mappability, peak caller sensitivity). Parameterised as $\text{softplus}(\gamma)/0.7$ with a tight Gamma prior centered at 1. When scaling covariates $t$ are registered, each covariate category gets its own factor; the per-cell scaling is selected via one-hot indicator ([`_multimodule.py:452–456`](src/regularizedvi/_multimodule.py#L452-L456) init, [`_multimodule.py:728–736`](src/regularizedvi/_multimodule.py#L728-L736) activation and selection, [`_multimodule.py:924–941`](src/regularizedvi/_multimodule.py#L924-L941) prior penalty):

$$\alpha^{(m)}_{t,f} = \text{softplus}(\gamma^{(m)}_{t,f})\,/\,0.7, \qquad \alpha^{(m)}_{t,f} \sim \text{Gamma}(200,\, 200)$$

**Expected mean counts** — decoder output plus optional background, scaled by library size and region factor ([`_components.py:467`](src/regularizedvi/_components.py#L467), [`_multimodule.py:736`](src/regularizedvi/_multimodule.py#L736)):

$$\mu^{(m)}_{nf} = \ell^{(m)}_n \cdot \big(\rho^{(m)}_{nf} + b^{(m)}_{f,s_n}\big) \cdot \alpha^{(m)}_{t_n,f}$$

**Hierarchical dispersion prior** — same two-level structure as single-modality, per modality and batch ([`_multimodule.py:889–922`](src/regularizedvi/_multimodule.py#L889-L922)):

$$\lambda^{(m)}_{s_n} \sim \text{Gamma}(9,\, 3), \qquad 1/\sqrt{\theta^{(m)}_{f,s_n}} \sim \text{Exponential}(\lambda^{(m)}_{s_n})$$

**Observation model** — GammaPoisson (= negative binomial) with mean $\mu^{(m)}\_{nf}$ and inverse dispersion $\theta^{(m)}\_{f,s\_n}$:

$$x^{(m)}_{nf} \sim \text{GammaPoisson}\!\Big(\text{concentration} = \theta^{(m)}_{f,s_n},\;\; \text{rate} = \frac{\theta^{(m)}_{f,s_n}}{\mu^{(m)}_{nf}}\Big)$$

#### Optional per-modality correction terms

| Term | Symbol | Prior | What it captures | RNA default | ATAC default |
|------|--------|-------|-----------------|-------------|--------------|
| Additive background | $b^{(m)}_{f,s_n} = \exp(\beta^{(m)}_{f,s_n})$ | $\text{Gamma}(1, 100)$, mean 0.01 | Per-feature, per-batch ambient contamination or assay baseline | **ON** | off |
| Region factor | $\alpha^{(m)}_{t,f} = \text{softplus}(\gamma^{(m)}_{t,f})/0.7$ | $\text{Gamma}(200, 200)$, mean 1.0 | Per-feature, per-covariate multiplicative bias (GC content, mappability) | off | **ON** |
| Learned library size | $\ell^{(m)}_n$ | $\text{LogNormal}$, 0.05 var scaling | Low-capacity encoder; observed totals include ambient | **always ON** | **always ON** |
| Dispersion regularisation | $1/\sqrt{\theta^{(m)}_{f,s_n}}$ | $\text{Exp}(\lambda)$, $\lambda \sim \text{Gamma}(9,3)$ | Containment prior regularising against excessive overdispersion | ON | ON |
| Batch-free decoder | — | — | Decoder conditioned only on categorical covariates $c_n$, not batch $s_n$ | ON | ON |

Setting $b^{(m)}\_{f,s\_n} = 0$ (no ambient) and $\alpha^{(m)}\_{t,f} = 1$ (no region factor) recovers the standard regularizedvi single-modality model for that modality. The defaults reflect domain knowledge for snRNA+ATAC multiome: ambient RNA contamination is substantial in single-nucleus RNA-seq and well-captured by an additive term, while ATAC peaks have systematic per-peak biases from GC content, mappability and peak caller thresholds. See the [bone marrow multiome tutorial](docs/notebooks/bone_marrow_multimodal_tutorial.ipynb) for a worked RNA+ATAC example.

#### Inference: per-modality encoders and posterior concatenation

**Per-modality encoder** — each modality's encoder takes its own observed counts as input and independently constructs a Gaussian posterior over its private latent slice. The RNA encoder sees only RNA counts; the ATAC encoder sees only ATAC counts. This forces the model to build a dedicated representation for each modality before combining them:

$$q_\eta(z^{(m)}_n \mid x^{(m)}_n, s_n, c_n) = \text{Normal}\!\big(\mu^{(m)}_\eta(x^{(m)}_n),\; (\sigma^{(m)}_\eta)^2(x^{(m)}_n)\big)$$

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

1. **Ambient RNA correction with Gamma prior**: Per-gene, per-batch additive background $b\_{g,s\_n} = \exp(\beta\_{g,s\_n})$ captures ambient RNA contamination, mirroring cell2location's $s\_g \cdot g\_{e,g}$ structure. A $\text{Gamma}(1, 100)$ prior pushes $b\_{g,s\_n}$ toward 0.01, keeping background small relative to biological signal. Initialised at `log(0.01)` (prior mean) with per-batch selection via one-hot encoding.

2. **Hierarchical dispersion regularisation**: Prior $1/\sqrt{\theta\_{g,s\_n}} \sim \text{Exponential}(\lambda)$ is a containment prior (Simpson et al. 2017) that penalises small $\theta$ (excessive overdispersion), regularising the NB toward the Poisson baseline during gradient-based training. The data likelihood provides the opposing force, pulling $\theta$ toward values that explain observed count variance. The rate $\lambda$ is learned per batch with a $\text{Gamma}(9, 3)$ hyper-prior (mean 3). Dispersion $\theta = \exp(\phi)$ is initialised at $\lambda^2 = 9$ (equilibrium). As used in cell2location/cell2fate.

3. **Batch-free decoder with separated correction paths**: The decoder $f\_w(z\_n, c\_n)$ receives only categorical covariates $c\_n$ (site, donor, protocol), **not** the batch indicator $s\_n$. This separates batch correction into structurally different paths: (a) a **constrained additive** path ($b\_{g,s\_n}$ with Gamma prior) for per-sample ambient RNA, (b) a **flexible multiplicative** path through categorical covariates in the decoder for systematic differences between donors, protocols, or sites (e.g. PCR bias, RT efficiency, 10x chemistry versions), and (c) batch-specific dispersion $\theta\_{g,s\_n}$ for per-batch variance differences. In standard scVI, the decoder handles all batch effects through a single flexible path, which can absorb biological variation. The separation is most beneficial when batches have high within-batch cell type diversity (e.g. whole-embryo samples), because the additive background can be cleanly identified as the baseline signal shared across all cells in a batch.

4. **Softplus activation**: Because $\rho\_{ng} + b\_{g,s\_n}$ must be non-negative but need not sum to 1 across genes, softmax is replaced with softplus. The library size $\ell\_n$ acts as a true normalisation factor.

5. **Learned library size with constrained prior**: The observed total counts include ambient RNA, so library size must be learned (not observed). Prior variance is scaled by 0.05 to prevent the library size from absorbing biological signal. Library encoder has low capacity (`n_hidden=16`).

6. **LayerNorm and dropout-on-input**: LayerNorm replaces BatchNorm (independent of batch composition). Dropout is applied before the linear layer (feature-level masking).

### Practical notes and caveats

- **Best suited for single-nucleus RNA-seq** (independent modality and multiome), which typically has substantial ambient RNA contamination. The ambient correction is less necessary for single-cell RNA-seq where ambient levels are lower.

- **Study design matters**: The structured assumptions (additive ambient + multiplicative categorical covariates) depend on the experimental design. With some study designs, every batch has both additive effects (ambient RNA) and multiplicative effects (PCR bias, RT differences, 10x 3' v1 vs v2 vs v3, 3' vs 5'). These assumptions may not hold for Smart-seq type data where every cell can have PCR bias and RT differences.

- **Using as standard scVI with ambient correction**: If you provide the batch covariate to both `batch_key` and `categorical_covariate_keys`, the model effectively operates as standard scVI with ambient RNA correction (batch effects handled through both additive and multiplicative paths).

- **Not a strict ambient correction model**: Unlike CellBender (Fleming et al. 2023), this model is not constrained by the ambient count distribution from empty droplets. However, because it does not require empty droplets data, it can be more easily applied to integration of published datasets where empty droplet profiles are unavailable.

- **Additivity in non-negative space**: The additive background operates in non-negative space ($b\_{g,s\_n} = \exp(\beta\_{g,s\_n})$), reflecting the ambient RNA correction mechanism. Without empty droplets data, the additive component can learn the minimal expression of each gene across cells — for many genes this reflects ambient levels, but for ubiquitously expressed genes it captures genuine baseline expression. The additive mechanism therefore works best when individual batches are composed of diverse cell types.

- **Regularised overdispersion alone likely helps**: The containment prior on overdispersion regularises the NB toward the Poisson baseline, preventing the model from absorbing residuals through excessive variance (small $\theta$). This forces the decoder to capture genuine biological signal through its mean structure rather than relying on high overdispersion to explain noise. This likely contributes to improved sensitivity, but needs more systematic testing.

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
