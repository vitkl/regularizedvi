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

**regularizedvi** adapts [cell2location](https://doi.org/10.1038/s41587-021-01139-4)/[cell2fate](https://doi.org/10.1038/s41592-025-02608-3) modelling principles to scVI:

$$z_n \sim \text{Normal}(0, I)$$
$$\ell_n \sim \text{LogNormal}(\ell_\mu^\top s_n,\; 0.05 \cdot \ell_{\sigma^2}^\top s_n)$$
$$\rho_{ng} = \text{softplus}\big(f_w(z_n, c_n)\big) \in \mathbb{R}_{\geq 0}^G$$
$$b_{g,s_n} = \exp(\beta_{g,s_n})$$
$$\sqrt{\theta_{g,s_n}} \sim \text{Exponential}(\lambda)$$
$$x_{ng} \sim \text{GammaPoisson}\Big(\text{concentration} = \theta_{g,s_n}, \text{rate} = \frac{\theta_{g,s_n}}{\ell_n \cdot \big(\rho_{ng} + b_{g,s_n}\big)}\Big)$$

or equivalently as negative binomial with mean $\mu = \ell_n \cdot \big(\rho_{ng} + b_{g,s_n}\big)$ and dispersion $\theta_{g,s_n}$.

where additionally:
- $b\_{g,s\_n} = \exp(\beta\_{g,s\_n})$ — per-gene, per-batch additive ambient RNA background (learnable parameter)
- $\beta\_{g,s\_n}$ — unconstrained ambient background parameter (code: `additive_background`), this is an implementational detail
- $c\_n$ — categorical covariates only (site, donor, etc.), **not** batch $s\_n$ (batch-free decoder)
- $\rho\_{ng} \in \mathbb{R}\_{\geq 0}^G$ — decoder output via softplus (no longer on the simplex), since $\rho\_{ng} + b\_{g,s\_n}$ need not sum to 1
- $\theta\_{g,s\_n} = \exp(\phi\_{g,s\_n})$ — inverse dispersion, parameterised in log-space (code: `px\_r` stores $\phi$)
- $\lambda = 3$ — rate for Exponential containment prior on $\sqrt{\theta\_{g,s\_n}}$
- $0.05$ — library prior variance scaling factor (constraining library size)

The NB variance is $\text{Var}(x) = \mu + \mu^2/\theta$, where $\theta$ is the inverse dispersion (= `GammaPoisson` concentration parameter). Large $\theta$ means less overdispersion and less variance (variance approaching Poisson $\mu^2/\theta = 0$ -> $\text{Var}(x) = \mu$).

### Key modifications

1. **Ambient RNA correction**: Per-gene, per-sample additive background $b\_{g,s\_n}$ captures ambient RNA contamination, mirroring cell2location's $(g\_{f,g} + b\_{e,g}) \cdot h\_e$ structure. Implemented as `nn.Parameter(torch.randn(n_genes, n_batch))` with per-batch selection via one-hot encoding.

2. **Dispersion regularisation**: Prior $\sqrt{\theta\_{g,s\_n}} \sim \text{Exponential}(\lambda)$ on the dispersion quantity (not overdispersion) penalises large $\theta$, preventing the NB from collapsing to Poisson during gradient-based training. Inspired by the containment prior of Simpson et al. (2017) as used in cell2location/cell2fate, but placed on the dispersion rather than overdispersion quantity — see comparison above.

3. **Batch-free decoder with separated correction paths**: The decoder $f\_w(z\_n, c\_n)$ receives only categorical covariates $c\_n$ (site, donor, protocol), **not** the batch indicator $s\_n$. This separates batch correction into two structurally different paths: (a) a **constrained additive** path ($b\_{g,s\_n}$) for per-sample ambient RNA, and (b) a **flexible multiplicative** path through categorical covariates in the decoder for systematic differences between donors, protocols, or sites (e.g. PCR bias, RT efficiency, 10x chemistry versions). In standard scVI, the decoder handles all batch effects through a single flexible path, which can absorb biological variation. The separation is most beneficial when batches have high within-batch cell type diversity (e.g. whole-embryo samples), because the additive background can be cleanly identified as the baseline signal shared across all cells in a batch. Batch-specific dispersion $\theta\_{g,s\_n}$ provides a third correction path for per-batch variance differences.

4. **Softplus activation**: Because $\rho\_{ng} + b\_{g,s\_n}$ must be non-negative but need not sum to 1 across genes, softmax is replaced with softplus. The library size $\ell\_n$ acts as a true normalisation factor.

5. **Learned library size with constrained prior**: The observed total counts include ambient RNA, so library size must be learned (not observed). Prior variance is scaled by 0.05 to prevent the library size from absorbing biological signal. Library encoder has low capacity (`n_hidden=16`).

6. **LayerNorm and dropout-on-input**: LayerNorm replaces BatchNorm (independent of batch composition). Dropout is applied before the linear layer (feature-level masking).

### Practical notes and caveats

- **Best suited for single-nucleus RNA-seq** (independent modality and multiome), which typically has substantial ambient RNA contamination. The ambient correction is less necessary for single-cell RNA-seq where ambient levels are lower.

- **Study design matters**: The structured assumptions (additive ambient + multiplicative categorical covariates) depend on the experimental design. With some study designs, every batch has both additive effects (ambient RNA) and multiplicative effects (PCR bias, RT differences, 10x 3' v1 vs v2 vs v3, 3' vs 5'). These assumptions may not hold for Smart-seq type data where every cell can have PCR bias and RT differences.

- **Using as standard scVI with ambient correction**: If you provide the batch covariate to both `batch_key` and `categorical_covariate_keys`, the model effectively operates as standard scVI with ambient RNA correction (batch effects handled through both additive and multiplicative paths).

- **Not a strict ambient correction model**: Unlike CellBender (Fleming et al. 2023), this model is not constrained by the ambient count distribution from empty droplets. However, because it does not require empty droplets data, it can be more easily applied to integration of published datasets where empty droplet profiles are unavailable.

- **Additivity in non-negative space**: The additive background operates in non-negative space ($b\_{g,s\_n} = \exp(\beta\_{g,s\_n})$), reflecting the ambient RNA correction mechanism. Without empty droplets data, the additive component can learn the minimal expression of each gene across cells — for many genes this reflects ambient levels, but for ubiquitously expressed genes it captures genuine baseline expression. The additive mechanism therefore works best when individual batches are composed of diverse cell types.

- **Regularised overdispersion alone likely helps**: Overdispersion regularisation prevents the NB from collapsing to Poisson during training, keeping the likelihood appropriately "loose" so the decoder captures genuine biological signal rather than overfitting individual count values. This likely contributes to improved sensitivity, but needs more systematic testing.

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

The model now uses **GammaPoisson likelihood** (cell2location-style) by default, which provides more flexible count modelling than NB and a regularised overdispersion prior to prevent overfitting. The default dispersion is `"gene-batch"`, providing per-gene, per-batch inverse dispersion parameters.

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
