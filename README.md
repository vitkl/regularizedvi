# regularizedvi

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/vitkl/regularizedvi/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/regularizedvi

Regularized scVI with ambient RNA correction and overdispersion regularisation, based on cell2location/cell2fate modelling principles (Kleshchevnikov et al. 2022, Simpson et al. 2017).

## Motivation

Standard scVI (Lopez et al. 2018) models observed counts as:

$$x_{ng} \sim \text{NB}(\mu = \ell_n \rho_{ng},\; \theta_g)$$

where $\rho_n \in \Delta^{G-1}$ (softmax output), $\ell_n$ is library size, and $\theta_g$ is per-gene inverse dispersion.

**regularizedvi** adapts cell2location modelling principles to scVI:

$$x_{ng} \sim \text{NB}\!\Big(\mu = \ell_n \cdot \big(\rho_{ng} + b_{g,s_n}\big),\;\theta_{g,s_n}\Big)$$

with a containment prior on overdispersion: $\sqrt{\exp(\phi_g)} \sim \text{Exponential}(\lambda)$

### Key modifications

1. **Ambient RNA correction**: Per-gene, per-sample additive background $b_{g,s_n}$ captures ambient RNA contamination, mirroring cell2location's $(g_{fg} + b_{eg}) \cdot h_e$ structure.

2. **Overdispersion regularisation**: Exponential prior pushes NB toward Poisson, forcing the model to explain count variation through the mean (biology) rather than inflated variance.

3. **Architectural changes**: Softplus activation (not softmax — expression is no longer on the simplex), learned library size, batch-free decoder (batch effects via additive background + categorical covariates), LayerNorm, dropout-on-input.

## Installation

### GPU environment (recommended)

```bash
export PYTHONNOUSERSITE="1"
module load ISG/conda
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

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

## References

- Lopez, R., Regier, J., Cole, M.B. et al. Deep generative modeling for single-cell transcriptomics. *Nat Methods* 15, 1053–1058 (2018).
- Kleshchevnikov, V., Shmatko, A., Dann, E. et al. Cell2location maps fine-grained cell types in spatial transcriptomics. *Nat Biotechnol* 40, 661–671 (2022).
- Simpson, D., Rue, H., Riebler, A. et al. Penalising Model Component Complexity: A Principled, Practical Approach to Constructing Priors. *Statist. Sci.* 32(1), 1-28 (2017).

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/vitkl/regularizedvi/issues
[tests]: https://github.com/vitkl/regularizedvi/actions/workflows/test.yaml
[documentation]: https://regularizedvi.readthedocs.io
[changelog]: https://regularizedvi.readthedocs.io/en/latest/changelog.html
[api documentation]: https://regularizedvi.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/regularizedvi
