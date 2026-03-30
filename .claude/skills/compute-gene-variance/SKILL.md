---
name: compute-gene-variance
description: "Use when computing per-gene variance on large h5ad files. TRIGGER when: user asks about gene variance, highly variable genes, feature selection statistics, or needs variance computation without loading full AnnData into memory. Uses chunked Welford's algorithm via h5py."
user-invocable: true
argument-hint: "[FILE.h5ad] [--layer LAYER] [--chunk-size N]"
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Compute Gene Variance

Efficient per-gene variance on large h5ad files using library-size normalization + log1p + chunked Welford's algorithm. Uses h5py + scipy.sparse — no full anndata load.

## Usage

```bash
# Default (X matrix)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/compute_gene_variance.py /path/to/file.h5ad

# Specific layer
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/compute_gene_variance.py file.h5ad --layer counts

# Filter by feature type
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/compute_gene_variance.py file.h5ad --feature-type "Gene Expression"

# Custom chunk size (for memory management)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/compute_gene_variance.py file.h5ad --chunk-size 5000
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--layer` | X | Which layer to compute variance on |
| `--feature-type` | None | Filter to specific feature type |
| `--chunk-size` | 10000 | Cells per chunk (lower = less memory) |
