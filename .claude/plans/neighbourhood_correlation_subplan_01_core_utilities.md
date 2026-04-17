# Sub-Plan 01: Core Utilities (Module Skeleton + Normalisation + cluster_averages)

**Parent plan**: `neighbourhood_correlation_plan.md`

## Review Corrections (applied)

1. **Zero-total-counts guard** in `normalise_counts`: use `scale = np.where(total_counts > 0, n_vars / total_counts, 0.0)` to avoid inf/NaN for empty cells.
2. **`compute_cluster_averages` default `use_raw=True` caveat**: add explicit docstring note that downstream callers should pass `use_raw=False` (or provide normalised data via `layer`). Downstream sub-plan 02 should use a matrix-taking helper to avoid layer mutation.

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (CREATE)

## Dependencies
- Read-only: `/nfs/team205/vk7/sanger_projects/BayraktarLab/cell2location/cell2location/cluster_averages/cluster_averages.py` (source to copy)
- Read-only: `src/regularizedvi/plt/_integration_metrics.py` (reference for style, imports, `TYPE_CHECKING` pattern)

## Tasks

### 1. Create module skeleton with imports and module docstring
```python
"""Cell-level neighbourhood marker gene correlation metrics.

Per-cell marker gene expression correlation with KNN neighbours, stratified by
library / dataset / technical covariate relationships. Distinguishes positive
vs negative integration failure modes. Label-free analysis.

See ``.claude/plans/neighbourhood_correlation_plan.md`` for design rationale.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    from anndata import AnnData
    from pathlib import Path
```

### 2. Implement `normalise_counts(X, n_vars=None)` function

Formula: `X * (n_vars / total_counts)` — sets per-gene average to 1.

```python
def normalise_counts(X, n_vars: int | None = None):
    """Normalise counts so per-gene average = 1.

    Formula: normalised = count * (n_vars / total_count_per_cell)

    Parameters
    ----------
    X : sparse or dense (n_cells, n_genes)
    n_vars : int, optional. If None, uses X.shape[1].

    Returns
    -------
    Same type as input (preserves sparsity).
    """
    if n_vars is None:
        n_vars = X.shape[1]
    total_counts = np.asarray(X.sum(axis=1)).flatten()
    # Zero-guard: cells with total_counts == 0 get scale 0 (not inf)
    scale = np.where(total_counts > 0, n_vars / total_counts, 0.0).astype(np.float32)
    if sp.issparse(X):
        return X.multiply(scale[:, None])  # broadcast column vector, stays sparse
    return X * scale[:, None]
```

### 3. Copy `compute_cluster_averages` from cell2location

Verbatim copy from `/nfs/team205/vk7/sanger_projects/BayraktarLab/cell2location/cell2location/cluster_averages/cluster_averages.py` (function at lines 6-53). Add attribution comment.

```python
def compute_cluster_averages(adata, labels, use_raw=True, layer=None):
    """Compute average expression of each gene in each cluster.

    Copied verbatim from cell2location
    (cell2location.cluster_averages.cluster_averages.compute_cluster_averages).
    Kept here to avoid cell2location dependency.

    [rest of original docstring]
    """
    # [verbatim body]
```

### 4. Add public API list (`__all__`)
```python
__all__ = [
    "normalise_counts",
    "compute_cluster_averages",
    # future: select_marker_genes, compute_marker_correlation, etc.
]
```

## Test cases
- `normalise_counts`: after normalisation, `X_norm.sum(axis=1)` ≈ `n_vars` for every cell
- `normalise_counts`: preserves sparsity (sparse in → sparse out)
- `compute_cluster_averages`: reproduce the cell2location test (cluster means match groupby mean)

## Verification
- Run `bash scripts/helper_scripts/run_python_cmd.sh -c "from regularizedvi.plt._neighbourhood_correlation import normalise_counts, compute_cluster_averages"` — imports succeed
- `bash scripts/helper_scripts/run_python_cmd.sh ~/.claude/shared-skills/scripts/syntax_check.py src/regularizedvi/plt/_neighbourhood_correlation.py` — passes
