# Sub-Plan 03: Mask Construction + Covariate Hierarchy Validation

**Parent plan**: `neighbourhood_correlation_plan.md` (Three-Level Covariate Hierarchy section)

## Review Corrections (applied)

1. **Keep `between_libraries` as a base mask** (subagent was WRONG to remove it). `between_libraries` = neighbours with different `library_key` value, REGARDLESS of dataset. It's always computed, alongside the more granular `cross_library`/`cross_dataset` split when dataset is provided. Useful as an aggregate "any cross-library" view and required for single-dataset graceful degradation. `list_active_masks` returns:
   - `dataset_key is None`: `["same_library", "between_libraries"]`
   - `dataset_key` provided: `["same_library", "between_libraries", "cross_library", "cross_dataset"]` — note `between_libraries = cross_library ∪ cross_dataset` but both are kept (different semantic views)
   - `+ technical_covariate_keys`: append `["within_{C}", "between_{C}"]` per covariate
2. **Primary implementation pattern** (replace `sp.find` approach): use sparse boolean mask × connectivity multiply:
   ```python
   conn = connectivities.copy()
   conn.setdiag(0)
   conn.eliminate_zeros()
   i, j = conn.nonzero()
   lib_codes = adata.obs[library_key].astype("category").cat.codes.to_numpy()
   same_lib_bool = lib_codes[i] == lib_codes[j]
   mask_bool = sp.csr_matrix((same_lib_bool, (i, j)), shape=conn.shape, dtype=bool)
   masked = conn.multiply(mask_bool)  # preserves connectivity weights
   ```
3. **Preprocessing**: explicit `connectivities.setdiag(0); connectivities.eliminate_zeros()` upfront (single pass) instead of per-edge `i != j` filter.
4. **Use `conn.nonzero()` not `sp.find`**: `sp.find` returns `(i, j, v)` but we only need `(i, j)` for mask bool construction.
5. **Partitioning test** (replaces old): verify `same_library + cross_library + cross_dataset` nnz equals `nonzero(connectivities after setdiag).nnz` (exact partition).

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD to module)

## Dependencies
- Sub-plan 01 complete (module skeleton)

## Tasks

### 1. Hierarchy validation

```python
def validate_covariate_hierarchy(
    adata,
    library_key: str,
    dataset_key: str | None = None,
) -> None:
    """Verify each library maps to exactly one dataset.

    Raises ValueError if a library value spans multiple dataset values.
    """
```

- If `dataset_key is None`: pass silently (single-dataset case)
- Extract unique `(library, dataset)` pairs from `adata.obs`
- Group by library, check each library has exactly one dataset
- On violation: `ValueError` listing offending library values + their datasets
- On success: optionally log n_libraries per dataset as diagnostic

### 2. Neighbour mask construction

```python
def construct_neighbour_masks(
    adata,
    connectivities,  # scipy.sparse CSR (n_cells, n_cells)
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
) -> dict[str, "sp.csr_matrix"]:
    """Build boolean masks over connectivity matrix for each covariate relationship.

    Returns
    -------
    dict mapping mask name to sparse boolean (n_cells, n_cells) matrix.
    Multiplying connectivities element-wise with a mask keeps only neighbours
    matching that relationship.

    Masks always include:
        - same_library
        - between_libraries  (different library, any dataset)

    If dataset_key provided:
        - cross_library     (different library, SAME dataset)
        - cross_dataset     (different dataset — always different library too)

    If technical_covariate_keys provided, for each covariate name C:
        - within_{C}        (same value of covariate C)
        - between_{C}       (different value of covariate C)
    """
```

Implementation approach (primary — use `multiply(sparse_bool_mask)`):

```python
# Preprocessing (once at start of function)
conn = connectivities.copy()
conn.setdiag(0)
conn.eliminate_zeros()  # removes self-loops cleanly

# Call validation
validate_covariate_hierarchy(adata, library_key, dataset_key)

# Get integer-encoded covariates
lib_codes = adata.obs[library_key].astype("category").cat.codes.to_numpy()

# Extract (i, j) pairs ONCE for reuse across all masks
i, j = conn.nonzero()

# For each mask, build sparse boolean mask matrix then multiply:
same_lib_bool = lib_codes[i] == lib_codes[j]
same_lib_mask = sp.csr_matrix(
    (same_lib_bool, (i, j)), shape=conn.shape, dtype=bool
)
masks["same_library"] = conn.multiply(same_lib_mask)  # preserves weights

# Between-libraries (always computed):
between_lib_bool = ~same_lib_bool
masks["between_libraries"] = conn.multiply(
    sp.csr_matrix((between_lib_bool, (i, j)), shape=conn.shape, dtype=bool)
)

# If dataset_key provided, also split between_libraries:
if dataset_key is not None:
    ds_codes = adata.obs[dataset_key].astype("category").cat.codes.to_numpy()
    same_ds_bool = ds_codes[i] == ds_codes[j]
    masks["cross_library"] = conn.multiply(
        sp.csr_matrix((between_lib_bool & same_ds_bool, (i, j)), shape=conn.shape, dtype=bool)
    )
    masks["cross_dataset"] = conn.multiply(
        sp.csr_matrix((~same_ds_bool, (i, j)), shape=conn.shape, dtype=bool)
    )

# Technical covariates: one pair of masks per covariate in the list
for tech_key in (technical_covariate_keys or []):
    t_codes = adata.obs[tech_key].astype("category").cat.codes.to_numpy()
    same_t_bool = t_codes[i] == t_codes[j]
    masks[f"within_{tech_key}"] = conn.multiply(
        sp.csr_matrix((same_t_bool, (i, j)), shape=conn.shape, dtype=bool)
    )
    masks[f"between_{tech_key}"] = conn.multiply(
        sp.csr_matrix((~same_t_bool, (i, j)), shape=conn.shape, dtype=bool)
    )
```

- Use `conn.nonzero()` (not `sp.find`) — we don't need weights at mask-construction time
- `multiply(sparse_bool_mask)` preserves connectivity weights automatically
- O(nnz) per mask; total O(nnz × n_masks) which is efficient

### 3. Helper: list active masks

```python
def list_active_masks(
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
) -> list[str]:
    """Return ordered list of mask names based on which keys are provided."""
```

Returns:
- Base (always): `["same_library", "between_libraries"]`
- If dataset provided: `["same_library", "between_libraries", "cross_library", "cross_dataset"]` — `between_libraries` kept as aggregate view alongside the dataset-split granular masks
- If technical provided: append `["within_{C}", "between_{C}"]` per covariate name

### 4. Add to `__all__`

Export: `validate_covariate_hierarchy`, `construct_neighbour_masks`, `list_active_masks`.

## Test cases
- Valid hierarchy passes silently
- Library spanning 2 datasets raises ValueError with informative message
- `construct_neighbour_masks` returns all expected masks for each config (library only; library+dataset; library+dataset+technical)
- Mask entries preserve connectivity weights
- Self-loops excluded from all masks
- `cross_library ∪ cross_dataset == between_libraries` (sparsity pattern check)

## Verification
- Test on immune integration data: `library_key="batch"`, `dataset_key="dataset"`, no technical → 4 masks
- Test on embryo: `library_key="sample_id"`, `dataset_key="Section"`, `technical_covariate_keys=["Embryo", "Experiment"]` → 8 masks
- Test on single-dataset (bone marrow only): `library_key="batch"`, no dataset → 2 masks
