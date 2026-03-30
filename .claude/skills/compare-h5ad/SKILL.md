---
name: compare-h5ad
description: "Use when comparing two h5ad/AnnData files. TRIGGER when: user asks to compare, diff, or check overlap between two .h5ad files, wants to verify obs/var alignment, check if layers match, or validate data pipeline output against reference."
user-invocable: true
argument-hint: "[FILE1.h5ad] [FILE2.h5ad]"
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Compare h5ad Files

Deep comparison of two h5ad files: obs/var overlap, layers, obsm, dtypes, encodings. Uses h5py only (no full anndata load).

## Usage

```bash
# Compare two files
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/compare_h5ad.py file1.h5ad file2.h5ad

# Show example values for differences
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/compare_h5ad.py file1.h5ad file2.h5ad --examples 5
```

## What it compares

- Shape (n_obs × n_vars)
- obs_names and var_names overlap (Jaccard, unique to each)
- obs and var columns (presence, dtypes)
- Layers (names, shapes, dtypes)
- obsm keys (shapes)
- obsp keys
- uns keys
