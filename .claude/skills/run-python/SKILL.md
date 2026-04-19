---
name: run-python
description: ALWAYS use this when executing Python code/scripts. Activates correct conda env for the current cluster (Crick or Sanger). NEVER use bare python3, conda run, or piped/chained Python (|, &&, &).
user-invocable: false
---

# Run Python

**ALWAYS** use `scripts/helper_scripts/run_python_cmd.sh` to run any Python code or scripts. The launcher auto-detects the cluster (Crick vs Sanger) and activates the correct conda env — call sites are identical on both systems.

## Usage

```bash
# Run a script (default regularizedvi env)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py /path/to/file.h5ad

# Run inline code (default regularizedvi env)
bash scripts/helper_scripts/run_python_cmd.sh -c "print('hello')"

# Use a different environment (Sanger only for now)
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state script.py
bash scripts/helper_scripts/run_python_cmd.sh --env hashfrag script.py
```

## Cluster-aware activation

The launcher picks the correct conda env prefix based on which cluster filesystems exist:

| Short name | Crick path | Sanger path |
|---|---|---|
| `regularizedvi` (default) | `/nemo/lab/briscoej/home/users/kleshcv/conda_environments/regularizedvi` | `/software/conda/users/vk7/regularizedvi` |
| `cell2state` | (not installed yet) | `/software/conda/users/vk7/cell2state_v2026_cuda124_torch25` |
| `hashfrag` | (not installed yet) | `/software/conda/users/vk7/hashfrag_env` |

**Activation flow:**

- **Crick** — the user's `~/.bashrc` contains the standard `conda init` block (managed by Miniconda3) and exports `PYTHONNOUSERSITE=TRUE` via a block appended by `setup_bashrc_crick.sh`. The launcher `source`s `~/.bashrc`, then sets `PYTHONNOUSERSITE=TRUE` again explicitly (belt-and-braces) immediately before `conda activate`.
- **Sanger** — the launcher loads `ISG/conda` (a site-customized module that pre-registers the conda shell function) and runs `conda activate <env>`. `PYTHONNOUSERSITE=TRUE` is set explicitly before activation.

Why `conda activate` (not direct binary invocation): activation is what attaches non-Python runtime deps — it prepends the env's `bin/` to PATH, sets `CONDA_PREFIX`, injects `LD_LIBRARY_PATH`, and runs per-env `etc/conda/activate.d/*.sh` hooks for CUDA/MKL. Calling `/path/to/env/bin/python` directly skips all of that.

## NEVER do these

```bash
# BAD: bare python
python3 -c "..."
python -c "..."

# BAD: manual env activation (skips activate.d hooks)
PYTHONNOUSERSITE=TRUE /software/conda/.../bin/python -c "..."
/nemo/lab/briscoej/.../conda_environments/regularizedvi/bin/python -c "..."

# BAD: conda run
conda run -n regularizedvi python -c "..."

# BAD: source conda.sh in user scripts (the launcher already handles activation)
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

# BAD: piping Python output through shell tools
bash run_python_cmd.sh -c "..." | grep pattern | head -5
```
