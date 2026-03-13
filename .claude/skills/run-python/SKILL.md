---
name: run-python
description: ALWAYS use this when executing Python code/scripts. Activates correct conda env. NEVER use bare python3, conda run, or piped/chained Python (|, &&, &).
user-invocable: false
---

# Run Python

**ALWAYS** use `scripts/helper_scripts/run_python_cmd.sh` to run any Python code or scripts.

## Usage

```bash
# Run a script (default regularizedvi env)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py /path/to/file.h5ad

# Run inline code (default regularizedvi env)
bash scripts/helper_scripts/run_python_cmd.sh -c "print('hello')"

# Use a different environment
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state script.py
bash scripts/helper_scripts/run_python_cmd.sh --env hashfrag script.py
```

## Available environments

| Short name | Conda path |
|---|---|
| `regularizedvi` (default) | `/software/conda/users/vk7/regularizedvi` |
| `cell2state` | `/software/conda/users/vk7/cell2state_v2026_cuda124_torch25` |
| `hashfrag` | `/software/conda/users/vk7/hashfrag_env` |

## NEVER do these

```bash
# BAD: bare python
python3 -c "..."
python -c "..."

# BAD: manual env activation
PYTHONNOUSERSITE=TRUE /software/conda/.../bin/python -c "..."

# BAD: conda run
conda run -n regularizedvi python -c "..."

# BAD: piping Python output through shell tools
bash run_python_cmd.sh -c "..." | grep pattern | head -5
```
