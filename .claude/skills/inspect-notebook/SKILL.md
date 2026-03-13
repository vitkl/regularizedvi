---
name: inspect-notebook
description: Use when examining Jupyter notebooks - structure, progress, errors, search, execution status, stdout tail, papermill params.
user-invocable: false
allowed-tools: Bash(bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py:*), Bash(bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_extract_params.py:*), Bash(bash scripts/helper_scripts/run_python_cmd.sh --env regularizedvi scripts/claude_helper_scripts/_inspect_notebook.py:*), Bash(bash scripts/helper_scripts/run_python_cmd.sh --env regularizedvi scripts/claude_helper_scripts/_extract_params.py:*)
---

# Inspect Notebook

Use these scripts to examine Jupyter notebooks. All go through `run_python_cmd.sh` for correct Python env activation.

## Scripts

### `_inspect_notebook.py` - Structure, search, progress, errors, execution status

```bash
# Show cell structure
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb

# Execution status per cell
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --exec

# Search for a pattern
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb -s "pattern"

# Show specific cell(s) with outputs
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --cell 17 --outputs

# Execution progress
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --progress

# Extract errors
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --errors

# Last N stdout lines from a cell (skips stderr/tqdm)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --stdout-tail 17
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --stdout-tail 17 -n 50

# Filter output
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --head 20 --grep "error"
```

### `_extract_params.py` - Papermill parameters

```bash
# All output notebooks
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_extract_params.py

# Specific notebook
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_extract_params.py notebook_out.ipynb

# Show only specific params
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_extract_params.py notebook_out.ipynb --param learning_rate --param n_epochs

# Params only (skip training/error info)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_extract_params.py notebook_out.ipynb --params-only
```

### Root-level scripts (tracked in git)

- `_check_nb.py` — alternative notebook checker with summary table: `bash scripts/helper_scripts/run_python_cmd.sh _check_nb.py notebook.ipynb`
- `_show_outputs.py` — show notebook outputs: `bash scripts/helper_scripts/run_python_cmd.sh _show_outputs.py notebook.ipynb`

## NEVER do this

```bash
# BAD: inline Python to parse notebooks
python -c "import json; nb = json.load(open('notebook.ipynb')); ..."
PYTHONNOUSERSITE=TRUE /software/.../bin/python -c "import json; ..."
```
