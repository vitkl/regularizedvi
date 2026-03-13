---
name: use-helper-scripts
description: Background rules for using helper scripts. Claude MUST use these instead of bare python3, conda run, inline Python, or pipe chains. When no existing script covers the use case, launch a subagent to extend or create one.
user-invocable: false
---

# Helper Scripts - Background Knowledge and Fallback

Claude MUST use helper scripts instead of bare python3, conda run, inline Python, or pipe chains.

## Routing Table

| Need | Script(s) |
|---|---|
| Run any Python | `scripts/helper_scripts/run_python_cmd.sh [--env NAME]` |
| Inspect notebook | `scripts/claude_helper_scripts/_inspect_notebook.py`, `_extract_params.py` |
| Inspect notebook (alt) | Root-level `_check_nb.py`, `_show_outputs.py` |
| Inspect h5ad | `scripts/claude_helper_scripts/inspect_h5ad.py` |
| Inspect pickle | `scripts/claude_helper_scripts/inspect_pickle.py` |
| Inspect parquet | `scripts/claude_helper_scripts/check_parquet.py` |
| Monitor jobs | Root-level `_check_alive.sh`, `_check_job_mem.sh`, `_monitor_process.sh` |
| Run tests | `bash run_tests.sh tests/test_model.py -x -q` |

All Python scripts are called via: `bash scripts/helper_scripts/run_python_cmd.sh [--env ENV] SCRIPT.py [args]`

## Fallback: When No Script Covers the Use Case

**DO NOT fall back to raw Python commands** for inspection/analysis tasks.

Instead:
1. Read existing helper scripts to understand patterns and identify the closest match
2. If an existing script can be extended: launch an Agent subagent to modify it
3. If no script is close: launch an Agent subagent to create a new `.py` script following the same patterns (argparse, `--head`/`--grep` options)
4. After the subagent creates/modifies the script, use it via `run_python_cmd.sh`
5. NEVER fall back to inline `python -c "..."` or `run_python_cmd.sh -c "..."` for inspection/analysis tasks

## Anti-Patterns to Avoid

### Anti-pattern 1: Inline Python with manual env activation
```bash
# BAD:
PYTHONNOUSERSITE=TRUE /software/conda/.../bin/python -c "import json; nb = json.load(open('notebook.ipynb')); ..."
# GOOD:
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/_inspect_notebook.py notebook.ipynb --cell 15
```

### Anti-pattern 2: Sleep + chain + pipe for job monitoring
```bash
# BAD:
sleep 360 && tail -3 /path/752345.err 2>/dev/null && echo "---" && bjobs 752345 2>&1 | head -3
# GOOD:
bash _check_job_mem.sh 752345
```

### Anti-pattern 3: Bare python3 for anything
```bash
# BAD:
python3 _check_nb.py notebook.ipynb
# GOOD:
bash scripts/helper_scripts/run_python_cmd.sh _check_nb.py notebook.ipynb
```
