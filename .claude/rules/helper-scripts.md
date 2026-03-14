---
description: Mandatory rules for running Python, using helper scripts, planning workflow, and bioinformatics tooling — loaded every session
---

# Helper Script Rules

## Python Execution
- ALWAYS use `bash scripts/helper_scripts/run_python_cmd.sh` to run Python
- Default env: `regularizedvi`
- Other envs: `--env cell2state`, `--env hashfrag`

## NEVER Do These
- Bare `python3` or `python`
- `conda run -n ... python`
- Manual env activation (`PYTHONNOUSERSITE=TRUE /software/.../bin/python`)
- `run_python_cmd.sh -c "..."` for inspection/analysis tasks
- Piping Python output through shell tools (`| grep`, `| head`)
- Sleep + chain + pipe for job monitoring
- Complex multi-command chains (e.g. `bjobs X Y 2>&1 | head -5; echo "---"; ls -lt ...`) — use skills instead (`/check-job`)

## Inspection Routing
| Need | Script |
|------|--------|
| Notebook structure/search/errors/progress | `_inspect_notebook.py` |
| Papermill params | `_extract_params.py` |
| h5ad files | `inspect_h5ad.py` |
| Pickle files | `inspect_pickle.py` |
| Parquet files | `check_parquet.py` |
| Job status/memory | `_check_alive.sh`, `_check_job_mem.sh`, `_monitor_process.sh` |

All Python scripts via: `bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/SCRIPT [args]`
Job scripts are at project root: `bash _check_alive.sh PID`

## Fallback: When No Script Covers the Use Case
1. Check existing helper scripts for closest match
2. Launch an Agent subagent to extend or create a new script
3. Use the new script via `run_python_cmd.sh`
4. NEVER fall back to inline Python for inspection/analysis

## Planning Workflow — MANDATORY
- **Always summarise the request and ask clarifying questions BEFORE starting implementation**
- Be active and persistent in clarifying implementation decisions — especially:
  - Data filtering thresholds (QC cutoffs, minimum cells/genes)
  - Hyperparameters (learning rate, n_epochs, n_hidden, n_latent)
  - Normalisation and scaling choices
  - Prior specifications and regularisation strengths
  - Matching obs_names/var_names between AnnData objects (e.g. RNA vs ATAC)
  - Choice between multiple bioinformatics tools for the same task
- Do NOT assume defaults for scientific parameters — always ask
- When similar analysis/implementation is requested, go through every option and ask clarifying questions whether descisions are the same.

## Bioinformatics Tooling Fallback
- When shell commands, CLI tools, or API calls for bioinformatics tasks fail or aren't available in the environment (e.g. genome database queries, sequence retrieval, file format conversions, annotation lookups), do NOT just give up
- Search online bioinformatics resources (documentation, forums, Bioconductor/Biostars/etc.) for the correct approach, alternative tools, or Python-based workarounds before reporting failure

## Data Type Rules
- Default float dtype: float32
- Integer dtypes are nuanced — always ask (e.g. chromosome position → int64, some layers → uint16)
