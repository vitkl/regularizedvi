---
description: Mandatory rules for running Python, using helper scripts, planning workflow, and bioinformatics tooling — loaded every session
---

# Helper Script Rules

## Cluster Detection
The helper scripts auto-detect which HPC cluster they are running on — you don't need to branch on cluster in your own code. Detection markers:

- **Crick** — `/nemo/lab/briscoej/` exists **and** `/camp/apps/eb/software/Miniconda3/` exists; hostname typically matches `*.thecrick.org` or `*.crick.ac.uk`.
- **Sanger farm22** — `/software/conda/users/vk7/` exists; hostname matches `farm22-*`.

Cluster-specific entry points: use `/check-job` for **LSF/bsub** (Sanger) and `/check-job-slurm` for **Slurm/sbatch** (Crick). `run_python_cmd.sh` handles cluster detection internally — one call site, both clusters.

One-time setup on Crick (before first use): `bash scripts/helper_scripts/setup_bashrc_crick.sh` — sanity-checks conda init + module availability and appends `PYTHONNOUSERSITE=TRUE` to `~/.bashrc` as a safety net.

## Python Execution
- ALWAYS use `bash scripts/helper_scripts/run_python_cmd.sh` to run Python
- Default env: `regularizedvi`
- Other envs: `--env cell2state`, `--env hashfrag` (Sanger only for now)

## NEVER Do These
- Bare `python3` or `python`
- `conda run -n ... python`
- Manual env activation (`PYTHONNOUSERSITE=TRUE /software/.../bin/python`)
- `run_python_cmd.sh -c "..."` for inspection/analysis tasks
- Piping Python output through shell tools (`| grep`, `| head`)
- Sleep + chain + pipe for job monitoring
- Complex multi-command chains (e.g. `bjobs X Y 2>&1 | head -5; echo "---"; ls -lt ...`) — use skills instead (`/check-job`)
- **NEVER use `cat > file << EOF`, `echo >`, or heredocs to create or write files** — always use the Write tool (for new files) or Edit tool (for modifications). This applies to ALL file types: scripts, plans, configs, reports, data files. If a subagent lacks Write/Edit tools, it must return the content to the parent agent which then uses Write/Edit. Subagents must NEVER work around missing Write/Edit by using Bash heredocs.
- **NEVER write fake verification reports** — all checks MUST actually execute the real tools (syntax_check.py, check_imports.py, inspect_h5ad.py, etc.). If you need to verify something, RUN the actual check — do not fabricate output.

## Inspection Routing — Global Skills (in `~/.claude/shared-skills/`)
| Need | Skill / Script |
|------|----------------|
| Slurm job status / monitoring (Crick) | `/check-job-slurm` skill → `bash ~/.claude/shared-skills/scripts/check_jobs_slurm.sh JOB_ID1 [JOB_ID2 ...]` |
| LSF/bsub job status / monitoring (Sanger) | `/check-job` skill → `bash ~/.claude/shared-skills/scripts/check_jobs.sh JOB_ID1 [JOB_ID2 ...]` |
| Notebook structure/search/errors/progress | `/inspect-notebook` skill → `bash scripts/helper_scripts/run_python_cmd.sh ~/.claude/shared-skills/scripts/_inspect_notebook.py NOTEBOOK [flags]` |
| Conversation JSONL inspection | `/inspect-conversation` skill → `python3 ~/.claude/shared-skills/scripts/inspect_conversation.py JSONL [flags]` |
| Notebook progress bars (tqdm) | `bash scripts/helper_scripts/run_python_cmd.sh ~/.claude/shared-skills/scripts/check_notebook_progress_bar.py NOTEBOOK` |
| Process alive check | `bash ~/.claude/shared-skills/scripts/_check_alive.sh PID` |
| Job memory (cgroup) | `bash ~/.claude/shared-skills/scripts/_check_job_mem.sh [--watch N]` |
| Memory watchdog | `bash ~/.claude/shared-skills/scripts/_monitor_process.sh PID MAX_GB [INTERVAL] [DURATION]` |

## Inspection Routing — Project-Specific
| Need | Script |
|------|--------|
| Papermill params | `_extract_params.py` (via `run_python_cmd.sh`) |
| h5ad files | `inspect_h5ad.py` |
| Pickle files | `inspect_pickle.py` |
| Parquet files | `check_parquet.py` |

All project Python scripts via: `bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/SCRIPT [args]`

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
- **Plan completion verification — MANDATORY**: When a plan reaches its final step (after all implementation/commits but **before** any irreversible action such as GPU job submission or `git push`), invoke the **`/verify-implementation`** skill. Do NOT launch an ad-hoc Agent subagent for this — use the skill specifically. It runs 7 parallel specialist audits (plan completion, post-plan user input, math-code matching, code structure, cross-file consistency, project-specific checks, notebook pre-submission) and produces a structured PASS/FAIL report. Only proceed past verification once the skill returns PASS (or each finding is explicitly acknowledged).
- When you are ready to present the plan but before presenting the plan to me invoke the **`/verify-plan-against-inputs`** skill (not an ad-hoc subagent). It extracts my verbatim input from the conversation JSONL and checks whether the plan accounts for every item. Only call `ExitPlanMode` after it returns ALL_COVERED (or after each gap has been addressed).

## Bioinformatics Tooling Fallback
- When shell commands, CLI tools, or API calls for bioinformatics tasks fail or aren't available in the environment (e.g. genome database queries, sequence retrieval, file format conversions, annotation lookups), do NOT just give up
- Search online bioinformatics resources (documentation, forums, Bioconductor/Biostars/etc.) for the correct approach, alternative tools, or Python-based workarounds before reporting failure

## Data Type Rules
- Default float dtype: float32
- Integer dtypes are nuanced — always ask (e.g. chromosome position → int64, some layers → uint16)
