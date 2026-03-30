---
name: qc-filter-summary
description: "Use when checking QC filter thresholds on h5ad files or summarising cell filtering. TRIGGER when: user asks about QC metrics, cell counts after filtering, scrublet doublet scores, ATAC QC, or per-dataset/per-batch quality summaries. Fast h5py-based obs loading with optional scrublet and ATAC QC joins."
user-invocable: true
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# QC Filter Summary

Fast per-dataset QC filter summary using h5py (obs-only loading, no full anndata).

## Script

`scripts/claude_helper_scripts/qc_filter_summary.py`

## Usage

```bash
# Defaults: counts>1100, genes>700, counts<80000, genes<10000, mt<0.20
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad

# Custom thresholds
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad --min-counts 1500 --min-genes 800

# With scrublet doublet scores
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad --scrublet scrublet_results.csv

# With ATAC QC metrics
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad --atac-qc atac_qc_metrics.csv

# Both scrublet + ATAC
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad --scrublet scrublet.csv --atac-qc atac_qc.csv --show-unmatched

# Per-batch breakdown
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad --per-batch

# Single dataset
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/qc_filter_summary.py FILE.h5ad --dataset bone_marrow
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--min-counts` | 1100 | Min total_counts |
| `--max-counts` | 80000 | Max total_counts |
| `--min-genes` | 700 | Min n_genes |
| `--max-genes` | 10000 | Max n_genes |
| `--max-mt` | 0.20 | Max mt_frac |
| `--scrublet CSV` | None | Scrublet results CSV (joins by index, NaN->0.0) |
| `--max-doublet` | 0.20 | Max doublet_score |
| `--atac-qc CSV` | None | ATAC QC metrics CSV (joins by index) |
| `--min-fragments` | 2500 | Min total_fragments |
| `--max-fragments` | 80000 | Max total_fragments |
| `--per-batch` | False | Group by batch instead of dataset |
| `--dataset NAME` | None | Filter to single dataset |
| `--show-unmatched` | False | Show join match diagnostics |
