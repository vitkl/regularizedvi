# scripts/immune_integration_v2/

Phase 0 helper scripts for the v2 immune integration onboarding. See [.claude/plans/implement-these-steps-in-tranquil-parasol.md](../../.claude/plans/implement-these-steps-in-tranquil-parasol.md) for the full plan and [.claude/plans/incorporating-immune-cells-from-abundant-pixel.md](../../.claude/plans/incorporating-immune-cells-from-abundant-pixel.md) for the parent plan.

## Scripts

| Script | Purpose | Step |
|---|---|---|
| `build_unified_annotations.py` | HTAN RNA∩ATAC inner-join on `(piece_id_stripped, barcode)` | 0.1 |
| `download_htan_sample_lookup.py` | HTAN Ding-lab donor lookup CSV (URL inline-pinned) | 0.2 |
| `inspect_gbm_space_h5ad.py` | h5py-only obs dump of 59 GB GBM-Space h5ad | 0.3 |
| `extract_lung_smoking_metadata.R` | Seurat RDS → meta.data CSV (Slurm, `seurat` env) | 0.4 |
| `submit_lung_smoking_metadata.sh` | Slurm submission wrapper for 0.4 | 0.4 |
| `inspect_dataset_annotations.py` | Cell-type vocab review for 6 labelled datasets | 0.5a/0.5b |
| `inspect_dataset_metadata.py` | Column-rename review (→ v1 STANDARD_OBS_COLS) for all 11 v2 datasets | 0.6 |

## Run all scripts via the project wrapper
```bash
bash scripts/helper_scripts/run_python_cmd.sh scripts/immune_integration_v2/<script>.py [args]
```
