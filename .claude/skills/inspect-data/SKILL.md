---
name: inspect-data
description: "Use when inspecting, examining, or exploring h5ad, pickle (.pkl), or parquet files. TRIGGER when: user mentions .h5ad, .pkl, .pickle, .parquet file paths, asks about AnnData structure, obs/var columns, layer shapes, pickle keys, or parquet schema. Auto-detects file type by extension."
user-invocable: true
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Inspect Data Files

Use these scripts to inspect h5ad, pickle, and parquet files. All go through `run_python_cmd.sh` for correct Python env activation.

## h5ad files - `inspect_h5ad.py`

```bash
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py /path/to/file.h5ad
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --section obs
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --train-test
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --var-names 10 --obs-names 5
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --layer-shapes
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --grep "train" --head 10
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --obs-unique tissue dataset batch
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_h5ad.py file.h5ad --obs-unique tissue --obs-unique-counts
```

## Pickle files - `inspect_pickle.py`

```bash
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_pickle.py /path/to/file.pkl
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_pickle.py file.pkl --psm
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_pickle.py file.pkl --key "some_key"
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_pickle.py file.pkl --pattern "*affinity*"
```

## Parquet files - `check_parquet.py`

```bash
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_parquet.py /path/to/file.parquet
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_parquet.py file.parquet --no-nunique
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_parquet.py file.parquet --head-rows 10
```

## NEVER do this

```bash
python -c "import h5py; f = h5py.File('file.h5ad'); ..."
python -c "import pickle; d = pickle.load(open('file.pkl','rb')); ..."
```
