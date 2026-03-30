---
name: inspect-package-source
description: "Use when inspecting source code of installed Python packages. TRIGGER when: user asks to see implementation of a class, function, or method from an installed package (scvi, torch, scanpy, anndata, etc.), asks about method resolution order (MRO), or wants to understand how a library function works internally."
user-invocable: true
argument-hint: "[MODULE.CLASS_OR_FUNC] [--method METHOD] [--mro] [--attrs]"
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Inspect Package Source

Query source code of installed Python packages by dotted path.

## Usage

```bash
# Full class source
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_package_source.py scvi.data.fields.CategoricalJointObsField

# Specific method
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_package_source.py scvi.data.fields.CategoricalJointObsField --method register_field

# Limit output lines
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_package_source.py torch.nn.Linear --limit 50

# Method resolution order
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_package_source.py regularizedvi.RegularizedVI --mro

# List public attributes/methods
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/inspect_package_source.py regularizedvi.RegularizedVI --attrs
```

## Options

| Flag | Description |
|------|-------------|
| `--method/-m METHOD` | Show source of a specific method |
| `--limit/-l N` | Max lines to print |
| `--mro` | Print method resolution order for classes |
| `--attrs` | List public attributes/methods |
