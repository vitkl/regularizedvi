---
name: syntax-check
description: "Use when checking Python files or Jupyter notebooks for syntax errors. TRIGGER when: user asks to syntax-check, validate, or verify Python code or notebook cells before running, or after editing .py or .ipynb files."
user-invocable: true
argument-hint: "[FILE1.py] [FILE2.ipynb] ..."
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Syntax Check

Check Python files and Jupyter notebook cells for syntax errors without executing them.

## Usage

```bash
# Check Python files
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/syntax_check.py script.py module.py

# Check Jupyter notebooks (checks each code cell, strips % and ! magics)
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/syntax_check.py notebook.ipynb

# Mix of both
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/syntax_check.py script.py notebook.ipynb another.py
```

## What it does

- **`.py` files**: Parses AST to check for syntax errors
- **`.ipynb` files**: Extracts each code cell, strips Jupyter magic commands (`%` and `!` lines), then compiles to check syntax
- Reports per-file/per-cell results, exits with code 1 if any errors found
