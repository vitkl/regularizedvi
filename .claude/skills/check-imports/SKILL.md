---
name: check-imports
description: "Use when checking that Python imports resolve correctly. TRIGGER when: user asks to verify imports, check for missing dependencies, or validate that all imported modules are installed. Works on both .py files and .ipynb notebooks."
user-invocable: true
argument-hint: "[FILE1.py] [FILE2.ipynb] ..."
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Check Imports

Verify that all import statements in Python files and notebook cells can be resolved without executing them.

## Usage

```bash
# Check imports in Python files
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_imports.py script.py

# Check imports in Jupyter notebooks
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_imports.py notebook.ipynb

# Check multiple files
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_imports.py script.py notebook.ipynb

# Use different env to check if imports resolve there
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state scripts/claude_helper_scripts/check_imports.py script.py
```

## What it does

- Extracts all `import X` and `from X import Y` statements using AST parsing
- Checks if the top-level package can be found via `importlib.util.find_spec()`
- For notebooks: strips Jupyter magics before parsing
- Deduplicates imports, reports first location for each
- Exits with code 1 if any imports are missing
