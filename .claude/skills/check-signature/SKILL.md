---
name: check-signature
description: "Use when checking a function or class signature, docstring, or verifying a specific import works. TRIGGER when: user asks what arguments a function takes, wants to see a docstring, or wants to verify a specific function/class can be imported. Lighter than inspect-package-source (no full source code)."
user-invocable: true
argument-hint: "[MODULE.FUNC] [MODULE.CLASS.METHOD] ..."
---

**MANDATORY**: All Python scripts MUST be run via `bash scripts/helper_scripts/run_python_cmd.sh`. NEVER run Python scripts directly with `python3`, `python`, or bare script paths.

# Check Signature

Show function/class signature and docstring without printing full source code. Also verifies the import resolves.

## Usage

```bash
# Single function
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_signature.py regularizedvi.RegularizedVI.train

# Multiple targets
bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/check_signature.py torch.nn.Linear numpy.array

# Different env
bash scripts/helper_scripts/run_python_cmd.sh --env cell2state scripts/claude_helper_scripts/check_signature.py cell2state.models.Cell2StateModel.train
```

## Output

For each target shows:
- Type (function, class, method, etc.)
- Signature with parameter names and defaults
- Full docstring

Exits with code 1 if any targets cannot be resolved.

## When to use this vs other skills

| Need | Skill |
|------|-------|
| "What args does X take?" | `/check-signature` (this) |
| "Show me the full source code of X" | `/inspect-package-source` |
| "Can all imports in this file resolve?" | `/check-imports` |
