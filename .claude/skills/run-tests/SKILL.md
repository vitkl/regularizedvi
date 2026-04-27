---
name: run-tests
description: "Use when running regularizedvi package tests. TRIGGER when: user asks to run tests, check if tests pass, run pytest, or verify code changes."
user-invocable: true
argument-hint: "[pytest args...]"
---

# Run Tests

Run regularizedvi package tests with correct conda environment (auto-detects Crick vs Sanger via `run_python_cmd.sh`).

## Usage

```bash
# All tests
bash scripts/helper_scripts/run_tests.sh tests/ -v --tb=short -q

# Specific test file
bash scripts/helper_scripts/run_tests.sh tests/test_model.py -v -s

# Run with keyword filter
bash scripts/helper_scripts/run_tests.sh tests/ -k "test_training" -v

# Stop at first failure
bash scripts/helper_scripts/run_tests.sh tests/test_model.py -x -q

# Use a different conda env
bash scripts/helper_scripts/run_tests.sh --env cell2state tests/
```

Passes all arguments directly to pytest.
