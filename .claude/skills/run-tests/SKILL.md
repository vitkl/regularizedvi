---
name: run-tests
description: "Use when running package tests. TRIGGER when: user asks to run tests, check if tests pass, run pytest, or verify code changes."
user-invocable: true
argument-hint: "[pytest args...]"
---

# Run Tests

Run regularizedvi package tests with correct conda environment.

## Usage

```bash
# All tests
bash run_tests.sh tests/

# Specific test file
bash run_tests.sh tests/test_model.py -v -s

# Run with keyword filter
bash run_tests.sh tests/ -k "test_training" -v

# Verbose with short traceback
bash run_tests.sh tests/ -v --tb=short -q

# Stop at first failure
bash run_tests.sh tests/ -x
```

Passes all arguments directly to pytest.
