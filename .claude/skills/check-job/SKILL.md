---
name: check-job
description: "Use when checking bsub/LSF job status, tailing job logs, or checking job memory usage. TRIGGER when: user asks about job status, bjobs, bsub output, .err/.out log files, or job memory (cgroup RSS)."
user-invocable: true
argument-hint: "[JOB_ID] [LOG_DIR]"
---

# Check Job

Use these scripts for bsub/LSF job monitoring. Pure bash - no Python needed. Scripts at project root.

## Scripts

### `_check_alive.sh` - Check if process is alive

```bash
bash _check_alive.sh PID
```

### `_check_job_mem.sh` - Job memory usage (cgroup)

```bash
bash _check_job_mem.sh JOB_ID
```

## NEVER do this

```bash
# BAD: sleep + chain + pipe for job monitoring
sleep 360 && tail -3 /path/752345.err 2>/dev/null && echo "---" && bjobs 752345 2>&1 | head -3
```
