---
name: check-job
description: Use when checking bsub/LSF job status, tailing logs, checking memory, monitoring processes.
user-invocable: false
allowed-tools: Bash(bash _check_alive.sh:*), Bash(bash _check_job_mem.sh:*), Bash(bash _monitor_process.sh:*)
---

# Check Job

Use these scripts for bsub/LSF job monitoring. Pure bash - no Python needed. These are at the project root.

## Scripts

### `_check_alive.sh` - Check if process is alive

```bash
bash _check_alive.sh PID
```

### `_check_job_mem.sh` - Memory usage

```bash
bash _check_job_mem.sh JOB_ID
```

### `_monitor_process.sh` - Process monitoring

```bash
bash _monitor_process.sh JOB_ID
```

## NEVER do this

```bash
# BAD: sleep + chain + pipe for job monitoring
sleep 360 && tail -3 /path/752345.err 2>/dev/null && echo "---" && bjobs 752345 2>&1 | head -3
```
