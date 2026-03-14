---
name: check-job
description: Use when checking bsub/LSF job status, tailing logs, checking memory, monitoring processes, or watching papermill memory usage.
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

### `_monitor_process.sh` - Memory watchdog (papermill + children)

Monitors RSS of a process AND all its children (e.g., papermill spawns a Jupyter kernel).
Kills with SIGKILL (-9) + children if threshold exceeded.

```bash
# Watch papermill PID, kill if total RSS > 700GB, check every 60s, for up to 24h
bash _monitor_process.sh <PID> 700

# Custom interval (30s) and duration (12h)
bash _monitor_process.sh <PID> 700 30 43200
```

## NEVER do this

```bash
# BAD: inline memory monitoring loops
while kill -0 $PID; do RSS=$(ps ...); ...; sleep 60; done

# BAD: sleep + chain + pipe for job monitoring
sleep 360 && tail -3 /path/752345.err 2>/dev/null && echo "---" && bjobs 752345 2>&1 | head -3
```
