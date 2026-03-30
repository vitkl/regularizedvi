---
name: monitor-process
description: "Use when monitoring a local process memory usage or setting up a memory watchdog. TRIGGER when: user asks to monitor a PID, watch process memory (RSS), set a memory kill threshold, or monitor papermill/Jupyter kernel memory usage."
user-invocable: true
argument-hint: "[PID] [MAX_GB] [INTERVAL_SEC] [DURATION_SEC]"
---

# Monitor Process

Memory watchdog for local processes. Monitors RSS of a process AND all its children (e.g., papermill spawns a Jupyter kernel). Kills with SIGKILL (-9) + children if memory threshold exceeded.

**WARNING**: This script can kill processes. Only use when you need memory-based OOM protection.

## Usage

```bash
# Watch papermill PID, kill if total RSS > 700GB, check every 60s, for up to 24h
bash _monitor_process.sh <PID> 700

# Custom interval (30s) and duration (12h)
bash _monitor_process.sh <PID> 700 30 43200
```

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| PID | required | Process ID to monitor |
| MAX_GB | required | Kill threshold in GB |
| INTERVAL_SEC | 60 | Check interval in seconds |
| DURATION_SEC | 86400 | Max monitoring duration (default 24h) |

## NEVER do this

```bash
# BAD: inline memory monitoring loops
while kill -0 $PID; do RSS=$(ps ...); ...; sleep 60; done
```
