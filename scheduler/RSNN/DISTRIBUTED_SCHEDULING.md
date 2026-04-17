# Distributed Multi-Node GPU Scheduling Guide

**Note (2026):** The Python `distributed_gpu_queue.py` helper described below is **not** shipped in this repository anymore. Use **Slurm**, **GNU Parallel**, or your site’s scheduler with the bundled **`jobs/r_sparse_nn_jobs.txt`** command lists. The rest of this document remains as general background on multi-node GPU scheduling.

## Overview

This guide covers options for scheduling GPU jobs across multiple nodes with a shared file system, ensuring no job runs multiple times.

## Requirements

- **Multiple compute nodes** with GPUs
- **Shared file system** (NFS, Lustre, etc.) accessible from all nodes
- **Job list** in a document/file (JSON or text format)
- **No duplicate execution** - each job runs exactly once

## Option Comparison

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **File-based Locking** (Custom) | Simple, no dependencies, works with shared FS | Less robust to node failures | Small-medium clusters, simple setup |
| **SLURM** | Industry standard, robust, feature-rich | Requires admin setup, learning curve | HPC clusters, large-scale |
| **Redis Queue** | Fast, scalable, good for many jobs | Requires Redis server | Medium-large clusters |
| **SQLite on Shared FS** | Simple, transactional, no server needed | File locking can be slow | Small-medium clusters |
| **Ray/Dask** | Distributed computing framework | Overhead, complexity | Complex distributed workloads |

## Option 1: File-Based Locking (Recommended for Your Setup)

### Implementation

I've created `distributed_gpu_queue.py` which extends your current queue manager with:

- **File locking** using `fcntl` for exclusive job execution
- **Status tracking** in shared directory
- **Automatic job skipping** if already completed/running
- **Multi-node coordination** via shared filesystem

### How It Works

1. **Lock Files**: Each job gets a lock file in shared `locks/` directory
2. **Status Files**: Job status stored in shared `status/` directory
3. **Atomic Operations**: File locking ensures only one node can claim a job
4. **Status Checking**: Nodes check status before starting jobs

### Usage

#### Step 1: Prepare Shared Directories

On the shared filesystem:

```bash
# Create shared directories (accessible from all nodes)
mkdir -p /shared/locks
mkdir -p /shared/status
```

#### Step 2: Start Queue Manager on Each Node

**Node 1:**
```bash
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/r_sparse_nn

nohup python distributed_gpu_queue.py \
    jobs/r_sparse_nn_advanced_jobs.json \
    --max-gpus 4 \
    --node-id node1 \
    --lock-dir /shared/locks \
    --status-dir /shared/status \
    --log-file logs/distributed_queue_node1.log \
    > logs/distributed_queue_node1.out 2>&1 &
```

**Node 2:**
```bash
nohup python distributed_gpu_queue.py \
    jobs/r_sparse_nn_advanced_jobs.json \
    --max-gpus 4 \
    --node-id node2 \
    --lock-dir /shared/locks \
    --status-dir /shared/status \
    --log-file logs/distributed_queue_node2.log \
    > logs/distributed_queue_node2.out 2>&1 &
```

**Node N:**
```bash
# Repeat for each node with unique --node-id
```

#### Step 3: Monitor

```bash
# Check status files
ls -la /shared/status/

# View a job status
cat /shared/status/job_<hash>.json

# Check locks
ls -la /shared/locks/

# Monitor logs
tail -f logs/distributed_queue_node*.log
```

### Features

- ✅ **No Duplicates**: File locking ensures exclusive execution
- ✅ **Fault Tolerant**: If node crashes, lock is released (file system dependent)
- ✅ **Status Tracking**: Know which jobs completed/failed
- ✅ **Automatic Retry**: Failed jobs can be retried
- ✅ **No Server Required**: Works with just shared filesystem

### Limitations

- File locking performance degrades with many nodes (100+)
- Node crashes may leave stale locks (can be cleaned manually)
- No central monitoring dashboard

---

## Option 2: SLURM (HPC Standard)

### Overview

SLURM (Simple Linux Utility for Resource Management) is the industry standard for HPC clusters.

### Setup

#### Create SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=sparse_nn
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Get job ID from SLURM
JOB_ID=$SLURM_ARRAY_TASK_ID

# Load job command from file
COMMAND=$(sed -n "${JOB_ID}p" jobs/job_commands.txt)

# Execute
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/r_sparse_nn
eval $COMMAND
```

#### Submit Jobs

```bash
# Submit as job array (one job per line in job_commands.txt)
sbatch --array=1-600 job_script.sh

# Or submit individual jobs
for i in {1..600}; do
    sbatch --array=$i job_script.sh
done
```

### Pros

- ✅ Industry standard, well-tested
- ✅ Automatic resource management
- ✅ Built-in job queuing
- ✅ Excellent monitoring tools (`squeue`, `sacct`)
- ✅ Handles node failures automatically

### Cons

- ❌ Requires SLURM installation (admin access)
- ❌ Learning curve
- ❌ May be overkill for small setups

---

## Option 3: Redis-Based Queue

### Overview

Use Redis as a distributed job queue with multiple workers.

### Setup

#### Install Redis

```bash
# On one node (or dedicated server)
sudo apt-get install redis-server
redis-server --daemonize yes
```

#### Python Worker Script

```python
import redis
import subprocess
import json
import socket

r = redis.Redis(host='redis-server-ip', port=6379, db=0)
node_id = socket.gethostname()

while True:
    # Get job from queue (blocking, atomic)
    job_data = r.blpop('job_queue', timeout=10)
    
    if job_data:
        job = json.loads(job_data[1])
        job_id = job['id']
        
        # Check if already completed
        if r.exists(f"job:{job_id}:status"):
            continue
        
        # Mark as running
        r.setex(f"job:{job_id}:status", 86400, f"running:{node_id}")
        
        # Execute job
        result = subprocess.run(job['command'], shell=True)
        
        # Mark as completed
        r.setex(f"job:{job_id}:status", 86400, "completed")
```

#### Load Jobs

```python
import redis
import json

r = redis.Redis(host='redis-server-ip', port=6379, db=0)

with open('jobs/r_sparse_nn_advanced_jobs.json') as f:
    jobs = json.load(f)['jobs']

for job in jobs:
    r.rpush('job_queue', json.dumps(job))
```

### Pros

- ✅ Fast and scalable
- ✅ Atomic operations (no race conditions)
- ✅ Good for many jobs (1000+)
- ✅ Built-in pub/sub for monitoring

### Cons

- ❌ Requires Redis server
- ❌ Additional dependency
- ❌ Network dependency

---

## Option 4: SQLite on Shared Filesystem

### Overview

Use SQLite database on shared filesystem for job tracking.

### Implementation

```python
import sqlite3
import fcntl
import os

DB_PATH = "/shared/jobs.db"

def get_job():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            command TEXT,
            status TEXT,
            node_id TEXT,
            started_at TIMESTAMP
        )
    """)
    
    # Atomic job claim
    cursor = conn.execute("""
        SELECT id, command FROM jobs 
        WHERE status = 'pending' 
        LIMIT 1
    """)
    job = cursor.fetchone()
    
    if job:
        conn.execute("""
            UPDATE jobs 
            SET status = 'running', node_id = ?, started_at = datetime('now')
            WHERE id = ? AND status = 'pending'
        """, (socket.gethostname(), job[0]))
        conn.commit()
    
    conn.close()
    return job
```

### Pros

- ✅ Simple, no server needed
- ✅ Transactional (ACID)
- ✅ Easy to query status

### Cons

- ❌ File locking can be slow with many nodes
- ❌ Not ideal for high concurrency

---

## Option 5: Ray/Dask

### Overview

Distributed computing frameworks with built-in scheduling.

### Ray Example

```python
import ray

ray.init(address='auto')  # Connect to Ray cluster

@ray.remote(num_gpus=1)
def run_job(job_command):
    subprocess.run(job_command, shell=True)
    return "completed"

# Submit all jobs
futures = [run_job.remote(job['command']) for job in jobs]

# Wait for completion
ray.get(futures)
```

### Pros

- ✅ Built for distributed computing
- ✅ Automatic load balancing
- ✅ Good monitoring

### Cons

- ❌ Overhead for simple job execution
- ❌ More complex setup
- ❌ May be overkill

---

## Recommendation for Your Use Case

**Use Option 1 (File-Based Locking)** because:

1. ✅ **Simple**: No additional dependencies or servers
2. ✅ **Works with your existing setup**: Extends current queue manager
3. ✅ **Shared filesystem**: You already have NFS
4. ✅ **Sufficient scale**: Works well for 10-50 nodes
5. ✅ **Easy to debug**: Status files are human-readable

### Quick Start

```bash
# On each node (node1, node2, ..., nodeN)
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/r_sparse_nn

# Create shared directories (once, on shared FS)
mkdir -p /shared/locks /shared/status

# Start queue manager
nohup python distributed_gpu_queue.py \
    jobs/r_sparse_nn_advanced_jobs.json \
    --max-gpus 4 \
    --node-id $(hostname) \
    --lock-dir /shared/locks \
    --status-dir /shared/status \
    > logs/distributed_$(hostname).log 2>&1 &
```

### Monitoring Script

```bash
#!/bin/bash
# monitor_jobs.sh

echo "=== Job Status Summary ==="
echo "Completed: $(ls /shared/status/*.json 2>/dev/null | xargs grep -l '"status": "completed"' | wc -l)"
echo "Running: $(ls /shared/status/*.json 2>/dev/null | xargs grep -l '"status": "running"' | wc -l)"
echo "Failed: $(ls /shared/status/*.json 2>/dev/null | xargs grep -l '"status": "failed"' | wc -l)"
echo ""
echo "=== Active Locks ==="
ls -la /shared/locks/*.lock 2>/dev/null | wc -l
```

---

## Troubleshooting

### Stale Locks

If a node crashes, locks may remain. Clean them:

```bash
# Check lock age
find /shared/locks -name "*.lock" -mtime +1 -ls

# Remove old locks (be careful!)
find /shared/locks -name "*.lock" -mtime +1 -delete
```

### Job Not Starting

1. Check if lock exists: `ls /shared/locks/job_*.lock`
2. Check status: `cat /shared/status/job_*.json`
3. Check logs: `tail -f logs/distributed_*.log`
4. Verify shared FS is accessible: `ls /shared/`

### Performance Issues

- Too many nodes checking locks: Add random delay before checking
- Slow filesystem: Use faster shared FS (Lustre instead of NFS)
- Many small files: Consider Redis option

---

## Advanced: Hybrid Approach

Combine file-based locking with a simple coordinator:

```python
# coordinator.py - runs on one node
# Periodically checks for stuck jobs and cleans up

import time
import os
from datetime import datetime, timedelta

while True:
    for lock_file in os.listdir('/shared/locks'):
        lock_path = os.path.join('/shared/locks', lock_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(lock_path))
        
        # If lock is older than 24 hours, likely stale
        if datetime.now() - mtime > timedelta(hours=24):
            # Check if process is still running
            with open(lock_path) as f:
                lock_info = json.load(f)
                pid = lock_info['pid']
                
            # Check if process exists
            if not os.path.exists(f"/proc/{pid}"):
                os.remove(lock_path)
                print(f"Removed stale lock: {lock_file}")
    
    time.sleep(3600)  # Check every hour
```

---

## Summary

For your use case (multiple nodes, shared filesystem, GPU jobs), **file-based locking** (Option 1) is the best balance of simplicity and functionality. It requires no additional infrastructure and works immediately with your existing setup.

