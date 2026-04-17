#!/usr/bin/env python3
"""
Monitor distributed job queue status across multiple nodes.
Shows summary of job statuses, active locks, and node activity.

python monitor_distributed_jobs.py --status-dir shared/status --lock-dir shared/locks

"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_job_statuses(status_dir):
    """Load all job status files and return both list and hash-to-status mapping"""
    statuses = []
    hash_to_status = {}  # Map hash (from filename) to status dict
    if not os.path.exists(status_dir):
        return statuses, hash_to_status
    
    for status_file in Path(status_dir).glob("job_*.json"):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
                statuses.append(status)
                # Extract hash from filename: job_{hash}.json -> {hash}
                hash_key = status_file.stem.replace('job_', '')
                hash_to_status[hash_key] = status
        except Exception as e:
            print(f"Error reading {status_file}: {e}")
    
    return statuses, hash_to_status


def check_locks(lock_dir):
    """Check active lock files"""
    locks = []
    if not os.path.exists(lock_dir):
        return locks
    
    for lock_file in Path(lock_dir).glob("job_*.lock"):
        try:
            mtime = datetime.fromtimestamp(lock_file.stat().st_mtime)
            with open(lock_file, 'r') as f:
                lock_info = json.load(f)
                lock_info['lock_file'] = str(lock_file)
                lock_info['lock_age'] = (datetime.now() - mtime).total_seconds() / 3600  # hours
                locks.append(lock_info)
        except Exception as e:
            print(f"Error reading {lock_file}: {e}")
    
    return locks


def get_job_duration(status):
    """Get job duration from status file. Returns duration in hours or None."""
    # The status file already contains 'duration' in seconds
    if 'duration' in status:
        duration_seconds = status['duration']
        if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
            return duration_seconds / 3600  # Convert seconds to hours
    return None


def print_summary(status_dir, lock_dir):
    """Print summary of job statuses"""
    statuses, hash_to_status = load_job_statuses(status_dir)
    locks = check_locks(lock_dir)
    
    # Filter out completed jobs from locks by matching hash in filename
    active_locks = []
    for lock in locks:
        lock_file_path = lock.get('lock_file', '')
        if lock_file_path:
            # Extract hash from lock filename: job_{hash}.lock -> {hash}
            lock_file = Path(lock_file_path)
            hash_key = lock_file.stem.replace('job_', '')
            
            # Check if corresponding status file exists and job is completed
            corresponding_status = hash_to_status.get(hash_key)
            if corresponding_status and corresponding_status.get('status') == 'completed':
                # Job is completed, skip this lock
                continue
        
        # Job is not completed (or no status file found), include in active locks
        active_locks.append(lock)
    
    # Count by status
    status_counts = defaultdict(int)
    node_counts = defaultdict(int)
    
    for status in statuses:
        status_counts[status.get('status', 'unknown')] += 1
        if 'node_id' in status:
            node_counts[status['node_id']] += 1
    
    # Calculate average job times using duration field from status files
    job_durations = []
    node_durations = defaultdict(list)
    
    for status in statuses:
        if status.get('status') == 'completed':
            duration = get_job_duration(status)
            if duration is not None:
                job_durations.append(duration)
                node_id = status.get('node_id')
                if node_id:
                    node_durations[node_id].append(duration)
    
    print("=" * 60)
    print("DISTRIBUTED JOB QUEUE STATUS")
    print("=" * 60)
    print(f"\nTotal Jobs Tracked: {len(statuses)}")
    print(f"\nStatus Breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:12s}: {count:4d}")
    
    print(f"\nJobs by Node:")
    for node, count in sorted(node_counts.items()):
        print(f"  {node:20s}: {count:4d}")
    
    print(f"\nActive Locks: {len(locks)}")
    print(f"Actually Running Jobs: {len(active_locks)}")
    if active_locks:
        print("\nCurrently Running Jobs:")
        for lock in active_locks:
            node = lock.get('node_id', 'unknown')
            age = lock.get('lock_age', 0)
            print(f"  Node: {node:20s} | Age: {age:6.2f} hours")
    
    # Show average job times
    if job_durations:
        avg_duration = sum(job_durations) / len(job_durations)
        print(f"\nAverage Job Duration: {avg_duration:.2f} hours ({len(job_durations)} completed jobs)")
        
        if node_durations:
            print("\nAverage Duration by Node:")
            for node in sorted(node_durations.keys()):
                node_avg = sum(node_durations[node]) / len(node_durations[node])
                node_count = len(node_durations[node])
                print(f"  {node:20s}: {node_avg:6.2f} hours ({node_count} jobs)")
    
    # Show recent failures
    failures = [s for s in statuses if s.get('status') == 'failed']
    if failures:
        print(f"\nRecent Failures ({len(failures)} total):")
        for failure in failures[-5:]:  # Last 5
            job_id = failure.get('job_id', 'unknown')
            node = failure.get('node_id', 'unknown')
            timestamp = failure.get('timestamp', 'unknown')
            print(f"  {job_id:20s} | Node: {node:15s} | {timestamp}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Monitor distributed job queue')
    parser.add_argument('--status-dir', default='status',
                       help='Directory containing status files')
    parser.add_argument('--lock-dir', default='locks',
                       help='Directory containing lock files')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode (refresh every 5 seconds)')
    
    args = parser.parse_args()
    
    if args.watch:
        import time
        try:
            while True:
                os.system('clear')  # Clear screen
                print_summary(args.status_dir, args.lock_dir)
                print("\nPress Ctrl+C to exit...")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print_summary(args.status_dir, args.lock_dir)


if __name__ == "__main__":
    main()

