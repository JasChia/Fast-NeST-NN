#!/usr/bin/env python3
"""
Cleanup script for failed experiments.
Deletes three things for each job:
1. Lock file in lock_dir (shared/locks/job_<hash>.lock)
2. Status file in status_dir (shared/status/job_<hash>.json)
3. Experiment directory in results/ (results/D<drug_id>/<job_id>)

This allows failed experiments to be rerun.

Usage:
    # Clean up a single experiment
    python cleanup_failed_experiment.py --job-id D5_0
    
    # Clean up multiple experiments
    python cleanup_failed_experiment.py --job-id D5_0 D5_1 D5_2
    
    # Clean up all failed experiments (status='failed')
    python cleanup_failed_experiment.py --all-failed
    
    # Clean up experiments matching a pattern
    python cleanup_failed_experiment.py --pattern "D5_*" --job-file jobs/r_sparse_nn_advanced_jobs.json
    
    # Clean up with custom lock/status directories
    python cleanup_failed_experiment.py --job-id D5_0 --lock-dir shared/locks --status-dir shared/status
    
    # Dry run (show what would be deleted without actually deleting)
    python cleanup_failed_experiment.py --job-id D5_0 D5_1 --dry-run
"""

import os
import sys
import json
import argparse
import hashlib
import shutil
import fnmatch
import re
from pathlib import Path


class ExperimentCleaner:
    def __init__(self, lock_dir="shared/locks", status_dir="shared/status", dry_run=False):
        """
        Initialize experiment cleaner.
        
        Args:
            lock_dir: Directory containing lock files
            status_dir: Directory containing status files
            dry_run: If True, only show what would be deleted without actually deleting
        """
        self.lock_dir = lock_dir
        self.status_dir = status_dir
        self.dry_run = dry_run
        
        # Ensure directories exist (for reading)
        if not os.path.exists(self.lock_dir):
            print(f"Warning: Lock directory does not exist: {self.lock_dir}")
        if not os.path.exists(self.status_dir):
            print(f"Warning: Status directory does not exist: {self.status_dir}")
    
    def _get_job_hash(self, job_id):
        """Get MD5 hash for a job ID (same as distributed_gpu_queue.py)"""
        return hashlib.md5(job_id.encode()).hexdigest()
    
    def _get_job_lock_file(self, job_id):
        """Get path to lock file for a job"""
        job_hash = self._get_job_hash(job_id)
        return os.path.join(self.lock_dir, f"job_{job_hash}.lock")
    
    def _get_job_status_file(self, job_id):
        """Get path to status file for a job"""
        job_hash = self._get_job_hash(job_id)
        return os.path.join(self.status_dir, f"job_{job_hash}.json")
    
    def _read_status_file(self, job_id):
        """Read and return status file contents"""
        status_file = self._get_job_status_file(job_id)
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read status file {status_file}: {e}")
                return None
        return None
    
    def _get_experiment_dir_from_job_id(self, job_id):
        """Get experiment directory path from job_id (e.g., D5_0 -> results/D5/D5_0)"""
        # Job IDs are like D5_0, D127_1, etc.
        # Experiment directories are like results/D5/D5_0, results/D127/D127_1, etc.
        match = re.match(r'^D(\d+)_(\d+)$', job_id)
        if match:
            drug_id = match.group(1)
            exp_id = match.group(2)
            return os.path.join("results", f"D{drug_id}", job_id)
        return None
    
    def cleanup_job(self, job_id):
        """
        Clean up a single job by deleting:
        1. Lock file in lock_dir
        2. Status file in status_dir
        3. Experiment directory in results/
        
        Args:
            job_id: Job ID to clean up (e.g., "D5_0")
        
        Returns:
            dict with cleanup results
        """
        results = {
            'job_id': job_id,
            'lock_file_deleted': False,
            'status_file_deleted': False,
            'experiment_dir_deleted': False,
            'errors': []
        }
        
        # Get experiment directory from job_id
        experiment_dir = self._get_experiment_dir_from_job_id(job_id)
        
        # Delete lock file
        lock_file = self._get_job_lock_file(job_id)
        if os.path.exists(lock_file):
            try:
                if not self.dry_run:
                    os.remove(lock_file)
                print(f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} lock file: {lock_file}")
                results['lock_file_deleted'] = True
            except OSError as e:
                error_msg = f"Could not delete lock file {lock_file}: {e}"
                print(f"Error: {error_msg}")
                results['errors'].append(error_msg)
        else:
            print(f"Lock file does not exist: {lock_file}")
        
        # Delete status file
        status_file = self._get_job_status_file(job_id)
        if os.path.exists(status_file):
            try:
                if not self.dry_run:
                    os.remove(status_file)
                print(f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} status file: {status_file}")
                results['status_file_deleted'] = True
            except OSError as e:
                error_msg = f"Could not delete status file {status_file}: {e}"
                print(f"Error: {error_msg}")
                results['errors'].append(error_msg)
        else:
            print(f"Status file does not exist: {status_file}")
        
        # Delete experiment directory
        if experiment_dir:
            if os.path.exists(experiment_dir):
                try:
                    if not self.dry_run:
                        shutil.rmtree(experiment_dir)
                    print(f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleted'} experiment directory: {experiment_dir}")
                    results['experiment_dir_deleted'] = True
                except OSError as e:
                    error_msg = f"Could not delete experiment directory {experiment_dir}: {e}"
                    print(f"Error: {error_msg}")
                    results['errors'].append(error_msg)
            else:
                print(f"Experiment directory does not exist: {experiment_dir}")
        else:
            print(f"Warning: Could not determine experiment directory for job {job_id} (expected format: D<drug_id>_<exp_id>)")
        
        return results
    
    def get_all_failed_jobs(self):
        """Get list of all job IDs with status='failed'"""
        failed_jobs = []
        if not os.path.exists(self.status_dir):
            return failed_jobs
        
        for filename in os.listdir(self.status_dir):
            if filename.startswith('job_') and filename.endswith('.json'):
                status_file = os.path.join(self.status_dir, filename)
                try:
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                        if status_data.get('status') == 'failed':
                            job_id = status_data.get('job_id')
                            if job_id:
                                failed_jobs.append(job_id)
                except (json.JSONDecodeError, IOError):
                    continue
        
        return failed_jobs
    
    def get_all_job_ids_from_file(self, job_file):
        """Get all job IDs from a job JSON file"""
        if not os.path.exists(job_file):
            return []
        
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
                return [job.get('id') for job in job_data.get('jobs', []) if job.get('id')]
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading job file {job_file}: {e}")
            return []
    
    def match_job_ids(self, pattern, job_file=None):
        """Get job IDs matching a pattern (supports wildcards)"""
        if job_file:
            all_job_ids = self.get_all_job_ids_from_file(job_file)
        else:
            # Try to find job IDs from status files
            all_job_ids = []
            if os.path.exists(self.status_dir):
                for filename in os.listdir(self.status_dir):
                    if filename.startswith('job_') and filename.endswith('.json'):
                        status_file = os.path.join(self.status_dir, filename)
                        try:
                            with open(status_file, 'r') as f:
                                status_data = json.load(f)
                                job_id = status_data.get('job_id')
                                if job_id:
                                    all_job_ids.append(job_id)
                        except (json.JSONDecodeError, IOError):
                            continue
        
        return [job_id for job_id in all_job_ids if fnmatch.fnmatch(job_id, pattern)]


def main():
    parser = argparse.ArgumentParser(
        description='Clean up failed experiments to allow rerunning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--job-id', nargs='+', help='Job ID(s) to clean up')
    parser.add_argument('--pattern', help='Pattern to match job IDs (e.g., "D5_*")')
    parser.add_argument('--all-failed', action='store_true', 
                       help='Clean up all jobs with status="failed"')
    parser.add_argument('--job-file', help='Path to job JSON file (for pattern matching)')
    parser.add_argument('--lock-dir', default='shared/locks', 
                       help='Directory containing lock files (default: shared/locks)')
    parser.add_argument('--status-dir', default='shared/status', 
                       help='Directory containing status files (default: shared/status)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    # Determine which jobs to clean up
    job_ids = []
    
    if args.all_failed:
        cleaner = ExperimentCleaner(args.lock_dir, args.status_dir, args.dry_run)
        job_ids = cleaner.get_all_failed_jobs()
        if not job_ids:
            print("No failed jobs found.")
            return
        print(f"Found {len(job_ids)} failed jobs")
    elif args.pattern:
        cleaner = ExperimentCleaner(args.lock_dir, args.status_dir, args.dry_run)
        job_ids = cleaner.match_job_ids(args.pattern, args.job_file)
        if not job_ids:
            print(f"No jobs found matching pattern: {args.pattern}")
            return
        print(f"Found {len(job_ids)} jobs matching pattern: {args.pattern}")
    elif args.job_id:
        job_ids = args.job_id
    else:
        parser.print_help()
        print("\nError: Must specify --job-id, --pattern, or --all-failed")
        sys.exit(1)
    
    # Clean up each job
    cleaner = ExperimentCleaner(args.lock_dir, args.status_dir, args.dry_run)
    
    if args.dry_run:
        print("\n=== DRY RUN MODE - No files will be deleted ===\n")
    
    all_results = []
    for job_id in job_ids:
        print(f"\n{'='*60}")
        print(f"Cleaning up job: {job_id}")
        print(f"{'='*60}")
        result = cleaner.cleanup_job(job_id)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs processed: {len(all_results)}")
    print(f"Lock files {'would be deleted' if args.dry_run else 'deleted'}: {sum(1 for r in all_results if r['lock_file_deleted'])}")
    print(f"Status files {'would be deleted' if args.dry_run else 'deleted'}: {sum(1 for r in all_results if r['status_file_deleted'])}")
    print(f"Experiment directories {'would be deleted' if args.dry_run else 'deleted'}: {sum(1 for r in all_results if r['experiment_dir_deleted'])}")
    
    errors = [e for r in all_results for e in r['errors']]
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    
    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to actually delete files.")


if __name__ == '__main__':
    main()

