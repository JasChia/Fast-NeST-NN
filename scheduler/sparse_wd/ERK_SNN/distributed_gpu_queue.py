#!/usr/bin/env python3
"""
Distributed GPU Queue Manager for Multi-Node Execution
Works across multiple nodes with a shared file system using file-based locking
to ensure no job is executed multiple times.

Usage:
    # On each node:
    python distributed_gpu_queue.py jobs/ERK_SNN_advanced_jobs.json \
        --max-gpus 4 \
        --node-id node1 \
        --lock-dir shared/locks \
        --status-dir shared/status \
        --log-file logs/distributed_queue.log

    L5: nohup python -u distributed_gpu_queue.py jobs/ERK_SNN_advanced_jobs.json --max-gpus 19 --node-id l5 --lock-dir shared/locks --status-dir shared/status --log-file logs/distributed_queue_l5.log > logs/distributed_queue_l5.out 2>&1 &
    L4: nohup python distributed_gpu_queue.py jobs/ERK_SNN_advanced_jobs.json --max-gpus 8 --node-id l4 --lock-dir shared/locks --status-dir shared/status --log-file logs/distributed_queue_l4.log > logs/distributed_queue_l4.out 2>&1 &
    L6: nohup python distributed_gpu_queue.py jobs/ERK_SNN_advanced_jobs.json --max-gpus 4 --node-id l6 --lock-dir shared/locks --status-dir shared/status --log-file logs/distributed_queue_l6.log > logs/distributed_queue_l6.out 2>&1 &
    L11: nohup python distributed_gpu_queue.py jobs/ERK_SNN_advanced_jobs.json --max-gpus 2 --node-id l11 --lock-dir shared/locks --status-dir shared/status --log-file logs/distributed_queue_l11.log > logs/distributed_queue_l11.out 2>&1 &
    L12: nohup python distributed_gpu_queue.py jobs/ERK_SNN_advanced_jobs.json --max-gpus 2 --node-id l12 --lock-dir shared/locks --status-dir shared/status --log-file logs/distributed_queue_l12.log > logs/distributed_queue_l12.out 2>&1 &
"""

import os
import sys
import time
import json
import subprocess
import threading
import argparse
from datetime import datetime
from pathlib import Path
import signal
import logging
import re
import fcntl  # For file locking
import socket  # For hostname detection
import hashlib


class DistributedGPUQueueManager:
    def __init__(self, max_gpus=None, log_file=None, node_id=None, 
                 lock_dir=None, status_dir=None, job_file=None):
        """
        Initialize distributed GPU queue manager.
        
        Args:
            max_gpus: Maximum GPUs to use on this node
            log_file: Log file path
            node_id: Unique identifier for this node (defaults to hostname)
            lock_dir: Directory for lock files (shared filesystem)
            status_dir: Directory for job status files (shared filesystem)
            job_file: Path to job file
        """
        # Node identification
        self.node_id = node_id or socket.gethostname()
        self.hostname = socket.gethostname()
        
        # Shared filesystem directories
        self.lock_dir = lock_dir or "locks"
        self.status_dir = status_dir or "status"
        os.makedirs(self.lock_dir, exist_ok=True)
        os.makedirs(self.status_dir, exist_ok=True)
        
        # Local GPU management
        available_gpus = self._get_gpu_count()
        self.max_gpus = max_gpus or available_gpus
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set log file path
        if log_file:
            self.log_file = log_file
        else:
            self.log_file = os.path.join(self.logs_dir, f"distributed_queue_{self.node_id}.log")
        
        # Job tracking
        self.running_jobs = {}  # Local running jobs
        self.job_file = job_file
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s [{self.node_id}] - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Initialized distributed queue manager on node: {self.node_id}")
        self.logger.info(f"Available GPUs: {available_gpus}, Using: {self.max_gpus}")
        self.logger.info(f"Lock directory: {self.lock_dir}")
        self.logger.info(f"Status directory: {self.status_dir}")
    
    def _get_gpu_count(self):
        """Get the number of available GPUs on this node"""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip().split('\n'))
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("nvidia-smi not found or failed to run")
            return 0
    
    def _get_job_lock_file(self, job_id):
        """Get path to lock file for a job"""
        # Use hash to avoid filesystem issues with special characters
        job_hash = hashlib.md5(job_id.encode()).hexdigest()
        return os.path.join(self.lock_dir, f"job_{job_hash}.lock")
    
    def _get_job_status_file(self, job_id):
        """Get path to status file for a job"""
        job_hash = hashlib.md5(job_id.encode()).hexdigest()
        return os.path.join(self.status_dir, f"job_{job_hash}.json")
    
    def _acquire_job_lock(self, job_id, timeout=5):
        """
        Try to acquire exclusive lock on a job.
        Returns lock file handle if successful, None otherwise.
        """
        lock_file_path = self._get_job_lock_file(job_id)
        lock_file = None
        
        try:
            # Open lock file in append mode (create if doesn't exist)
            lock_file = open(lock_file_path, 'a')
            
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write node info to lock file
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(json.dumps({
                'node_id': self.node_id,
                'hostname': self.hostname,
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat()
            }))
            lock_file.flush()
            
            self.logger.debug(f"Acquired lock for job {job_id}")
            return lock_file
            
        except (IOError, BlockingIOError):
            # Lock is held by another process
            if lock_file:
                lock_file.close()
            return None
        except Exception as e:
            self.logger.error(f"Error acquiring lock for {job_id}: {e}")
            if lock_file:
                lock_file.close()
            return None
    
    def _release_job_lock(self, lock_file):
        """Release a job lock"""
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception as e:
                self.logger.error(f"Error releasing lock: {e}")
    
    def _is_job_completed(self, job_id):
        """Check if job is already completed"""
        status_file = self._get_job_status_file(job_id)
        if not os.path.exists(status_file):
            return False
        
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
                return status.get('status') == 'completed'
        except Exception as e:
            self.logger.warning(f"Error reading status file for {job_id}: {e}")
            return False
    
    def _is_job_running(self, job_id):
        """Check if job is currently running (by checking lock file)"""
        lock_file_path = self._get_job_lock_file(job_id)
        if not os.path.exists(lock_file_path):
            return False
        
        try:
            # Try to acquire lock - if we can't, it's being held (running)
            with open(lock_file_path, 'r') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # We got the lock, so it's not running
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return False
                except (IOError, BlockingIOError):
                    # Lock is held, job is running
                    return True
        except Exception:
            return False
    
    def _update_job_status(self, job_id, status, **kwargs):
        """Update job status in shared status file"""
        status_file = self._get_job_status_file(job_id)
        
        status_data = {
            'job_id': job_id,
            'status': status,
            'node_id': self.node_id,
            'hostname': self.hostname,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        # Use file locking to update status atomically
        try:
            with open(status_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(status_data, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            self.logger.error(f"Error updating status for {job_id}: {e}")
    
    def _get_job_status(self, job_id):
        """Get current job status"""
        status_file = self._get_job_status_file(job_id)
        if not os.path.exists(status_file):
            return None
        
        try:
            with open(status_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                status = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return status
        except Exception as e:
            self.logger.warning(f"Error reading status for {job_id}: {e}")
            return None
    
    def find_available_gpu(self):
        """Find the next available GPU on this node"""
        for gpu in range(self.max_gpus):
            # Check if this GPU is already being used by our queue manager
            gpu_in_use = any(job_info['gpu'] == gpu for job_info in self.running_jobs.values())
            if gpu_in_use:
                continue
            
            # Check if this GPU has other processes running
            if self._is_gpu_available(gpu):
                return gpu
        return None
    
    def _is_gpu_available(self, gpu_id):
        """Check if a specific GPU is available on this node"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid', 
                '--format=csv,noheader,nounits', '-i', str(gpu_id)
            ], capture_output=True, text=True, check=True)
            return not result.stdout.strip()
        except subprocess.CalledProcessError:
            return False
    
    def load_jobs_from_json(self, json_file):
        """Load jobs from JSON file"""
        with open(json_file, 'r') as f:
            jobs_data = json.load(f)
        
        jobs = []
        for job_config in jobs_data['jobs']:
            job_id = job_config.get('id', f"job_{len(jobs) + 1}")
            
            # Skip if already completed
            if self._is_job_completed(job_id):
                self.logger.info(f"Skipping completed job: {job_id}")
                continue
            
            # Skip if currently running
            if self._is_job_running(job_id):
                self.logger.info(f"Skipping running job: {job_id}")
                continue
            
            jobs.append({
                'id': job_id,
                'command': job_config['command'],
                'priority': job_config.get('priority', 0),
                'max_retries': job_config.get('max_retries', 1),
                'retry_count': 0
            })
        
        # Sort by priority
        jobs.sort(key=lambda x: x.get('priority', 0), reverse=True)
        self.logger.info(f"Loaded {len(jobs)} pending jobs from {json_file}")
        return jobs
    
    def _create_experiment_directory(self, command):
        """Create organized experiment directory"""
        out_match = re.search(r'-output_dir\s+(\S+)', command)
        if out_match:
            experiment_dir = out_match.group(1)
            os.makedirs(experiment_dir, exist_ok=True)
            return experiment_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fallback_dir = os.path.join("results", f"experiment_{timestamp}")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir
    
    def start_job(self, job_info, gpu_id):
        """Start a job on a specific GPU with distributed locking"""
        job_id = job_info['id']
        command = job_info['command']
        
        # Try to acquire lock for this job
        lock_file = self._acquire_job_lock(job_id)
        if not lock_file:
            self.logger.debug(f"Could not acquire lock for {job_id}, skipping")
            return False
        
        try:
            # Double-check status after acquiring lock
            if self._is_job_completed(job_id):
                self.logger.info(f"Job {job_id} was completed by another node, releasing lock")
                self._release_job_lock(lock_file)
                return False
            
            # Update status to running
            self._update_job_status(job_id, 'running', gpu_id=gpu_id, node_id=self.node_id)
            
            # Create experiment directory
            experiment_dir = self._create_experiment_directory(command)
            
            # Modify command to redirect output
            if '>' in command:
                log_match = re.search(r'>\s*(\S+)', command)
                if log_match:
                    log_file = log_match.group(1)
                    command = re.sub(r'>\s*\S+', f'> {experiment_dir}/{log_file}', command)
            
            # Wrap command with directory change and CUDA_VISIBLE_DEVICES
            # Only expose the specific GPU being used to avoid conflicts
            # When CUDA_VISIBLE_DEVICES is set to one GPU, it appears as GPU 0
            if '-cuda' not in command:
                command += f" -cuda 0"
            else:
                # Replace existing -cuda flag with 0 since only one GPU is visible
                command = re.sub(r'-cuda\s+\d+', '-cuda 0', command)
            
            conda_command = (
                f"cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/sparse_wd/ERK_SNN && "
                f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
                f"{command}"
            )
            
            self.logger.info(f"Starting {job_id} on GPU {gpu_id} (node: {self.node_id})")
            
            # Start process
            process = subprocess.Popen(
                conda_command,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Store job info (but don't store lock_file - we'll release it when done)
            self.running_jobs[job_id] = {
                'pid': process.pid,
                'gpu': gpu_id,
                'start_time': time.time(),
                'command': command,
                'process': process,
                'lock_file': lock_file  # Keep reference to release later
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job_id}: {e}")
            self._release_job_lock(lock_file)
            return False
    
    def check_job_status(self):
        """Check status of running jobs"""
        completed_jobs = []
        
        for job_id, job_info in list(self.running_jobs.items()):
            process = job_info['process']
            lock_file = job_info.get('lock_file')
            
            if process.poll() is not None:
                # Process finished
                stdout, stderr = process.communicate()
                exit_code = process.returncode
                
                duration = time.time() - job_info['start_time']
                
                if exit_code == 0:
                    self._update_job_status(
                        job_id, 'completed',
                        exit_code=exit_code,
                        duration=duration,
                        node_id=self.node_id
                    )
                    self.logger.info(f"Job {job_id} completed successfully on {self.node_id}")
                else:
                    self._update_job_status(
                        job_id, 'failed',
                        exit_code=exit_code,
                        duration=duration,
                        node_id=self.node_id,
                        error=stderr.decode('utf-8')[:2000]  # First 2000 chars
                    )
                    self.logger.error(f"Job {job_id} failed with exit code {exit_code}")
                
                # Release lock
                if lock_file:
                    self._release_job_lock(lock_file)
                
                completed_jobs.append(job_id)
        
        # Remove completed jobs
        for job_id in completed_jobs:
            del self.running_jobs[job_id]
    
    def run_queue(self):
        """Main queue processing loop"""
        self.logger.info(f"Starting distributed GPU queue manager on node: {self.node_id}")
        
        # Load jobs
        if not self.job_file:
            self.logger.error("No job file specified")
            return
        
        jobs = self.load_jobs_from_json(self.job_file)
        self.logger.info(f"Total pending jobs: {len(jobs)}")
        
        job_index = 0
        shutdown = False
        
        while not shutdown:
            # Check for completed jobs
            self.check_job_status()
            
            # Start new jobs if GPUs are available
            while job_index < len(jobs) and len(self.running_jobs) < self.max_gpus:
                available_gpu = self.find_available_gpu()
                if available_gpu is not None:
                    job_info = jobs[job_index]
                    
                    # Try to start job (will acquire lock internally)
                    if self.start_job(job_info, available_gpu):
                        job_index += 1
                        self.logger.info(f"Queued jobs remaining: {len(jobs) - job_index}")
                    else:
                        # Couldn't acquire lock, skip this job for now
                        job_index += 1
                else:
                    break
            
            # Reload jobs periodically to pick up any new jobs
            if job_index >= len(jobs):
                time.sleep(30)  # Wait longer when no jobs
                jobs = self.load_jobs_from_json(self.job_file)
                job_index = 0
            else:
                time.sleep(10)  # Check every 10 seconds
        
        # Wait for remaining jobs
        while self.running_jobs:
            self.check_job_status()
            time.sleep(5)
        
        self.logger.info("Distributed queue manager shutting down")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        # Release all locks
        for job_id, job_info in list(self.running_jobs.items()):
            lock_file = job_info.get('lock_file')
            if lock_file:
                self._release_job_lock(lock_file)
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Distributed GPU Queue Manager for Multi-Node Execution'
    )
    parser.add_argument('job_file', help='Job file (JSON format)')
    parser.add_argument('--max-gpus', type=int, help='Maximum number of GPUs to use on this node')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--node-id', help='Unique node identifier (defaults to hostname)')
    parser.add_argument('--lock-dir', help='Directory for lock files (shared filesystem)', 
                       default='locks')
    parser.add_argument('--status-dir', help='Directory for status files (shared filesystem)',
                       default='status')
    
    args = parser.parse_args()
    
    manager = DistributedGPUQueueManager(
        max_gpus=args.max_gpus,
        log_file=args.log_file,
        node_id=args.node_id,
        lock_dir=args.lock_dir,
        status_dir=args.status_dir,
        job_file=args.job_file
    )
    
    try:
        manager.run_queue()
    except KeyboardInterrupt:
        manager.logger.info("Interrupted by user")
        manager._signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()

