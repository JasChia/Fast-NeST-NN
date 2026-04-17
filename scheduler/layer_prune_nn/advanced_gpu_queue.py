#!/usr/bin/env python3
"""
Advanced GPU Queue Manager for Layer Pruned Fully Connected Neural Network Hyperparameter Tuning
Supports job queuing, GPU monitoring, and automatic job distribution

Run with: nohup python advanced_gpu_queue.py jobs/layer_prune_nn_advanced_jobs.json --max-gpus 16 --log-file logs/gpu_queue_manager.log > logs/queue_output.log 2>&1 &
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

class GPUQueueManager:
    def __init__(self, max_gpus=None, log_file=None):
        # Use all available GPUs but limit CUDA_VISIBLE_DEVICES to 16 for PyTorch compatibility
        available_gpus = self._get_gpu_count()
        self.max_gpus = max_gpus or available_gpus
        
        if available_gpus > 16:
            print(f"Info: System has {available_gpus} GPUs. Using CUDA_VISIBLE_DEVICES to limit PyTorch to 16 GPUs.")
            print(f"Jobs will see only GPUs 0-15, but queue manager can use all available GPUs.")
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set log file path within logs directory
        if log_file:
            self.log_file = log_file
        else:
            self.log_file = os.path.join(self.logs_dir, "gpu_queue.log")
        self.running_jobs = {}  # {job_id: {'pid': pid, 'gpu': gpu, 'start_time': time, 'command': cmd}}
        self.completed_jobs = []
        self.failed_jobs = []
        self.job_queue = []
        self.shutdown = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _get_gpu_count(self):
        """Get the number of available GPUs"""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip().split('\n'))
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("nvidia-smi not found or failed to run")
            return 0
    
    def _format_duration(self, seconds):
        """Convert seconds to human-readable duration string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"
    
    def _create_experiment_directory(self, command):
        """Create organized experiment directory and return its path.
        Uses the exact path provided to -output_dir if present; otherwise falls back.
        """
        out_match = re.search(r'-output_dir\s+(\S+)', command)
        if out_match:
            experiment_dir = out_match.group(1)
            os.makedirs(experiment_dir, exist_ok=True)
            self.logger.info(f"Created experiment directory: {experiment_dir}")
            return experiment_dir
        # Fallback: timestamped directory under results/
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fallback_dir = os.path.join("results", f"experiment_{timestamp}")
        os.makedirs(fallback_dir, exist_ok=True)
        self.logger.warning(f"Could not parse experiment info, using fallback directory: {fallback_dir}")
        return fallback_dir
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown = True
        self._cleanup_jobs()
    
    def _cleanup_jobs(self):
        """Clean up running jobs on shutdown"""
        for job_id, job_info in self.running_jobs.items():
            try:
                os.kill(job_info['pid'], signal.SIGTERM)
                self.logger.info(f"Terminated job {job_id} (PID: {job_info['pid']})")
            except ProcessLookupError:
                pass
    
    def is_gpu_available(self, gpu_id):
        """Check if a specific GPU is available"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid', 
                '--format=csv,noheader,nounits', '-i', str(gpu_id)
            ], capture_output=True, text=True, check=True)
            return not result.stdout.strip()
        except subprocess.CalledProcessError:
            return False
    
    def find_available_gpu(self):
        """Find the next available GPU"""
        for gpu in range(self.max_gpus):
            # Check if this GPU is already being used by our queue manager
            gpu_in_use = any(job_info['gpu'] == gpu for job_info in self.running_jobs.values())
            if gpu_in_use:
                self.logger.debug(f"GPU {gpu} is in use by queue manager")
                continue
            
            # Check if this GPU has other processes running
            if self.is_gpu_available(gpu):
                self.logger.debug(f"Found available GPU: {gpu}")
                return gpu
            else:
                self.logger.debug(f"GPU {gpu} has other processes running")
        self.logger.debug("No available GPUs found")
        return None
    
    def load_jobs_from_file(self, job_file):
        """Load jobs from a text file"""
        with open(job_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    self.job_queue.append({
                        'id': f"job_{line_num}",
                        'command': line,
                        'line_number': line_num
                    })
        
        self.logger.info(f"Loaded {len(self.job_queue)} jobs from {job_file}")
    
    def load_jobs_from_json(self, json_file):
        """Load jobs from a JSON file with more detailed configuration"""
        with open(json_file, 'r') as f:
            jobs_data = json.load(f)
        
        for job_config in jobs_data['jobs']:
            self.job_queue.append({
                'id': job_config.get('id', f"job_{len(self.job_queue) + 1}"),
                'command': job_config['command'],
                'priority': job_config.get('priority', 0),
                'max_retries': job_config.get('max_retries', 1),
                'retry_count': 0
            })
        
        # Sort by priority (higher priority first)
        self.job_queue.sort(key=lambda x: x.get('priority', 0), reverse=True)
        self.logger.info(f"Loaded {len(self.job_queue)} jobs from {json_file}")
    
    def start_job(self, job_info, gpu_id):
        """Start a job on a specific GPU"""
        command = job_info['command']
        
        # Extract experiment info and create organized directory structure
        experiment_dir = self._create_experiment_directory(command)
        
        # Modify command to redirect output to experiment directory
        if '>' in command:
            # Extract the log file name from the command
            log_match = re.search(r'>\s*(\S+)', command)
            if log_match:
                log_file = log_match.group(1)
                # Change the log file path to be in the experiment directory
                command = re.sub(r'>\s*\S+', f'> {experiment_dir}/{log_file}', command)
        
        # Add CUDA device if not present (AFTER handling output redirection)
        if '-cuda' not in command:
            command += f" -cuda {gpu_id}"
        
        # Wrap command with directory change and CUDA_VISIBLE_DEVICES
        # Change to the scheduler directory where the layer_prune_nn scripts are located
        conda_command = f"cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/layer_prune_nn && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 && {command}"
        
        self.logger.info(f"Starting {job_info['id']} on GPU {gpu_id}: {command}")
        
        try:
            # Start the process using bash explicitly
            process = subprocess.Popen(
                conda_command,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.running_jobs[job_info['id']] = {
                'pid': process.pid,
                'gpu': gpu_id,
                'start_time': time.time(),
                'command': command,
                'process': process
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job_info['id']}: {e}")
            return False
    
    def check_job_status(self):
        """Check status of running jobs and move completed ones"""
        completed_jobs = []
        
        for job_id, job_info in list(self.running_jobs.items()):
            process = job_info['process']
            
            # Check if process is still running
            if process.poll() is not None:
                # Process has finished
                stdout, stderr = process.communicate()
                exit_code = process.returncode
                
                job_result = {
                    'id': job_id,
                    'gpu': job_info['gpu'],
                    'start_time': job_info['start_time'],
                    'start_time_readable': datetime.fromtimestamp(job_info['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': time.time(),
                    'end_time_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': time.time() - job_info['start_time'],
                    'duration_readable': self._format_duration(time.time() - job_info['start_time']),
                    'exit_code': exit_code,
                    'stdout': stdout.decode('utf-8'),
                    'stderr': stderr.decode('utf-8'),
                    'command': job_info['command']
                }
                
                if exit_code == 0:
                    self.completed_jobs.append(job_result)
                    duration_str = self._format_duration(job_result['duration'])
                    self.logger.info(f"Job {job_id} completed successfully on GPU {job_info['gpu']} in {duration_str}")
                else:
                    self.failed_jobs.append(job_result)
                    duration_str = self._format_duration(job_result['duration'])
                    self.logger.error(f"Job {job_id} failed with exit code {exit_code} on GPU {job_info['gpu']} after {duration_str}")
                
                completed_jobs.append(job_id)
        
        # Remove completed jobs from running_jobs
        for job_id in completed_jobs:
            del self.running_jobs[job_id]
    
    def run_queue(self):
        """Main queue processing loop"""
        self.logger.info(f"Starting GPU Queue Manager with {self.max_gpus} GPUs")
        self.logger.info(f"Total jobs in queue: {len(self.job_queue)}")
        
        while (self.job_queue or self.running_jobs) and not self.shutdown:
            # Check for completed jobs
            self.check_job_status()
            
            # Start new jobs if GPUs are available
            while self.job_queue and len(self.running_jobs) < self.max_gpus and not self.shutdown:
                available_gpu = self.find_available_gpu()
                if available_gpu is not None:
                    job_info = self.job_queue.pop(0)
                    if self.start_job(job_info, available_gpu):
                        self.logger.info(f"Queued jobs remaining: {len(self.job_queue)}")
                else:
                    break
            
            # Wait a bit before checking again
            time.sleep(10)
        
        # Wait for remaining jobs to complete
        while self.running_jobs and not self.shutdown:
            self.check_job_status()
            time.sleep(5)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of job execution"""
        self.logger.info("=" * 50)
        self.logger.info("JOB EXECUTION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total jobs processed: {len(self.completed_jobs) + len(self.failed_jobs)}")
        self.logger.info(f"Successfully completed: {len(self.completed_jobs)}")
        self.logger.info(f"Failed: {len(self.failed_jobs)}")
        
        if self.completed_jobs:
            avg_duration = sum(job['duration'] for job in self.completed_jobs) / len(self.completed_jobs)
            avg_duration_str = self._format_duration(avg_duration)
            self.logger.info(f"Average job duration: {avg_duration_str}")
        
        if self.failed_jobs:
            self.logger.info("\nFailed jobs:")
            for job in self.failed_jobs:
                self.logger.info(f"  {job['id']}: Exit code {job['exit_code']}")
        
        # Save detailed results to JSON
        results = {
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'summary': {
                'total_jobs': len(self.completed_jobs) + len(self.failed_jobs),
                'completed': len(self.completed_jobs),
                'failed': len(self.failed_jobs),
                'start_time': datetime.now().isoformat()
            }
        }
        
        results_file = os.path.join(self.logs_dir, "queue_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Detailed results saved to: {results_file}")
        
        # Flush all logger handlers to ensure immediate output
        for handler in self.logger.handlers:
            handler.flush()
        sys.stdout.flush()

def main():
    
    parser = argparse.ArgumentParser(description='Advanced GPU Queue Manager for Layer Pruned Fully Connected Neural Network Hyperparameter Tuning')
    parser.add_argument('job_file', help='Job file (txt or json)')
    parser.add_argument('--max-gpus', type=int, help='Maximum number of GPUs to use')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--format', choices=['txt', 'json'], 
                       help='Job file format (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Auto-detect format if not specified
    if not args.format:
        if args.job_file.endswith('.json'):
            args.format = 'json'
        else:
            args.format = 'txt'
    
    # Create queue manager
    manager = GPUQueueManager(max_gpus=args.max_gpus, log_file=args.log_file)
    
    # Load jobs
    if args.format == 'json':
        manager.load_jobs_from_json(args.job_file)
    else:
        manager.load_jobs_from_file(args.job_file)
    
    # Run the queue
    try:
        manager.run_queue()
    except KeyboardInterrupt:
        manager.logger.info("Interrupted by user")
        manager._cleanup_jobs()

if __name__ == "__main__":
    main() 