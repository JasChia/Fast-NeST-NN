#!/usr/bin/env python3
"""
Job Generator for Uniform Random DO+DI (Direct Output + Direct Input) Sparse Neural Network Hyperparameter Tuning
Generates jobs files with drug experiments for the GPU queue manager, targeting
uniform_random_do_di_snn_hparam_tuner.py and writing outputs under results/...

Should be run from the scheduler/uniform_random_DO_DI directory.

Usage notes (run location):
- Run this script from the scheduler/uniform_random_DO_DI directory so relative paths resolve correctly.
  Example:
    cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/uniform_random_DO_DI
    python generate_jobs.py
"""

import os
from pathlib import Path
import json
from datetime import datetime


def generate_do_di_jobs():
	"""Generate jobs.txt for Uniform Random DO+DI Sparse NN hyperparameter tuner."""
	# Drug IDs
	drugs = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]

	# Base paths
	base_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"

	# Hyperparameter tuning parameters
	n_trials = 100

	# Output file
	output_file = "jobs/uniform_random_do_di_snn_jobs.txt"

	# Create jobs directory if it doesn't exist
	os.makedirs("jobs", exist_ok=True)

	jobs = []
	job_count = 0

	print(f"Generating Uniform Random DO+DI Sparse NN hyperparameter tuning jobs for {len(drugs)} drugs...")

	for drug in drugs:
		cell2id_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_cell2ind.txt"
		print(f"Processing drug {drug}...")

		for i in range(50):  # 50 experiments per drug
			seed = i * 1000
			job_count += 1

			# Gene expression data file
			ge_data_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_GE_Data.txt"
			log_file = f"D{drug}_{i}.log"
			# Nested structure: results/D{drug}/D{drug}_{i}
			output_dir = f"results/D{drug}/D{drug}_{i}"

			# Training, validation, and test data files
			train_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/true_training_data.txt"
			val_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/validation_data.txt"
			test_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt"

			# Build the command for Uniform Random DO+DI Sparse NN hyperparameter tuning
			command = (
				f"python -u uniform_random_do_di_snn_hparam_tuner.py "
				f"-drug {drug} "
				f"-train_file {train_file} "
				f"-val_file {val_file} "
				f"-test_file {test_file} "
				f"-cell2id {cell2id_file} "
				f"-ge_data {ge_data_file} "
				f"-n_trials {n_trials} "
				f"-seed {seed} "
				f"-output_dir {output_dir} "
				f"> {log_file}"
			)

			jobs.append(command)

			# Progress indicator
			if job_count % 100 == 0:
				print(f"Generated {job_count} jobs...")

	# Write jobs to file
	print(f"Writing {len(jobs)} jobs to {output_file}...")

	with open(output_file, 'w') as f:
		f.write("# Uniform Random DO+DI (Direct Output + Direct Input) Sparse NN Hyperparameter Tuning Jobs\n")
		f.write(f"# Generated automatically for {len(drugs)} drugs\n")
		f.write(f"# Total jobs: {len(jobs)}\n")
		f.write(f"# Drugs: {drugs}\n")
		f.write(f"# Experiments per drug: 50\n")
		f.write(f"# Hyperparameter tuning trials: {n_trials}\n")
		f.write(f"# Seed: {seed}\n")
		f.write("#\n")
		f.write("# Format: drug_experiment\n")
		f.write("# Example: D5_0, D5_1, D5_2, etc.\n")
		f.write("#\n\n")

		for i, job in enumerate(jobs, 1):
			f.write(f"# Job {i}\n")
			f.write(f"{job}\n\n")

	print(f"Successfully generated {len(jobs)} jobs in {output_file}")
	print(f"Jobs cover {len(drugs)} drugs with 50 experiments each")
	return output_file, len(jobs)


def generate_do_di_jobs_json():
	"""Generate advanced_jobs.json file with Uniform Random DO+DI Sparse NN hyperparameter tuning experiments."""
	# Drug IDs
	drugs = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]

	# Base paths
	base_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"

	# Hyperparameter tuning parameters
	n_trials = 100

	# Output file
	output_file = "jobs/uniform_random_do_di_snn_advanced_jobs.json"

	# Create jobs directory if it doesn't exist
	os.makedirs("jobs", exist_ok=True)

	jobs = []
	job_count = 0

	print(f"Generating JSON jobs for Uniform Random DO+DI Sparse NN hyperparameter tuning for {len(drugs)} drugs...")

	for drug in drugs:
		cell2id_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_cell2ind.txt"
		print(f"Processing drug {drug}...")

		for i in range(50):  # 50 experiments per drug
			job_count += 1
			seed = i * 1000

			# Gene expression data file
			ge_data_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_GE_Data.txt"
			log_file = f"D{drug}_{i}.log"
			# Nested structure: results/D{drug}/D{drug}_{i}
			output_dir = f"results/D{drug}/D{drug}_{i}"

			# Training, validation, and test data files
			train_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/true_training_data.txt"
			val_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/validation_data.txt"
			test_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt"

			# Build the command for Uniform Random DO+DI Sparse NN hyperparameter tuning
			command = (
				f"python -u uniform_random_do_di_snn_hparam_tuner.py "
				f"-drug {drug} "
				f"-train_file {train_file} "
				f"-val_file {val_file} "
				f"-test_file {test_file} "
				f"-cell2id {cell2id_file} "
				f"-ge_data {ge_data_file} "
				f"-n_trials {n_trials} "
				f"-seed {seed} "
				f"-output_dir {output_dir} "
				f"> {log_file}"
			)

			# Create job configuration
			job_config = {
				"id": f"D{drug}_{i}",
				"command": command,
				"priority": 5,  # Medium priority
				"max_retries": 2,
				"description": f"Drug {drug} experiment {i} - Uniform Random DO+DI Sparse NN Hyperparameter Tuning",
				"metadata": {
					"drug_id": drug,
					"experiment_id": i,
					"output_dir": output_dir,
					"log_file": log_file,
					"n_trials": n_trials,
					"seed": seed
				}
			}

			jobs.append(job_config)

			# Progress indicator
			if job_count % 100 == 0:
				print(f"Generated {job_count} jobs...")

	# Create JSON structure
	json_data = {
		"description": "Uniform Random DO+DI (Direct Output + Direct Input) Sparse NN Hyperparameter Tuning Jobs",
		"generated_at": datetime.now().strftime("%Y-%m-%d"),
		"total_jobs": len(jobs),
		"drugs": drugs,
		"experiments_per_drug": 50,
		"hyperparameter_tuning": {
			"n_trials": n_trials,
			"seed_base": 0,  # Seeds are i * 1000 for experiment i
			"algorithm": "Optuna TPE Sampler"
		},
		"jobs": jobs
	}

	# Write JSON file
	print(f"Writing {len(jobs)} jobs to {output_file}...")
	with open(output_file, 'w') as f:
		json.dump(json_data, f, indent=2)

	print(f"Successfully generated {len(jobs)} jobs in {output_file}")
	return output_file, len(jobs)


def main():
	"""Main function to generate Uniform Random DO+DI Sparse NN job files (text and json)."""
	print("Uniform Random DO+DI (Direct Output + Direct Input) Sparse NN Hyperparameter Tuning Job Generator")
	print("=" * 80)

	print("\n1. Generating simple jobs.txt...")
	txt_file, txt_count = generate_do_di_jobs()

	print("\n2. Generating advanced jobs.json...")
	json_file, json_count = generate_do_di_jobs_json()

	print("\n" + "=" * 80)
	print("GENERATION COMPLETE")
	print("=" * 80)
	print(f"Simple jobs file: {txt_file} ({txt_count} jobs)")
	print(f"Advanced jobs file: {json_file} ({json_count} jobs)")
	print(f"Total jobs generated: {txt_count}")

	print("\nUsage:")
	print(f"python distributed_gpu_queue.py jobs/uniform_random_do_di_snn_advanced_jobs.json --max-gpus 16 --node-id node1 --lock-dir shared/locks --status-dir shared/status")
	print(f"tail -f logs/distributed_queue.log")


if __name__ == "__main__":
	main()
