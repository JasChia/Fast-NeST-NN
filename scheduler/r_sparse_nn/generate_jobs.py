#!/usr/bin/env python3
"""
Job Generator for Sparse Neural Network (Sparse_NN) Hyperparameter Tuning
Generates jobs files with drug experiments for the GPU queue manager, targeting
r_sparse_nn/r_sparse_nn_hparam_tuner.py and writing outputs under scheduler/r_sparse_nn/results/...

Should be run from the scheduler/r_sparse_nn directory.

Usage notes (run location):
- Run this script from the scheduler/r_sparse_nn directory so relative paths resolve correctly.
  Example:
    cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/r_sparse_nn
    python generate_jobs.py
"""

import os
from pathlib import Path
import json


def generate_sparse_jobs():
	"""Generate jobs.txt for Sparse_NN hyperparameter tuner."""
	# Drug IDs
	drugs = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]

	# Base paths
	base_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"

	# Hyperparameter tuning parameters
	n_trials = 100

	# Output file
	output_file = "jobs/r_sparse_nn_jobs.txt"

	# Create jobs directory if it doesn't exist
	os.makedirs("jobs", exist_ok=True)

	jobs = []
	job_count = 0

	print(f"Generating Sparse_NN hyperparameter tuning jobs for {len(drugs)} drugs...")

	for drug in drugs:
		cell2id_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_cell2ind.txt"
		print(f"Processing drug {drug}...")

		for i in range(50):  # 50 experiments per drug
			seed = i * 1000
			job_count += 1

			# Gene expression data file
			ge_data_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_GE_Data.txt"
			log_file = f"D{drug}_{i}.log"
			# Nested structure: r_sparse_nn/results/D{drug}/D{drug}_{i}
			output_dir = f"results/D{drug}/D{drug}_{i}"

			# Training, validation, and test data files
			train_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/true_training_data.txt"
			val_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/validation_data.txt"
			test_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt"

			# Build the command for Sparse_NN hyperparameter tuning
			command = (
				f"python -u r_sparse_nn_hparam_tuner.py "
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
		f.write("# Sparse_NN Hyperparameter Tuning Jobs\n")
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


def generate_sparse_jobs_json():
	"""Generate advanced_jobs.json file with Sparse_NN hyperparameter tuning experiments."""
	# Drug IDs
	drugs = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]

	# Base paths
	base_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"

	# Hyperparameter tuning parameters
	n_trials = 100

	# Output file
	output_file = "jobs/r_sparse_nn_advanced_jobs.json"

	# Create jobs directory if it doesn't exist
	os.makedirs("jobs", exist_ok=True)

	jobs = []
	job_count = 0

	print(f"Generating JSON jobs for Sparse_NN hyperparameter tuning for {len(drugs)} drugs...")

	for drug in drugs:
		cell2id_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_cell2ind.txt"
		print(f"Processing drug {drug}...")

		for i in range(50):  # 50 experiments per drug
			job_count += 1
			seed = i * 1000

			# Gene expression data file
			ge_data_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/D{drug}_GE_Data.txt"
			log_file = f"D{drug}_{i}.log"
			# Nested structure: r_sparse_nn/results/D{drug}/D{drug}_{i}
			output_dir = f"results/D{drug}/D{drug}_{i}"

			# Training, validation, and test data files
			train_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/true_training_data.txt"
			val_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/validation_data.txt"
			test_file = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt"

			# Build the command for Sparse_NN hyperparameter tuning
			command = (
				f"python -u r_sparse_nn_hparam_tuner.py "
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
				"description": f"Drug {drug} experiment {i} - Sparse_NN Hyperparameter Tuning",
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
		"description": "Sparse_NN Hyperparameter Tuning Jobs",
		"generated_at": "2024-01-15",
		"total_jobs": len(jobs),
		"drugs": drugs,
		"experiments_per_drug": 50,
		"hyperparameter_tuning": {
			"n_trials": n_trials,
			"seed": seed,
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
	"""Main function to generate Sparse_NN job files (text and json)."""
	print("Sparse_NN Hyperparameter Tuning Job Generator")
	print("=" * 60)

	print("\n1. Generating Sparse_NN simple jobs.txt...")
	txt_file, txt_count = generate_sparse_jobs()

	print("\n2. Generating Sparse_NN advanced jobs.json...")
	json_file, json_count = generate_sparse_jobs_json()

	print("\n" + "=" * 60)
	print("GENERATION COMPLETE")
	print("=" * 60)
	print(f"Sparse_NN simple jobs file: {txt_file} ({txt_count} jobs)")
	print(f"Sparse_NN advanced jobs file: {json_file} ({json_count} jobs)")
	print(f"Total Sparse_NN jobs generated: {txt_count}")

	print("\nUsage:")
	print(f"python advanced_gpu_queue.py jobs/r_sparse_nn_advanced_jobs.json --max-gpus 16")
	print(f"tail -f r_sparse_nn/logs/gpu_queue_manager.log")


if __name__ == "__main__":
	main()


