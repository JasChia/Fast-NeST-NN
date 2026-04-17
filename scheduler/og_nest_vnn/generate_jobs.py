#!/usr/bin/env python3
"""
Job Generator for NeST VNN (Neural Network with Visible Structure) Hyperparameter Tuning
Generates jobs files with drug experiments for the GPU queue manager, targeting
nest_vnn_hparam_tuner.py and writing outputs under scheduler/og_nest_vnn/results/...

Uses command structure similar to Old_eNest.

Should be run from the scheduler/og_nest_vnn directory.

Usage notes (run location):
- Run this script from the scheduler/og_nest_vnn directory so relative paths resolve correctly.
  Example:
    cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/og_nest_vnn
    python generate_jobs.py
"""

import os
from pathlib import Path
import json
from datetime import datetime


def generate_nest_vnn_jobs():
	"""Generate jobs.txt for NeST VNN hyperparameter tuner."""
	# Drug IDs
	drugs = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]

	# Base paths
	base_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"
	nest_data_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM"

	# NeST VNN requires shared ontology and gene2id files (not drug-specific)
	# These are the same for all drugs
	gene2id_file = f"{base_path}/red_gene2ind.txt"
	ontology_file = f"{base_path}/red_ontology.txt"

	# Hyperparameter tuning parameters
	n_trials = 100

	# Output file
	output_file = "jobs/nest_vnn_jobs.txt"

	# Create jobs directory if it doesn't exist
	os.makedirs("jobs", exist_ok=True)

	jobs = []
	job_count = 0

	print(f"Generating NeST VNN hyperparameter tuning jobs for {len(drugs)} drugs...")

	for drug in drugs:
		# Drug-specific files
		cell2id_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/D{drug}_cell2ind.txt"
		ge_data_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/D{drug}_GE_Data.txt"
		
		print(f"Processing drug {drug}...")

		for i in range(50):  # 50 experiments per drug
			seed = i * 1000
			job_count += 1

			log_file = f"D{drug}_{i}.log"
			# Nested structure: og_nest_vnn/results/D{drug}/D{drug}_{i}
			output_dir = f"results/D{drug}/D{drug}_{i}"

			# Training, validation, and test data files
			train_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/true_training_data.txt"
			val_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/validation_data.txt"
			test_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt"

			# Build the command for NeST VNN hyperparameter tuning
			command = (
				f"python -u nest_vnn_hparam_tuner.py "
				f"-drug {drug} "
				f"-onto {ontology_file} "
				f"-train_file {train_file} "
				f"-val_file {val_file} "
				f"-test_file {test_file} "
				f"-cell2id {cell2id_file} "
				f"-gene2id {gene2id_file} "
				f"-transcriptomic {ge_data_file} "
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
		f.write("# NeST VNN Hyperparameter Tuning Jobs\n")
		f.write(f"# Generated automatically for {len(drugs)} drugs\n")
		f.write(f"# Total jobs: {len(jobs)}\n")
		f.write(f"# Drugs: {drugs}\n")
		f.write(f"# Experiments per drug: 50\n")
		f.write(f"# Hyperparameter tuning trials: {n_trials}\n")
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


def generate_nest_vnn_jobs_json():
	"""Generate advanced_jobs.json file with NeST VNN hyperparameter tuning experiments."""
	# Drug IDs
	drugs = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]

	# Base paths
	base_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"
	nest_data_path = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM"

	# NeST VNN requires shared ontology and gene2id files (not drug-specific)
	# These are the same for all drugs
	gene2id_file = f"{base_path}/red_gene2ind.txt"
	ontology_file = f"{base_path}/red_ontology.txt"

	# Hyperparameter tuning parameters
	n_trials = 100

	# Output file
	output_file = "jobs/nest_vnn_advanced_jobs.json"

	# Create jobs directory if it doesn't exist
	os.makedirs("jobs", exist_ok=True)

	jobs = []
	job_count = 0

	print(f"Generating JSON jobs for NeST VNN hyperparameter tuning for {len(drugs)} drugs...")

	for drug in drugs:
		# Drug-specific files
		cell2id_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/D{drug}_cell2ind.txt"
		ge_data_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/D{drug}_GE_Data.txt"
		
		print(f"Processing drug {drug}...")

		for i in range(50):  # 50 experiments per drug
			job_count += 1
			seed = i * 1000

			log_file = f"D{drug}_{i}.log"
			# Nested structure: og_nest_vnn/results/D{drug}/D{drug}_{i}
			output_dir = f"results/D{drug}/D{drug}_{i}"

			# Training, validation, and test data files
			train_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/true_training_data.txt"
			val_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/validation_data.txt"
			test_file = f"{nest_data_path}/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt"

			# Build the command for NeST VNN hyperparameter tuning
			command = (
				f"python -u nest_vnn_hparam_tuner.py "
				f"-drug {drug} "
				f"-onto {ontology_file} "
				f"-train_file {train_file} "
				f"-val_file {val_file} "
				f"-test_file {test_file} "
				f"-cell2id {cell2id_file} "
				f"-gene2id {gene2id_file} "
				f"-transcriptomic {ge_data_file} "
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
				"description": f"Drug {drug} experiment {i} - NeST VNN Hyperparameter Tuning",
				"metadata": {
					"drug_id": drug,
					"experiment_id": i,
					"output_dir": output_dir,
					"log_file": log_file,
					"n_trials": n_trials,
					"seed": seed,
					"ontology_file": ontology_file
				}
			}

			jobs.append(job_config)

			# Progress indicator
			if job_count % 100 == 0:
				print(f"Generated {job_count} jobs...")

	# Create JSON structure
	json_data = {
		"description": "NeST VNN Hyperparameter Tuning Jobs",
		"generated_at": datetime.now().strftime("%Y-%m-%d"),
		"total_jobs": len(jobs),
		"drugs": drugs,
		"experiments_per_drug": 50,
		"hyperparameter_tuning": {
			"n_trials": n_trials,
			"seed_base": 0,  # Seeds are i * 1000 for experiment i
			"algorithm": "Optuna TPE Sampler",
			"hyperparameters": {
				"lr": {"min": 1e-5, "max": 1e-2, "scale": "log"},
				"wd": {"min": 1e-5, "max": 1e-2, "scale": "log"},
				"l1": {"min": 1e-5, "max": 1e-2, "scale": "log"},
				"dropout_fraction": {"min": 0.0, "max": 0.7, "step": 0.1},
				"alpha": {"min": 0.0, "max": 1.0},
				"batch_size_power": {"min": 2, "max": 5},
				"activation": ["Tanh", "ReLU"],
				"min_dropout_layer": {"min": 1, "max": 4}
			}
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
	"""Main function to generate NeST VNN job files (text and json)."""
	print("NeST VNN Hyperparameter Tuning Job Generator")
	print("=" * 60)

	print("\n1. Generating NeST VNN simple jobs.txt...")
	txt_file, txt_count = generate_nest_vnn_jobs()

	print("\n2. Generating NeST VNN advanced jobs.json...")
	json_file, json_count = generate_nest_vnn_jobs_json()

	print("\n" + "=" * 60)
	print("GENERATION COMPLETE")
	print("=" * 60)
	print(f"NeST VNN simple jobs file: {txt_file} ({txt_count} jobs)")
	print(f"NeST VNN advanced jobs file: {json_file} ({json_count} jobs)")
	print(f"Total NeST VNN jobs generated: {txt_count}")

	print("\nUsage (Distributed Multi-Node):")
	print("  # Run on each node (adjust --max-gpus and --node-id for each node):")
	print(f"  nohup python -u distributed_gpu_queue.py jobs/nest_vnn_advanced_jobs.json \\")
	print(f"      --max-gpus 16 --node-id l5 \\")
	print(f"      --lock-dir shared/locks --status-dir shared/status \\")
	print(f"      --log-file logs/distributed_queue_l5.log \\")
	print(f"      > logs/distributed_queue_l5.out 2>&1 &")
	print(f"\n  # Monitor progress:")
	print(f"  tail -f logs/distributed_queue_l5.log")


if __name__ == "__main__":
	main()

