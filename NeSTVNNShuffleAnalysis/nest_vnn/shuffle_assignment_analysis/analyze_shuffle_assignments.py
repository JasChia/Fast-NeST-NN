#!/usr/bin/env python3
"""
Analyze gene-to-assembly assignments after shuffling.

This script processes graphs with correct and shuffled gene2id files to determine
how many genes were assigned to incorrect assemblies after shuffling.

Minimal adaptation of original code - preserves graph processing logic from
training_data_wrapper.py while adding analysis functionality.

Usage:

    # From original_nest_vnn/ (or run ``python analyze_shuffle_assignments.py``
    # from shuffle_assignment_analysis/ with the same arguments):
    python shuffle_assignment_analysis/analyze_shuffle_assignments.py \\
        -onto /path/to/red_ontology.txt \\
        -gene2id /path/to/red_gene2ind.txt \\
        -drug 5 \\
        -max_experiments 50 \\
        -output shuffle_analysis_results.json \\
        -verbose

    For the all-drugs batch, run ``shuffle_assignment_analysis/run_shuffle_analysis.sh``
    (writes JSON and log under ``shuffle_assignment_analysis/outputs/``).

The script will:
1. Process the reference graph with the correct gene2id file
2. Find all shuffled gene2id files for the specified drug
3. Process each shuffled graph and compare gene-to-assembly assignments
4. Report statistics on how many genes were assigned to incorrect assemblies
"""

import sys
import os
import argparse
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from collections import defaultdict
import json
import numpy as np
import random
import tempfile
from pathlib import Path

# Import original modules (training code lives in nest_vnn/src)
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.normpath(os.path.join(script_dir, '..', 'src'))
if os.path.exists(src_dir):
	sys.path.insert(0, src_dir)
import util


class GraphProcessor:
	"""
	Minimal wrapper around original graph processing code.
	Extracts gene-to-assembly assignment logic from TrainingDataWrapper.
	"""
	
	def __init__(self, ontology_file, gene2id_file):
		"""
		Initialize graph processor with ontology and gene2id mapping.
		
		Args:
			ontology_file: Path to ontology file
			gene2id_file: Path to gene2id mapping file
		"""
		self.ontology_file = ontology_file
		self.gene_id_mapping = util.load_mapping(gene2id_file, 'genes')
		self.load_ontology(ontology_file)
	
	def load_ontology(self, file_name):
		"""
		Load ontology and build gene-to-assembly mappings.
		This is the original code from TrainingDataWrapper.load_ontology()
		"""
		dG = nx.DiGraph()
		term_direct_gene_map = {}
		term_size_map = {}
		gene_set = set()

		file_handle = open(file_name)
		for line in file_handle:
			line = line.rstrip().split()
			if line[2] == 'default':
				dG.add_edge(line[0], line[1])
			else:
				if line[1] not in self.gene_id_mapping:
					continue
				if line[0] not in term_direct_gene_map:
					term_direct_gene_map[line[0]] = set()
				term_direct_gene_map[line[0]].add(self.gene_id_mapping[line[1]])
				gene_set.add(line[1])
		file_handle.close()

		for term in dG.nodes():
			term_gene_set = set()
			if term in term_direct_gene_map:
				term_gene_set = term_direct_gene_map[term]
			deslist = nxadag.descendants(dG, term)
			for child in deslist:
				if child in term_direct_gene_map:
					term_gene_set = term_gene_set | term_direct_gene_map[child]
			if len(term_gene_set) == 0:
				print('There is empty terms, please delete term:', term)
				sys.exit(1)
			else:
				term_size_map[term] = len(term_gene_set)

		roots = [n for n in dG.nodes if dG.in_degree(n) == 0]

		uG = dG.to_undirected()
		connected_subG_list = list(nxacc.connected_components(uG))

		if len(roots) > 1:
			print('There are more than 1 root of ontology. Please use only one root.')
			sys.exit(1)
		if len(connected_subG_list) > 1:
			print('There are more than connected components. Please connect them.')
			sys.exit(1)

		self.dG = dG
		self.root = roots[0]
		self.term_size_map = term_size_map
		self.term_direct_gene_map = term_direct_gene_map
	
	def get_gene_to_assemblies(self):
		"""
		Get mapping from gene symbols to assemblies (terms) that contain them.
		
		Returns:
			dict: gene_symbol -> set of assembly (term) names
		"""
		# Create reverse mapping: gene_id -> gene_symbol
		gene_id_to_symbol = {gene_id: symbol for symbol, gene_id in self.gene_id_mapping.items()}
		
		# Build gene_symbol -> assemblies mapping
		gene_to_assemblies = defaultdict(set)
		
		# For each term, find all genes (by symbol) that are in it
		for term in self.dG.nodes():
			# Get all descendant terms including the term itself
			all_terms = [term] + list(nxadag.descendants(self.dG, term))
			
			# For each term, get the genes directly associated with it
			for t in all_terms:
				if t in self.term_direct_gene_map:
					# term_direct_gene_map contains gene IDs
					for gene_id in self.term_direct_gene_map[t]:
						# Map gene ID back to gene symbol using reverse mapping
						if gene_id in gene_id_to_symbol:
							gene_symbol = gene_id_to_symbol[gene_id]
							gene_to_assemblies[gene_symbol].add(term)
		
		return dict(gene_to_assemblies)
	
	def get_gene_id_to_assemblies(self):
		"""
		Get mapping from gene IDs (positions) to assemblies (terms) that contain them.
		This is needed to compare by position rather than by symbol.
		
		Returns:
			dict: gene_id -> set of assembly (term) names
		"""
		# Build gene_id -> assemblies mapping
		gene_id_to_assemblies = defaultdict(set)
		
		# For each term, find all gene IDs that are in it
		for term in self.dG.nodes():
			# Get all descendant terms including the term itself
			all_terms = [term] + list(nxadag.descendants(self.dG, term))
			
			# For each term, get the genes directly associated with it
			for t in all_terms:
				if t in self.term_direct_gene_map:
					# term_direct_gene_map contains gene IDs
					for gene_id in self.term_direct_gene_map[t]:
						gene_id_to_assemblies[gene_id].add(term)
		
		return dict(gene_id_to_assemblies)


def compare_assignments(reference_assignments, shuffled_assignments):
	"""
	Compare gene-to-assembly assignments between reference and shuffled graphs.
	
	Args:
		reference_assignments: dict mapping gene_symbol -> set of assemblies (from correct graph)
		shuffled_assignments: dict mapping gene_symbol -> set of assemblies (from shuffled graph)
	
	Returns:
		dict with comparison statistics
	"""
	# Get all genes present in both
	all_genes = set(reference_assignments.keys()) | set(shuffled_assignments.keys())
	
	correct_assignments = 0
	incorrect_assignments = 0
	gene_mismatches = []
	
	for gene in all_genes:
		ref_assemblies = reference_assignments.get(gene, set())
		shuffled_assemblies = shuffled_assignments.get(gene, set())
		
		if ref_assemblies == shuffled_assemblies:
			correct_assignments += 1
		else:
			incorrect_assignments += 1
			gene_mismatches.append({
				'gene': gene,
				'reference_assemblies': sorted(list(ref_assemblies)),
				'shuffled_assemblies': sorted(list(shuffled_assemblies)),
				'added_assemblies': sorted(list(shuffled_assemblies - ref_assemblies)),
				'removed_assemblies': sorted(list(ref_assemblies - shuffled_assemblies))
			})
	
	return {
		'total_genes': len(all_genes),
		'correct_assignments': correct_assignments,
		'incorrect_assignments': incorrect_assignments,
		'correct_percentage': (correct_assignments / len(all_genes) * 100) if all_genes else 0,
		'incorrect_percentage': (incorrect_assignments / len(all_genes) * 100) if all_genes else 0,
		'gene_mismatches': gene_mismatches
	}


def compare_assignments_by_position(reference_id_assignments, shuffled_id_assignments, permutation):
	"""
	Compare gene-to-assembly assignments by checking each assembly individually.
	
	For each position i in the shuffled graph:
	- The gene at shuffled position i came from original position permutation[i]
	- Get the assemblies this gene should have (from reference, using original position)
	- Get the assemblies this gene actually has in shuffled graph (at position i)
	- For each assembly in the shuffled set:
	  * If the gene from original position permutation[i] was supposed to be in that assembly → correct
	  * If the gene from original position permutation[i] was NOT supposed to be in that assembly → incorrect
	
	Args:
		reference_id_assignments: dict mapping gene_id -> set of assemblies (from correct graph)
		shuffled_id_assignments: dict mapping gene_id -> set of assemblies (from shuffled graph)
		permutation: list where permutation[i] = original position of gene now at position i
	
	Returns:
		dict with comparison statistics
	"""
	# Get all gene IDs present in both
	all_gene_ids = set(reference_id_assignments.keys()) | set(shuffled_id_assignments.keys())
	
	total_assembly_assignments = 0
	correct_assembly_assignments = 0
	incorrect_assembly_assignments = 0
	gene_mismatches = []
	
	# Compare assembly-by-assembly for each position
	for position in all_gene_ids:
		if position >= len(permutation):
			continue
			
		# The gene at shuffled position i came from original position permutation[i]
		original_position = permutation[position]
		
		# Get assemblies for the gene that SHOULD be at this position (from reference, using position)
		# This is the gene that was originally at position i
		expected_assemblies = reference_id_assignments.get(position, set())
		
		# Get assemblies for the gene that IS at this position in shuffled graph
		# This is a different gene (from original position permutation[i])
		actual_assemblies = shuffled_id_assignments.get(position, set())
		
		# Track correct and incorrect assemblies for this position
		correct_assemblies = []
		incorrect_assemblies = []
		
		# Check each assembly in the shuffled set
		# For each assembly assigned to the gene at this position, check if the gene
		# that SHOULD be at this position (from original) was supposed to be in that assembly
		for assembly in actual_assemblies:
			total_assembly_assignments += 1
			if assembly in expected_assemblies:
				# The gene that should be at this position was supposed to be in this assembly → correct
				correct_assembly_assignments += 1
				correct_assemblies.append(assembly)
			else:
				# The gene that should be at this position was NOT supposed to be in this assembly → incorrect
				incorrect_assembly_assignments += 1
				incorrect_assemblies.append(assembly)
		
		# Also count missing assemblies (expected but not present) as incorrect
		missing_assemblies = expected_assemblies - actual_assemblies
		if missing_assemblies:
			incorrect_assembly_assignments += len(missing_assemblies)
			total_assembly_assignments += len(missing_assemblies)
		
		# Store mismatch details if there are any incorrect assemblies
		if incorrect_assemblies or missing_assemblies:
			gene_mismatches.append({
				'position': position,
				'original_gene_came_from_position': original_position,
				'expected_assemblies': sorted(list(expected_assemblies)),
				'actual_assemblies': sorted(list(actual_assemblies)),
				'correct_assemblies': sorted(correct_assemblies),
				'incorrect_assemblies': sorted(incorrect_assemblies),
				'missing_assemblies': sorted(list(missing_assemblies))
			})
	
	# Calculate percentages
	correct_percentage = (correct_assembly_assignments / total_assembly_assignments * 100) if total_assembly_assignments > 0 else 0
	incorrect_percentage = (incorrect_assembly_assignments / total_assembly_assignments * 100) if total_assembly_assignments > 0 else 0
	
	return {
		'total_genes': len(all_gene_ids),
		'total_assembly_assignments': total_assembly_assignments,
		'correct_assignments': correct_assembly_assignments,
		'incorrect_assignments': incorrect_assembly_assignments,
		'correct_percentage': correct_percentage,
		'incorrect_percentage': incorrect_percentage,
		'gene_mismatches': gene_mismatches
	}


def get_shuffling_permutation(num_genes, experiment_id, base_seed=42):
	"""
	Recreate the shuffling permutation used in create_train_test_splits.py.
	
	The shuffling uses: random_seed = base_seed + experiment_id
	Then creates a permutation of gene indices [0, 1, 2, ..., num_genes-1]
	
	Args:
		num_genes: Number of genes
		experiment_id: Experiment ID number
		base_seed: Base random seed (42 in create_train_test_splits.py)
	
	Returns:
		list: Permutation of gene indices (shuffled order)
	"""
	# Recreate the same random seed used in create_train_test_splits.py
	split_seed = base_seed + experiment_id
	random.seed(split_seed)
	
	# Create the same permutation as in shuffle_gene_expression_data
	gene_indices = list(range(num_genes))
	random.shuffle(gene_indices)
	
	return gene_indices


def create_shuffled_gene2id_mapping(original_gene2id_file, experiment_id, base_seed=42):
	"""
	Create a shuffled gene2id mapping based on the column permutation used in shuffling.
	
	When gene expression columns are shuffled, the gene2id mapping is effectively shuffled too.
	The permutation tells us: shuffled_column[i] = original_column[permutation[i]]
	
	So in the shuffled gene2id:
	- The gene_symbol that ends up in shuffled column i maps to gene_id i
	- That gene_symbol is the one that was originally at position permutation[i]
	
	Args:
		original_gene2id_file: Path to original gene2id file
		experiment_id: Experiment ID number
		base_seed: Base random seed (42 in create_train_test_splits.py)
	
	Returns:
		dict: Shuffled gene2id mapping (gene_symbol -> gene_id)
	"""
	# Load original gene2id mapping
	original_mapping = util.load_mapping(original_gene2id_file, 'genes')
	
	# Get number of genes
	num_genes = len(original_mapping)
	
	# Get the shuffling permutation
	permutation = get_shuffling_permutation(num_genes, experiment_id, base_seed)
	
	# Create mapping: original_gene_id -> gene_symbol
	original_id_to_symbol = {}
	for symbol, gene_id in original_mapping.items():
		original_id_to_symbol[gene_id] = symbol
	
	# Create shuffled mapping
	# permutation[i] tells us which original column position is now in shuffled column i
	# So: shuffled column i contains original gene at position permutation[i]
	# The gene_symbol at original position permutation[i] now maps to gene_id i in shuffled data
	shuffled_mapping = {}
	
	# Sort original gene IDs to get ordered list (gene_id 0, 1, 2, ...)
	sorted_original_ids = sorted(original_id_to_symbol.keys())
	
	for shuffled_column in range(num_genes):
		# shuffled_column is the new column position (0, 1, 2, ...)
		# permutation[shuffled_column] tells us which original column this came from
		original_column = permutation[shuffled_column]
		
		# Get the original gene_id at that original column position
		original_gene_id = sorted_original_ids[original_column]
		
		# Get the gene symbol for that original gene
		gene_symbol = original_id_to_symbol[original_gene_id]
		
		# In shuffled mapping, this gene_symbol now maps to shuffled_column (the new gene_id)
		shuffled_mapping[gene_symbol] = shuffled_column
	
	return shuffled_mapping


def find_shuffled_experiments(base_path, drug_id, max_experiments=None):
	"""
	Find all shuffled experiment directories for a given drug.
	
	Args:
		base_path: Base path to data directory
		drug_id: Drug ID number
		max_experiments: Maximum number of experiments to process (None for all)
	
	Returns:
		list of (experiment_id, experiment_dir) tuples
	"""
	experiments = []
	drug_dir = f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug_id}/D{drug_id}_CL"
	
	# Check for experiment directories
	experiment_dirs = []
	if os.path.exists(f"{drug_dir}/train_test_splits"):
		for item in os.listdir(f"{drug_dir}/train_test_splits"):
			exp_path = f"{drug_dir}/train_test_splits/{item}"
			if os.path.isdir(exp_path) and item.startswith("experiment_"):
				experiment_dirs.append(exp_path)
	
	# Sort by experiment number
	experiment_dirs.sort(key=lambda x: int(x.split("experiment_")[-1]))
	
	if max_experiments:
		experiment_dirs = experiment_dirs[:max_experiments]
	
	for exp_dir in experiment_dirs:
		exp_id = int(exp_dir.split("experiment_")[-1])
		
		# Verify shuffled gene expression file exists
		shuffled_ge_file = f"{exp_dir}/shuffled_gene_expression.txt"
		if os.path.exists(shuffled_ge_file):
			experiments.append((exp_id, exp_dir))
		else:
			if len(experiments) == 0:  # Only print warning if we haven't found any yet
				print(f"Warning: No shuffled_gene_expression.txt found for experiment {exp_id}")
	
	return experiments


def _default_data_base_path():
	"""NeSTVNNShuffleAnalysis/Data (repo root is parent of nest_vnn/)."""
	return str((Path(__file__).resolve().parent.parent.parent / "Data"))


# Drug IDs used in the NeST-VNN shuffle experiments (12 drugs × 50 runs)
DRUG_IDS = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]
EXPERIMENTS_PER_DRUG = 50


def process_single_drug(ontology_file, gene2id_file, base_path, drug_id, max_experiments, output_prefix, verbose=False):
	"""
	Process a single drug and return results.
	
	Returns:
		dict with results or None if no shuffled files found
	"""
	print(f"\n{'='*80}")
	print(f"Processing Drug {drug_id}")
	print(f"{'='*80}")
	
	# Process reference graph with correct gene2id
	if verbose:
		print("\n[1/3] Processing reference graph with correct gene2id...")
	reference_processor = GraphProcessor(ontology_file, gene2id_file)
	reference_id_assignments = reference_processor.get_gene_id_to_assemblies()
	if verbose:
		print(f"Reference graph processed: {len(reference_id_assignments)} gene positions assigned to assemblies")
	
	# Find all shuffled experiments
	if verbose:
		print(f"\n[2/3] Finding shuffled experiments for Drug {drug_id}...")
	shuffled_experiments = find_shuffled_experiments(base_path, drug_id, max_experiments)
	
	if not shuffled_experiments:
		print(f"No shuffled experiments found for Drug {drug_id}")
		return None
	
	if verbose:
		print(f"Found {len(shuffled_experiments)} shuffled experiments")
	
	# Process each shuffled graph and compare
	if verbose:
		print(f"\n[3/3] Processing shuffled graphs and comparing assignments...")
	all_results = []
	
	for exp_id, exp_dir in shuffled_experiments:
		if verbose:
			print(f"\nProcessing experiment {exp_id}...")
		
		try:
			# Get the shuffling permutation for this experiment
			# Load original mapping to get number of genes
			original_mapping = util.load_mapping(gene2id_file, 'genes')
			num_genes = len(original_mapping)
			permutation = get_shuffling_permutation(num_genes, exp_id, base_seed=42)
			
			# Create shuffled gene2id mapping based on the shuffling permutation
			shuffled_gene2id_mapping = create_shuffled_gene2id_mapping(
				gene2id_file, 
				exp_id, 
				base_seed=42  # Same as in create_train_test_splits.py
			)
			
			# Create a temporary gene2id file with the shuffled mapping
			with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
				# Write shuffled mapping in gene2id format: id\tgene_symbol
				# Sort by gene_id to maintain order
				for symbol, gene_id in sorted(shuffled_gene2id_mapping.items(), key=lambda x: x[1]):
					tmp_file.write(f"{gene_id}\t{symbol}\n")
				tmp_gene2id_file = tmp_file.name
			
			try:
				# Process shuffled graph
				shuffled_processor = GraphProcessor(ontology_file, tmp_gene2id_file)
				shuffled_id_assignments = shuffled_processor.get_gene_id_to_assemblies()
				
				# Compare by position: does the gene from original position i end up in same assemblies?
				comparison = compare_assignments_by_position(
					reference_id_assignments, 
					shuffled_id_assignments, 
					permutation
				)
				comparison['experiment_id'] = exp_id
				comparison['experiment_dir'] = exp_dir
				
				all_results.append(comparison)
			finally:
				# Clean up temporary file
				if os.path.exists(tmp_gene2id_file):
					os.unlink(tmp_gene2id_file)
			
			if verbose:
				print(f"  Experiment {exp_id}: {comparison['incorrect_assignments']}/{comparison['total_assembly_assignments']} "
				      f"assembly assignments ({comparison['incorrect_percentage']:.3f}%) are incorrect")
			else:
				if len(all_results) % 10 == 0:
					print(f"  Processed {len(all_results)}/{len(shuffled_experiments)} experiments...")
		
		except Exception as e:
			print(f"  Error processing experiment {exp_id}: {e}")
			continue
	
	return {
		'drug_id': drug_id,
		'reference_gene2id': gene2id_file,
		'ontology_file': ontology_file,
		'total_experiments': len(all_results),
		'experiment_results': all_results
	}


def main():
	parser = argparse.ArgumentParser(description='Analyze gene-to-assembly assignments after shuffling')
	parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str, required=True)
	parser.add_argument('-gene2id', help='Correct gene to ID mapping file', type=str, required=True)
	parser.add_argument('-base_path', help='Base path to data directory (contains nest_shuffle_data/, red_ontology.txt)', type=str,
	                   default=_default_data_base_path())
	parser.add_argument('-drug', help='Drug ID to analyze (if not specified, processes all drugs)', type=int, default=None)
	parser.add_argument('-all_drugs', help='Process all drugs in DRUG_IDS (NeST-VNN shuffle study)', action='store_true')
	parser.add_argument('-max_experiments', help='Maximum number of experiments to process per drug', type=int, default=None)
	parser.add_argument('-output', help='Output file for results (JSON)', type=str, default='shuffle_analysis_results.json')
	parser.add_argument('-verbose', help='Print detailed information', action='store_true')
	
	args = parser.parse_args()
	
	# Determine which drugs to process
	if args.all_drugs:
		drugs_to_process = DRUG_IDS
		print("=" * 80)
		print("Gene-to-Assembly Assignment Analysis - ALL DRUGS")
		print("=" * 80)
		print(f"Processing {len(drugs_to_process)} drugs: {drugs_to_process}")
		print(f"Experiments per drug: {EXPERIMENTS_PER_DRUG}")
	elif args.drug:
		drugs_to_process = [args.drug]
		print("=" * 80)
		print("Gene-to-Assembly Assignment Analysis")
		print("=" * 80)
		print(f"Processing single drug: {args.drug}")
	else:
		print("Error: Must specify either -drug <drug_id> or -all_drugs")
		sys.exit(1)
	
	print(f"Ontology file: {args.onto}")
	print(f"Correct gene2id file: {args.gene2id}")
	print(f"Base path: {args.base_path}")
	if args.max_experiments:
		print(f"Max experiments per drug: {args.max_experiments}")
	else:
		print(f"Max experiments per drug: All available (up to {EXPERIMENTS_PER_DRUG})")
	print("=" * 80)
	
	# Process all drugs
	all_drug_results = []
	
	for drug_id in drugs_to_process:
		drug_result = process_single_drug(
			args.onto, 
			args.gene2id, 
			args.base_path, 
			drug_id, 
			args.max_experiments,
			args.output,
			args.verbose
		)
		
		if drug_result:
			all_drug_results.append(drug_result)
	
	if not all_drug_results:
		print("\nNo results to report for any drug.")
		sys.exit(1)
	
	# Calculate overall summary statistics
	print("\n" + "=" * 80)
	print("OVERALL SUMMARY STATISTICS")
	print("=" * 80)
	
	total_experiments = sum(r['total_experiments'] for r in all_drug_results)
	total_drugs_processed = len(all_drug_results)
	
	# Aggregate statistics across all drugs
	all_experiment_results = []
	for drug_result in all_drug_results:
		all_experiment_results.extend(drug_result['experiment_results'])
	
	if all_experiment_results:
		total_genes = all_experiment_results[0]['total_genes']  # Should be same for all
		
		# Calculate overall statistics with mean and standard deviation
		all_incorrect_pcts = [r['incorrect_percentage'] for r in all_experiment_results]
		all_incorrect_counts = [r['incorrect_assignments'] for r in all_experiment_results]
		
		overall_pct_mean = np.mean(all_incorrect_pcts)
		overall_pct_std = np.std(all_incorrect_pcts, ddof=1)
		overall_count_mean = np.mean(all_incorrect_counts)
		overall_count_std = np.std(all_incorrect_counts, ddof=1)
		
		print(f"Total drugs processed: {total_drugs_processed}")
		print(f"Total experiments analyzed: {total_experiments}")
		print(f"Total genes analyzed: {total_genes}")
		print(f"\nOverall Statistics (across all drugs and experiments):")
		print(f"  Incorrect assembly assignments %: mean={overall_pct_mean:.3f}%, std={overall_pct_std:.3f}%")
		print(f"    Range: {min(all_incorrect_pcts):.3f}% - {max(all_incorrect_pcts):.3f}%")
		print(f"  Incorrect assembly assignments count: mean={overall_count_mean:.3f}, std={overall_count_std:.3f}")
		print(f"    Range: {min(all_incorrect_counts)} - {max(all_incorrect_counts)}")
		
		# Per-drug statistics with mean and standard deviation
		print("\nPer-Drug Statistics:")
		per_drug_summaries = []
		for drug_result in all_drug_results:
			if drug_result['experiment_results']:
				# Calculate mean and std dev for incorrect percentage
				incorrect_pcts = [r['incorrect_percentage'] for r in drug_result['experiment_results']]
				drug_mean = np.mean(incorrect_pcts)
				drug_std = np.std(incorrect_pcts, ddof=1)  # Sample standard deviation
				
				# Calculate mean and std dev for incorrect assignments count
				incorrect_counts = [r['incorrect_assignments'] for r in drug_result['experiment_results']]
				count_mean = np.mean(incorrect_counts)
				count_std = np.std(incorrect_counts, ddof=1)
				
				print(f"  Drug {drug_result['drug_id']}: {len(drug_result['experiment_results'])} experiments")
				print(f"    Incorrect assembly assignments %: mean={drug_mean:.3f}%, std={drug_std:.3f}%")
				print(f"    Incorrect assembly assignments count: mean={count_mean:.3f}, std={count_std:.3f}")
				
				# Store for JSON output
				per_drug_summaries.append({
					'drug_id': drug_result['drug_id'],
					'num_experiments': len(drug_result['experiment_results']),
					'incorrect_percentage': {
						'mean': float(drug_mean),
						'std': float(drug_std),
						'min': float(min(incorrect_pcts)),
						'max': float(max(incorrect_pcts))
					},
					'incorrect_assignments': {
						'mean': float(count_mean),
						'std': float(count_std),
						'min': int(min(incorrect_counts)),
						'max': int(max(incorrect_counts))
					}
				})
		
		print("=" * 80)
		
		# Save results
		output_data = {
			'reference_gene2id': args.gene2id,
			'ontology_file': args.onto,
			'base_path': args.base_path,
			'drugs_processed': drugs_to_process,
			'total_drugs_processed': total_drugs_processed,
			'total_experiments': total_experiments,
			'overall_summary': {
				'total_genes': total_genes,
				'incorrect_percentage': {
					'mean': float(overall_pct_mean),
					'std': float(overall_pct_std),
					'min': float(min(all_incorrect_pcts)),
					'max': float(max(all_incorrect_pcts))
				},
				'incorrect_assignments': {
					'mean': float(overall_count_mean),
					'std': float(overall_count_std),
					'min': int(min(all_incorrect_counts)),
					'max': int(max(all_incorrect_counts))
				}
			},
			'per_drug_summaries': per_drug_summaries,  # Mean and std dev per drug
			'drug_results': all_drug_results  # Full detailed results per drug
		}
		
		with open(args.output, 'w') as f:
			json.dump(output_data, f, indent=2)
		
		print(f"\nDetailed results saved to: {args.output}")
		
		# Print sample mismatches from first drug/experiment
		if all_drug_results and all_drug_results[0]['experiment_results']:
			first_exp = all_drug_results[0]['experiment_results'][0]
			if first_exp.get('gene_mismatches'):
				print(f"\nSample mismatches (first 5 from Drug {all_drug_results[0]['drug_id']}, experiment {first_exp['experiment_id']}):")
				for mismatch in first_exp['gene_mismatches'][:5]:
					position = mismatch.get('position', 'unknown')
					orig_pos = mismatch.get('original_gene_came_from_position', 'unknown')
					print(f"  Position {position} (gene originally from position {orig_pos}):")
					if mismatch.get('incorrect_assemblies'):
						incorrect = mismatch['incorrect_assemblies']
						print(f"    Incorrectly assigned to: {incorrect[:3]}..." if len(incorrect) > 3 else f"    Incorrectly assigned to: {incorrect}")
					if mismatch.get('missing_assemblies'):
						missing = mismatch['missing_assemblies']
						print(f"    Missing from: {missing[:3]}..." if len(missing) > 3 else f"    Missing from: {missing}")
					if mismatch.get('correct_assemblies'):
						correct = mismatch['correct_assemblies']
						print(f"    Correctly assigned to: {correct[:3]}..." if len(correct) > 3 else f"    Correctly assigned to: {correct}")
	else:
		print("\nNo experiment results to report.")


if __name__ == "__main__":
	main()

