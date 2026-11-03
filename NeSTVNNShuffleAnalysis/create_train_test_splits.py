#!/usr/bin/env python3
#NeST Folds
#Usage (run from Data/nest_shuffle_data/; paths are relative to it):
# python create_train_test_splits.py --drugid 5
# python create_train_test_splits.py --drugid 80
# python create_train_test_splits.py --drugid 99
# python create_train_test_splits.py --drugid 127
# python create_train_test_splits.py --drugid 151
# python create_train_test_splits.py --drugid 188
# python create_train_test_splits.py --drugid 244
# python create_train_test_splits.py --drugid 273
# python create_train_test_splits.py --drugid 298
# python create_train_test_splits.py --drugid 380

"""
Script to create 100 different train/test splits from drug response data.
Each split will have 80% training data and 20% test data.
Also includes shuffled gene expression data in each experiment folder.
"""

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

def shuffle_gene_expression_data(ge_data, random_seed):
    """
    Shuffle the order of genes (columns) in the gene expression data.
    
    Args:
        ge_data (list): List of gene expression data lines
        random_seed (int): Random seed for reproducibility
    
    Returns:
        list: Shuffled gene expression data
    """
    # Set random seed for this shuffle
    random.seed(random_seed)
    
    # Parse the first line to get the number of genes
    first_line = ge_data[0].strip()
    genes = first_line.split(',')
    num_genes = len(genes)
    
    # Create a permutation of gene indices
    gene_indices = list(range(num_genes))
    random.shuffle(gene_indices)
    
    # Shuffle each line according to the gene permutation
    shuffled_data = []
    for line in ge_data:
        line = line.strip()
        if line:  # Skip empty lines
            values = line.split(',')
            # Reorder values according to shuffled gene indices
            shuffled_values = [values[i] for i in gene_indices]
            shuffled_data.append(','.join(shuffled_values))
    
    return shuffled_data

def create_train_test_splits(input_file, ge_file, output_dir, num_splits=100, train_ratio=0.8, random_seed=42):
    """
    Create multiple train/test splits from the input data file.
    
    Args:
        input_file (str): Path to the drug response data file
        ge_file (str): Path to the gene expression data file
        output_dir (str): Directory to save the split files
        num_splits (int): Number of different splits to create
        train_ratio (float): Ratio of data to use for training (0.8 = 80%)
        random_seed (int): Random seed for reproducibility
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the drug response data
    print(f"Reading drug response data from {input_file}...")
    with open(input_file, 'r') as f:
        drug_data_lines = f.readlines()
    
    # Remove any empty lines and strip whitespace
    drug_data_lines = [line.strip() for line in drug_data_lines if line.strip()]
    
    # Read the gene expression data
    print(f"Reading gene expression data from {ge_file}...")
    with open(ge_file, 'r') as f:
        ge_data_lines = f.readlines()
    
    # Remove any empty lines and strip whitespace
    ge_data_lines = [line.strip() for line in ge_data_lines if line.strip()]
    
    print(f"Total drug response data points: {len(drug_data_lines)}")
    print(f"Total gene expression data points: {len(ge_data_lines)}")
    print(f"Creating {num_splits} train/test splits...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    for i in range(num_splits):
        # Create a new random seed for each split to ensure different splits
        split_seed = random_seed + i
        
        # Split the drug response data
        train_drug_data, test_drug_data = train_test_split(
            drug_data_lines, 
            train_size=train_ratio, 
            random_state=split_seed,
            shuffle=True
        )
        
        # Create shuffled gene expression data for this experiment
        shuffled_ge_data = shuffle_gene_expression_data(ge_data_lines, split_seed)
        1
        
        # Create experiment folder
        experiment_folder = os.path.join(output_dir, f"experiment_{i}")
        os.makedirs(experiment_folder, exist_ok=True)
        
        # Write training drug response data
        train_filename = os.path.join(experiment_folder, "training_data.txt")
        with open(train_filename, 'w') as f:
            for line in train_drug_data:
                f.write(line + '\n')
        
        # Write test drug response data
        test_filename = os.path.join(experiment_folder, "test_data.txt")
        with open(test_filename, 'w') as f:
            for line in test_drug_data:
                f.write(line + '\n')
        
        # Write shuffled gene expression data
        ge_filename = os.path.join(experiment_folder, "shuffled_gene_expression.txt")
        with open(ge_filename, 'w') as f:
            for line in shuffled_ge_data:
                f.write(line + '\n')
        
        # Print progress every 10 splits
        if (i + 1) % 10 == 0:
            print(f"Created experiment {i+1}/{num_splits}")
    
    print(f"\nSuccessfully created {num_splits} train/test splits!")
    print(f"Training set size: {len(train_drug_data)} samples ({train_ratio*100:.0f}%)")
    print(f"Test set size: {len(test_drug_data)} samples ({(1-train_ratio)*100:.0f}%)")
    print(f"Gene expression data: {len(shuffled_ge_data)} cell lines × {len(shuffled_ge_data[0].split(','))} genes")
    print(f"Experiment folders saved in: {output_dir}")

def main():
    # Configuration
    # INSERT_YOUR_CODE
    import argparse

    parser = argparse.ArgumentParser(description="Create train/test splits for drug response data.")
    parser.add_argument('--drugid', type=int, required=True, help='Drug ID to process')
    args = parser.parse_args()
    drugid = args.drugid
    input_file = f"CombatLog2TPM/Drug{drugid}/D{drugid}_CL/D{drugid}_all_data.txt"
    ge_file = f"CombatLog2TPM/Drug{drugid}/D{drugid}_CL/D{drugid}_GE_Data.txt"
    output_dir = f"CombatLog2TPM/Drug{drugid}/D{drugid}_CL/train_test_splits"
    num_splits = 100
    train_ratio = 0.8  # 80% training, 20% test
    random_seed = 42
    
    # Check if input files exist
    if not os.path.exists(input_file):
        print(f"Error: Drug response file '{input_file}' not found!")
        return
    
    if not os.path.exists(ge_file):
        print(f"Error: Gene expression file '{ge_file}' not found!")
        return
    
    # Create the splits
    create_train_test_splits(
        input_file=input_file,
        ge_file=ge_file,
        output_dir=output_dir,
        num_splits=num_splits,
        train_ratio=train_ratio,
        random_seed=random_seed
    )

if __name__ == "__main__":
    main() 