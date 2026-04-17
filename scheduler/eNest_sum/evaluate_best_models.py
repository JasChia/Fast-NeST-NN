#!/usr/bin/env python3
"""
Evaluate best models for eNest_sum experiments.
Loads the best model for each experiment, computes R² and Pearson correlation
on the test data. Parallelizes by drug across multiple GPUs.
"""

import os
import sys
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import r2_score
from multiprocessing import Process, Queue, Manager, Value, Lock
import queue

# Suppress PearsonR constant input warnings
warnings.filterwarnings('ignore', category=stats.PearsonRConstantInputWarning)

# Add the eNest_sum directory to the path to import eNest
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from eNest import eNest


def load_mapping(mapping_file):
    """Load cell line to index mapping"""
    mapping = {}
    with open(mapping_file) as f:
        for line in f:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])
    return mapping


def load_pred_data(test_file, cell2id):
    """Load and preprocess test data"""
    test_df = pd.read_csv(test_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])
    
    feature = []
    label = []
    for row in test_df.values:
        feature.append([[cell2id[row[0]]]])
        label.append([float(row[2])])
    return feature, label


def build_input_vector(input_data, cell_features):
    """Build input vector from cell line IDs by looking up gene expression data"""
    num_genes = cell_features.shape[1]
    batch_size = input_data.size()[0]
    
    feature = np.zeros((batch_size, num_genes))
    for i in range(batch_size):
        cell_id = int(input_data[i, 0])
        feature[i] = cell_features[cell_id]
    
    return torch.from_numpy(feature).float()


def pearson_correlation(pred, true, drug_id=None, exp_id=None):
    """Calculate Pearson correlation coefficient"""
    pred_np = pred.flatten()
    true_np = true.flatten()
    
    if np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np)):
        return np.nan
    if np.any(np.isnan(true_np)) or np.any(np.isinf(true_np)):
        return np.nan
    
    # Check for constant arrays (all same value)
    pred_std = np.std(pred_np)
    true_std = np.std(true_np)
    if pred_std == 0 or true_std == 0:
        if drug_id is not None and exp_id is not None:
            if pred_std == 0:
                print(f"  Warning: D{drug_id}_{exp_id} - Constant predictions detected (std={pred_std:.6f})")
            if true_std == 0:
                print(f"  Warning: D{drug_id}_{exp_id} - Constant true labels detected (std={true_std:.6f})")
        return np.nan
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=stats.PearsonRConstantInputWarning)
            return stats.pearsonr(pred_np, true_np)[0]
    except (ValueError, RuntimeWarning):
        return np.nan


def evaluate_single_experiment(exp_dir: Path, drug_id: int, exp_id: int, device: str = 'cpu') -> Optional[Dict[str, Any]]:
    """
    Evaluate a single experiment by loading the best model and computing metrics on test data.
    """
    # Check if required files exist
    model_path = exp_dir / 'best_model' / 'model_best.pt'
    results_path = exp_dir / 'final_results.json'
    hparams_path = exp_dir / 'best_model' / 'hyperparameters.json'
    
    if not model_path.exists():
        return None
    
    try:
        # Load hyperparameters - try multiple sources
        hparams = {}
        if hparams_path.exists():
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)
        elif results_path.exists():
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        hparams = results_json.get('best_hyperparameters', {})
        
        # Build data file paths
        data_dir = Path('/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM')
        drug_dir = data_dir / f'Drug{drug_id}' / f'D{drug_id}_CL'
        test_file = drug_dir / 'train_test_splits' / f'experiment_{exp_id}' / 'test_data.txt'
        cell2id_file = drug_dir / f'D{drug_id}_cell2ind.txt'
        ge_data_file = drug_dir / f'D{drug_id}_GE_Data.txt'
        
        if not test_file.exists() or not cell2id_file.exists() or not ge_data_file.exists():
            return None
        
        # Load data
        cell2id = load_mapping(str(cell2id_file))
        ge_data = pd.read_csv(ge_data_file, sep=',', header=None).values
        test_features, test_labels = load_pred_data(str(test_file), cell2id)
        
        test_data = torch.Tensor(test_features)
        test_labels = torch.FloatTensor(test_labels).squeeze()
        test_input = build_input_vector(test_data, ge_data)
        
        # Get activation function
        activation_name = hparams.get('activation', 'Tanh')
        if activation_name == 'Tanh':
            activation = nn.Tanh
        elif activation_name == 'ReLU':
            activation = nn.ReLU
        else:
            activation = nn.Tanh
        
        dropout = hparams.get('dropout_fraction', 0.0)
        output_method = hparams.get('output_method', 'final')
        batchnorm_position = hparams.get('batchnorm_position', 'before_activation')
        
        # Ensure device is set correctly
        if device.startswith('cuda:'):
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
        
        # Create model
        model = eNest(
            nodes_per_assembly=4,
            dropout=dropout,
            activation=activation,
            output_method=output_method,
            batchnorm_position=batchnorm_position,
            verbosity=-1
        ).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Run inference
        with torch.no_grad():
            test_input_dev = test_input.to(device)
            final_pred, hidden_asm_pred = model(test_input_dev)
            
            final_pred_np = final_pred.cpu().numpy().flatten()
            true_labels = test_labels.numpy().flatten()
        
        # Compute metrics
        r2 = r2_score(true_labels, final_pred_np)
        pearson = pearson_correlation(final_pred_np, true_labels, drug_id, exp_id)
        
        return {
            'drug_id': drug_id,
            'experiment_id': exp_id,
            'r2_test': r2,
            'pearson_test': pearson,
            'output_method': output_method,
            'batchnorm_position': batchnorm_position
        }
        
    except Exception as e:
        print(f"  Error evaluating D{drug_id}_{exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_drug(drug_id: int, results_dir: Path, device: str, result_queue: Queue, total_counter: Value, lock: Lock):
    """
    Process all experiments for a single drug on a specific GPU.
    Puts results into the queue and updates the total counter.
    """
    # Explicitly set the CUDA device for this process
    if device.startswith('cuda:'):
        gpu_id = int(device.split(':')[1])
        torch.cuda.set_device(gpu_id)
        actual_device = torch.cuda.current_device()
        print(f"  Process for Drug {drug_id}: Set CUDA device to {gpu_id}, current device: {actual_device}")
    
    drug_folder = results_dir / f'D{drug_id}'
    if not drug_folder.exists():
        return
    
    drug_results = []
    exp_count = 0
    
    for exp_folder in sorted(drug_folder.iterdir()):
        if not exp_folder.is_dir():
            continue
        
        match = re.match(r'D\d+_(\d+)', exp_folder.name)
        if not match:
            continue
        
        exp_id = int(match.group(1))
        
        result = evaluate_single_experiment(exp_folder, drug_id, exp_id, device)
        if result:
            drug_results.append(result)
            exp_count += 1
            
            with lock:
                total_counter.value += 1
                total = total_counter.value
                if total % 100 == 0:
                    print(f"  [PROGRESS] Completed {total} experiments total")
    
    print(f"  Drug {drug_id} on {device}: Evaluated {exp_count} experiments")
    
    for result in drug_results:
        result_queue.put(result)


def main():
    base_dir = Path('/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/eNest_sum')
    results_dir = base_dir / 'results'
    output_dir = base_dir / 'evaluation_results'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("EVALUATING ENEST_SUM BEST MODELS")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s):")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        devices = [f'cuda:{i}' for i in range(num_gpus)]
    else:
        print("No GPUs available, using CPU")
        devices = ['cpu']
        num_gpus = 1
    
    # Find all drug folders
    drug_ids = []
    skip_drugs = {57, 201}  # Skip these drugs if problematic
    for drug_folder in sorted(results_dir.iterdir()):
        if drug_folder.is_dir() and drug_folder.name.startswith('D'):
            try:
                drug_id = int(drug_folder.name[1:])
                if drug_id not in skip_drugs:
                    drug_ids.append(drug_id)
                else:
                    print(f"Skipping drug {drug_id} (as configured)")
            except ValueError:
                continue
    
    print(f"\nFound {len(drug_ids)} drugs to process: {drug_ids}")
    
    # Use multiprocessing with one drug per GPU
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    manager = Manager()
    result_queue = manager.Queue()
    total_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    processes = []
    
    print("\nStarting parallel processing...")
    print(f"Processing {len(drug_ids)} drugs across {len(devices)} device(s)")
    print("Progress will be logged every 100 experiments completed\n")
    
    batch_size = len(devices)
    
    for batch_start in range(0, len(drug_ids), batch_size):
        batch_drugs = drug_ids[batch_start:batch_start + batch_size]
        batch_processes = []
        
        for i, drug_id in enumerate(batch_drugs):
            device = devices[i % len(devices)]
            print(f"Starting Drug {drug_id} on {device}...")
            p = Process(target=process_drug, args=(drug_id, results_dir, device, result_queue, total_counter, lock))
            batch_processes.append(p)
            p.start()
        
        for p in batch_processes:
            p.join()
            processes.append(p)
        
        print(f"Completed batch {batch_start // batch_size + 1}/{(len(drug_ids) + batch_size - 1) // batch_size}")
    
    print("\nAll processes completed. Collecting results...")
    
    # Collect all results from the queue
    all_results = []
    while True:
        try:
            result = result_queue.get_nowait()
            all_results.append(result)
        except queue.Empty:
            break
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"Collected {len(all_results)} experiment results")
    
    # Save per-experiment results to CSV
    df = pd.DataFrame(all_results)
    per_exp_csv = output_dir / 'per_experiment_metrics.csv'
    df.to_csv(per_exp_csv, index=False)
    print(f"\nPer-experiment metrics saved to: {per_exp_csv}")
    
    # Compute summary statistics by drug
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS BY DRUG")
    print("=" * 60)
    
    summary_data = []
    
    for drug_id in sorted(df['drug_id'].unique()):
        drug_data = df[df['drug_id'] == drug_id]
        
        summary_row = {
            'drug_id': drug_id,
            'n_experiments': len(drug_data),
            'r2_test_mean': drug_data['r2_test'].mean(),
            'r2_test_std': drug_data['r2_test'].std(),
            'pearson_test_mean': drug_data['pearson_test'].mean(),
            'pearson_test_std': drug_data['pearson_test'].std(),
        }
        summary_data.append(summary_row)
        
        print(f"\nDrug D{drug_id} (n={len(drug_data)}):")
        print(f"  R² Test:      {summary_row['r2_test_mean']:.4f} ± {summary_row['r2_test_std']:.4f}")
        print(f"  Pearson Test: {summary_row['pearson_test_mean']:.4f} ± {summary_row['pearson_test_std']:.4f}")
    
    # Save summary statistics to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'summary_statistics_by_drug.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary statistics saved to: {summary_csv}")
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  1. {per_exp_csv.name} (per-experiment metrics)")
    print(f"  2. {summary_csv.name} (summary statistics)")


if __name__ == '__main__':
    main()
