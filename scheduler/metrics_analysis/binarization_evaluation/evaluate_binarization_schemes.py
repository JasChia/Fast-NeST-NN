#!/usr/bin/env python3
"""
Binarization Scheme Evaluation for eNest_linear_fair with Fine-tuning

This script evaluates two AUC binarization schemes on the eNest_linear_fair model:
1. Median-based binary: AUC > median = Sensitive (1), AUC <= median = Resistant (0)
2. 3-Category scheme: AUC > median + 1*std = Sensitive (1), 
                      AUC < median - 1*std = Resistant (0),
                      Otherwise = Undefined (excluded from metrics)

For each saved model, the script:
- Loads the best model from results/D{drug_id}/D{drug_id}_{exp_id}/best_model/model_best.pt
- Uses the same test folds from training (computed programmatically)
- Computes binarization thresholds (median, median ± std) from TRAINING set
- Fine-tunes the model for up to 10 epochs on binarized training labels
- Applies these thresholds to TEST set labels for binarization
- Computes Pearson correlation on the binarized test targets

Usage:
    python evaluate_binarization_schemes.py [--drugs 5,80,99] [--output_dir ./results]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
SCHEDULER_DIR = SCRIPT_DIR.parent.parent
ENEST_DIR = SCHEDULER_DIR / "eNest_linear_fair"
sys.path.insert(0, str(ENEST_DIR))

from eNest import eNest


# ========================== Configuration ==========================
BASE_DATA_PATH = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"
ENEST_RESULTS_DIR = ENEST_DIR / "results"
DEFAULT_DRUGS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
EXPERIMENTS_PER_DRUG = 50

# Fine-tuning parameters
FINETUNE_EPOCHS = 10
FINETUNE_PATIENCE = 3
FINETUNE_LR = 1e-4
VAL_IMPROVEMENT_EPS = 1e-4


# ========================== Data Loading Functions ==========================
def load_mapping(mapping_file: str) -> Dict[str, int]:
    """Load cell line to index mapping."""
    mapping: Dict[str, int] = {}
    with open(mapping_file) as fh:
        for line in fh:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])
    return mapping


def load_test_data(test_file: str, cell2id: Dict[str, int]) -> Tuple[List, List]:
    """Load test data - returns cell line IDs and AUC labels."""
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"])
    # Feature is just cell line ID for lookup
    feature = [[[cell2id[row[0]]]] for row in test_df.values]
    label = [float(row[2]) for row in test_df.values]
    return feature, label


def load_train_val_data(train_file: str, val_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training and validation AUC values for threshold computation and fine-tuning.
    
    Training/Val file format: cell_line_id<TAB>auc_value (2 columns)
    """
    train_df = pd.read_csv(train_file, sep="\t", header=None, names=["cell_line_id", "auc"])
    val_df = pd.read_csv(val_file, sep="\t", header=None, names=["cell_line_id", "auc"])
    train_auc = train_df["auc"].values.astype(np.float32)
    val_auc = val_df["auc"].values.astype(np.float32)
    train_ids = train_df["cell_line_id"].values.astype(np.int64)
    val_ids = val_df["cell_line_id"].values.astype(np.int64)
    return train_ids, train_auc, val_ids, val_auc


def build_input_vector(input_data: torch.Tensor, cell_features: np.ndarray) -> torch.Tensor:
    """Build input vector from cell line IDs by looking up gene expression data."""
    num_genes = cell_features.shape[1]
    batch_size = input_data.size(0)
    feature = np.zeros((batch_size, num_genes), dtype=np.float32)
    for i in range(batch_size):
        cell_id = int(input_data[i, 0])
        feature[i] = cell_features[cell_id]
    return torch.from_numpy(feature)


def get_data_paths(drug_id: int, experiment_id: int) -> Dict[str, str]:
    """Get data file paths for a specific drug and experiment (mirrors generate_jobs.py)."""
    base_path = BASE_DATA_PATH
    return {
        "cell2id": f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug_id}/D{drug_id}_CL/D{drug_id}_cell2ind.txt",
        "ge_data": f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug_id}/D{drug_id}_CL/D{drug_id}_GE_Data.txt",
        "train_file": f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug_id}/D{drug_id}_CL/train_test_splits/experiment_{experiment_id}/true_training_data.txt",
        "val_file": f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug_id}/D{drug_id}_CL/train_test_splits/experiment_{experiment_id}/validation_data.txt",
        "test_file": f"{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug_id}/D{drug_id}_CL/train_test_splits/experiment_{experiment_id}/test_data.txt",
    }


# ========================== Binarization Functions ==========================
def compute_training_thresholds(train_auc: np.ndarray) -> Dict[str, float]:
    """
    Compute binarization thresholds from training set AUC values.
    
    Returns:
        Dictionary with 'median', 'std', 'upper_threshold', 'lower_threshold'
    """
    median_val = np.median(train_auc)
    std_val = np.std(train_auc)
    return {
        "median": float(median_val),
        "std": float(std_val),
        "upper_threshold": float(median_val + std_val),
        "lower_threshold": float(median_val - std_val)
    }


def binarize_median(auc_values: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scheme 1: Median-based binary classification.
    
    - Sensitive (1): AUC > threshold (median from training)
    - Resistant (0): AUC <= threshold
    
    Args:
        auc_values: AUC values to binarize
        threshold: Median value computed from training set
    
    Returns:
        binary_labels: Array of 0/1 labels
        mask: All True (no samples excluded)
    """
    binary_labels = (auc_values > threshold).astype(np.float32)
    mask = np.ones(len(auc_values), dtype=bool)
    return binary_labels, mask


def categorize_3class(auc_values: np.ndarray, upper_threshold: float, 
                      lower_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scheme 2: 3-category classification with undefined middle zone.
    
    - Sensitive (1): AUC > upper_threshold (median + std from training)
    - Resistant (0): AUC < lower_threshold (median - std from training)
    - Undefined: lower_threshold <= AUC <= upper_threshold (excluded from metrics)
    
    Args:
        auc_values: AUC values to categorize
        upper_threshold: median + std computed from training set
        lower_threshold: median - std computed from training set
    
    Returns:
        categorical_labels: Array of 0/1 labels (undefined samples set to -1)
        mask: Boolean mask, True for samples to include (sensitive/resistant only)
    """
    # Initialize as undefined (-1)
    categorical_labels = np.full(len(auc_values), -1, dtype=np.float32)
    
    # Sensitive: AUC > upper threshold
    sensitive_mask = auc_values > upper_threshold
    categorical_labels[sensitive_mask] = 1.0
    
    # Resistant: AUC < lower threshold
    resistant_mask = auc_values < lower_threshold
    categorical_labels[resistant_mask] = 0.0
    
    # Mask: only include sensitive and resistant samples
    valid_mask = sensitive_mask | resistant_mask
    
    return categorical_labels, valid_mask


# ========================== Metrics Functions ==========================
def compute_pearson(predictions: np.ndarray, true_labels: np.ndarray, 
                    mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute Pearson correlation between predictions and binarized labels.
    
    Args:
        predictions: Model predictions (continuous values)
        true_labels: Binarized true labels
        mask: Optional boolean mask to exclude samples
        
    Returns:
        Dictionary with pearson_r, pearson_p
    """
    if mask is not None:
        predictions = predictions[mask]
        true_labels = true_labels[mask]
    
    if len(predictions) < 2:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "n_samples": len(predictions)}
    
    # Check for constant arrays (would result in NaN)
    if np.std(true_labels) < 1e-10 or np.std(predictions) < 1e-10:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "n_samples": len(predictions)}
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(predictions.flatten(), true_labels.flatten())
    
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p, 
        "n_samples": len(predictions)
    }


# ========================== Model Loading and Inference ==========================
def load_model(model_path: Path, hyperparams_path: Path, device: torch.device) -> nn.Module:
    """Load eNest model with saved hyperparameters."""
    # Load hyperparameters
    with open(hyperparams_path) as f:
        hparams = json.load(f)
    
    # Map activation string to class
    activation_map = {"Tanh": nn.Tanh, "ReLU": nn.ReLU}
    activation = activation_map.get(hparams.get("activation", "Tanh"), nn.Tanh)
    
    # Create model
    model = eNest(
        nodes_per_assembly=4,  # Fixed from the trainer
        dropout=hparams.get("dropout_fraction", 0.0),
        activation=activation,
        output_method=hparams.get("output_method", "linear"),
        verbosity=-1
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, hparams


def run_inference(model: nn.Module, test_input: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run model inference and return predictions."""
    model.eval()
    with torch.no_grad():
        predictions, _ = model(test_input.to(device))
    return predictions.cpu().numpy().flatten()


# ========================== Fine-tuning ==========================
def finetune_model(model: nn.Module, 
                   train_input: torch.Tensor, train_labels: torch.Tensor,
                   val_input: torch.Tensor, val_labels: torch.Tensor,
                   device: torch.device, hparams: Dict) -> nn.Module:
    """
    Fine-tune the model for up to FINETUNE_EPOCHS with binarized labels.
    
    Uses the same training loop as eNest_hparam_tuner.py except:
    - Labels are binarized
    - Selection is based on best validation Pearson instead of R²
    - Maximum 10 epochs with early stopping (patience=3)
    
    Returns the model with best validation Pearson.
    """
    model.train()
    model.register_grad_hooks()
    
    # Get hyperparameters from original training
    wd = hparams.get("wd", 0.0) if hparams.get("wd_bool", False) else 0.0
    l1 = hparams.get("l1", 0.0) if hparams.get("l1_bool", False) else 0.0
    batch_size_power = hparams.get("batch_size_power", 5)
    batch_size = int(2 ** batch_size_power)
    
    # Move data to device
    train_input = train_input.to(device)
    train_labels = train_labels.to(device)
    val_input = val_input.to(device)
    val_labels = val_labels.to(device)
    
    if train_labels.dim() == 1:
        train_labels = train_labels.unsqueeze(1)
    if val_labels.dim() == 1:
        val_labels = val_labels.unsqueeze(1)
    
    # Create dataloader
    dataloader = DataLoader(
        TensorDataset(train_input, train_labels), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, betas=(0.9, 0.99), eps=1e-8, weight_decay=wd)
    loss_fn = nn.MSELoss()
    
    best_val_pearson = float('-inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        # Training
        model.train()
        for input_data, labels in dataloader:
            optimizer.zero_grad()
            
            if not input_data.is_cuda:
                input_data = input_data.to(device)
            if not labels.is_cuda:
                labels = labels.to(device)
            
            pred, _ = model(input_data)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            mse_loss = loss_fn(pred, labels)
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            total_loss = mse_loss + l1_loss * l1
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_input)
            val_pred_np = val_pred.cpu().numpy().flatten()
            val_labels_np = val_labels.cpu().numpy().flatten()
            
            # Compute Pearson
            if np.std(val_labels_np) > 1e-10 and np.std(val_pred_np) > 1e-10:
                val_pearson, _ = pearsonr(val_pred_np, val_labels_np)
            else:
                val_pearson = float('-inf')
        
        # Check for improvement
        if val_pearson > best_val_pearson + VAL_IMPROVEMENT_EPS:
            best_val_pearson = val_pearson
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_PATIENCE:
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    return model


# ========================== Single Experiment Evaluation ==========================
def evaluate_single_experiment(drug_id: int, experiment_id: int, 
                                device: torch.device) -> Optional[Dict]:
    """
    Evaluate both binarization schemes for a single experiment with fine-tuning.
    
    Returns:
        Dictionary with results for both schemes, or None if model not found or all labels same
    """
    # Check if model exists
    model_dir = ENEST_RESULTS_DIR / f"D{drug_id}" / f"D{drug_id}_{experiment_id}" / "best_model"
    model_path = model_dir / "model_best.pt"
    hyperparams_path = model_dir / "hyperparameters.json"
    
    if not model_path.exists() or not hyperparams_path.exists():
        return None
    
    try:
        # Get data paths
        data_paths = get_data_paths(drug_id, experiment_id)
        
        # Load data
        cell2id = load_mapping(data_paths["cell2id"])
        ge_data = pd.read_csv(data_paths["ge_data"], sep=",", header=None).values
        
        # Load training and validation data
        train_ids, train_auc, val_ids, val_auc = load_train_val_data(
            data_paths["train_file"], data_paths["val_file"]
        )
        thresholds = compute_training_thresholds(train_auc)
        
        # Build training/val input vectors
        train_data = torch.tensor(train_ids.reshape(-1, 1), dtype=torch.float32)
        val_data = torch.tensor(val_ids.reshape(-1, 1), dtype=torch.float32)
        train_input = build_input_vector(train_data, ge_data)
        val_input = build_input_vector(val_data, ge_data)
        
        # Load test data
        test_features, test_labels = load_test_data(data_paths["test_file"], cell2id)
        test_data = torch.tensor(test_features, dtype=torch.float32)
        test_input = build_input_vector(test_data, ge_data)
        test_labels_np = np.array(test_labels)
        
        results = {
            "drug_id": drug_id,
            "experiment_id": experiment_id,
            "n_test_samples": len(test_labels_np),
            "n_train_samples": len(train_auc),
        }
        
        # ==================== Scheme 1: Median-based binary ====================
        # Binarize labels
        train_binary, train_mask = binarize_median(train_auc, thresholds["median"])
        val_binary, val_mask = binarize_median(val_auc, thresholds["median"])
        test_binary, test_mask = binarize_median(test_labels_np, thresholds["median"])
        
        # Check if all labels are the same (would cause NaN)
        if np.std(train_binary) < 1e-10 or np.std(val_binary) < 1e-10:
            results["median_scheme"] = {
                "threshold": thresholds["median"],
                "skipped": True,
                "reason": "constant_labels"
            }
        else:
            # Load fresh model and fine-tune
            model, hparams = load_model(model_path, hyperparams_path, device)
            train_labels_tensor = torch.tensor(train_binary, dtype=torch.float32)
            val_labels_tensor = torch.tensor(val_binary, dtype=torch.float32)
            
            model = finetune_model(
                model, train_input, train_labels_tensor,
                val_input, val_labels_tensor, device, hparams
            )
            
            # Evaluate on test set
            predictions = run_inference(model, test_input, device)
            metrics = compute_pearson(predictions, test_binary, test_mask)
            
            results["median_scheme"] = {
                "threshold": thresholds["median"],
                "skipped": False,
                **metrics
            }
        
        # ==================== Scheme 2: 3-category ====================
        # Binarize labels
        train_cat, train_cat_mask = categorize_3class(
            train_auc, thresholds["upper_threshold"], thresholds["lower_threshold"]
        )
        val_cat, val_cat_mask = categorize_3class(
            val_auc, thresholds["upper_threshold"], thresholds["lower_threshold"]
        )
        test_cat, test_cat_mask = categorize_3class(
            test_labels_np, thresholds["upper_threshold"], thresholds["lower_threshold"]
        )
        
        # Filter to only valid samples for training/val
        train_cat_valid = train_cat[train_cat_mask]
        val_cat_valid = val_cat[val_cat_mask]
        train_input_valid = train_input[train_cat_mask]
        val_input_valid = val_input[val_cat_mask]
        
        # Check if all labels are the same or too few samples
        if (len(train_cat_valid) < 10 or len(val_cat_valid) < 5 or 
            np.std(train_cat_valid) < 1e-10 or np.std(val_cat_valid) < 1e-10):
            results["categorical_scheme"] = {
                "upper_threshold": thresholds["upper_threshold"],
                "lower_threshold": thresholds["lower_threshold"],
                "n_sensitive": int(np.sum(test_cat == 1)),
                "n_resistant": int(np.sum(test_cat == 0)),
                "n_undefined": int(np.sum(test_cat == -1)),
                "skipped": True,
                "reason": "insufficient_valid_samples"
            }
        else:
            # Load fresh model and fine-tune
            model, hparams = load_model(model_path, hyperparams_path, device)
            train_labels_tensor = torch.tensor(train_cat_valid, dtype=torch.float32)
            val_labels_tensor = torch.tensor(val_cat_valid, dtype=torch.float32)
            
            model = finetune_model(
                model, train_input_valid, train_labels_tensor,
                val_input_valid, val_labels_tensor, device, hparams
            )
            
            # Evaluate on test set (only valid samples)
            predictions = run_inference(model, test_input, device)
            metrics = compute_pearson(predictions, test_cat, test_cat_mask)
            
            results["categorical_scheme"] = {
                "upper_threshold": thresholds["upper_threshold"],
                "lower_threshold": thresholds["lower_threshold"],
                "n_sensitive": int(np.sum(test_cat == 1)),
                "n_resistant": int(np.sum(test_cat == 0)),
                "n_undefined": int(np.sum(test_cat == -1)),
                "skipped": False,
                **metrics
            }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating D{drug_id}_{experiment_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================== Aggregation Functions ==========================
def aggregate_drug_results(drug_results: List[Dict]) -> Dict:
    """Aggregate results across experiments for a single drug."""
    if not drug_results:
        return {}
    
    drug_id = drug_results[0]["drug_id"]
    
    # Extract metrics for median scheme
    median_pearson = []
    median_skipped = 0
    for r in drug_results:
        if r["median_scheme"].get("skipped", False):
            median_skipped += 1
        elif not np.isnan(r["median_scheme"].get("pearson_r", np.nan)):
            median_pearson.append(r["median_scheme"]["pearson_r"])
    
    # Extract metrics for categorical scheme
    cat_pearson = []
    cat_skipped = 0
    for r in drug_results:
        if r["categorical_scheme"].get("skipped", False):
            cat_skipped += 1
        elif not np.isnan(r["categorical_scheme"].get("pearson_r", np.nan)):
            cat_pearson.append(r["categorical_scheme"]["pearson_r"])
    
    return {
        "drug_id": drug_id,
        "n_experiments": len(drug_results),
        "median_scheme": {
            "pearson_mean": np.mean(median_pearson) if median_pearson else np.nan,
            "pearson_std": np.std(median_pearson) if median_pearson else np.nan,
            "n_valid": len(median_pearson),
            "n_skipped": median_skipped,
        },
        "categorical_scheme": {
            "pearson_mean": np.mean(cat_pearson) if cat_pearson else np.nan,
            "pearson_std": np.std(cat_pearson) if cat_pearson else np.nan,
            "n_valid": len(cat_pearson),
            "n_skipped": cat_skipped,
        }
    }


# ========================== Main Evaluation ==========================
def evaluate_all_experiments(drugs: List[int], output_dir: Path, 
                             device: torch.device) -> None:
    """Evaluate all experiments for given drugs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    drug_summaries = []
    
    total_median_skipped = 0
    total_cat_skipped = 0
    total_experiments = 0
    
    for drug_id in drugs:
        print(f"\nEvaluating Drug {drug_id}...")
        drug_results = []
        
        for exp_id in range(EXPERIMENTS_PER_DRUG):
            result = evaluate_single_experiment(drug_id, exp_id, device)
            if result:
                drug_results.append(result)
                all_results.append(result)
                total_experiments += 1
                
                # Print progress
                if (exp_id + 1) % 10 == 0:
                    print(f"  Completed {exp_id + 1}/{EXPERIMENTS_PER_DRUG} experiments")
                
        if drug_results:
            summary = aggregate_drug_results(drug_results)
            drug_summaries.append(summary)
            
            total_median_skipped += summary["median_scheme"]["n_skipped"]
            total_cat_skipped += summary["categorical_scheme"]["n_skipped"]
            
            print(f"  Drug {drug_id}: {len(drug_results)} experiments evaluated")
            print(f"    Median scheme - Pearson: {summary['median_scheme']['pearson_mean']:.4f} ± {summary['median_scheme']['pearson_std']:.4f} (skipped: {summary['median_scheme']['n_skipped']})")
            print(f"    Categorical scheme - Pearson: {summary['categorical_scheme']['pearson_mean']:.4f} ± {summary['categorical_scheme']['pearson_std']:.4f} (skipped: {summary['categorical_scheme']['n_skipped']})")
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    
    # Save drug summaries
    with open(output_dir / "drug_summaries.json", "w") as f:
        json.dump(drug_summaries, f, indent=2, default=float)
    
    # Create summary CSV tables
    create_summary_tables(drug_summaries, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments evaluated: {total_experiments}")
    print(f"Median scheme - Total skipped: {total_median_skipped}")
    print(f"Categorical scheme - Total skipped: {total_cat_skipped}")


def create_summary_tables(drug_summaries: List[Dict], output_dir: Path) -> None:
    """Create CSV summary tables for both schemes."""
    
    # Median scheme table
    median_rows = []
    for summary in drug_summaries:
        median_rows.append({
            "Drug": f"D{summary['drug_id']}",
            "N_Experiments": summary["n_experiments"],
            "Pearson_Mean": summary["median_scheme"]["pearson_mean"],
            "Pearson_Std": summary["median_scheme"]["pearson_std"],
            "N_Valid": summary["median_scheme"]["n_valid"],
            "N_Skipped": summary["median_scheme"]["n_skipped"],
        })
    median_df = pd.DataFrame(median_rows)
    median_df.to_csv(output_dir / "median_binarization_results.csv", index=False)
    
    # Categorical scheme table
    cat_rows = []
    for summary in drug_summaries:
        cat_rows.append({
            "Drug": f"D{summary['drug_id']}",
            "N_Experiments": summary["n_experiments"],
            "Pearson_Mean": summary["categorical_scheme"]["pearson_mean"],
            "Pearson_Std": summary["categorical_scheme"]["pearson_std"],
            "N_Valid": summary["categorical_scheme"]["n_valid"],
            "N_Skipped": summary["categorical_scheme"]["n_skipped"],
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(output_dir / "categorical_binarization_results.csv", index=False)
    
    # Combined comparison table with formatted output
    comparison_rows = []
    for summary in drug_summaries:
        comparison_rows.append({
            "Drug": f"D{summary['drug_id']}",
            "Median_Pearson": f"{summary['median_scheme']['pearson_mean']:.4f} ± {summary['median_scheme']['pearson_std']:.4f}",
            "Median_Skipped": summary["median_scheme"]["n_skipped"],
            "Categorical_Pearson": f"{summary['categorical_scheme']['pearson_mean']:.4f} ± {summary['categorical_scheme']['pearson_std']:.4f}",
            "Categorical_Skipped": summary["categorical_scheme"]["n_skipped"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "binarization_comparison.csv", index=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AUC binarization schemes on eNest_linear_fair models with fine-tuning"
    )
    parser.add_argument(
        "--drugs", 
        type=str, 
        default=None,
        help="Comma-separated list of drug IDs (default: all drugs)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device index (-1 for CPU)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse drugs
    if args.drugs:
        drugs = [int(d.strip()) for d in args.drugs.split(",")]
    else:
        drugs = DEFAULT_DRUGS
    
    # Setup device
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("AUC Binarization Scheme Evaluation for eNest_linear_fair")
    print("(with fine-tuning on binarized labels)")
    print("=" * 60)
    print(f"Drugs: {drugs}")
    print(f"Output: {output_dir}")
    print(f"Fine-tuning: up to {FINETUNE_EPOCHS} epochs, patience={FINETUNE_PATIENCE}")
    print(f"Schemes:")
    print("  1. Median-based binary (thresholds from training set)")
    print("     Sensitive: AUC > training_median, Resistant: AUC <= training_median")
    print("  2. 3-Category (thresholds from training set)")
    print("     Sensitive: AUC > training_median+std, Resistant: AUC < training_median-std")
    print("     Undefined samples excluded from training and metrics")
    print("=" * 60)
    
    evaluate_all_experiments(drugs, output_dir, device)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
