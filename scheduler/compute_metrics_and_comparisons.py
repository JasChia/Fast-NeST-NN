#!/usr/bin/env python3
"""
Extract metrics from all folders with results/D{drug_id}/D{drug_id}_{experiment_id}/best_model/metrics.csv
Also generates tables performance tables (mean ± std) and p-values (paired t-tests and Wilcoxon rank-sum tests with and without BH column-wise adjustment).
"""

import os
import csv
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Mapping from method names to display names for CSV columns
# Only methods listed here will be searched for and included in the combined CSV
METHOD_DISPLAY_NAMES = {
    'sparse_wd/ERK_SNN': 'ERK SNN',
    # 'sparse_wd/GraNet_DI': 'GraNet Direct Input',
    # 'sparse_wd/Old_GraNet_DI': 'Old GraNet Direct Input',
    # 'eNest': 'eNest',
    'eNest_linear_fair': 'eNest',
    'Old_eNest': 'Old_eNest',
    'fc_nn': 'Fully Connected NN',
    # 'fc_nn_direct_input' and 'uniform_random_direct_output_SNN' removed from repo
    'fc_nn_DI_Layer_Pred': 'Fully Connected NN Direct Input Direct Layer Predictions',
    # 'global_prune_nn': 'Global Prune NN',
    'global_prune_nn_warmup': 'Global Prune NN',
    'layer_prune_nn': 'Layer-wise Prune NN',
    'og_nest_vnn': 'Nest VNN',
    'r_sparse_nn': 'Uniform Random Sparse NN',
    'relaxed_global_prune_nn_warmup': 'Relaxed Global Prune NN',
    'uniform_random_DO_DI' : 'Direct Output Direct Input Uniform Random Sparse NN'
}

# List of drug IDs to process (as strings for consistent comparison)
ALLOWED_DRUG_IDS = {'5', '80', '99', '127', '151', '188', '244', '273', '298', '380'}


def find_all_method_folders(base_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find method folders that are in METHOD_DISPLAY_NAMES and have results directories.
    Only checks for methods listed in METHOD_DISPLAY_NAMES.
    Handles both direct methods (e.g., 'fc_nn') and nested methods (e.g., 'sparse_wd/ERK_SNN').
    Returns list of (folder_path, method_name) tuples.
    """
    methods = []
    
    # Get all method names from METHOD_DISPLAY_NAMES
    valid_method_names = set(METHOD_DISPLAY_NAMES.keys())
    
    # Separate direct methods (no '/') and nested methods (with '/')
    direct_methods = {m for m in valid_method_names if '/' not in m}
    nested_methods = {m for m in valid_method_names if '/' in m}
    
    # Check direct subdirectories (e.g., 'fc_nn', 'eNest')
    for method_name in direct_methods:
        method_dir = base_dir / method_name
        if method_dir.exists() and method_dir.is_dir():
            results_dir = method_dir / 'results'
            if results_dir.exists() and results_dir.is_dir():
                methods.append((method_dir, method_name))
    
    # Check nested subdirectories (e.g., 'sparse_wd/ERK_SNN')
    for nested_path in nested_methods:
        parts = nested_path.split('/')
        if len(parts) == 2:
            parent_dir, sub_dir = parts
            method_dir = base_dir / parent_dir / sub_dir
            if method_dir.exists() and method_dir.is_dir():
                results_dir = method_dir / 'results'
                if results_dir.exists() and results_dir.is_dir():
                    methods.append((method_dir, nested_path))
    
    return methods

def extract_drug_and_experiment(path: Path) -> Optional[Tuple[str, str]]:
    """
    Extract drug_id and experiment_number from path like:
    results/D{drug_id}/D{drug_id}_{experiment_number}/best_model/metrics.csv
    """
    # Pattern to match D{drug_id}/D{drug_id}_{experiment_number}
    pattern = r'D(\d+)/D\d+_(\d+)/best_model/metrics\.csv'
    match = re.search(pattern, str(path))
    
    if match:
        drug_id = match.group(1)
        experiment_number = match.group(2)
        return (drug_id, experiment_number)
    
    return None

def find_all_metrics_files(method_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Find all metrics.csv files for a given method folder.
    Uses the fixed structure: results/D{drug_id}/D{drug_id}_{experiment_number}/best_model/metrics.csv
    Returns list of tuples: (path, drug_id, experiment_number)
    """
    results = []
    results_dir = method_dir / 'results'
    
    if not results_dir.exists():
        return results
    
    # Use the fixed structure: results/D{drug_id}/D{drug_id}_{experiment_number}/best_model/metrics.csv
    # Find all D* directories in results
    for drug_dir in results_dir.iterdir():
        if not drug_dir.is_dir():
            continue
        
        # Check if directory name matches D{drug_id} pattern
        drug_pattern = r'^D(\d+)$'
        drug_match = re.match(drug_pattern, drug_dir.name)
        if not drug_match:
            continue
        
        drug_id = drug_match.group(1)
        
        # Only process allowed drug IDs
        if drug_id not in ALLOWED_DRUG_IDS:
            continue
        
        # Find all experiment directories D{drug_id}_{experiment_number}
        for exp_dir in drug_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Check if directory name matches D{drug_id}_{experiment_number} pattern
            exp_pattern = rf'^D{drug_id}_(\d+)$'
            exp_match = re.match(exp_pattern, exp_dir.name)
            if not exp_match:
                continue
            
            experiment_number = exp_match.group(1)
            
            # Check for best_model/metrics.csv
            metrics_file = exp_dir / 'best_model' / 'metrics.csv'
            if metrics_file.exists():
                results.append((metrics_file, drug_id, experiment_number))
    
    return results


def extract_metrics_from_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    """
    Extract R² Score, Pearson Correlation, and Spearman Correlation
    from a metrics.csv file.
    
    Returns a dictionary with keys:
    - r2_validation, r2_test
    - pearson_validation, pearson_test
    - spearman_validation, spearman_test
    """
    metrics = {}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_name = row.get('Metric', '').strip()
                validation_val = row.get('Validation', '').strip()
                test_val = row.get('Test', '').strip()
                
                # Extract R² Score
                if 'R²' in metric_name or 'R^2' in metric_name or 'R2' in metric_name:
                    if validation_val and validation_val != 'N/A':
                        try:
                            metrics['r2_validation'] = float(validation_val)
                        except ValueError:
                            pass
                    if test_val and test_val != 'N/A':
                        try:
                            metrics['r2_test'] = float(test_val)
                        except ValueError:
                            pass
                
                # Extract Pearson Correlation
                elif 'Pearson' in metric_name:
                    if validation_val and validation_val != 'N/A':
                        try:
                            metrics['pearson_validation'] = float(validation_val)
                        except ValueError:
                            pass
                    if test_val and test_val != 'N/A':
                        try:
                            metrics['pearson_test'] = float(test_val)
                        except ValueError:
                            pass
                
                # Extract Spearman Correlation
                elif 'Spearman' in metric_name:
                    if validation_val and validation_val != 'N/A':
                        try:
                            metrics['spearman_validation'] = float(validation_val)
                        except ValueError:
                            pass
                    if test_val and test_val != 'N/A':
                        try:
                            metrics['spearman_test'] = float(test_val)
                        except ValueError:
                            pass
        
        # Check if we got all required metrics
        required = ['r2_validation', 'r2_test', 'pearson_validation', 
                   'pearson_test', 'spearman_validation', 'spearman_test']
        if all(key in metrics for key in required):
            return metrics
        else:
            return None
            
    except Exception as e:
        return None




def process_single_metrics_file(metrics_file: Path, method: str, drug_id: str, exp_num: str) -> Optional[Dict[str, Any]]:
    """
    Process a single metrics file and return a result row dictionary.
    """
    metrics = extract_metrics_from_csv(metrics_file)
    
    if metrics:
        return {
            'method': method,
            'drug_id': drug_id,
            'experiment_number': exp_num,
            'r2_validation': metrics.get('r2_validation'),
            'r2_test': metrics.get('r2_test'),
            'pearson_validation': metrics.get('pearson_validation'),
            'pearson_test': metrics.get('pearson_test'),
            'spearman_validation': metrics.get('spearman_validation'),
            'spearman_test': metrics.get('spearman_test')
        }
    else:
        return None


def compute_paired_tests(df, baseline_method='eNest_linear_fair', metric='r2_test'):
    """
    Compute paired t-tests comparing each method to baseline.
    All methods are compared to the specified baseline_method.
    If baseline_method doesn't exist for a drug, p-values are set to np.nan.
    
    Returns:
    - Dictionary: {drug_id -> {method -> {'mean': float, 'std': float, 'p_value': float}}}
    """
    methods = sorted(df['method'].unique())
    drugs = sorted(df['drug_id'].unique())
    
    results = {}
    
    for drug_id in drugs:
        drug_data = df[df['drug_id'] == drug_id]
        results[drug_id] = {}
        
        # Use specified baseline method
        baseline_data = drug_data[drug_data['method'] == baseline_method].copy()
        
        if len(baseline_data) == 0:
            # No baseline available for this drug
            # Still add all methods but with NA p-values
            for method in methods:
                method_data = drug_data[drug_data['method'] == method].copy()
                if len(method_data) > 0:
                    method_values = method_data[metric].dropna()
                    if len(method_values) > 0:
                        results[drug_id][method] = {
                            'mean': method_values.mean(),
                            'std': method_values.std(),
                            'p_value': np.nan  # NA when no baseline
                        }
            continue
        
        # Add baseline to results (no p-value)
        baseline_values = baseline_data[metric].dropna()
        if len(baseline_values) > 0:
            results[drug_id][baseline_method] = {
                'mean': baseline_values.mean(),
                'std': baseline_values.std(),
                'p_value': np.nan
            }
        
        # Compare ALL methods to baseline
        for method in methods:
            # Skip baseline itself (baseline has no p-value)
            if method == baseline_method:
                continue
            
            method_data = drug_data[drug_data['method'] == method].copy()
            
            if len(method_data) == 0:
                continue
            
            # Pair experiments by experiment_number
            merged = pd.merge(
                baseline_data[['experiment_number', metric]].dropna(),
                method_data[['experiment_number', metric]].dropna(),
                on='experiment_number',
                suffixes=('_baseline', '_method')
            )
            
            # Calculate mean and std from all method data (not just paired)
            method_values = method_data[metric].dropna()
            if len(method_values) > 0:
                method_mean = method_values.mean()
                method_std = method_values.std()
            else:
                method_mean = np.nan
                method_std = np.nan
            
            # Compute p-value from paired data
            if len(merged) >= 3:  # Need at least 3 pairs for t-test
                try:
                    baseline_values_paired = merged[f'{metric}_baseline'].values
                    method_values_paired = merged[f'{metric}_method'].values
                    statistic, p_value = stats.ttest_rel(baseline_values_paired, method_values_paired)
                except Exception as e:
                    p_value = np.nan
            else:
                p_value = np.nan
            
            results[drug_id][method] = {
                'mean': method_mean,
                'std': method_std,
                'p_value': p_value
            }
    
    return results


def compute_paired_rank_sum_tests(df, baseline_method='eNest_linear_fair', metric='r2_test'):
    """
    Compute paired Wilcoxon signed-rank tests comparing each method to baseline.
    All methods are compared to the specified baseline_method.
    If baseline_method doesn't exist for a drug, p-values are set to np.nan.
    
    Returns:
    - Dictionary: {drug_id -> {method -> {'mean': float, 'std': float, 'p_value': float}}}
    """
    methods = sorted(df['method'].unique())
    drugs = sorted(df['drug_id'].unique())
    
    results = {}
    
    for drug_id in drugs:
        drug_data = df[df['drug_id'] == drug_id]
        results[drug_id] = {}
        
        # Use specified baseline method
        baseline_data = drug_data[drug_data['method'] == baseline_method].copy()
        
        if len(baseline_data) == 0:
            # No baseline available for this drug
            # Still add all methods but with NA p-values
            for method in methods:
                method_data = drug_data[drug_data['method'] == method].copy()
                if len(method_data) > 0:
                    method_values = method_data[metric].dropna()
                    if len(method_values) > 0:
                        results[drug_id][method] = {
                            'mean': method_values.mean(),
                            'std': method_values.std(),
                            'p_value': np.nan  # NA when no baseline
                        }
            continue
        
        # Add baseline to results (no p-value)
        baseline_values = baseline_data[metric].dropna()
        if len(baseline_values) > 0:
            results[drug_id][baseline_method] = {
                'mean': baseline_values.mean(),
                'std': baseline_values.std(),
                'p_value': np.nan
            }
        
        # Compare ALL methods to baseline
        for method in methods:
            # Skip baseline itself (baseline has no p-value)
            if method == baseline_method:
                continue
            
            method_data = drug_data[drug_data['method'] == method].copy()
            
            if len(method_data) == 0:
                continue
            
            # Pair experiments by experiment_number
            merged = pd.merge(
                baseline_data[['experiment_number', metric]].dropna(),
                method_data[['experiment_number', metric]].dropna(),
                on='experiment_number',
                suffixes=('_baseline', '_method')
            )
            
            # Calculate mean and std from all method data (not just paired)
            method_values = method_data[metric].dropna()
            if len(method_values) > 0:
                method_mean = method_values.mean()
                method_std = method_values.std()
            else:
                method_mean = np.nan
                method_std = np.nan
            
            # Compute p-value from paired data using Wilcoxon signed-rank test
            if len(merged) >= 3:  # Need at least 3 pairs for rank-sum test
                try:
                    baseline_values_paired = merged[f'{metric}_baseline'].values
                    method_values_paired = merged[f'{metric}_method'].values
                    # Compute differences for Wilcoxon signed-rank test
                    differences = method_values_paired - baseline_values_paired
                    # Remove zero differences (they don't contribute to the test)
                    differences = differences[differences != 0]
                    if len(differences) >= 3:  # Need at least 3 non-zero differences
                        statistic, p_value = stats.wilcoxon(baseline_values_paired, method_values_paired, alternative='two-sided')
                    else:
                        p_value = np.nan
                except Exception as e:
                    p_value = np.nan
            else:
                p_value = np.nan
            
            results[drug_id][method] = {
                'mean': method_mean,
                'std': method_std,
                'p_value': p_value
            }
    
    return results


def apply_bh_correction_column_wise(results, methods, drugs):
    """
    Apply BH correction column-wise (per method).
    For each method, collect all p-values across drugs and adjust them.
    """
    adjusted_results = {}
    
    # For each method, collect p-values across all drugs
    for method in methods:
        p_value_list = []  # List of (drug_id, p_value) tuples
        
        for drug_id in drugs:
            if drug_id in results and method in results[drug_id]:
                p_val = results[drug_id][method]['p_value']
                if pd.notna(p_val):
                    p_value_list.append((drug_id, p_val))
        
        # Apply BH correction to this method's p-values
        if len(p_value_list) > 0:
            p_values = [p_val for _, p_val in p_value_list]
            adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            
            # Create adjusted results for this method
            for (drug_id, _), adjusted_p in zip(p_value_list, adjusted_p_values):
                if drug_id not in adjusted_results:
                    adjusted_results[drug_id] = {}
                adjusted_results[drug_id][method] = results[drug_id][method].copy()
                adjusted_results[drug_id][method]['p_value'] = adjusted_p
    
    # Copy baseline and methods without p-values
    for drug_id in drugs:
        if drug_id in results:
            if drug_id not in adjusted_results:
                adjusted_results[drug_id] = {}
            for method in results[drug_id]:
                if method not in adjusted_results[drug_id]:
                    adjusted_results[drug_id][method] = results[drug_id][method].copy()
    
    return adjusted_results


def create_comparison_table(results, methods, drugs, baseline_method='eNest_linear_fair', use_bh_adjustment=False):
    """
    Create a formatted comparison table.
    Handles the specified baseline method.
    """
    table_data = []
    # Use display names for columns, fallback to method name if not in mapping
    column_names = ['Drug'] + [METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
    
    # Determine which methods are baselines
    baseline_methods = {baseline_method}
    
    for drug_id in drugs:
        row = [f'D{drug_id}']
        
        for method in methods:
            if drug_id in results and method in results[drug_id]:
                method_result = results[drug_id][method]
                mean_val = method_result['mean']
                std_val = method_result['std']
                p_val = method_result['p_value']
                
                if pd.notna(mean_val) and pd.notna(std_val):
                    value_str = f'{mean_val:.4f} ± {std_val:.4f}'
                    
                    # Add p-value if method is not a baseline and p-value exists
                    if method not in baseline_methods:
                        if pd.notna(p_val):
                            if p_val < 0.001:
                                p_str = '(p<0.001)'
                            else:
                                p_str = f'(p={p_val:.3f})'
                            value_str += f' {p_str}'
                        else:
                            # Show NA when p-value is not available
                            value_str += ' (p=NA)'
                    
                    row.append(value_str)
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data, columns=column_names)
    return table_df


def main():
    """Main function"""
    base_dir = Path('/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler')
    output_dir = base_dir / 'metrics_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Check if combined CSV already exists
    combined_csv_path = output_dir / 'combined_metrics_all_methods.csv'
    
    if combined_csv_path.exists():
        print("=" * 60)
        print("LOADING EXISTING METRICS DATA")
        print("=" * 60)
        print(f"Loading from: {combined_csv_path}")
        df = pd.read_csv(combined_csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Filter to only include methods that are keys in METHOD_DISPLAY_NAMES
        # (commented-out methods in METHOD_DISPLAY_NAMES will be automatically excluded)
        valid_methods = set(METHOD_DISPLAY_NAMES.keys())
        initial_count = len(df)
        df = df[df['method'].isin(valid_methods)]
        
        # Additional exclusion for any data quality issues (e.g., 'method' column name issues)
        excluded_methods = {'method'}  # Exclude any rows with 'method' as the method name (data quality issue)
        df = df[~df['method'].isin(excluded_methods)]
        
        # Filter to only include allowed drug IDs
        df['drug_id'] = df['drug_id'].astype(str)
        df = df[df['drug_id'].isin(ALLOWED_DRUG_IDS)]
        
        print(f"After filtering: {len(df)} rows (removed {initial_count - len(df)} rows)")
        print(f"Methods: {sorted(df['method'].unique())}")
        print(f"Drugs: {sorted(df['drug_id'].unique())}")
    else:
        print("=" * 60)
        print("EXTRACTING METRICS FROM ALL FOLDERS")
        print("=" * 60)
        print(f"Base directory: {base_dir}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Find all method folders
        print("Scanning for method folders with results directories...")
        method_folders = find_all_method_folders(base_dir)
        print(f"Found {len(method_folders)} method folders")
        
        # Collect all metrics files
        print("\nCollecting all metrics files...")
        all_tasks = []
        
        for method_dir, method_name in method_folders:
            metrics_files = find_all_metrics_files(method_dir)
            print(f"  {method_name}: Found {len(metrics_files)} metrics files")
            for metrics_file, drug_id, exp_num in metrics_files:
                all_tasks.append((metrics_file, method_name, drug_id, exp_num))
        
        if not all_tasks:
            print("No metrics files found!")
            return
        
        print(f"\nProcessing {len(all_tasks)} metrics files...")
        
        # Process all metrics files in parallel
        all_results = []
        max_workers = min(40, len(all_tasks))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_metrics_file, metrics_file, method, drug_id, exp_num): 
                (metrics_file, method, drug_id, exp_num)
                for metrics_file, method, drug_id, exp_num in all_tasks
            }
            
            completed = 0
            start_time = time.time()
            
            for future in as_completed(future_to_task):
                completed += 1
                if completed % 100 == 0 or completed == len(all_tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {completed}/{len(all_tasks)} files processed "
                          f"({rate:.1f} files/sec)")
                
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    metrics_file, method, drug_id, exp_num = future_to_task[future]
                    print(f"  Error processing {metrics_file}: {e}")
        
        if not all_results:
            print("No results found!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Filter to only include methods that are keys in METHOD_DISPLAY_NAMES
        # (commented-out methods in METHOD_DISPLAY_NAMES will be automatically excluded)
        valid_methods = set(METHOD_DISPLAY_NAMES.keys())
        df = df[df['method'].isin(valid_methods)]
        
        # Additional exclusion for any data quality issues (e.g., 'method' column name issues)
        excluded_methods = {'method'}  # Exclude any rows with 'method' as the method name (data quality issue)
        df = df[~df['method'].isin(excluded_methods)]
        
        # Filter to only include allowed drug IDs
        df['drug_id'] = df['drug_id'].astype(str)
        df = df[df['drug_id'].isin(ALLOWED_DRUG_IDS)]
        
        # Write combined CSV
        df.to_csv(combined_csv_path, index=False)
        print(f"\nCombined metrics saved to: {combined_csv_path}")
        print(f"Total experiments: {len(df)}")
        print(f"Methods: {sorted(df['method'].unique())}")
        print(f"Drugs: {sorted(df['drug_id'].unique())}")
    
    # Print summary by method
    print("\nSummary by method:")
    for method in sorted(df['method'].unique()):
        method_results = df[df['method'] == method]
        print(f"  {method}: {len(method_results)} experiments")
    
    # Set baseline method
    baseline_method = 'eNest_linear_fair'
    
    # Check if baseline method exists
    if baseline_method not in df['method'].unique():
        print(f"\nWarning: {baseline_method} not found in results. Cannot create comparison tables.")
        return
    
    # Create comparison tables - R² test, 4 tables total
    print("\n" + "=" * 60)
    print(f"CREATING R² TEST COMPARISON TABLES (vs {baseline_method})")
    print("=" * 60)
    
    methods = sorted(df['method'].unique())
    # Exclude D57 and D201 from table generation
    # Convert drug_id to string for consistent comparison (handles both int and string types)
    df['drug_id'] = df['drug_id'].astype(str)
    all_drugs = sorted(df['drug_id'].unique())
    excluded_drugs = {'57', '201'}  # Drug IDs to exclude (as strings)
    drugs = [d for d in all_drugs if d not in excluded_drugs]
    print(f"Excluding drugs D57 and D201 from tables. Including {len(drugs)} drugs: {[f'D{d}' for d in drugs]}")
    
    # Filter dataframe to exclude D57 and D201
    df_filtered = df[~df['drug_id'].isin(excluded_drugs)].copy()
    
    # ===== PAIRED T-TEST TABLES =====
    print("\n" + "-" * 60)
    print("PAIRED T-TEST TABLES")
    print("-" * 60)
    
    # Compute paired t-tests
    print(f"\nComputing paired t-tests comparing each method to {baseline_method}...")
    ttest_results = compute_paired_tests(df_filtered, baseline_method=baseline_method, metric='r2_test')
    
    # Create unadjusted table
    print("\nCreating unadjusted p-values table...")
    ttest_unadjusted_table = create_comparison_table(
        ttest_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=False
    )
    ttest_unadjusted_path = output_dir / 'r2_test_comparison_vs_eNest_sum_t_test_unadjusted.csv'
    ttest_unadjusted_table.to_csv(ttest_unadjusted_path, index=False)
    print(f"  Saved: {ttest_unadjusted_path}")
    
    # Apply BH correction column-wise (per method)
    print("\nApplying BH correction column-wise (per method)...")
    ttest_adjusted_results = apply_bh_correction_column_wise(ttest_results, methods, drugs)
    
    # Create BH-adjusted table
    print("Creating BH-adjusted p-values table...")
    ttest_adjusted_table = create_comparison_table(
        ttest_adjusted_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=True
    )
    ttest_adjusted_path = output_dir / 'r2_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'
    ttest_adjusted_table.to_csv(ttest_adjusted_path, index=False)
    print(f"  Saved: {ttest_adjusted_path}")
    
    # ===== WILCOXON RANK-SUM TEST TABLES =====
    print("\n" + "-" * 60)
    print("WILCOXON RANK-SUM TEST TABLES")
    print("-" * 60)
    
    print(f"\nComputing paired Wilcoxon rank-sum tests comparing each method to {baseline_method}...")
    wilcoxon_results = compute_paired_rank_sum_tests(df_filtered, baseline_method=baseline_method, metric='r2_test')
    
    # Create unadjusted table
    print("\nCreating unadjusted p-values table...")
    wilcoxon_unadjusted_table = create_comparison_table(
        wilcoxon_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=False
    )
    wilcoxon_unadjusted_path = output_dir / 'r2_test_comparison_vs_eNest_sum_wilcoxon_unadjusted.csv'
    wilcoxon_unadjusted_table.to_csv(wilcoxon_unadjusted_path, index=False)
    print(f"  Saved: {wilcoxon_unadjusted_path}")
    
    # Apply BH correction column-wise (per method)
    print("\nApplying BH correction column-wise (per method)...")
    wilcoxon_adjusted_results = apply_bh_correction_column_wise(wilcoxon_results, methods, drugs)
    
    # Create BH-adjusted table
    print("Creating BH-adjusted p-values table...")
    wilcoxon_adjusted_table = create_comparison_table(
        wilcoxon_adjusted_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=True
    )
    wilcoxon_adjusted_path = output_dir / 'r2_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv'
    wilcoxon_adjusted_table.to_csv(wilcoxon_adjusted_path, index=False)
    print(f"  Saved: {wilcoxon_adjusted_path}")
    
    # Create comparison tables - Pearson test, 4 tables total
    print("\n" + "=" * 60)
    print(f"CREATING PEARSON TEST COMPARISON TABLES (vs {baseline_method})")
    print("=" * 60)
    
    # ===== PAIRED T-TEST TABLES =====
    print("\n" + "-" * 60)
    print("PAIRED T-TEST TABLES")
    print("-" * 60)
    
    # Compute paired t-tests
    print(f"\nComputing paired t-tests comparing each method to {baseline_method}...")
    pearson_ttest_results = compute_paired_tests(df_filtered, baseline_method=baseline_method, metric='pearson_test')
    
    # Create unadjusted table
    print("\nCreating unadjusted p-values table...")
    pearson_ttest_unadjusted_table = create_comparison_table(
        pearson_ttest_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=False
    )
    pearson_ttest_unadjusted_path = output_dir / 'pearson_test_comparison_vs_eNest_sum_t_test_unadjusted.csv'
    pearson_ttest_unadjusted_table.to_csv(pearson_ttest_unadjusted_path, index=False)
    print(f"  Saved: {pearson_ttest_unadjusted_path}")
    
    # Apply BH correction column-wise (per method)
    print("\nApplying BH correction column-wise (per method)...")
    pearson_ttest_adjusted_results = apply_bh_correction_column_wise(pearson_ttest_results, methods, drugs)
    
    # Create BH-adjusted table
    print("Creating BH-adjusted p-values table...")
    pearson_ttest_adjusted_table = create_comparison_table(
        pearson_ttest_adjusted_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=True
    )
    pearson_ttest_adjusted_path = output_dir / 'pearson_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'
    pearson_ttest_adjusted_table.to_csv(pearson_ttest_adjusted_path, index=False)
    print(f"  Saved: {pearson_ttest_adjusted_path}")
    
    # ===== WILCOXON RANK-SUM TEST TABLES =====
    print("\n" + "-" * 60)
    print("WILCOXON RANK-SUM TEST TABLES")
    print("-" * 60)
    
    print(f"\nComputing paired Wilcoxon rank-sum tests comparing each method to {baseline_method}...")
    pearson_wilcoxon_results = compute_paired_rank_sum_tests(df_filtered, baseline_method=baseline_method, metric='pearson_test')
    
    # Create unadjusted table
    print("\nCreating unadjusted p-values table...")
    pearson_wilcoxon_unadjusted_table = create_comparison_table(
        pearson_wilcoxon_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=False
    )
    pearson_wilcoxon_unadjusted_path = output_dir / 'pearson_test_comparison_vs_eNest_sum_wilcoxon_unadjusted.csv'
    pearson_wilcoxon_unadjusted_table.to_csv(pearson_wilcoxon_unadjusted_path, index=False)
    print(f"  Saved: {pearson_wilcoxon_unadjusted_path}")
    
    # Apply BH correction column-wise (per method)
    print("\nApplying BH correction column-wise (per method)...")
    pearson_wilcoxon_adjusted_results = apply_bh_correction_column_wise(pearson_wilcoxon_results, methods, drugs)
    
    # Create BH-adjusted table
    print("Creating BH-adjusted p-values table...")
    pearson_wilcoxon_adjusted_table = create_comparison_table(
        pearson_wilcoxon_adjusted_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=True
    )
    pearson_wilcoxon_adjusted_path = output_dir / 'pearson_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv'
    pearson_wilcoxon_adjusted_table.to_csv(pearson_wilcoxon_adjusted_path, index=False)
    print(f"  Saved: {pearson_wilcoxon_adjusted_path}")
    
    # Create comparison tables - Spearman test, 4 tables total
    print("\n" + "=" * 60)
    print(f"CREATING SPEARMAN TEST COMPARISON TABLES (vs {baseline_method})")
    print("=" * 60)
    
    # ===== PAIRED T-TEST TABLES =====
    print("\n" + "-" * 60)
    print("PAIRED T-TEST TABLES")
    print("-" * 60)
    
    # Compute paired t-tests
    print(f"\nComputing paired t-tests comparing each method to {baseline_method}...")
    spearman_ttest_results = compute_paired_tests(df_filtered, baseline_method=baseline_method, metric='spearman_test')
    
    # Create unadjusted table
    print("\nCreating unadjusted p-values table...")
    spearman_ttest_unadjusted_table = create_comparison_table(
        spearman_ttest_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=False
    )
    spearman_ttest_unadjusted_path = output_dir / 'spearman_test_comparison_vs_eNest_sum_t_test_unadjusted.csv'
    spearman_ttest_unadjusted_table.to_csv(spearman_ttest_unadjusted_path, index=False)
    print(f"  Saved: {spearman_ttest_unadjusted_path}")
    
    # Apply BH correction column-wise (per method)
    print("\nApplying BH correction column-wise (per method)...")
    spearman_ttest_adjusted_results = apply_bh_correction_column_wise(spearman_ttest_results, methods, drugs)
    
    # Create BH-adjusted table
    print("Creating BH-adjusted p-values table...")
    spearman_ttest_adjusted_table = create_comparison_table(
        spearman_ttest_adjusted_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=True
    )
    spearman_ttest_adjusted_path = output_dir / 'spearman_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'
    spearman_ttest_adjusted_table.to_csv(spearman_ttest_adjusted_path, index=False)
    print(f"  Saved: {spearman_ttest_adjusted_path}")
    
    # ===== WILCOXON RANK-SUM TEST TABLES =====
    print("\n" + "-" * 60)
    print("WILCOXON RANK-SUM TEST TABLES")
    print("-" * 60)
    
    print(f"\nComputing paired Wilcoxon rank-sum tests comparing each method to {baseline_method}...")
    spearman_wilcoxon_results = compute_paired_rank_sum_tests(df_filtered, baseline_method=baseline_method, metric='spearman_test')
    
    # Create unadjusted table
    print("\nCreating unadjusted p-values table...")
    spearman_wilcoxon_unadjusted_table = create_comparison_table(
        spearman_wilcoxon_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=False
    )
    spearman_wilcoxon_unadjusted_path = output_dir / 'spearman_test_comparison_vs_eNest_sum_wilcoxon_unadjusted.csv'
    spearman_wilcoxon_unadjusted_table.to_csv(spearman_wilcoxon_unadjusted_path, index=False)
    print(f"  Saved: {spearman_wilcoxon_unadjusted_path}")
    
    # Apply BH correction column-wise (per method)
    print("\nApplying BH correction column-wise (per method)...")
    spearman_wilcoxon_adjusted_results = apply_bh_correction_column_wise(spearman_wilcoxon_results, methods, drugs)
    
    # Create BH-adjusted table
    print("Creating BH-adjusted p-values table...")
    spearman_wilcoxon_adjusted_table = create_comparison_table(
        spearman_wilcoxon_adjusted_results, methods, drugs, baseline_method=baseline_method, use_bh_adjustment=True
    )
    spearman_wilcoxon_adjusted_path = output_dir / 'spearman_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv'
    spearman_wilcoxon_adjusted_table.to_csv(spearman_wilcoxon_adjusted_path, index=False)
    print(f"  Saved: {spearman_wilcoxon_adjusted_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  1. {combined_csv_path.name} (all metrics)")
    print(f"  2. {ttest_unadjusted_path.name} (R² test paired t-test unadjusted p-values)")
    print(f"  3. {ttest_adjusted_path.name} (R² test paired t-test BH column-wise adjusted p-values)")
    print(f"  4. {wilcoxon_unadjusted_path.name} (R² test Wilcoxon rank-sum unadjusted p-values)")
    print(f"  5. {wilcoxon_adjusted_path.name} (R² test Wilcoxon rank-sum BH column-wise adjusted p-values)")
    print(f"  6. {pearson_ttest_unadjusted_path.name} (Pearson test paired t-test unadjusted p-values)")
    print(f"  7. {pearson_ttest_adjusted_path.name} (Pearson test paired t-test BH column-wise adjusted p-values)")
    print(f"  8. {pearson_wilcoxon_unadjusted_path.name} (Pearson test Wilcoxon rank-sum unadjusted p-values)")
    print(f"  9. {pearson_wilcoxon_adjusted_path.name} (Pearson test Wilcoxon rank-sum BH column-wise adjusted p-values)")
    print(f" 10. {spearman_ttest_unadjusted_path.name} (Spearman test paired t-test unadjusted p-values)")
    print(f" 11. {spearman_ttest_adjusted_path.name} (Spearman test paired t-test BH column-wise adjusted p-values)")
    print(f" 12. {spearman_wilcoxon_unadjusted_path.name} (Spearman test Wilcoxon rank-sum unadjusted p-values)")
    print(f" 13. {spearman_wilcoxon_adjusted_path.name} (Spearman test Wilcoxon rank-sum BH column-wise adjusted p-values)")


if __name__ == '__main__':
    main()

