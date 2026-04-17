#!/usr/bin/env python3
"""
Compute BH column-wise adjusted table from test_r2_per_experiment.csv
Uses three methods for multiple testing correction:
1. Benjamini-Hochberg (fdr_bh)
2. Benjamini-Yekutieli (fdr_by) 
3. Bonferroni (bonferroni)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path


def compute_bh_adjusted_table(csv_path, output_dir=None, baseline_model='ERK_SNN'):
    """
    Compute BH column-wise adjusted table from R² test data.
    
    Parameters:
    - csv_path: Path to test_r2_per_experiment.csv
    - output_dir: Directory to save output files (default: same as csv_path)
    - baseline_model: Model to use as baseline for comparisons (default: 'ERK_SNN')
    
    Returns:
    - Dictionary with results for each correction method
    """
    # Read CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['Model'].unique()}")
    print(f"Drugs: {sorted(df['Drug_ID'].unique())}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    models = sorted(df['Model'].unique())
    drugs = sorted(df['Drug_ID'].unique())
    comparison_models = [m for m in models if m != baseline_model]
    
    # Correction methods to apply
    correction_methods = {
        'fdr_bh': 'Benjamini-Hochberg',
        'fdr_by': 'Benjamini-Yekutieli',
        'bonferroni': 'Bonferroni'
    }
    
    all_results = {}
    
    # Collect p-values per drug for column-wise correction
    drug_p_value_info = {}  # drug_id -> list of (model, p_value, result_index)
    metric_results = []
    
    # First pass: calculate means, stds, and p-values
    for drug_id in drugs:
        drug_data = df[df['Drug_ID'] == drug_id].copy()
        
        # Get baseline model data for this drug
        baseline_data = drug_data[drug_data['Model'] == baseline_model].copy()
        
        if len(baseline_data) == 0:
            print(f"  Warning: No baseline data for {baseline_model} on D{drug_id}")
            continue
        
        # Calculate baseline summary
        baseline_values = baseline_data['Test_R2'].dropna()
        if len(baseline_values) > 0:
            baseline_mean = baseline_values.mean()
            baseline_std = baseline_values.std()
        else:
            baseline_mean = np.nan
            baseline_std = np.nan
        
        # Add baseline row
        baseline_result_idx = len(metric_results)
        metric_results.append({
            'drug_id': drug_id,
            'model': baseline_model,
            'mean': baseline_mean,
            'std': baseline_std,
            'p_value': np.nan,  # No p-value for baseline
            'n': len(baseline_values)
        })
        
        # Initialize list for this drug's p-values
        if drug_id not in drug_p_value_info:
            drug_p_value_info[drug_id] = []
        
        # Compare each model to baseline
        for model in comparison_models:
            model_data = drug_data[drug_data['Model'] == model].copy()
            
            if len(model_data) == 0:
                continue
            
            # Pair experiments by Experiment_ID
            merged = pd.merge(
                baseline_data[['Experiment_ID', 'Test_R2']].dropna(),
                model_data[['Experiment_ID', 'Test_R2']].dropna(),
                on='Experiment_ID',
                suffixes=('_baseline', '_model')
            )
            
            result_idx = len(metric_results)
            
            if len(merged) < 3:  # Need at least 3 pairs for t-test
                model_values = model_data['Test_R2'].dropna()
                if len(model_values) > 0:
                    model_mean = model_values.mean()
                    model_std = model_values.std()
                else:
                    model_mean = np.nan
                    model_std = np.nan
                p_value = np.nan
            else:
                baseline_values_paired = merged['Test_R2_baseline'].values
                model_values_paired = merged['Test_R2_model'].values
                
                model_values = model_data['Test_R2'].dropna()
                if len(model_values) > 0:
                    model_mean = model_values.mean()
                    model_std = model_values.std()
                else:
                    model_mean = np.nan
                    model_std = np.nan
                
                try:
                    # Perform paired t-test
                    statistic, p_value = stats.ttest_rel(baseline_values_paired, model_values_paired)
                    # Store p-value info for this drug (column-wise correction)
                    drug_p_value_info[drug_id].append((model, p_value, result_idx))
                except Exception as e:
                    print(f"  Warning: Could not compute t-test for {model} vs {baseline_model} on D{drug_id}: {e}")
                    p_value = np.nan
            
            metric_results.append({
                'drug_id': drug_id,
                'model': model,
                'mean': model_mean,
                'std': model_std,
                'p_value': p_value,
                'n': len(model_data['Test_R2'].dropna())
            })
    
    # Apply each correction method
    for method_key, method_name in correction_methods.items():
        print(f"\nApplying {method_name} correction...")
        
        # Create a copy of metric_results for this correction method
        corrected_results = [r.copy() for r in metric_results]
        
        # Apply correction column-wise (per drug)
        for drug_id in drug_p_value_info:
            p_value_list = drug_p_value_info[drug_id]
            
            if len(p_value_list) == 0:
                continue
            
            # Extract p-values and their indices
            p_values = [p_val for _, p_val, _ in p_value_list]
            result_indices = [idx for _, _, idx in p_value_list]
            
            # Filter out NaN p-values for correction
            valid_indices = [i for i, p in enumerate(p_values) if not pd.isna(p)]
            
            if len(valid_indices) > 0:
                valid_p_values = [p_values[i] for i in valid_indices]
                
                # Apply correction for this drug (column)
                adjusted_p_values = multipletests(valid_p_values, method=method_key)[1]
                
                # Update corrected_results with adjusted p-values
                for valid_idx, adjusted_p in zip(valid_indices, adjusted_p_values):
                    result_idx = result_indices[valid_idx]
                    corrected_results[result_idx]['p_value'] = adjusted_p
        
        # Convert to DataFrame
        results_df = pd.DataFrame(corrected_results)
        all_results[method_key] = results_df
        
        # Create formatted table: rows = drugs, columns = models
        table_data = []
        columns = ['Drug'] + models
        
        for drug_id in drugs:
            row = [f'D{drug_id}']
            
            for model in models:
                model_data = results_df[
                    (results_df['drug_id'] == drug_id) & 
                    (results_df['model'] == model)
                ]
                
                if len(model_data) > 0:
                    row_data = model_data.iloc[0]
                    mean_val = row_data['mean']
                    std_val = row_data['std']
                    p_val = row_data['p_value']
                    
                    if pd.notna(mean_val) and pd.notna(std_val):
                        if pd.notna(p_val):
                            # Format: mean ± std (p=value)
                            if p_val < 0.001:
                                p_str = 'p<0.001'
                            else:
                                p_str = f'p={p_val:.3f}'
                            row.append(f'{mean_val:.4f} ± {std_val:.4f} ({p_str})')
                        else:
                            # No p-value (baseline or insufficient data)
                            row.append(f'{mean_val:.4f} ± {std_val:.4f}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            
            table_data.append(row)
        
        # Create DataFrame for table
        table_df = pd.DataFrame(table_data, columns=columns)
        
        # Save table
        table_filename = output_dir / f'bh_adjusted_table_{method_key}.csv'
        table_df.to_csv(table_filename, index=False)
        print(f"  Saved: {table_filename}")
        
        # Save detailed results
        detailed_filename = output_dir / f'bh_adjusted_detailed_{method_key}.csv'
        results_df.to_csv(detailed_filename, index=False)
        print(f"  Saved: {detailed_filename}")
    
    return all_results


def main():
    """Main function"""
    base_dir = Path('/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/sparse_wd')
    csv_path = base_dir / 'test_r2_per_experiment.csv'
    output_dir = base_dir / 'bh_adjusted_tables'
    
    print("=" * 60)
    print("COMPUTING BH COLUMN-WISE ADJUSTED TABLES")
    print("=" * 60)
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    results = compute_bh_adjusted_table(csv_path, output_dir, baseline_model='ERK_SNN')
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nCorrection methods applied:")
    print("  1. Benjamini-Hochberg (fdr_bh)")
    print("  2. Benjamini-Yekutieli (fdr_by)")
    print("  3. Bonferroni (bonferroni)")


if __name__ == '__main__':
    main()













