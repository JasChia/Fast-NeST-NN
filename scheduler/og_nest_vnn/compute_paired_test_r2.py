#!/usr/bin/env python3
"""
Compute paired test R² values and statistical tests comparing og_nest_vnn vs Old_eNest.
Performs paired t-test and Wilcoxon signed-rank test for each drug and overall.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Constants
INVALID_R2 = -1000.0
DRUGS = [5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]

# Paths
OG_NEST_VNN_RESULTS = Path("/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/og_nest_vnn/results")
OLD_ENEST_RESULTS = Path("/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/Old_eNest/results")
OUTPUT_FILE = Path("/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/og_nest_vnn/paired_test_r2_comparison.csv")


def collect_paired_test_r2_values():
    """Collect paired test R² values by matching experiment IDs between methods."""
    paired_data = defaultdict(list)  # drug -> list of (og_r2, old_r2, exp_id) tuples
    
    if not OG_NEST_VNN_RESULTS.exists():
        print(f"Warning: og_nest_vnn results directory does not exist: {OG_NEST_VNN_RESULTS}")
        return paired_data
    
    if not OLD_ENEST_RESULTS.exists():
        print(f"Warning: Old_eNest results directory does not exist: {OLD_ENEST_RESULTS}")
        return paired_data
    
    # Collect og_nest_vnn results
    og_results = defaultdict(dict)  # drug -> {exp_id: r2}
    for drug in DRUGS:
        drug_dir = OG_NEST_VNN_RESULTS / f"D{drug}"
        if not drug_dir.exists():
            continue
        
        for exp_dir in drug_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_id = exp_dir.name  # e.g., "D5_0"
            results_file = exp_dir / "best_model_results.csv"
            if results_file.exists():
                try:
                    df = pd.read_csv(results_file)
                    if 'Test R2' in df.columns:
                        test_r2 = df['Test R2'].iloc[0]
                        if pd.notna(test_r2):
                            test_r2 = float(test_r2)
                            # Only include valid R² values
                            if test_r2 != INVALID_R2 and not np.isnan(test_r2) and not np.isinf(test_r2):
                                og_results[drug][exp_id] = test_r2
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
                    continue
    
    # Collect Old_eNest results and match with og_nest_vnn
    for drug in DRUGS:
        drug_dir = OLD_ENEST_RESULTS / f"D{drug}"
        if not drug_dir.exists():
            continue
        
        for exp_dir in drug_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_id = exp_dir.name  # e.g., "D5_0"
            results_file = exp_dir / "best_model_results.csv"
            if results_file.exists():
                try:
                    df = pd.read_csv(results_file)
                    if 'Test R2' in df.columns:
                        test_r2 = df['Test R2'].iloc[0]
                        if pd.notna(test_r2):
                            test_r2 = float(test_r2)
                            # Only include valid R² values
                            if test_r2 != INVALID_R2 and not np.isnan(test_r2) and not np.isinf(test_r2):
                                # Check if we have a matching og_nest_vnn result
                                if exp_id in og_results[drug]:
                                    og_r2 = og_results[drug][exp_id]
                                    paired_data[drug].append((og_r2, test_r2, exp_id))
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
                    continue
    
    return paired_data


def perform_paired_tests(og_values, old_values):
    """
    Perform paired statistical tests.
    
    Returns:
        dict with test statistics and p-values
    """
    if len(og_values) != len(old_values) or len(og_values) < 2:
        return {
            'n_pairs': len(og_values),
            'paired_t_test_statistic': np.nan,
            'paired_t_test_pvalue': np.nan,
            'wilcoxon_statistic': np.nan,
            'wilcoxon_pvalue': np.nan,
            'mean_diff': np.nan,
            'std_diff': np.nan,
            'mean_og': np.nan,
            'mean_old': np.nan
        }
    
    # Calculate differences
    differences = np.array(og_values) - np.array(old_values)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(og_values, old_values)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(og_values, old_values)
    except ValueError as e:
        # This can happen if all differences are zero
        wilcoxon_stat = np.nan
        wilcoxon_pvalue = np.nan
    
    return {
        'n_pairs': len(og_values),
        'paired_t_test_statistic': t_stat,
        'paired_t_test_pvalue': t_pvalue,
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_pvalue': wilcoxon_pvalue,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'mean_og': np.mean(og_values),
        'mean_old': np.mean(old_values),
        'median_og': np.median(og_values),
        'median_old': np.median(old_values)
    }


def main():
    print("Collecting paired test R² values...")
    paired_data = collect_paired_test_r2_values()
    
    # Process each drug
    drug_results = []
    all_og_values = []
    all_old_values = []
    
    for drug in DRUGS:
        if drug not in paired_data or len(paired_data[drug]) == 0:
            drug_results.append({
                'Drug': f'D{drug}',
                'N_Pairs': 0,
                'og_nest_vnn_Mean': np.nan,
                'og_nest_vnn_Median': np.nan,
                'Old_eNest_Mean': np.nan,
                'Old_eNest_Median': np.nan,
                'Mean_Difference': np.nan,
                'Std_Difference': np.nan,
                'Paired_t_test_Statistic': np.nan,
                'Paired_t_test_Pvalue': np.nan,
                'Wilcoxon_Statistic': np.nan,
                'Wilcoxon_Pvalue': np.nan
            })
            continue
        
        # Extract paired values
        og_values = [pair[0] for pair in paired_data[drug]]
        old_values = [pair[1] for pair in paired_data[drug]]
        
        # Perform statistical tests
        test_results = perform_paired_tests(og_values, old_values)
        
        drug_results.append({
            'Drug': f'D{drug}',
            'N_Pairs': test_results['n_pairs'],
            'og_nest_vnn_Mean': test_results['mean_og'],
            'og_nest_vnn_Median': test_results['median_og'],
            'Old_eNest_Mean': test_results['mean_old'],
            'Old_eNest_Median': test_results['median_old'],
            'Mean_Difference': test_results['mean_diff'],
            'Std_Difference': test_results['std_diff'],
            'Paired_t_test_Statistic': test_results['paired_t_test_statistic'],
            'Paired_t_test_Pvalue': test_results['paired_t_test_pvalue'],
            'Wilcoxon_Statistic': test_results['wilcoxon_statistic'],
            'Wilcoxon_Pvalue': test_results['wilcoxon_pvalue']
        })
        
        # Collect for overall analysis
        all_og_values.extend(og_values)
        all_old_values.extend(old_values)
    
    # Overall analysis (across all drugs)
    overall_results = perform_paired_tests(all_og_values, all_old_values)
    
    # Create DataFrame
    df = pd.DataFrame(drug_results)
    
    # Add overall row
    overall_row = pd.DataFrame([{
        'Drug': 'Overall',
        'N_Pairs': overall_results['n_pairs'],
        'og_nest_vnn_Mean': overall_results['mean_og'],
        'og_nest_vnn_Median': overall_results['median_og'],
        'Old_eNest_Mean': overall_results['mean_old'],
        'Old_eNest_Median': overall_results['median_old'],
        'Mean_Difference': overall_results['mean_diff'],
        'Std_Difference': overall_results['std_diff'],
        'Paired_t_test_Statistic': overall_results['paired_t_test_statistic'],
        'Paired_t_test_Pvalue': overall_results['paired_t_test_pvalue'],
        'Wilcoxon_Statistic': overall_results['wilcoxon_statistic'],
        'Wilcoxon_Pvalue': overall_results['wilcoxon_pvalue']
    }])
    df = pd.concat([df, overall_row], ignore_index=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPaired comparison saved to: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "="*100)
    print("PAIRED TEST R² COMPARISON SUMMARY")
    print("="*100)
    
    print(f"\nTotal paired experiments: {overall_results['n_pairs']}")
    print(f"Overall mean difference (og_nest_vnn - Old_eNest): {overall_results['mean_diff']:.6f}")
    print(f"Overall std of differences: {overall_results['std_diff']:.6f}")
    print(f"\nOverall og_nest_vnn: mean={overall_results['mean_og']:.6f}, median={overall_results['median_og']:.6f}")
    print(f"Overall Old_eNest: mean={overall_results['mean_old']:.6f}, median={overall_results['median_old']:.6f}")
    
    print(f"\n{'='*100}")
    print("STATISTICAL TESTS (Overall)")
    print(f"{'='*100}")
    print(f"Paired t-test: t={overall_results['paired_t_test_statistic']:.6f}, p={overall_results['paired_t_test_pvalue']:.6e}")
    print(f"Wilcoxon signed-rank test: statistic={overall_results['wilcoxon_statistic']:.6f}, p={overall_results['wilcoxon_pvalue']:.6e}")
    
    if overall_results['paired_t_test_pvalue'] < 0.05:
        print(f"\n*** Significant difference detected (p < 0.05) ***")
    else:
        print(f"\nNo significant difference detected (p >= 0.05)")
    
    print(f"\n{'='*100}")
    print("BY DRUG (Drugs with paired data)")
    print(f"{'='*100}")
    for row in drug_results:
        if row['N_Pairs'] > 0:
            sig_marker = "***" if row['Paired_t_test_Pvalue'] < 0.05 else ""
            print(f"{row['Drug']}: N={row['N_Pairs']}, "
                  f"Mean diff={row['Mean_Difference']:.6f}, "
                  f"t-test p={row['Paired_t_test_Pvalue']:.6e} {sig_marker}")
    
    # Count significant drugs
    significant_drugs = [r for r in drug_results if r['N_Pairs'] > 0 and r['Paired_t_test_Pvalue'] < 0.05]
    print(f"\n{'='*100}")
    print(f"Drugs with significant differences (p < 0.05): {len(significant_drugs)}/{sum(1 for r in drug_results if r['N_Pairs'] > 0)}")
    if significant_drugs:
        print("Significant drugs:", ", ".join([r['Drug'] for r in significant_drugs]))


if __name__ == "__main__":
    main()

