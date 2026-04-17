#!/usr/bin/env python3
"""
Extract all non-significant p-values (p > 0.05) from comparison tables.
"""

import pandas as pd
import re
from pathlib import Path

def parse_p_value(value_str):
    """Extract p-value from a string like '0.3298 ± 0.0862 (p=0.137)' or '0.3298 ± 0.0862 (p<0.001)'"""
    if pd.isna(value_str) or value_str == 'N/A':
        return None
    
    # Pattern to match p-value: (p=0.137) or (p<0.001)
    p_pattern = r'\(p([<>=]?)(\d+\.?\d*)\)'
    match = re.search(p_pattern, str(value_str))
    
    if match:
        operator = match.group(1)
        p_val_str = match.group(2)
        
        if operator == '<':
            # p<0.001 means p is less than 0.001, so it's significant
            return float(p_val_str)  # Return the threshold value
        elif operator == '=':
            return float(p_val_str)
        else:
            return None
    else:
        # Check for (p=NA)
        if '(p=NA)' in str(value_str):
            return None
    
    return None

def extract_non_significant_from_table(file_path, metric_type, test_type, adjustment_type):
    """Extract non-significant p-values from a comparison table."""
    df = pd.read_csv(file_path)
    
    non_sig_results = []
    
    # Get drug column (first column)
    drug_col = df.columns[0]
    
    # Iterate through each row (drug)
    for idx, row in df.iterrows():
        drug = row[drug_col]
        
        # Skip if drug is empty or N/A
        if pd.isna(drug) or drug == '':
            continue
        
        # Iterate through each method column (skip first column which is Drug)
        for method_col in df.columns[1:]:
            # Exclude Old_eNest
            if method_col == 'Old_eNest':
                continue
            
            value_str = row[method_col]
            p_value = parse_p_value(value_str)
            
            if p_value is not None and p_value > 0.05:
                non_sig_results.append({
                    'Statistic': metric_type,
                    'Drug': drug,
                    'Method': method_col,
                    'Test': test_type,
                    'Adjustment': adjustment_type,
                    'P-value': p_value
                })
    
    return non_sig_results

def main():
    """Main function"""
    base_dir = Path('/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/metrics_analysis')
    
    all_results = []
    
    # Process all comparison tables
    tables = [
        ('R²', 't-test', 'unadjusted', 'r2_test_comparison_vs_eNest_sum_t_test_unadjusted.csv'),
        ('R²', 't-test', 'BH-adjusted', 'r2_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'),
        ('R²', 'Wilcoxon', 'unadjusted', 'r2_test_comparison_vs_eNest_sum_wilcoxon_unadjusted.csv'),
        ('R²', 'Wilcoxon', 'BH-adjusted', 'r2_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv'),
        ('Pearson', 't-test', 'unadjusted', 'pearson_test_comparison_vs_eNest_sum_t_test_unadjusted.csv'),
        ('Pearson', 't-test', 'BH-adjusted', 'pearson_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'),
        ('Pearson', 'Wilcoxon', 'unadjusted', 'pearson_test_comparison_vs_eNest_sum_wilcoxon_unadjusted.csv'),
        ('Pearson', 'Wilcoxon', 'BH-adjusted', 'pearson_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv'),
        # ('Spearman', 't-test', 'unadjusted', 'spearman_test_comparison_vs_eNest_sum_t_test_unadjusted.csv'),
        # ('Spearman', 't-test', 'BH-adjusted', 'spearman_test_comparison_vs_eNest_sum_t_test_BH_adjusted.csv'),
        # ('Spearman', 'Wilcoxon', 'unadjusted', 'spearman_test_comparison_vs_eNest_sum_wilcoxon_unadjusted.csv'),
        # ('Spearman', 'Wilcoxon', 'BH-adjusted', 'spearman_test_comparison_vs_eNest_sum_wilcoxon_BH_adjusted.csv'),
    ]
    
    for metric_type, test_type, adjustment_type, filename in tables:
        file_path = base_dir / filename
        if file_path.exists():
            print(f"Processing {filename}...")
            results = extract_non_significant_from_table(file_path, metric_type, test_type, adjustment_type)
            all_results.extend(results)
        else:
            print(f"Warning: {filename} not found")
    
    # Create DataFrame and sort
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values(['Statistic', 'Test', 'Adjustment', 'Drug', 'Method'])
        
        # Save to CSV
        output_path = base_dir / 'non_significant_pvalues.csv'
        df_results.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"Found {len(df_results)} non-significant p-values (p > 0.05)")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}\n")
        
        # Print summary
        print("Summary by Statistic and Test:")
        summary = df_results.groupby(['Statistic', 'Test', 'Adjustment']).size().reset_index(name='Count')
        print(summary.to_string(index=False))
        print()
        
        # Print all results
        print("All non-significant p-values:")
        print(df_results.to_string(index=False))
    else:
        print("No non-significant p-values found (p > 0.05)")

if __name__ == '__main__':
    main()

