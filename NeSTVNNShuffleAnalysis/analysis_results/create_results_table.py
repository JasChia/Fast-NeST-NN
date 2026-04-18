#!/usr/bin/env python3
"""
Create a clean, formatted results table for statistical significance analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent

def create_clean_results_table():
    """Create a clean, formatted results table."""
    
    # Load the summary data
    df = pd.read_csv(_HERE / "statistical_summary_table.csv")
    
    # Create a cleaner table focusing on key results
    results = []
    
    for _, row in df.iterrows():
        # Determine overall significance (either T-test or Mann-Whitney significant)
        val_sig = (row['Val_Significant_T'] == 'Yes') or (row['Val_Significant_MW'] == 'Yes')
        test_sig = (row['Test_Significant_T'] == 'Yes') or (row['Test_Significant_MW'] == 'Yes')
        
        # Get all p-values
        val_p_t = float(row['Val_P_Value_T']) if row['Val_P_Value_T'] != 'N/A' else 1.0
        val_p_mw = float(row['Val_P_Value_MW']) if row['Val_P_Value_MW'] != 'N/A' else 1.0
        
        test_p_t = float(row['Test_P_Value_T']) if row['Test_P_Value_T'] != 'N/A' else 1.0
        test_p_mw = float(row['Test_P_Value_MW']) if row['Test_P_Value_MW'] != 'N/A' else 1.0
        
        results.append({
            'Drug': f"D{row['Drug_ID']}",
            'Metric': row['Metric'],
            'Val_Shuffled': row['Val_Shuffled_Mean'],
            'Val_Unshuffled': row['Val_Unshuffled_Mean'],
            'Val_Diff': row['Val_Difference'],
            'Val_Significant': 'Yes' if val_sig else 'No',
            'Val_P_Value_T': f"{val_p_t:.4f}",
            'Val_P_Value_MW': f"{val_p_mw:.4f}",
            'Test_Shuffled': row['Test_Shuffled_Mean'],
            'Test_Unshuffled': row['Test_Unshuffled_Mean'],
            'Test_Diff': row['Test_Difference'],
            'Test_Significant': 'Yes' if test_sig else 'No',
            'Test_P_Value_T': f"{test_p_t:.4f}",
            'Test_P_Value_MW': f"{test_p_mw:.4f}",
            'N_Shuffled': row['N_Shuffled'],
            'N_Unshuffled': row['N_Unshuffled']
        })
    
    results_df = pd.DataFrame(results)
    
    # Save the clean table
    results_df.to_csv(_HERE / "clean_statistical_results.csv", index=False)
    
    # Create summary statistics
    print("="*100)
    print("STATISTICAL SIGNIFICANCE ANALYSIS RESULTS")
    print("="*100)
    
    print(f"\nTotal comparisons: {len(results_df)}")
    print(f"Validation data - Significant differences: {len(results_df[results_df['Val_Significant'] == 'Yes'])}")
    print(f"Test data - Significant differences: {len(results_df[results_df['Test_Significant'] == 'Yes'])}")
    
    # Group by metric
    print("\n" + "="*80)
    print("RESULTS BY METRIC")
    print("="*80)
    
    for metric in results_df['Metric'].unique():
        metric_data = results_df[results_df['Metric'] == metric]
        val_sig_count = len(metric_data[metric_data['Val_Significant'] == 'Yes'])
        test_sig_count = len(metric_data[metric_data['Test_Significant'] == 'Yes'])
        
        print(f"\n{metric}:")
        print(f"  Validation - Significant: {val_sig_count}/{len(metric_data)}")
        print(f"  Test - Significant: {test_sig_count}/{len(metric_data)}")
    
    # Show significant results
    print("\n" + "="*80)
    print("SIGNIFICANT RESULTS (p < 0.05)")
    print("="*80)
    
    sig_results = results_df[(results_df['Val_Significant'] == 'Yes') | (results_df['Test_Significant'] == 'Yes')]
    
    if len(sig_results) > 0:
        print(sig_results.to_string(index=False))
    else:
        print("No statistically significant differences found.")
    
    # Show all results in a clean table
    print("\n" + "="*120)
    print("COMPLETE RESULTS TABLE")
    print("="*120)
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    results_df = create_clean_results_table()
