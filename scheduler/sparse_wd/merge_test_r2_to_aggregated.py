#!/usr/bin/env python3
"""
Convert test_r2_per_experiment.csv to aggregated_metrics.csv format
and merge with existing aggregated_metrics.csv (if it exists).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_test_r2_to_aggregated_format(test_r2_path, output_path=None):
    """
    Convert test_r2_per_experiment.csv to aggregated_metrics.csv format.
    
    Parameters:
    - test_r2_path: Path to test_r2_per_experiment.csv
    - output_path: Path to save converted data (default: test_r2_aggregated_format.csv)
    
    Returns:
    - DataFrame in aggregated_metrics format
    """
    print(f"Reading {test_r2_path}...")
    df = pd.read_csv(test_r2_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['Model'].unique()}")
    print(f"Drugs: {sorted(df['Drug_ID'].unique())}")
    
    # Convert to aggregated_metrics format
    # Map: Model -> method, Drug_ID -> drug_id, Experiment_ID -> experiment_number, Test_R2 -> r2_test
    converted_df = pd.DataFrame({
        'method': df['Model'],
        'drug_id': df['Drug_ID'].str.replace('D', '').astype(int),  # Remove 'D' prefix
        'experiment_number': df['Experiment_ID'],
        'r2_test': df['Test_R2'],
        # Set validation metrics to NaN (not available in test_r2_per_experiment.csv)
        'r2_validation': np.nan,
        'pearson_validation': np.nan,
        'pearson_test': np.nan,
        'spearman_validation': np.nan,
        'spearman_test': np.nan
    })
    
    if output_path:
        converted_df.to_csv(output_path, index=False)
        print(f"\nConverted data saved to: {output_path}")
        print(f"Converted {len(converted_df)} rows")
    
    return converted_df


def merge_with_aggregated(converted_df, aggregated_path, output_path):
    """
    Merge converted test_r2 data with existing aggregated_metrics.csv.
    
    Parameters:
    - converted_df: DataFrame from convert_test_r2_to_aggregated_format
    - aggregated_path: Path to existing aggregated_metrics.csv
    - output_path: Path to save merged data
    """
    print(f"\nReading existing aggregated_metrics.csv: {aggregated_path}")
    if not Path(aggregated_path).exists():
        print(f"  Warning: {aggregated_path} not found. Using only test_r2 data.")
        merged_df = converted_df
    else:
        existing_df = pd.read_csv(aggregated_path)
        print(f"  Loaded {len(existing_df)} rows from aggregated_metrics.csv")
        print(f"  Methods in existing data: {sorted(existing_df['method'].unique())}")
        
        # Merge the dataframes
        merged_df = pd.concat([existing_df, converted_df], ignore_index=True)
        print(f"\nMerged data: {len(merged_df)} total rows")
        print(f"  Methods in merged data: {sorted(merged_df['method'].unique())}")
    
    # Save merged data
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to: {output_path}")
    
    return merged_df


def main():
    """Main function"""
    base_dir = Path('/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler')
    sparse_wd_dir = base_dir / 'sparse_wd'
    
    test_r2_path = sparse_wd_dir / 'test_r2_per_experiment.csv'
    aggregated_path = base_dir / 'aggregated_metrics.csv'
    output_path = base_dir / 'aggregated_metrics_with_test_r2.csv'
    
    print("=" * 60)
    print("CONVERTING test_r2_per_experiment.csv TO aggregated_metrics FORMAT")
    print("=" * 60)
    
    # Convert test_r2 data
    converted_df = convert_test_r2_to_aggregated_format(test_r2_path)
    
    # Save converted data separately
    converted_output = sparse_wd_dir / 'test_r2_aggregated_format.csv'
    converted_df.to_csv(converted_output, index=False)
    print(f"\nConverted data also saved to: {converted_output}")
    
    # Merge with existing aggregated_metrics.csv
    print("\n" + "=" * 60)
    print("MERGING WITH EXISTING aggregated_metrics.csv")
    print("=" * 60)
    merged_df = merge_with_aggregated(converted_df, aggregated_path, output_path)
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"\nTo use the merged data, update analyze_metrics.py to read from:")
    print(f"  {output_path}")
    print("\nOr replace aggregated_metrics.csv with the merged file.")


if __name__ == '__main__':
    main()













