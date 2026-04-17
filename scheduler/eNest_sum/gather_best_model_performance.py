#!/usr/bin/env python3
"""Gather performance metrics from best_model/metrics.csv files."""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_test_r2_from_metrics(metrics_file: Path) -> float:
    """Extract test R² from metrics.csv file."""
    try:
        df = pd.read_csv(metrics_file)
        # Find the row with R² Score
        r2_row = df[df['Metric'] == 'R² Score']
        if not r2_row.empty:
            test_r2 = float(r2_row['Test'].values[0])
            return test_r2
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
    return None

def main():
    base_dir = Path("/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/eNest_sum/results/D5")
    
    # Find all best_model/metrics.csv files
    test_r2_values = []
    experiment_ids = []
    
    for exp_dir in sorted(base_dir.glob("D5_*")):
        if exp_dir.is_dir():
            metrics_file = exp_dir / "best_model" / "metrics.csv"
            if metrics_file.exists():
                test_r2 = extract_test_r2_from_metrics(metrics_file)
                if test_r2 is not None:
                    test_r2_values.append(test_r2)
                    experiment_ids.append(exp_dir.name)
                    print(f"{exp_dir.name}: Test R² = {test_r2:.4f}")
                else:
                    print(f"{exp_dir.name}: Could not extract test R²")
            else:
                print(f"{exp_dir.name}: metrics.csv not found")
    
    if len(test_r2_values) == 0:
        print("No valid test R² values found!")
        return
    
    # Calculate statistics
    test_r2_array = np.array(test_r2_values)
    mean_r2 = np.mean(test_r2_array)
    std_r2 = np.std(test_r2_array, ddof=1)  # Sample standard deviation
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(test_r2_values)}")
    print(f"\nTest R² Statistics:")
    print(f"  Mean: {mean_r2:.4f}")
    print(f"  Std:  {std_r2:.4f}")
    print(f"  Min:  {np.min(test_r2_array):.4f}")
    print(f"  Max:  {np.max(test_r2_array):.4f}")
    print(f"  Median: {np.median(test_r2_array):.4f}")
    print(f"\nMean ± Std: {mean_r2:.4f} ± {std_r2:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()



