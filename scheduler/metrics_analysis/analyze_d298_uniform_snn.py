#!/usr/bin/env python3
"""
Analyze which experiments caused Uniform SNN (r_sparse_nn) to have low R² on D298.
"""

import csv
import sys

# Read the CSV file
csv_file = 'combined_metrics_all_methods.csv'
data = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['method'] == 'r_sparse_nn' and int(row['drug_id']) == 298:
            data.append({
                'exp': int(row['experiment_number']),
                'r2_test': float(row['r2_test']),
                'r2_val': float(row['r2_validation']),
                'pearson_test': float(row['pearson_test']),
                'pearson_val': float(row['pearson_validation']),
                'spearman_test': float(row['spearman_test']),
                'spearman_val': float(row['spearman_validation'])
            })

# Sort by R² test (worst first)
data.sort(key=lambda x: x['r2_test'])

print("="*100)
print("Uniform Random Sparse NN (r_sparse_nn) - D298 Analysis")
print("="*100)
print(f"\nTotal experiments: {len(data)}")
print(f"Mean R² test: {sum(d['r2_test'] for d in data) / len(data):.4f}")
print(f"Median R² test: {sorted([d['r2_test'] for d in data])[len(data)//2]:.4f}")
print(f"Min R² test: {min(d['r2_test'] for d in data):.4f}")
print(f"Max R² test: {max(d['r2_test'] for d in data):.4f}")

print("\n" + "="*100)
print("WORST PERFORMING EXPERIMENTS (R² test < 0.1)")
print("="*100)
print(f"{'Exp':<6} {'R²_test':<10} {'R²_val':<10} {'Pearson_test':<14} {'Pearson_val':<14} {'Spearman_test':<15} {'Spearman_val':<15}")
print("-"*100)

worst_experiments = [d for d in data if d['r2_test'] < 0.1]
for d in worst_experiments:
    print(f"{d['exp']:<6} {d['r2_test']:<10.4f} {d['r2_val']:<10.4f} {d['pearson_test']:<14.4f} {d['pearson_val']:<14.4f} {d['spearman_test']:<15.4f} {d['spearman_val']:<15.4f}")

print("\n" + "="*100)
print("KEY OBSERVATIONS:")
print("="*100)

# Count experiments with zero Spearman
zero_spearman = [d for d in worst_experiments if abs(d['spearman_test']) < 0.0001]
print(f"\n1. Experiments with Spearman_test ≈ 0: {len(zero_spearman)}/{len(worst_experiments)}")
if zero_spearman:
    print(f"   Experiments: {sorted([d['exp'] for d in zero_spearman])}")

# Count experiments with negative R²
negative_r2 = [d for d in worst_experiments if d['r2_test'] < 0]
print(f"\n2. Experiments with negative R²_test: {len(negative_r2)}/{len(worst_experiments)}")
if negative_r2:
    print(f"   Experiments: {sorted([d['exp'] for d in negative_r2])}")

# Count experiments with low Pearson
low_pearson = [d for d in worst_experiments if abs(d['pearson_test']) < 0.2]
print(f"\n3. Experiments with |Pearson_test| < 0.2: {len(low_pearson)}/{len(worst_experiments)}")
if low_pearson:
    print(f"   Experiments: {sorted([d['exp'] for d in low_pearson])}")

# Check validation vs test discrepancy
high_val_low_test = [d for d in worst_experiments if d['r2_val'] > 0.3 and d['r2_test'] < 0]
print(f"\n4. Experiments with high R²_val (>0.3) but negative R²_test: {len(high_val_low_test)}/{len(worst_experiments)}")
if high_val_low_test:
    print(f"   Experiments: {sorted([d['exp'] for d in high_val_low_test])}")
    print("   (Possible overfitting or train/test distribution mismatch)")

print("\n" + "="*100)
print("TOP 10 WORST EXPERIMENTS:")
print("="*100)
print(f"{'Exp':<6} {'R²_test':<10} {'R²_val':<10} {'Pearson_test':<14} {'Spearman_test':<15}")
print("-"*60)
for d in data[:10]:
    print(f"{d['exp']:<6} {d['r2_test']:<10.4f} {d['r2_val']:<10.4f} {d['pearson_test']:<14.4f} {d['spearman_test']:<15.4f}")



