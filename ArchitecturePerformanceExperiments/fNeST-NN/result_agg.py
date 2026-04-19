"""Aggregate test R² across CV folds under results/ (optional utility)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Repo: ArchitecturePerformanceExperiments/fNeST-NN/results/D{drug}/...
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

drug_ids = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
for drug_id in drug_ids:
    r2_test_values = []
    drug_dir = RESULTS_ROOT / f"D{drug_id}"
    if not drug_dir.is_dir():
        continue
    for exp_dir in sorted(os.listdir(drug_dir)):
        p = drug_dir / exp_dir
        if not p.is_dir():
            continue
        metrics_file = p / "best_model_results.csv"
        if not metrics_file.is_file():
            continue
        metrics_df = pd.read_csv(metrics_file)
        r2_test_values.append(metrics_df["Test R2"].values[0])
    if r2_test_values:
        print(f"Drug {drug_id}: {np.mean(r2_test_values)} ± {np.std(r2_test_values)}")
