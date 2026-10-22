"""Aggregate per-experiment LightGBM results into one tidy CSV.

Columns mirror what the neural methods report so the table can be concatenated
with theirs: method, drug, experiment, seed, val/test R2, Pearson, Spearman.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

M = Path(__file__).resolve().parent
rows = []
for fr in sorted(M.glob("results/D*/D*_*/final_results.json")):
    d = json.loads(fr.read_text())
    run = fr.parent
    m = re.match(r"D(\d+)_(\d+)$", run.name)
    rec = {"method": "LightGBM", "drug": int(m.group(1)), "experiment": int(m.group(2)),
           "seed": d.get("seed"), "total_trials": d.get("total_trials"),
           "val_r2": d.get("best_validation_r2"), "test_r2": d.get("best_test_r2")}
    mc = run / "best_model" / "metrics.csv"
    if mc.exists():
        mm = pd.read_csv(mc).set_index("Metric")
        for key, label in [("pearson", "Pearson Correlation"), ("spearman", "Spearman Correlation")]:
            rec[f"val_{key}"] = float(mm.loc[label, "Validation"])
            rec[f"test_{key}"] = float(mm.loc[label, "Test"])
    bp = d.get("best_hyperparameters") or {}
    rec["best_iteration"] = bp.get("best_iteration")
    rows.append(rec)

df = pd.DataFrame(rows).sort_values(["drug", "experiment"]).reset_index(drop=True)
out = M / "lightgbm_results.csv"
df.to_csv(out, index=False)
print(f"wrote {out}  rows={len(df)}")
print(f"\nper-drug test R2 (n experiments, mean +/- sd):")
g = df.groupby("drug")["test_r2"].agg(["count", "mean", "std"])
for drug, r in g.iterrows():
    print(f"  D{drug:<5} n={int(r['count']):<3} {r['mean']:+.4f} +/- {r['std']:.4f}")
print(f"\noverall: n={len(df)} test R2 {df.test_r2.mean():+.4f} +/- {df.test_r2.std():.4f} "
      f"| test Pearson {df.test_pearson.mean():.4f} | test Spearman {df.test_spearman.mean():.4f}")
print(f"median boosting rounds at best trial: {df.best_iteration.median():.0f}")
