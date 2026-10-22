"""Drug-level paired comparison: LightGBM vs fNeST-NN.

Both methods run the identical protocol on identical folds -- same
true_training_data / validation_data / test_data per (drug, experiment),
same Optuna budget (100 trials), both selecting the best trial by validation R2
and reporting that trial's test metrics.  So runs pair exactly on
(drug, experiment).

The 50 experiments within a drug are overlapping 80/20 resamples of one
cell-line pool, not independent replicates, so each drug is collapsed to its
mean paired difference and the Wilcoxon signed-rank test is run across the
10 drugs (exact; min two-sided p = 2/2^10 = 0.00195).
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

# This file sits in <repo>/ArchitecturePerformanceExperiments/PostTrainingStatistics/
_APE = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser(description="Drug-level paired comparison: LightGBM vs fNeST-NN")
_ap.add_argument("--lightgbm-dir", type=Path, default=_APE / "LightGBM",
                 help="directory holding lightgbm_results.csv (default: ../LightGBM)")
_ap.add_argument("--fnest-results", type=Path, default=_APE / "fNeST-NN" / "results",
                 help="tree searched recursively for final_results.json "
                      "(default: ../fNeST-NN/results)")
_ap.add_argument("--out", type=Path, default=Path("scores/lightgbm_vs_fnest_paired.csv"),
                 help="paired table to write (default: ./scores/lightgbm_vs_fnest_paired.csv)")
_args = _ap.parse_args()
A, FN, OUT = _args.lightgbm_dir, _args.fnest_results, _args.out

lgb = pd.read_csv(A / "lightgbm_results.csv")[
    ["drug", "experiment", "val_r2", "test_r2", "test_pearson", "test_spearman"]]
lgb.columns = ["drug", "experiment", "lgb_val_r2", "lgb_test_r2", "lgb_test_pearson", "lgb_test_spearman"]

rows = []
for f in FN.rglob("final_results.json"):
    m = re.search(r"/D(\d+)_(\d+)/final_results\.json$", str(f))
    if not m:
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    rec = {"drug": int(m.group(1)), "experiment": int(m.group(2)),
           "fn_val_r2": d.get("best_validation_r2"), "fn_test_r2": d.get("best_test_r2")}
    mc = f.parent / "best_model" / "metrics.csv"
    if mc.exists():
        mm = pd.read_csv(mc).set_index("Metric")
        rec["fn_test_pearson"] = float(mm.loc["Pearson Correlation", "Test"])
        rec["fn_test_spearman"] = float(mm.loc["Spearman Correlation", "Test"])
    rows.append(rec)
fn = pd.DataFrame(rows)
print(f"fNeST-NN runs found : {len(fn)}   drugs: {sorted(fn.drug.unique()) if len(fn) else '-'}")

j = lgb.merge(fn, on=["drug", "experiment"], how="inner").dropna(subset=["lgb_test_r2", "fn_test_r2"])
print(f"paired (drug,experiment): {len(j)}   drugs: {sorted(j.drug.unique())}")
OUT.parent.mkdir(parents=True, exist_ok=True)
j.to_csv(OUT, index=False)

for metric, lcol, fcol in [("test R2", "lgb_test_r2", "fn_test_r2"),
                           ("test Pearson", "lgb_test_pearson", "fn_test_pearson"),
                           ("test Spearman", "lgb_test_spearman", "fn_test_spearman")]:
    if fcol not in j or j[fcol].isna().all():
        continue
    sub = j.dropna(subset=[lcol, fcol])
    tmp = sub.assign(_d=sub[lcol] - sub[fcol])
    per = (tmp.groupby("drug")
              .agg(lgb=(lcol, "mean"), fnest=(fcol, "mean"), diff=("_d", "mean"), n=("_d", "size"))
              .reset_index())
    W, p = wilcoxon(per.lgb, per.fnest)
    print(f"\n{'='*72}\n{metric.upper()}  -- drug-level paired Wilcoxon (n={len(per)} drugs)\n{'='*72}")
    print(f"{'drug':<7}{'n':>4}{'LightGBM':>11}{'fNeST-NN':>11}{'diff':>10}")
    for _, r in per.iterrows():
        print(f"D{int(r.drug):<6}{int(r.n):>4}{r.lgb:>11.4f}{r.fnest:>11.4f}{r['diff']:>+10.4f}")
    print(f"{'MEAN':<7}{int(per.n.sum()):>4}{per.lgb.mean():>11.4f}{per.fnest.mean():>11.4f}{per['diff'].mean():>+10.4f}")
    print(f"  Wilcoxon W={W:.1f}  p={p:.5f}   LightGBM higher in {(per['diff']>0).sum()}/{len(per)} drugs")
    print(f"  per-experiment win rate: {100*(sub[lcol]>sub[fcol]).mean():.1f}% of {len(sub)}")
    # paired effect size across drugs
    d = per["diff"]
    print(f"  mean paired diff {d.mean():+.4f}  (sd {d.std():.4f}, Cohen dz {d.mean()/d.std():+.2f})")
print(f"\nwrote {A/'lightgbm_vs_fnest_paired.csv'}")
