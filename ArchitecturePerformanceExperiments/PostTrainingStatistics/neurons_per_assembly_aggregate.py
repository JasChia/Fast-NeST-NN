"""Aggregate the neurons-per-assembly sensitivity grid.

Reads scoring/sens/sens_D<drug>_<exp>.csv (one per experiment, 10 rows: k=1..10),
checks the architecture actually varied, and reports performance vs k.

Pairing: every k shares the same (drug, experiment) split and the same pinned
hyperparameters, so k-values pair exactly within a run. The 50 experiments per drug are
overlapping resamples of one cell-line pool, so the headline test collapses each drug to
its mean and runs Wilcoxon across drugs (n=12, or n=10 for the paper's drug set).
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

_ap = argparse.ArgumentParser(description="Aggregate the neurons-per-assembly sweep")
_ap.add_argument("--sweep-dir", type=Path, default=Path("sweep"),
                 help="directory of per-experiment sweep CSVs (default: ./sweep)")
_ap.add_argument("--out-dir", type=Path, default=Path("scores"),
                 help="where to write the summaries (default: ./scores)")
_args = _ap.parse_args()
S, OUT = _args.sweep_dir, _args.out_dir
# Drugs reported in the paper (Data_archives/ and the bundled job lists). The trained
# grid may hold more; this subsets it.
DRUG_IDS = {5, 80, 99, 127, 151, 188, 244, 273, 298, 380}
METRICS = ["test_r2", "test_spearman", "test_rmse"]
REF_K = 4

files = sorted(f for f in S.glob("sens_D*_*.csv") if "pilot" not in f.name)
if not files:
    sys.exit("no sensitivity CSVs yet")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df.status == "ok"].copy()
for c in METRICS + ["k", "k_used", "n_params"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"files={len(files)}  rows={len(df)}  drugs={df.drug.nunique()}  "
      f"experiments={len(df.groupby(['drug','experiment']))}")

# ---- architecture actually varied? ----
bad = df[df.k_used != df.k]
print(f"k_used != k : {len(bad)} rows" + ("  <-- INVESTIGATE" if len(bad) else "  (ok)"))
pc = df.groupby("k").n_params.agg(["min", "max", "nunique"])
print("\nparameter count by k (must be strictly increasing; one value per k):")
print(pc.to_string())
if pc["nunique"].max() == 1 and pc["min"].is_monotonic_increasing:
    print("-> architecture varies with k exactly as intended")
else:
    print("-> WARNING: parameter count is not a clean function of k")

# ---- k=4 control vs the stored original run ----
ctl = df[(df.k == REF_K) & df.stored_test_r2_k4.notna()].copy()
if len(ctl):
    d = (ctl.test_r2 - pd.to_numeric(ctl.stored_test_r2_k4, errors="coerce")).abs()
    print(f"\nk=4 control vs stored original run (n={len(d)}):")
    print(f"  |delta test R2|  median={d.median():.4f}  mean={d.mean():.4f}  "
          f"p90={d.quantile(.9):.4f}  max={d.max():.4f}")
    print("  (these are RETRAINED models, so this is the training-noise floor against")
    print("   which any k effect must be judged -- not a reproduction failure)")

# ---- performance vs k ----
rows = []
for k, g in df.groupby("k"):
    r = {"k": int(k), "n_runs": len(g), "n_params": int(g.n_params.iloc[0])}
    for m in METRICS:
        r[f"{m}_mean"] = g[m].mean(); r[f"{m}_sd"] = g[m].std(ddof=1)
        r[f"{m}_median"] = g[m].median()
    rows.append(r)
summ = pd.DataFrame(rows).sort_values("k")
summ.to_csv(OUT / "sens_summary_by_k.csv", index=False)

df.groupby(["drug", "k"])[METRICS].agg(["mean", "std", "median"]).to_csv(
    OUT / "sens_summary_by_drug_k.csv")
df.to_csv(OUT / "sens_all_runs.csv", index=False)

# ---- drug-level paired Wilcoxon, each k vs k=4 ----
out = []
for subset, label in ((None, "all"), (DRUG_IDS, "reported")):
    d2 = df if subset is None else df[df.drug.isin(subset)]
    per = d2.groupby(["drug", "k"])[METRICS].mean().reset_index()
    ref = per[per.k == REF_K].set_index("drug")
    for k, g in per.groupby("k"):
        if k == REF_K: continue
        g = g.set_index("drug"); drugs = ref.index.intersection(g.index)
        row = {"subset": label, "k": int(k), "n_drugs": len(drugs)}
        for m in METRICS:
            a, b = g.loc[drugs, m].to_numpy(float), ref.loc[drugs, m].to_numpy(float)
            keep = np.isfinite(a) & np.isfinite(b); a, b = a[keep], b[keep]
            row[f"{m}_delta"] = (a - b).mean() if len(a) else np.nan
            row[f"{m}_p"] = (stats.wilcoxon(a, b)[1]
                             if len(a) >= 5 and not np.allclose(a, b) else np.nan)
        out.append(row)
if out:
    w = pd.DataFrame(out)
    for label in w.subset.unique():          # BH within each subset, per metric
        for m in METRICS:
            idx = w.index[w.subset == label]
            p = np.nan_to_num(w.loc[idx, f"{m}_p"].to_numpy(float), nan=1.0)
            order = np.argsort(p); adj = p[order] * len(p) / (np.arange(len(p)) + 1)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            q = np.empty_like(p); q[order] = np.minimum(adj, 1.0)
            w.loc[idx, f"{m}_q_BH"] = q
    w.to_csv(OUT / "sens_paired_vs_k4.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== TEST PERFORMANCE vs NEURONS PER ASSEMBLY ===")
disp = summ[["k", "n_params", "n_runs", "test_r2_mean", "test_r2_median",
             "test_spearman_mean", "test_spearman_sd", "test_rmse_mean", "test_rmse_sd"]].copy()
for c in disp.columns:
    if disp[c].dtype.kind == "f": disp[c] = disp[c].map(lambda v: f"{v:.4f}")
print(disp.to_string(index=False))
if out:
    print("\n=== DRUG-LEVEL PAIRED WILCOXON vs k=4 ===")
    ww = w[w.subset == "reported"][["k","n_drugs","test_spearman_delta","test_spearman_p",
                                   "test_spearman_q_BH","test_rmse_delta","test_rmse_p"]].copy()
    for c in ww.columns:
        if ww[c].dtype.kind == "f": ww[c] = ww[c].map(lambda v: f"{v:.4g}")
    print(ww.to_string(index=False))
