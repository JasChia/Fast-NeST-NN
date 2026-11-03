"""Per-drug paired tests for the neurons-per-assembly sweep.

Reads the `sens_all_runs.csv` written by `neurons_per_assembly_aggregate.py` and tests
every k against k=4 *within* each drug, pairing the 50 cross-validation experiments by
split: experiment e at width k is compared with experiment e at k=4, which shares the
same train/validation/test partition and the same pinned hyperparameters.

Why per drug and not one pooled test over all 500 runs: the 50 experiments of a drug are
overlapping resamples of that drug's cell-line pool (mean Jaccard ~0.11 between test
sets, every line appearing in ~10 of the 50), so they are exchangeable within a drug but
not across drugs. Pooling all 500 pairs into a single test would treat correlated runs as
independent and shrink the p-values accordingly. `neurons_per_assembly_aggregate.py`
keeps the complementary drug-level view (one mean per drug, n=10 across drugs).

It also writes the pooled version of the same split-level pairing -- all 500 pairs in one
test, one p-value per (metric, k) -- because that is the single number the paper's table
reports. Read it together with the per-drug counts: pooling is the more sensitive view and
the less conservative one.

Convention: Wilcoxon signed-rank with `zero_method="wilcox"`, no continuity correction and
the normal approximation -- the variant the published n=50 tables used -- then
Benjamini-Hochberg across the nine k comparisons within each drug and metric.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

_ap = argparse.ArgumentParser(description="Per-drug paired tests for the sweep")
_ap.add_argument("--scores-dir", type=Path, default=Path("scores"),
                 help="directory holding sens_all_runs.csv (default: ./scores)")
_ap.add_argument("--out-dir", type=Path, default=None,
                 help="where to write the three CSVs (default: --scores-dir)")
_ap.add_argument("--degenerate", choices=["zero", "drop"], default="zero",
                 help="runs that collapsed to a constant prediction have undefined "
                      "correlations: 'zero' scores them 0 (default, as in the paper), "
                      "'drop' leaves them out")
_args = _ap.parse_args()
OUT = _args.out_dir or _args.scores_dir
OUT.mkdir(parents=True, exist_ok=True)

DRUG_IDS = {5, 80, 99, 127, 151, 188, 244, 273, 298, 380}
METRICS = ["test_r2", "test_pearson", "test_spearman", "test_rmse"]
CORR = ["test_pearson", "test_spearman"]       # undefined for a constant predictor
REF_K = 4

def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p)
    adj = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.minimum(adj, 1.0)
    return q

df = pd.read_csv(_args.scores_dir / "sens_all_runs.csv")
df = df[(df.status == "ok") & df.drug.isin(DRUG_IDS)].copy()
n_undefined = int(df[CORR].isna().to_numpy().sum())
if _args.degenerate == "zero":
    df[CORR] = df[CORR].fillna(0.0)
print(f"rows={len(df)}  drugs={df.drug.nunique()}  "
      f"undefined correlations={n_undefined} ({_args.degenerate})")

# ---- per-drug paired tests, each k vs k=4 ----
rows = []
for drug, g in df.groupby("drug"):
    for m in METRICS:
        wide = g.pivot_table(index="experiment", columns="k", values=m)
        if REF_K not in wide.columns:
            continue
        for k in sorted(c for c in wide.columns if c != REF_K):
            pair = wide[[k, REF_K]].dropna()          # same split in both arms
            a, b = pair[k].to_numpy(float), pair[REF_K].to_numpy(float)
            p = (stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                                method="approx")[1]
                 if len(a) >= 5 and not np.allclose(a, b) else np.nan)
            rows.append({"drug": int(drug), "metric": m, "k": int(k),
                         "n_pairs": len(a), "mean_k": a.mean(), "mean_k4": b.mean(),
                         "delta_vs_k4": (a - b).mean(), "p": p})
t = pd.DataFrame(rows)
for (drug, m), idx in t.groupby(["drug", "metric"]).groups.items():   # BH within drug
    t.loc[idx, "q_BH"] = bh(np.nan_to_num(t.loc[idx, "p"].to_numpy(float), nan=1.0))
t.to_csv(OUT / "sens_tests_by_drug.csv", index=False)

# ---- pooled test: every run against the k=4 run of the same drug and split ----
# One number per (metric, k), which is what the paper's table reports. It is the more
# sensitive of the two views and the less conservative: the 500 pairs are not fully
# independent, so read a small pooled p next to the per-drug counts above.
pooled = []
for m in METRICS:
    wide = df.pivot_table(index=["drug", "experiment"], columns="k", values=m)
    if REF_K not in wide.columns:
        continue
    for k in sorted(c for c in wide.columns if c != REF_K):
        pair = wide[[k, REF_K]].dropna()
        a, b = pair[k].to_numpy(float), pair[REF_K].to_numpy(float)
        pooled.append({"metric": m, "k": int(k), "n_pairs": len(a),
                       "mean_k": a.mean(), "sd_k": pair[k].std(ddof=1),
                       "delta_vs_k4": (a - b).mean(),
                       "p": stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                                           method="approx")[1]})
pl = pd.DataFrame(pooled)
for m, idx in pl.groupby("metric").groups.items():          # BH within metric
    pl.loc[idx, "q_BH"] = bh(pl.loc[idx, "p"].to_numpy(float))
pl.to_csv(OUT / "sens_tests_pooled.csv", index=False)

# ---- how many drugs move, and which way ----
summ = (t.assign(sig_lower=(t.q_BH <= 0.05) & (t.delta_vs_k4 < 0),
                 sig_higher=(t.q_BH <= 0.05) & (t.delta_vs_k4 > 0))
         .groupby(["metric", "k"])
         .agg(n_drugs=("drug", "nunique"), n_sig_lower=("sig_lower", "sum"),
              n_sig_higher=("sig_higher", "sum"), mean_delta=("delta_vs_k4", "mean"))
         .reset_index())
summ.to_csv(OUT / "sens_tests_summary.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== DRUGS DIFFERING FROM k=4 (paired over the 50 splits, BH within drug) ===")
for m in METRICS:
    s = summ[summ.metric == m]
    print(f"\n{m}:")
    print("  k        mean delta   lower   higher")
    for _, r in s.iterrows():
        print(f"  {int(r.k):<2}      {r.mean_delta:>10.4f}   {int(r.n_sig_lower):>5}"
              f"   {int(r.n_sig_higher):>6}    (of {int(r.n_drugs)} drugs)")
print("\n=== POOLED OVER ALL RUNS, PAIRED BY (DRUG, SPLIT) ===")
for m in METRICS:
    s = pl[pl.metric == m]
    print(f"\n{m}:   k   mean +/- sd        delta vs k=4    q_BH")
    for _, r in s.iterrows():
        print(f"         {int(r.k):<2}  {r.mean_k:.4f} +/- {r.sd_k:.4f}   "
              f"{r.delta_vs_k4:>8.4f}      {r.q_BH:.4f}")

print(f"\nwrote {OUT/'sens_tests_by_drug.csv'}, {OUT/'sens_tests_summary.csv'} "
      f"and {OUT/'sens_tests_pooled.csv'}")
