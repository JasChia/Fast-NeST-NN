"""Merge per-method score files into (a) one pairing-ready long table and
(b) summary statistics. Paired tests are done on the INTERSECTION of
(drug, experiment) keys, because methods do not all cover the same drugs."""
import argparse, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

# Drugs reported in the paper (Data_archives/). Others may be present in the per-run
# score files; they are filtered out before reporting.
DRUG_IDS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]

_ap = argparse.ArgumentParser(description="Merge per-method score files and run paired tests")
_ap.add_argument("--scores-dir", type=Path, default=Path("scores"),
                 help="directory holding <Method>_scores.csv from the score_* scripts; "
                      "summaries are written back into it (default: ./scores)")
_ap.add_argument("--drugs", default=",".join(map(str, DRUG_IDS)),
                 help="comma-separated drugs to report on, or 'all' to keep every drug "
                      "found (default: the reported 10)")
_args = _ap.parse_args()
OUT = _args.scores_dir
KEEP = None if _args.drugs.strip().lower() == "all" else [int(x) for x in _args.drugs.split(",")]
METRICS = ["test_spearman", "test_rmse", "test_r2", "test_pearson"]

frames = []
SELF = {"all_methods_scores.csv"}
for f in sorted(OUT.glob("*_scores.csv")):
    if f.name in SELF or f.name.startswith("_"):
        continue   # never re-ingest our own output (or smoke-test leftovers)
    df = pd.read_csv(f)
    if df.empty:
        print(f"  skip empty {f.name}"); continue
    frames.append(df)
    print(f"  loaded {f.name}: {len(df)} rows, verify_ok={int(df['verify_ok'].sum())}")
if not frames:
    sys.exit("no score files found")
all_df = pd.concat(frames, ignore_index=True)
if KEEP is not None:
    dropped = sorted(set(all_df.drug.unique()) - set(KEEP))
    before = len(all_df)
    all_df = all_df[all_df.drug.isin(KEEP)].reset_index(drop=True)
    print(f"  drug filter: kept {sorted(KEEP)}; dropped {dropped} "
          f"({before - len(all_df)} rows)")
all_df.to_csv(OUT / "all_methods_scores.csv", index=False)
print(f"\nwrote all_methods_scores.csv: {len(all_df)} rows, "
      f"{all_df['method'].nunique()} methods")

bad = all_df[all_df["verify_ok"] != 1]
if len(bad):
    print(f"\n!! {len(bad)} rows FAILED verification -- excluded from statistics:")
    print(bad[["method","drug","experiment","note"]].to_string(index=False))
ok = all_df[all_df["verify_ok"] == 1].copy()

# ---- summary per method ----
rows = []
for m, g in ok.groupby("method"):
    r = {"method": m, "n_runs": len(g), "n_drugs": g["drug"].nunique(),
         "n_degenerate": int(g["note"].astype(str).str.contains("DEGENERATE").sum())}
    for met in METRICS:
        r[f"{met}_mean"] = g[met].mean()          # pandas skips NaN
        r[f"{met}_sd"] = g[met].std(ddof=1)
        # R2 means are dominated by a few catastrophic runs on low-variance drugs
        # (RSNN D201_10: R2 = -42 with a perfectly ordinary RMSE of 0.158, because
        # D201's label variance is 20x smaller than every other drug's). Report the
        # median alongside so the summary is not driven by those tails.
        r[f"{met}_median"] = g[met].median()
        r[f"{met}_n"] = int(g[met].notna().sum())  # how many runs actually contributed
    rows.append(r)
summ = pd.DataFrame(rows).sort_values("test_spearman_mean", ascending=False)
summ.to_csv(OUT / "summary_by_method.csv", index=False)

# ---- summary per method x drug (for drug-level paired tests) ----
pd_rows = []
for (m, d), g in ok.groupby(["method", "drug"]):
    r = {"method": m, "drug": d, "n_runs": len(g)}
    for met in METRICS:
        r[f"{met}_mean"] = g[met].mean()
        r[f"{met}_sd"] = g[met].std(ddof=1)
    pd_rows.append(r)
pd.DataFrame(pd_rows).to_csv(OUT / "summary_by_method_drug.csv", index=False)

# ---- experiment-level paired Wilcoxon, every method vs the reference ----
REF = "fNeST-NN"
if REF in set(ok["method"]):
    out = []
    ref = ok[ok["method"] == REF].set_index(["drug", "experiment"])
    for m, g in ok.groupby("method"):
        if m == REF: continue
        g = g.set_index(["drug", "experiment"])
        keys = ref.index.intersection(g.index)
        if len(keys) < 10: continue
        row = {"method": m, "reference": REF, "n_paired": len(keys),
               "n_drugs_paired": len({k[0] for k in keys})}
        for met in METRICS:
            a = g.loc[keys, met].to_numpy(float)
            b = ref.loc[keys, met].to_numpy(float)
            # drop pairs where EITHER side is NaN (collapsed/constant-prediction models)
            keep = np.isfinite(a) & np.isfinite(b)
            row[f"{met}_n_pairs"] = int(keep.sum())
            row[f"{met}_n_dropped_nan"] = int((~keep).sum())
            a, b = a[keep], b[keep]
            if len(a) < 10:
                row[f"{met}_delta_mean"] = np.nan
                row[f"{met}_W"], row[f"{met}_p"] = np.nan, np.nan
                continue
            d = a - b
            row[f"{met}_delta_mean"] = d.mean()
            if np.allclose(d, 0):
                row[f"{met}_W"], row[f"{met}_p"] = np.nan, 1.0
            else:
                W, p = stats.wilcoxon(a, b)
                row[f"{met}_W"], row[f"{met}_p"] = W, p
        out.append(row)
    if out:
        w = pd.DataFrame(out)
        # Benjamini-Hochberg across methods, per metric
        for met in METRICS:
            p = np.nan_to_num(w[f"{met}_p"].to_numpy(float), nan=1.0)
            order = np.argsort(p); ranked = np.empty_like(p)
            adj = p[order] * len(p) / (np.arange(len(p)) + 1)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            ranked[order] = np.minimum(adj, 1.0)
            w[f"{met}_q_BH"] = ranked
        w.to_csv(OUT / "paired_wilcoxon_vs_fNeST-NN.csv", index=False)
        print(f"wrote paired_wilcoxon_vs_fNeST-NN.csv ({len(w)} comparisons)")

# ---- DRUG-LEVEL paired Wilcoxon (the statistically defensible headline test) ----
# The 50 experiments per drug are repeated random subsamples of the SAME cell lines, so
# experiment-level pairs are not independent and their p-values are anticonservative.
# Averaging within drug first gives one independent observation per drug (n=10-12).
if REF in set(ok["method"]):
    out = []
    rmean = ok[ok["method"] == REF].groupby("drug")[METRICS].mean()
    for m, g in ok.groupby("method"):
        if m == REF: continue
        gm = g.groupby("drug")[METRICS].mean()
        drugs = rmean.index.intersection(gm.index)
        if len(drugs) < 5: continue
        row = {"method": m, "reference": REF, "n_drugs": len(drugs)}
        for met in METRICS:
            a, b = gm.loc[drugs, met].to_numpy(float), rmean.loc[drugs, met].to_numpy(float)
            keep = np.isfinite(a) & np.isfinite(b); a, b = a[keep], b[keep]
            row[f"{met}_n"] = int(keep.sum())
            row[f"{met}_delta_mean"] = (a - b).mean() if len(a) else np.nan
            if len(a) >= 5 and not np.allclose(a, b):
                W, pv = stats.wilcoxon(a, b)
                row[f"{met}_W"], row[f"{met}_p"] = W, pv
            else:
                row[f"{met}_W"], row[f"{met}_p"] = np.nan, np.nan
        out.append(row)
    if out:
        dw = pd.DataFrame(out)
        for met in METRICS:
            pv = np.nan_to_num(dw[f"{met}_p"].to_numpy(float), nan=1.0)
            order = np.argsort(pv); ranked = np.empty_like(pv)
            adj = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            ranked[order] = np.minimum(adj, 1.0)
            dw[f"{met}_q_BH"] = ranked
        dw.to_csv(OUT / "paired_wilcoxon_druglevel_vs_fNeST-NN.csv", index=False)
        print(f"wrote paired_wilcoxon_druglevel_vs_fNeST-NN.csv ({len(dw)} comparisons)")

pd.set_option("display.width", 200)
print("\n=== TEST-SET SUMMARY (mean +/- SD over runs) ===")
disp = summ[["method","n_runs","n_drugs","n_degenerate","test_spearman_mean","test_spearman_sd",
             "test_spearman_median","test_rmse_mean","test_rmse_sd","test_rmse_median",
             "test_r2_mean","test_r2_median"]].copy()
for c in disp.columns:
    if disp[c].dtype.kind == "f": disp[c] = disp[c].map(lambda v: f"{v:.4f}")
print(disp.to_string(index=False))
