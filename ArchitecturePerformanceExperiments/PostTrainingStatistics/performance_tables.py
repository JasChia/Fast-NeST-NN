"""Per-drug performance tables: every method (LightGBM included) vs fNeST-NN.

Reads the <Method>_scores.csv files written by score_models.py / score_nestvnn.py /
score_lightgbm.py and writes:

  perf_<metric>_all_methods.csv   one row per (drug, method): n, mean, sd, p, q_BH vs fNeST-NN,
                                  LightGBM first as in the printed tables
                                  (r2: main Table 5; pearson: Supp S4; rmse: S5; spearman: S6)
  perf_lightgbm_vs_fnest.csv      one row per (metric, drug): both means/SDs, p, q_BH

Statistics (the variant that reproduces every cell of Table 5 and S4-S6, rounded once to the
printed digits):
  * mean and SD (ddof=1) over the 50 cross-validation experiments of a drug;
  * paired Wilcoxon signed-rank vs fNeST-NN on the 50 experiments of that drug, pairing by
    experiment (shared split), zero_method="wilcox", no continuity correction, normal
    approximation. scipy >= 1.13 switches n=50 to the exact test under the default method,
    so the method is pinned here;
  * Benjamini-Hochberg within each method and metric, across the 10 drugs, so adding a method
    leaves the other columns' adjusted p-values unchanged.
  * --degenerate zero|drop: runs that collapsed to a constant prediction have undefined
    Pearson/Spearman. "zero" scores them 0 (default, as in aggregate_scores.py and the
    printed tables); "drop" leaves them out of the mean and the test. R2 and RMSE are
    unaffected.

Run:
  python performance_tables.py --scores-dir scores        # Table 5, S4, S5, S6
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

DRUG_IDS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
METHODS = ["LightGBM", "FCNN", "Dense-fNeST", "UGP-NN", "GP-NN", "LP-NN", "RSNN", "ERK_SNN", "RP-fNeST",
           "NeST-VNN", "fNeST-NN"]
METRICS = {"r2": "test_r2", "pearson": "test_pearson", "spearman": "test_spearman", "rmse": "test_rmse"}
LOWER_IS_BETTER = {"rmse"}
REF = "fNeST-NN"

_ap = argparse.ArgumentParser(description="Per-drug performance tables with paired Wilcoxon + BH")
_ap.add_argument("--scores-dir", type=Path, default=Path("scores"),
                 help="directory holding <Method>_scores.csv (default: ./scores)")
_ap.add_argument("--out-dir", type=Path, default=None, help="where to write (default: --scores-dir)")
_ap.add_argument("--degenerate", choices=["zero", "drop"], default="zero",
                 help="score constant-prediction runs' correlations 0 (default) or drop them")
_ap.add_argument("--values", choices=["recomputed", "stored"], default="recomputed",
                 help="'recomputed' (default): full-precision re-scored metrics; 'stored': the 4-dp "
                      "values each tuner recorded "
                      "(RMSE was never stored and stays recomputed)")
_args = _ap.parse_args()
OUT = _args.out_dir or _args.scores_dir
OUT.mkdir(parents=True, exist_ok=True)


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p)
    adj = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.minimum(adj, 1.0)
    return q


def wilcoxon_p(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 10 or np.allclose(a, b):
        return np.nan, int(keep.sum())
    return stats.wilcoxon(a, b, zero_method="wilcox", correction=False, method="approx").pvalue, int(keep.sum())


def load():
    frames = [pd.read_csv(f) for f in sorted(_args.scores_dir.glob("*_scores.csv"))
              if not f.name.startswith(("all_", "_"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.drug.isin(DRUG_IDS) & (df.verify_ok == 1)].copy()
    if _args.values == "stored":
        df["test_r2"] = df["stored_test_r2"].round(4)
        df["test_pearson"] = df["stored_test_pearson"]
        df["test_spearman"] = df["stored_test_spearman"]
    if _args.degenerate == "zero":
        corr = ["test_pearson", "test_spearman"]
        df[corr] = df[corr].fillna(0.0)
    return df


def per_drug(df, methods):
    rows = []
    for met, col in METRICS.items():
        for m in methods:
            for d in DRUG_IDS:
                g = df[(df.method == m) & (df.drug == d)].set_index("experiment")[col]
                r = df[(df.method == REF) & (df.drug == d)].set_index("experiment")[col]
                row = {"metric": met, "drug": d, "method": m,
                       "n": int(g.notna().sum()), "mean": g.mean(), "sd": g.std(ddof=1)}
                if m != REF:
                    keys = g.index.intersection(r.index)
                    row["p"], row["n_pairs"] = wilcoxon_p(g.loc[keys].to_numpy(float), r.loc[keys].to_numpy(float))
                    row["delta_vs_ref"] = g.loc[keys].mean() - r.loc[keys].mean()
                rows.append(row)
    t = pd.DataFrame(rows)
    t["q_BH"] = np.nan
    for _, g in t[t["p"].notna()].groupby(["metric", "method"]):   # BH within method, across drugs
        t.loc[g.index, "q_BH"] = bh(g["p"])
    return t


df = load()
t = per_drug(df, METHODS)
for met in METRICS:
    t[t.metric == met].to_csv(OUT / f"perf_{met}_all_methods.csv", index=False)

lg = per_drug(df, ["LightGBM", REF])
a = lg[lg.method == "LightGBM"].set_index(["metric", "drug"])
b = lg[lg.method == REF].set_index(["metric", "drug"])
h = pd.DataFrame({"lgb_mean": a["mean"], "lgb_sd": a.sd, "fn_mean": b["mean"], "fn_sd": b.sd,
                  "delta": a["mean"] - b["mean"], "p": a.p, "q_BH": a.q_BH}).reset_index()
h.to_csv(OUT / "perf_lightgbm_vs_fnest.csv", index=False)

pd.set_option("display.width", 200)
print("method means across the 10 drugs:")
print(t.pivot_table(index="method", columns="metric", values="mean").reindex(METHODS + []).round(4).to_string())
sig = h[h.q_BH < 0.05]
lgb_better = np.where(sig.metric.isin(LOWER_IS_BETTER), sig.delta < 0, sig.delta > 0)
print(f"\ndegenerate={_args.degenerate}; LightGBM vs fNeST-NN: {len(sig)}/{len(h)} significant, "
      f"{int(lgb_better.sum())} favour LightGBM, {int((~lgb_better).sum())} favour fNeST-NN")
print(f"wrote perf_*_all_methods.csv and perf_lightgbm_vs_fnest.csv to {OUT}")
