"""Degree-preserving vs unshuffled NeST-VNN: per-drug summary and paired tests
(Degree-Preserving column of main Table 3 and Supplementary Tables S1-S3, and the across-drug
test).

Reads, for each of the 10 reported drugs and 50 experiments,
    nest_vnn/nest_vnn_logs/D{drug}_unshuffled_exp_{e}/
    degree_preserving_shuffle/nest_vnn_logs/D{drug}_degshuf_exp_{e}/
R2, Pearson and Spearman come from test_metrics.txt (4 dp, as written by vnn_trainer.py);
RMSE is not stored there and is computed from test_predictions.txt / test_true_labels.txt.
The two arms share the test set of every experiment (checked here).

Convention (the one that reproduces the published tables): mean and SD (ddof=1) over the 50
experiments; paired Wilcoxon signed-rank, zero_method="wilcox", no continuity correction,
normal approximation; Benjamini-Hochberg across the 10 drugs within each metric. Across
drugs: exact Wilcoxon over the 10 per-drug means.

Usage (from NeSTVNNShuffleAnalysis/):
    python degree_preserving_shuffle/degree_preserving_tables.py
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

_SA = Path(__file__).resolve().parent.parent          # NeSTVNNShuffleAnalysis/
_ap = argparse.ArgumentParser(description="Degree-preserving vs unshuffled tables")
_ap.add_argument("--unshuffled-logs", type=Path, default=_SA / "nest_vnn" / "nest_vnn_logs")
_ap.add_argument("--degshuf-logs", type=Path, default=_SA / "degree_preserving_shuffle" / "nest_vnn_logs")
_ap.add_argument("--out-dir", type=Path, default=_SA / "degree_preserving_shuffle" / "tables")
_args = _ap.parse_args()
OUT = _args.out_dir; OUT.mkdir(parents=True, exist_ok=True)

DRUG_IDS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
ARMS = {"unshuffled": (_args.unshuffled_logs, "D{d}_unshuffled_exp_{e}"),
        "degshuf": (_args.degshuf_logs, "D{d}_degshuf_exp_{e}")}
KEYS = {"R² Score": "r2", "Pearson Correlation": "pearson", "Spearman Correlation": "spearman"}
METRICS = ["r2", "pearson", "spearman", "rmse"]
LOWER_IS_BETTER = {"rmse"}


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p)
    adj = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.minimum(adj, 1.0)
    return q


def read_run(run):
    m = {}
    for line in open(run / "test_metrics.txt", encoding="utf-8"):
        k, _, v = line.rpartition(":")
        if k.strip() in KEYS:
            m[KEYS[k.strip()]] = float(v)
    y = np.loadtxt(run / "test_true_labels.txt"); yh = np.loadtxt(run / "test_predictions.txt")
    m["rmse"] = float(np.sqrt(np.mean((y - yh) ** 2)))
    return m, y


rows = []
for d in DRUG_IDS:
    for e in range(50):
        labels = {}
        for arm, (root, pat) in ARMS.items():
            m, labels[arm] = read_run(root / pat.format(d=d, e=e))
            rows.append(dict(drug=d, experiment=e, arm=arm, **m))
        assert np.array_equal(labels["unshuffled"], labels["degshuf"]), f"test sets differ: D{d} exp {e}"
runs = pd.DataFrame(rows)
runs.to_csv(OUT / "degree_preserving_all_runs.csv", index=False)

summary, across = [], []
for m in METRICS:
    w = runs.pivot_table(index=["drug", "experiment"], columns="arm", values=m)
    p = [stats.wilcoxon(w.loc[d, "degshuf"], w.loc[d, "unshuffled"], zero_method="wilcox",
                        correction=False, method="approx").pvalue for d in DRUG_IDS]
    for d, pi, qi in zip(DRUG_IDS, p, bh(p)):
        x, u = w.loc[d, "degshuf"], w.loc[d, "unshuffled"]
        summary.append(dict(metric=m, drug=d, degshuf_mean=x.mean(), degshuf_sd=x.std(ddof=1),
                            unshuffled_mean=u.mean(), unshuffled_sd=u.std(ddof=1), p=pi, q_BH=qi))
    mu = w.groupby(level=0).mean()
    worse = (mu.degshuf > mu.unshuffled) if m in LOWER_IS_BETTER else (mu.degshuf < mu.unshuffled)
    across.append(dict(metric=m, unshuffled_mean=mu.unshuffled.mean(), degshuf_mean=mu.degshuf.mean(),
                       diff=mu.degshuf.mean() - mu.unshuffled.mean(), drugs_worse=int(worse.sum()),
                       p_drug_level_exact=stats.wilcoxon(mu.degshuf, mu.unshuffled, method="exact").pvalue))
summary = pd.DataFrame(summary); across = pd.DataFrame(across)
summary.to_csv(OUT / "degree_preserving_by_drug.csv", index=False)
across.to_csv(OUT / "degree_preserving_across_drugs.csv", index=False)
pd.set_option("display.width", 200)
print(summary.round(5).to_string(index=False)); print(); print(across.round(5).to_string(index=False))
print(f"\nwrote {OUT}")
