"""Gene-permuted ("shuffled") vs unshuffled NeST-VNN test performance: the Gene Permuted column
and its p-values in main Table 3 (R2) and Supp Tables S1-S3 (Pearson, Spearman, RMSE).
The Degree-Preserving column comes from ../../degree_preserving_shuffle/degree_preserving_tables.py.

Reads every nest_vnn_logs/D<drug>_{shuffled,unshuffled}_exp_<k>/test_metrics.txt written by
src/vnn_trainer.py (4-dp values) and pairs the two conditions on (drug, experiment). RMSE is
not stored there and is computed from test_predictions.txt / test_true_labels.txt. Paired
runs share the test set and the gene-order seed 42+k (../create_train_test_splits.py).

Per drug: mean +- SD (ddof=1) over the 50 experiments for each condition, and the
BH-adjusted p-value of a paired Wilcoxon signed-rank test (normal approximation,
zero_method="wilcox", no continuity correction), BH across the reported drugs.
This variant reproduces every printed Gene Permuted / Unshuffled cell of Table 3 and S1-S3.

Usage (from NeSTVNNShuffleAnalysis/nest_vnn):
    python shuffle_assignment_analysis/shuffle_performance_table.py
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

# Drugs reported in the paper
DRUG_IDS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
METRICS = {"R2": "R² Score", "Pearson": "Pearson Correlation", "Spearman": "Spearman Correlation"}

_NV = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser(description="Shuffled vs unshuffled NeST-VNN paired Wilcoxon table")
_ap.add_argument("--logs", type=Path, default=_NV / "nest_vnn_logs",
                 help="per-run output tree (default: nest_vnn/nest_vnn_logs)")
_ap.add_argument("--experiments", type=int, default=50)
_ap.add_argument("--out", type=Path, default=_NV / "shuffle_assignment_analysis" / "outputs" / "shuffle_performance.csv")
_args = _ap.parse_args()


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p)
    adj = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.minimum(adj, 1.0)
    return q


def read_metrics(run):
    txt = (run / "test_metrics.txt").read_text(encoding="utf-8")
    m = {k: float(re.search(re.escape(lbl) + r":\s*(\S+)", txt).group(1)) for k, lbl in METRICS.items()}
    y = np.loadtxt(run / "test_true_labels.txt"); yh = np.loadtxt(run / "test_predictions.txt")
    m["RMSE"] = float(np.sqrt(np.mean((y - yh) ** 2)))
    return m, y


rows = []
for d in DRUG_IDS:
    for k in range(_args.experiments):
        labels = {}
        for cond in ("shuffled", "unshuffled"):
            run = _args.logs / f"D{d}_{cond}_exp_{k}"
            if not (run / "test_metrics.txt").exists():
                raise SystemExit(f"missing run: {run}")
            m, labels[cond] = read_metrics(run)
            rows.append({"drug": d, "experiment": k, "condition": cond, **m})
        assert np.array_equal(labels["shuffled"], labels["unshuffled"]), f"test sets differ: D{d} exp {k}"
runs = pd.DataFrame(rows)

table = []
for m in [*METRICS, "RMSE"]:
    wide = runs.pivot_table(index=["drug", "experiment"], columns="condition", values=m)
    ps = [wilcoxon(wide.loc[d, "shuffled"], wide.loc[d, "unshuffled"], zero_method="wilcox",
                   correction=False, method="approx").pvalue for d in DRUG_IDS]
    for d, q in zip(DRUG_IDS, bh(ps)):
        s, u = wide.loc[d, "shuffled"], wide.loc[d, "unshuffled"]
        table.append({"metric": m, "drug": d, "shuffled_mean": s.mean(), "shuffled_sd": s.std(ddof=1),
                      "unshuffled_mean": u.mean(), "unshuffled_sd": u.std(ddof=1), "q_BH": q})
table = pd.DataFrame(table)
_args.out.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(_args.out, index=False)
with pd.option_context("display.width", 160, "display.float_format", "{:.5f}".format):
    print(table.to_string(index=False))
print(f"\nwrote {_args.out}")
