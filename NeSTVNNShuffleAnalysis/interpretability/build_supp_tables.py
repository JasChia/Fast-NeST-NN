"""Importance agreement between unshuffled and perturbed NeST-VNN models on the same split
(main Table 4).

Run after collect_importance.py. Reads assembly_importance.csv and gene_importance.csv from
this directory and writes table4_assembly.csv, table4_overlap.csv and table4_gene.csv for
--perturbed shuffled (gene order permutation, the default), or the same three files with a
_degshuf suffix for --perturbed degshuf (degree-preserving first-layer swapping). Main Table 4
is the assembly mean/sd, the gene mean/sd and the overlap si_ge_0.5 column of both runs.

For each drug and each of the 50 cross-validation splits i, the unshuffled and perturbed
NeST-VNN models trained on split i are compared:

  assembly  Spearman(SI_unshuffled[i], SI_shuffled[i]) over the 131 assemblies
  overlap   top-k shared = |top_k(U_i) & top_k(S_i)| / k      (k = 5, 10, 25; stable tie order)
            SI >= 0.5 shared = Jaccard |A & B| / |A | B| of the two important sets (NaN if both empty)
  gene      Spearman(|Rho|_unshuffled[i], |Rho|_shuffled[i]) over the 689 genes, with shuffled-model
            input slot j attributed to gene pi_i[j], the gene whose expression fed it; undefined Rho
            (constant gene or prediction) scored 0. Degree-preserving swapping leaves the expression
            matrix untouched, so for --perturbed degshuf slot j is gene j and no attribution is done

The two models of a split share the test set the scores are computed on.

Reference (ref): for split i, the mean of the same statistic between the unshuffled model of
split i and the unshuffled models of the other 49 splits -- the agreement expected from
resampling alone. Per-drug p-values are a paired Wilcoxon signed-rank test of the 50
same-split values against their 50 reference values (normal approximation, no continuity
correction), BH-adjusted across the 10 drugs. They test agreement against the reference,
not against zero. The summary line uses one mean per drug: exact Wilcoxon of the 10 drug
means against the 10 reference means (smallest attainable two-sided p = 2/2^10), unadjusted.
"""
import argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

I = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser(description="Importance agreement tables (main Table 4)")
_ap.add_argument("--perturbed", choices=["shuffled", "degshuf"], default="shuffled",
                 help="perturbed arm compared with the unshuffled arm (default: shuffled)")
_ap.add_argument("--importance-dir", type=Path, default=I,
                 help="directory holding the two importance CSVs (default: this directory)")
_ap.add_argument("--out-dir", type=Path, default=None, help="default: --importance-dir")
_args = _ap.parse_args()
OUT = _args.out_dir or _args.importance_dir
OUT.mkdir(parents=True, exist_ok=True)
ARM = _args.perturbed
SUF = "" if ARM == "shuffled" else f"_{ARM}"

DRUG_IDS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
NEXP = 50
GENE_ORDER = [l.split()[1] for l in open(ROOT / "Data/red_gene2ind.txt")]
NG = len(GENE_ORDER)


def pi_for(exp):
    """Column permutation of experiment `exp`, as in create_train_test_splits.py:
    shuffled column j holds original column pi[j]."""
    random.seed(42 + exp)
    idx = list(range(NG))
    random.shuffle(idx)
    return np.asarray(idx)


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); order = np.argsort(p)
    adj = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.minimum(adj, 1.0)
    return q


def wx(x, y, method="approx"):
    return wilcoxon(x, y, zero_method="wilcox", correction=False, method=method)[1]


def mat(df, drug, arm, key, val, index=None):
    """(element x experiment) matrix of one drug and arm."""
    m = df[(df.drug == drug) & (df.arm == arm)].pivot_table(
        index=key, columns="experiment", values=val, dropna=False)
    m = m.reindex(index) if index is not None else m.sort_index()
    assert list(m.columns) == list(range(NEXP)), (drug, arm, "missing experiments")
    return m.to_numpy(float)


def same_and_ref(U, S, f):
    same = np.array([f(U[:, i], S[:, i]) for i in range(NEXP)])
    ref = np.array([np.nanmean([f(U[:, i], U[:, j]) for j in range(NEXP) if j != i])
                    for i in range(NEXP)])
    return same, ref


def rho(a, b):
    return spearmanr(a, b)[0]


def topk(k):
    def f(a, b):
        A = set(np.argsort(-a, kind="stable")[:k]); B = set(np.argsort(-b, kind="stable")[:k])
        return len(A & B) / k
    return f


def jaccard(a, b, thr=0.5):
    A, B = set(np.where(a >= thr)[0]), set(np.where(b >= thr)[0])
    return len(A & B) / len(A | B) if A | B else np.nan


a = pd.read_csv(_args.importance_dir / "assembly_importance.csv")
g = pd.read_csv(_args.importance_dir / "gene_importance.csv")
assembly, overlap, gene = [], [], []
for d in DRUG_IDS:
    U, S = mat(a, d, "unshuffled", "Term", "SI"), mat(a, d, ARM, "Term", "SI")
    same, ref = same_and_ref(U, S, rho)
    assembly.append(dict(drug=d, mean=same.mean(), sd=same.std(ddof=1), ref=ref.mean(), p=wx(same, ref)))
    row = dict(drug=d)
    for lab, f in [("top5", topk(5)), ("top10", topk(10)), ("top25", topk(25)),
                   ("si_ge_0.5", jaccard)]:
        same, ref = same_and_ref(U, S, f)
        row.update({lab: 100 * np.nanmean(same), f"{lab}_sd": 100 * np.nanstd(same, ddof=1),
                    f"{lab}_ref": 100 * np.nanmean(ref)})
    overlap.append(row)
    Ug = mat(g, d, "unshuffled", "Gene", "Rho", GENE_ORDER)
    Sg = mat(g, d, ARM, "Gene", "Rho", GENE_ORDER)
    if ARM == "shuffled":
        Sid = np.full_like(Sg, np.nan)
        for c in range(NEXP):
            Sid[pi_for(c), c] = Sg[:, c]          # slot j's score belongs to gene pi[j]
    else:
        Sid = Sg                                  # expression matrix untouched: slot j is gene j
    Ug, Sid = np.nan_to_num(np.abs(Ug)), np.nan_to_num(np.abs(Sid))
    same, ref = same_and_ref(Ug, Sid, rho)
    gene.append(dict(drug=d, mean=same.mean(), sd=same.std(ddof=1), ref=ref.mean(), p=wx(same, ref)))

pd.set_option("display.width", 200)
for name, rows in [("assembly", assembly), ("gene", gene)]:
    t = pd.DataFrame(rows); t["q_BH"] = bh(t.p)
    t.to_csv(OUT / f"table4_{name}{SUF}.csv", index=False)
    print(f"\n{name}\n{t.to_string(index=False, float_format=lambda x: f'{x:.4g}')}")
    print(f"Mean {t['mean'].mean():.3f} +/- {t['mean'].std(ddof=1):.3f} (SD across drug means); "
          f"drug-level exact Wilcoxon vs reference p = {wx(t['mean'], t.ref, 'exact'):.4f}")
t = pd.DataFrame(overlap); t.to_csv(OUT / f"table4_overlap{SUF}.csv", index=False)
print(f"\noverlap\n{t.round(1).to_string(index=False)}")
print("Mean", t[["top5", "top10", "top25", "si_ge_0.5"]].mean().round(1).to_dict())
