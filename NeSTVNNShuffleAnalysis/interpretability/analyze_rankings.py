"""Shuffled vs unshuffled importance-ranking comparison (paper's SI / Rho).

Contrasts (i != j index experiments within a drug):
  BASELINE  unshuffled(i) vs unshuffled(j)   same gene order, different split  -> ceiling
  PERMUTED  unshuffled(i) vs shuffled(j)     both differ                       -> effect
  MATCHED   unshuffled(i) vs shuffled(i)     same split, gene order differs    -> shown for reference

BASELINE and PERMUTED both involve different splits, so cell-sampling noise
enters each side equally and the difference isolates gene permutation.

Gene alignment.  The shuffle permutes GE columns; the ontology is unchanged, so
ontology slot j of a shuffled model receives the expression of gene pi[j].
  slot-aligned     : compare slot j to slot j       -> architectural prior
  identity-aligned : attribute slot j's score to gene pi[j]  -> biological faithfulness
pi is regenerated exactly as create_train_test_splits.py produced it and is
verified against the shuffled GE matrix.

Quantity reported: DESTRUCTION INDEX  D = (baseline - permuted) / (baseline - chance),
the fraction of structurally achievable ranking agreement that permutation
destroys.  D=0 permutation changes nothing; D=1 agreement falls to chance.
chance = 0 for Spearman, k/N for top-k overlap.

Inference at the DRUG level (the 50 experiments per drug are overlapping
resamples, not independent replicates): each drug -> one mean, then Wilcoxon
signed-rank across drugs.
"""
import itertools, random
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

# Resolve inside this clone: <repo>/NeSTVNNShuffleAnalysis/interpretability/
from pathlib import Path

I = str(Path(__file__).resolve().parent)
ROOT = str(Path(__file__).resolve().parents[2])
DRUG_IDS = [5, 80, 99, 127, 151, 188, 244, 273, 298, 380]
NPAIR = 12

GENE_ORDER = [l.split()[1] for l in open(f"{ROOT}/Data/red_gene2ind.txt")]
NG = len(GENE_ORDER)

def pi_for(exp):
    """Column permutation used by experiment `exp`: shuffled col j = original col pi[j]."""
    random.seed(42 + exp)
    idx = list(range(NG))
    random.shuffle(idx)
    return np.asarray(idx)

def srho(x, y):
    m = np.isfinite(x) & np.isfinite(y)          # pairwise-complete: 0.02% of gene Rho are NaN
    return spearmanr(x[m], y[m])[0] if m.sum() > 2 else np.nan

def topov(x, y, k):
    m = np.isfinite(x) & np.isfinite(y)
    xi, yi = np.where(m)[0], np.where(m)[0]
    a = set(xi[np.argsort(-x[m])[:k]]); b = set(yi[np.argsort(-y[m])[:k]])
    return len(a & b) / k

def analyze(mat_u, mat_s, drugs_exps, ks, N):
    """mat_u/mat_s: dict drug -> (entities x experiments) arrays, already aligned."""
    rows = []
    for drug in sorted(mat_u):
        u, s = mat_u[drug], mat_s[drug]
        exps = list(range(u.shape[1]))
        pairs = list(itertools.combinations(exps, 2))[:NPAIR]
        rec = {"drug": drug}
        rec["base_rho"] = np.nanmean([srho(u[:, i], u[:, j]) for i, j in pairs])
        rec["perm_rho"] = np.nanmean([srho(u[:, i], s[:, j]) for i, j in pairs])
        rec["match_rho"] = np.nanmean([srho(u[:, i], s[:, i]) for i in exps])
        for k in ks:
            rec[f"base_top{k}"] = np.nanmean([topov(u[:, i], u[:, j], k) for i, j in pairs])
            rec[f"perm_top{k}"] = np.nanmean([topov(u[:, i], s[:, j], k) for i, j in pairs])
            rec[f"match_top{k}"] = np.nanmean([topov(u[:, i], s[:, i], k) for i in exps])
        rows.append(rec)
    return pd.DataFrame(rows)

def report(R, title, ks, N):
    print(f"\n{'='*80}\n{title}   (drug-level, n={len(R)} drugs)\n{'='*80}")
    print(f"{'metric':<10}{'baseline':>10}{'permuted':>10}{'loss':>9}{'D':>7}{'retain':>8}"
          f"{'W':>6}{'p':>9}{'lose':>7}{'matched':>9}")
    for key, chance in [("rho", 0.0)] + [(f"top{k}", k / N) for k in ks]:
        b, p, m = R[f"base_{key}"], R[f"perm_{key}"], R[f"match_{key}"]
        d = b - p
        D = (b.mean() - p.mean()) / (b.mean() - chance)
        try:
            W, pv = wilcoxon(b, p); Ws, ps = f"{W:.1f}", f"{pv:.5f}"
        except Exception:
            Ws, ps = "n/a", "n/a"
        print(f"{key:<10}{b.mean():>10.4f}{p.mean():>10.4f}{d.mean():>+9.4f}{D:>7.2f}"
              f"{100*p.mean()/b.mean():>7.1f}%{Ws:>6}{ps:>9}{(d>0).sum():>5}/{len(d)}{m.mean():>9.4f}")

# ---------------- assemblies: one alignment (structural) ----------------
a = pd.read_csv(f"{I}/assembly_importance.csv")
a = a[a.drug.isin(DRUG_IDS)]
MU, MS = {}, {}
for drug, g in a.groupby("drug"):
    U = g[g.arm == "unshuffled"].pivot_table(index="Term", columns="experiment", values="SI")
    S = g[g.arm == "shuffled"].pivot_table(index="Term", columns="experiment", values="SI")
    S = S.reindex(U.index)
    MU[drug], MS[drug] = U.to_numpy(), S.to_numpy()
RA = analyze(MU, MS, None, [5, 10, 25], 131)
RA.to_csv(f"{I}/ranking_assembly_by_drug.csv", index=False)
report(RA, "ASSEMBLY  SI  (131 NeST assemblies)", [5, 10, 25], 131)

# ---------------- genes: slot-aligned AND identity-aligned ----------------
gdf = pd.read_csv(f"{I}/gene_importance.csv")
gdf = gdf[gdf.drug.isin(DRUG_IDS)]
for mode in ["slot", "identity"]:
    MU, MS = {}, {}
    for drug, g in gdf.groupby("drug"):
        U = g[g.arm == "unshuffled"].pivot_table(index="Gene", columns="experiment", values="Rho").reindex(GENE_ORDER)
        S = g[g.arm == "shuffled"].pivot_table(index="Gene", columns="experiment", values="Rho").reindex(GENE_ORDER)
        u, s = U.to_numpy(), S.to_numpy()
        if mode == "identity":
            s2 = np.full_like(s, np.nan)
            for c, exp in enumerate(S.columns):
                s2[pi_for(int(exp)), c] = s[:, c]      # slot j's score belongs to gene pi[j]
            s = s2
        MU[drug], MS[drug] = u, s
    RG = analyze(MU, MS, None, [10, 25, 50], NG)
    RG.to_csv(f"{I}/ranking_gene_by_drug_{mode}.csv", index=False)
    report(RG, f"GENE  Rho  (689 genes) -- {mode.upper()}-ALIGNED", [10, 25, 50], NG)
print("\nD = destruction index = (baseline - permuted)/(baseline - chance); 1.0 = agreement falls to chance")
