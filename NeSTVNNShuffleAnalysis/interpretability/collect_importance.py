"""Aggregate per-run rlipp.out / gene_rho.out into two tidy CSVs.

Emits long-format tables keyed by (drug, arm, experiment) so the unshuffled and
shuffled arms can be paired directly -- within a pair both models share the same
train/test split and differ only in gene order.
"""
import re
from pathlib import Path
import pandas as pd

I = Path(__file__).resolve().parent
RES = I / "results"
rx = re.compile(r"^D(\d+)_(unshuffled|shuffled)_exp_(\d+)$")

terms, genes, missing = [], [], []
for run in sorted(RES.iterdir()):
    m = rx.match(run.name)
    if not m:
        continue
    drug, arm, exp = int(m.group(1)), m.group(2), int(m.group(3))
    fr, fg = run / "rlipp.out", run / "gene_rho.out"
    if not (fr.exists() and fg.exists() and fr.stat().st_size and fg.stat().st_size):
        missing.append(run.name); continue
    t = pd.read_csv(fr, sep="\t"); t.insert(0, "experiment", exp)
    t.insert(0, "arm", arm); t.insert(0, "drug", drug); terms.append(t)
    g = pd.read_csv(fg, sep="\t"); g.insert(0, "experiment", exp)
    g.insert(0, "arm", arm); g.insert(0, "drug", drug); genes.append(g)

td = pd.concat(terms, ignore_index=True)
gd = pd.concat(genes, ignore_index=True)
td.rename(columns={"P_rho": "SI"}, inplace=True)        # paper's SI == P_rho
td.to_csv(I / "assembly_importance.csv", index=False)
gd.to_csv(I / "gene_importance.csv", index=False)

print(f"runs aggregated : {len(terms)}   incomplete: {len(missing)}")
if missing:
    print("  missing:", ", ".join(missing[:10]), "..." if len(missing) > 10 else "")
print(f"assembly rows   : {len(td):,}  ({td.Term.nunique()} terms)")
print(f"gene rows       : {len(gd):,}  ({gd.Gene.nunique()} genes)")
print(f"\nSI by arm:")
for arm, g in td.groupby("arm"):
    print(f"  {arm:11s} mean SI={g.SI.mean():.4f}  SI>=0.5: {100*(g.SI>=0.5).mean():5.1f}% of assembly-runs"
          f"  | RLIPP median={g.RLIPP.median():.4f}")
print(f"\ngene |Rho| by arm:")
for arm, g in gd.groupby("arm"):
    print(f"  {arm:11s} mean|Rho|={g.Rho.abs().mean():.4f}")
