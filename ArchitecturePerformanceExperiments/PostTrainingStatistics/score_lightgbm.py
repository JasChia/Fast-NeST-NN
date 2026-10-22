"""Recompute Spearman + RMSE (and verify Pearson/R2) for the BEST saved booster of
every (drug, experiment) LightGBM run. No retraining.

The tuner (`lightgbm_hparam_tuner.py`) selects the best trial by VALIDATION R2
(`if final_val_r2 > self.best:` at line 167) and, on each improvement, saves
`model_best.txt` with `num_iteration=best_iter` into the trial dir and copytree's
that dir to `best_model/`.  So `best_model/model_best.txt` is already truncated to
the winning iteration and `booster.predict(X)` with no `num_iteration` reproduces
the run's own numbers.  This script asserts that (`--check-numiter`).

Self-verifying: every run's recomputed test R2 / Pearson / Spearman is compared
against `final_results.json` (full precision R2) and `best_model/metrics.csv`
(4 dp), and the recomputed test predictions are cross-checked against the stored
`test_predictions.txt` (%.4e, ~5 s.f. -- a cross-check, never the source of truth).

Usage:
  python score_lightgbm.py --results-root <.../LightGBM> --code-dir <.../LightGBM> \
      --data-root <.../CombatLog2TPM> --out <.../LightGBM_scores.csv> [--limit N]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

METHOD = "LightGBM"

FIELDS = ["method","drug","experiment","seed","best_trial","activation","dropout",
          "n_val","val_pearson","val_spearman","val_r2","val_mse","val_rmse",
          "n_test","test_pearson","test_spearman","test_r2","test_mse","test_rmse",
          "stored_val_r2","stored_test_r2","d_val_r2","d_test_r2",
          "stored_test_pearson","stored_test_spearman","d_test_pearson","d_test_spearman",
          "verify_ok","note"]


def metrics(pred, y) -> dict:
    """Canonical metrics in float64."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    resid = p - t
    mse = float(np.mean(resid ** 2))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return {"pearson": float(stats.pearsonr(p, t)[0]),
            "spearman": float(stats.spearmanr(p, t)[0]),
            "r2": float(1.0 - ss_res / ss_tot),
            "mse": mse, "rmse": float(math.sqrt(mse)), "n": int(t.size)}


class DataCache:
    """Per-drug cell2id + gene expression matrix, loaded once."""
    def __init__(self, data_root: Path, T):
        self.root, self.T, self._c = Path(data_root), T, {}

    def get(self, drug):
        if drug not in self._c:
            cl = self.root / f"Drug{drug}/D{drug}_CL"
            cell2id = self.T.load_mapping(str(cl / f"D{drug}_cell2ind.txt"), "cell lines")
            ge = pd.read_csv(cl / f"D{drug}_GE_Data.txt", sep=",", header=None).values
            self._c[drug] = (cl, cell2id, ge)
        return self._c[drug]


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_trial_number(rdir: Path, bdir: Path):
    """Recover the winning trial number.

    `best_model/` carries no trial log, but `copy_best_model` copytree's the whole
    winning trial dir, so `best_model/model_best.txt` is byte-identical to exactly
    one `trials/trial_<N>/model_best.txt`.  Returns (number, note).
    """
    src = bdir / "model_best.txt"
    if not src.exists():
        return None, "no best_model/model_best.txt"
    want = _md5(src)
    hits = []
    for td in (rdir / "trials").glob("trial_*"):
        cand = td / "model_best.txt"
        if cand.exists() and cand.stat().st_size == src.stat().st_size and _md5(cand) == want:
            hits.append(int(td.name.split("_")[1]))
    if len(hits) == 1:
        return hits[0], ""
    if not hits:
        return None, "best_trial unmatched"
    return max(hits), f"best_trial ambiguous{sorted(hits)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="dir containing results/D*/")
    ap.add_argument("--code-dir", required=True, help="dir with lightgbm_hparam_tuner.py")
    ap.add_argument("--data-root", required=True, help=".../CombatLog2TPM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.code_dir).resolve()))
    T = __import__("lightgbm_hparam_tuner")
    print(f"[{METHOD}] lightgbm={lgb.__version__} loaders from {T.__file__}", flush=True)

    cache = DataCache(Path(a.data_root), T)

    runs = []
    for ddir in sorted((Path(a.results_root) / "results").glob("D*")):
        if not ddir.is_dir() or not ddir.name[1:].isdigit():
            continue
        drug = int(ddir.name[1:])
        for rdir in sorted(ddir.glob(f"{ddir.name}_*"), key=lambda p: int(p.name.split("_")[1])):
            runs.append((drug, int(rdir.name.split("_")[1]), rdir))
    runs.sort(key=lambda r: (r[0], r[1]))
    if a.limit:
        runs = runs[: a.limit]
    print(f"[{METHOD}] {len(runs)} runs found", flush=True)

    rows, nok, nbad = [], 0, 0
    max_pred_disc = 0.0           # max |recomputed - stored| on test predictions
    max_pred_rel = 0.0            # max relative version of the same
    pred_disc_where = ""
    numiter_needed = 0            # runs where predict() != predict(num_iteration=best_iter)
    treecount_mismatch = 0        # runs where num_trees() != best_iteration

    for drug, exp, rdir in runs:
        row = {k: "" for k in FIELDS}
        row.update(method=METHOD, drug=drug, experiment=exp)
        notes = []
        try:
            bdir = rdir / "best_model"
            fr = json.loads((rdir / "final_results.json").read_text())
            hp = fr["best_hyperparameters"]
            row["seed"] = int(fr.get("seed", 0))
            bt, bt_note = best_trial_number(rdir, bdir)
            row["best_trial"] = "" if bt is None else bt
            if bt_note:
                notes.append(bt_note)
            # activation / dropout intentionally left blank: N/A for LightGBM

            cl, cell2id, ge = cache.get(drug)
            sp = cl / f"train_test_splits/experiment_{exp}"
            _, va_f, _, va_l = T.load_train_data(str(sp / "true_training_data.txt"),
                                                 str(sp / "validation_data.txt"))
            te_f, te_l = T.load_pred_data(str(sp / "test_data.txt"), cell2id)
            X_val = T.build_input_matrix(va_f, ge)
            X_test = T.build_input_matrix(te_f, ge)
            y_val = np.asarray(va_l, dtype=float).ravel()
            y_test = np.asarray(te_l, dtype=float).ravel()

            booster = lgb.Booster(model_file=str(bdir / "model_best.txt"))
            ntrees = int(booster.num_trees())
            stored_iter = hp.get("best_iteration")
            if stored_iter is not None and int(stored_iter) != ntrees:
                treecount_mismatch += 1
                notes.append(f"num_trees={ntrees}!=best_iteration={stored_iter}")

            p_val = booster.predict(X_val)
            p_test = booster.predict(X_test)
            # evidence that the file is already truncated: forcing num_iteration changes nothing
            if stored_iter is not None:
                p_test_fix = booster.predict(X_test, num_iteration=int(stored_iter))
                dmax = float(np.max(np.abs(np.asarray(p_test) - np.asarray(p_test_fix))))
                if dmax > 0.0:
                    numiter_needed += 1
                    notes.append(f"num_iteration matters (dmax={dmax:.3e})")

            got = {"val": metrics(p_val, y_val), "test": metrics(p_test, y_test)}
            for s in ("val", "test"):
                g = got[s]
                row[f"n_{s}"] = g["n"]
                for k in ("pearson", "spearman", "r2", "mse", "rmse"):
                    row[f"{s}_{k}"] = f"{g[k]:.12g}"

            # ---------------- verification ----------------
            sv, st = fr.get("best_validation_r2"), fr.get("best_test_r2")
            row["stored_val_r2"] = "" if sv is None else f"{float(sv):.12g}"
            row["stored_test_r2"] = "" if st is None else f"{float(st):.12g}"
            dv = abs(got["val"]["r2"] - float(sv)) if sv is not None else float("nan")
            dt = abs(got["test"]["r2"] - float(st)) if st is not None else float("nan")
            row["d_val_r2"], row["d_test_r2"] = f"{dv:.3e}", f"{dt:.3e}"
            checks = [bool(dt < 1e-4)]

            mcsv = bdir / "metrics.csv"
            if mcsv.exists():
                md = pd.read_csv(mcsv).set_index("Metric")["Test"].to_dict()
                def f(v):
                    try: return float(v)
                    except Exception: return None
                pe_s = f(md.get("Pearson Correlation"))
                sp_s = f(md.get("Spearman Correlation"))
                row["stored_test_pearson"] = "" if pe_s is None else f"{pe_s:.12g}"
                row["stored_test_spearman"] = "" if sp_s is None else f"{sp_s:.12g}"
                if pe_s is not None:
                    d = abs(got["test"]["pearson"] - pe_s)
                    row["d_test_pearson"] = f"{d:.3e}"; checks.append(bool(d < 1.5e-4))
                if sp_s is not None:
                    d = abs(got["test"]["spearman"] - sp_s)
                    row["d_test_spearman"] = f"{d:.3e}"; checks.append(bool(d < 1.5e-4))
            else:
                notes.append("no metrics.csv")

            # cross-check vs stored %.4e predictions (NOT a pass/fail gate)
            tp = bdir / "test_predictions.txt"
            if tp.exists():
                stored_p = np.loadtxt(tp, dtype=np.float64).ravel()
                if stored_p.size == p_test.size:
                    d = np.abs(np.asarray(p_test, dtype=np.float64).ravel() - stored_p)
                    dm = float(d.max())
                    rel = float(np.max(d / np.maximum(np.abs(stored_p), 1e-12)))
                    if dm > max_pred_disc:
                        max_pred_disc, pred_disc_where = dm, f"D{drug}_{exp}"
                    max_pred_rel = max(max_pred_rel, rel)
                    if rel > 1e-3:
                        notes.append(f"pred mismatch rel={rel:.2e}")
                        checks.append(False)
                else:
                    notes.append("test_predictions.txt length mismatch")
                    checks.append(False)
            # labels sanity: stored true labels vs loaded y_test
            tl = bdir / "test_true_labels.txt"
            if tl.exists():
                stored_y = np.loadtxt(tl, dtype=np.float64).ravel()
                if stored_y.size != y_test.size or \
                   float(np.max(np.abs(stored_y - y_test))) > 1e-3:
                    notes.append("test labels differ from stored")
                    checks.append(False)

            ok = all(checks)
            row["verify_ok"] = int(ok)
            nok, nbad = nok + int(ok), nbad + int(not ok)
            if not ok:
                notes.insert(0, "VERIFY_FAIL")
                print(f"  !! VERIFY FAIL D{drug}_{exp} dT_r2={row['d_test_r2']} "
                      f"dPear={row.get('d_test_pearson')} dSpear={row.get('d_test_spearman')} "
                      f"{'; '.join(notes)}", flush=True)
            row["note"] = "; ".join(notes)[:300]
        except Exception as e:
            nbad += 1
            row["verify_ok"] = 0
            row["note"] = f"ERROR: {type(e).__name__}: {e}"[:300]
            print(f"  !! ERROR D{drug}_{exp}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  [{METHOD}] {len(rows)}/{len(runs)} ok={nok} bad={nbad}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    # -------- summary --------
    def col(name):
        v = [float(r[name]) for r in rows if r[name] not in ("", None)]
        return np.asarray(v, dtype=np.float64)
    ts, tr = col("test_spearman"), col("test_rmse")
    dr2 = col("d_test_r2"); dsp = col("d_test_spearman"); dpe = col("d_test_pearson")
    print(f"[{METHOD}] DONE rows={len(rows)} verified_ok={nok} failed={nbad} -> {a.out}", flush=True)
    if dr2.size:
        print(f"[{METHOD}] worst d_test_r2={dr2.max():.3e} worst d_test_pearson="
              f"{(dpe.max() if dpe.size else float('nan')):.3e} worst d_test_spearman="
              f"{(dsp.max() if dsp.size else float('nan')):.3e}", flush=True)
    print(f"[{METHOD}] max |pred - stored test_predictions| = {max_pred_disc:.3e} "
          f"(at {pred_disc_where}); max relative = {max_pred_rel:.3e}", flush=True)
    print(f"[{METHOD}] runs where num_iteration changed predictions: {numiter_needed}; "
          f"runs where num_trees() != best_iteration: {treecount_mismatch}", flush=True)
    if ts.size:
        print(f"[{METHOD}] test_spearman mean={ts.mean():.6f} sd={ts.std(ddof=1):.6f} (n={ts.size})", flush=True)
    if tr.size:
        print(f"[{METHOD}] test_rmse     mean={tr.mean():.6f} sd={tr.std(ddof=1):.6f} (n={tr.size})", flush=True)
    return 0 if nbad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
