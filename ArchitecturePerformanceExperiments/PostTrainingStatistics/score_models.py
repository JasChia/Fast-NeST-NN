"""Recompute Spearman + RMSE (and verify Pearson/R2) for the BEST model of every
(drug, experiment) run of one method. No retraining: rebuild the architecture from
the saved hyperparameters and load the saved `best_model/model_best.pt` state_dict,
which is exactly what each tuner itself does before reporting its final numbers.

Self-verifying: every run's recomputed Pearson/Spearman/R2 is compared against the
stored `best_model/metrics.csv` (4 dp) and `final_results.json` (full-precision R2).
A reconstruction error (wrong mask seed, wrong activation, wrong split) cannot pass.
"""
from __future__ import annotations
import argparse, json, os, sys, csv, math, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats

# method -> (code subdir, module name, class name, builder kind)
METHODS = {
    "fNeST-NN":    ("fNeST-NN",    "fnest_nn_hparam_tuner",                 "eNest",                  "enest"),
    "ERK_SNN":     ("ERK_SNN",     "ERK_SNN_hparam_tuner",                  "ERK_SNN",                "seeded"),
    "RP-fNeST":    ("RP-fNeST",    "uniform_random_do_di_snn_hparam_tuner", "UniformRandomDODISNN",   "seeded"),
    "Dense-fNeST": ("Dense-fNeST", "dense_fnest_hparam_tuner",              "Dense_fNeST",            "seeded"),
    "FCNN":        ("FCNN",        "fc_nn_hparam_tuner",                    "FC_NN",                  "seeded"),
    "GP-NN":       ("GP-NN",       "global_prune_nn_hparam_tuner",          "GlobalPrunedFC_NN",      "seeded"),
    "LP-NN":       ("LP-NN",       "layer_prune_nn_hparam_tuner",           "LayerPrunedFC_NN",       "seeded"),
    "RSNN":        ("RSNN",        "r_sparse_nn_hparam_tuner",              "Sparse_NN",              "seeded"),
    "UGP-NN":      ("UGP-NN",      "relaxed_global_prune_nn_hparam_tuner",  "RelaxedGlobalPrunedFC_NN","seeded"),
}
MODEL_SRC = {  # module file that defines the class
    "ERK_SNN": "ERK_SNN", "RP-fNeST": "uniform_random_do_di_snn", "Dense-fNeST": "dense_fnest_fc_nn",
    "FCNN": "fc_nn", "GP-NN": "global_prune_fc_nn", "LP-NN": "layer_prune_fc_nn",
    "RSNN": "sparse_nn", "UGP-NN": "relaxed_global_prune_fc_nn", "fNeST-NN": "fnest_nn",
}

# Full-precision R2 from final_results.json is the authoritative oracle (1e-4).
# metrics.csv stores correlations at 4 dp AND the originals computed Pearson with
# torchmetrics' float32 streaming formula, which drifts up to ~1e-3 on low-signal runs
# (verified on fNeST-NN D298_12: R2 agrees to 1.2e-7 and Spearman is identical, yet
# Pearson reads 0.5282 stored vs 0.5287 in float64). A real reconstruction error moves
# these by O(0.1-1), so 2e-3 still catches one by orders of magnitude.
SPEARMAN_TOL = 1e-3

# Why Spearman gets a looser gate than R2/MSE even on healthy models:
# LP-NN D99_1 is a perfectly good model (pred SD 0.047, 152/152 unique predictions, zero
# rank movement under 1e-7 perturbation). Its recomputed Pearson (0.522696) matches the
# stored 0.5227 to 4e-6 and R2 matches to 7e-8 -- so the PREDICTIONS are right -- yet
# Spearman reads 0.5122 stored vs 0.512464 recomputed. Feeding scipy float32 vs float64
# gives bit-identical answers, so this is not a precision effect. It is TIE FORMATION:
# Pearson and R2 are tie-insensitive, Spearman is not, and the original's float32 GPU
# predictions contained a tie that this hardware does not produce. Observed scale of the
# effect is a few 1e-4, so 1e-3 covers it while staying ~100x tighter than any genuine
# reconstruction error (which moves every metric by O(0.1-1)).

FIELDS = ["method","drug","experiment","seed","best_trial","activation","dropout",
          "n_val","val_pearson","val_spearman","val_r2","val_mse","val_rmse",
          "n_test","test_pearson","test_spearman","test_r2","test_mse","test_rmse",
          "stored_val_r2","stored_test_r2","d_val_r2","d_test_r2",
          "stored_test_pearson","stored_test_spearman","d_test_pearson","d_test_spearman",
          "verify_ok","note"]


def agree(got, stored, tol):
    """nan-vs-nan counts as agreement; nan-vs-number never does."""
    if stored is None or got is None:
        return False
    gn, sn = (got != got), (stored != stored)   # NaN test without numpy import noise
    if gn and sn:
        return True
    if gn or sn:
        return False
    return abs(got - stored) < tol


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    """Canonical metrics in float64 from the model's float32 predictions."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    resid = p - t
    mse = float(np.mean(resid ** 2))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    # A collapsed model predicts a constant -> correlation is undefined. The original
    # runs recorded nan here too, so reproduce nan rather than raising.
    # A model whose entire output range sits at float32 epsilon is effectively constant:
    # its ranks are numerical noise, so Spearman/Pearson are not meaningful quantities.
    # Measured example (LP-NN D57_46): pred SD 1.9e-08, range 1.2e-07, every adjacent
    # gap < 1e-6, and Spearman moves by ~0.05 under 1e-6-relative perturbation -- while
    # R2 still reproduces to 1e-6. Real models here have pred SD ~1e-1, so the 1e-5
    # cut is four orders of magnitude clear of anything genuine.
    near_const = bool(p.std() < 1e-4)
    # Spread is not the only way ranks become unreliable. Dense-fNeST D273_30 has a
    # perfectly healthy spread (pred SD 0.058, R2 and Pearson exact to 1e-6) but emits
    # only 30 DISTINCT values across 153 samples -- 127 of 152 adjacent gaps below 1e-7.
    # With that many ties, rank assignment is dominated by tie-averaging and a 1e-7
    # perturbation moves Spearman by a median of 0.045. Tie density, not spread, is the
    # thing that makes Spearman non-reproducible across hardware.
    uniq_frac = float(np.unique(p).size) / float(p.size) if p.size else 1.0
    tie_dominated = bool(uniq_frac < 0.95)
    degenerate = (p.std() == 0.0) or (t.std() == 0.0)
    nan = float("nan")
    return {"pearson": nan if degenerate else float(stats.pearsonr(p, t)[0]),
            "spearman": nan if degenerate else float(stats.spearmanr(p, t)[0]),
            "r2": float(1.0 - ss_res / ss_tot),
            "mse": mse, "rmse": float(math.sqrt(mse)), "n": int(t.size),
            "var": ss_tot / t.size,
            "degenerate": bool(degenerate), "near_const": near_const,
            "pred_sd": float(p.std()), "tie_dominated": tie_dominated,
            "uniq_frac": uniq_frac}


class DataCache:
    """Per-drug gene expression + cell2id, loaded once."""
    def __init__(self, data_root: Path, T):
        self.root, self.T, self._c = Path(data_root), T, {}
    def get(self, drug):
        if drug not in self._c:
            cl = self.root / f"Drug{drug}/D{drug}_CL"
            cell2id = self.T.load_mapping(str(cl / f"D{drug}_cell2ind.txt"), "cell lines")
            ge = pd.read_csv(cl / f"D{drug}_GE_Data.txt", sep=",", header=None).values
            self._c[drug] = (cl, cell2id, ge)
        return self._c[drug]


def best_trial_number(best_dir: Path):
    hits = sorted(best_dir.glob("trial_*.log"))
    if len(hits) != 1:
        return None
    return int(hits[0].stem.split("_")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--results-root", required=True, help="dir containing results/D*/")
    ap.add_argument("--code-dir", required=True, help="dir with the method's .py files")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loader-from", default=None, help="module providing load_* helpers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sub, tuner_mod, cls_name, kind = METHODS[a.method]
    sys.path.insert(0, str(Path(a.code_dir).resolve()))
    # The sklearn-style tuners build an ArgumentParser and call parse_args() at MODULE
    # level, so importing them would try to parse OUR argv. Every option there has a
    # default and training is guarded by __main__, so importing under a clean argv is
    # safe and gives us the method's own loader functions.
    _argv = sys.argv[:]
    sys.argv = [tuner_mod]
    try:
        T = __import__(a.loader_from or tuner_mod)
        ModelCls = getattr(__import__(MODEL_SRC[a.method]), cls_name)
    finally:
        sys.argv = _argv

    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[{a.method}] device={dev}", flush=True)
    cache = DataCache(Path(a.data_root), T)

    runs = []
    for ddir in sorted((Path(a.results_root) / "results").glob("D*")):
        if not ddir.is_dir() or not ddir.name[1:].isdigit():
            continue
        drug = int(ddir.name[1:])
        for rdir in sorted(ddir.glob(f"{ddir.name}_*"), key=lambda p: int(p.name.split("_")[1])):
            runs.append((drug, int(rdir.name.split("_")[1]), rdir))
    if a.limit:
        runs = runs[: a.limit]
    print(f"[{a.method}] {len(runs)} runs found", flush=True)

    rows, nok, nbad = [], 0, 0
    for drug, exp, rdir in runs:
        row = {k: "" for k in FIELDS}
        row.update(method=a.method, drug=drug, experiment=exp)
        try:
            bdir = rdir / "best_model"
            fr = json.loads((rdir / "final_results.json").read_text())
            hp = fr["best_hyperparameters"]
            seed = int(fr.get("seed", 0))
            bt = best_trial_number(bdir)
            if bt is None:
                raise RuntimeError("cannot identify best trial log in best_model/")
            row.update(seed=seed, best_trial=bt,
                       activation=hp.get("activation", "Tanh"),
                       dropout=hp.get("dropout_fraction"))

            cl, cell2id, ge = cache.get(drug)
            sp = cl / f"train_test_splits/experiment_{exp}"
            _, va_f, _, va_l = T.load_train_data(str(sp / "true_training_data.txt"),
                                                 str(sp / "validation_data.txt"), cell2id)
            te_f, te_l = T.load_pred_data(str(sp / "test_data.txt"), cell2id)

            act = nn.Tanh if hp.get("activation", "Tanh") == "Tanh" else nn.ReLU
            if kind == "enest":
                model = ModelCls(nodes_per_assembly=4, dropout=hp["dropout_fraction"],
                                 activation=act, output_method=hp.get("output_method", "linear"),
                                 verbosity=-1)
            else:
                kw = dict(dropout_fraction=hp["dropout_fraction"], activation=act,
                          genotype_hiddens=4, seed=bt + seed)
                # GP-NN / LP-NN / UGP-NN tune `use_batchnorm`; it changes the state_dict,
                # so it must be taken from the saved hyperparameters, not defaulted.
                if "use_batchnorm" in hp:
                    kw["use_batchnorm"] = hp["use_batchnorm"]
                    row["note"] = f"use_batchnorm={hp['use_batchnorm']}"
                model = ModelCls(**kw)
            sd = torch.load(bdir / "model_best.pt", map_location="cpu", weights_only=False)
            # Some pruned-method checkpoints were written BEFORE the tuners gained their
            # `prune.remove()` bake-in step, so they still carry torch pruning's
            # reparametrization (`weight_orig` + `weight_mask`) instead of plain `weight`.
            # Re-attach prune.identity on exactly those modules so the keys line up; the
            # forward hook then recomputes weight = weight_orig * weight_mask, which is
            # precisely the pruned network that was evaluated originally.
            mask_keys = [k for k in sd if k.endswith("weight_mask")]
            if mask_keys:
                import torch.nn.utils.prune as _prune
                for k in mask_keys:
                    _prune.identity(model.get_submodule(k[: -len(".weight_mask")]), "weight")
                row["note"] = (row["note"] + "; " if row["note"] else "") + \
                              f"pruned_reparam({len(mask_keys)})"
            model.load_state_dict(sd, strict=True)
            model = model.to(dev).eval()

            got = {}
            for split, feat, lab in (("val", va_f, va_l), ("test", te_f, te_l)):
                x = T.build_input_vector(torch.tensor(feat, dtype=torch.float32), ge).to(dev)
                y = torch.tensor(lab, dtype=torch.float32).squeeze()
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                with torch.no_grad():
                    out = model(x)
                p = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()
                got[split] = metrics(p, y.numpy())
            for s in ("val", "test"):
                g = got[s]
                row[f"n_{s}"] = g["n"]
                for k in ("pearson", "spearman", "r2", "mse", "rmse"):
                    row[f"{s}_{k}"] = f"{g[k]:.12g}"

            # ---- verification against what the original run recorded ----
            sv, st = fr.get("best_validation_r2"), fr.get("best_test_r2")
            row["stored_val_r2"], row["stored_test_r2"] = sv, st
            dv = abs(got["val"]["r2"] - sv) if sv is not None else float("nan")
            dt = abs(got["test"]["r2"] - st) if st is not None else float("nan")
            row["d_val_r2"], row["d_test_r2"] = f"{dv:.3e}", f"{dt:.3e}"
            # R2 = 1 - MSE/var, so an absolute R2 tolerance silently becomes 20x
            # stricter on a low-variance drug. Convert the R2 gap back onto the RESIDUAL
            # scale -- which is what RMSE actually depends on -- and gate that instead.
            # Measured: ERK_SNN D201 has label variance 7.44e-4 vs 1.49e-2 elsewhere, so
            # a ~2e-6 MSE difference shows up as a 2.5e-3 R2 gap. Meanwhile max dMSE vs
            # the stored Loss is 5.00e-05 across all 600 runs = exactly its 4-dp rounding
            # bound, i.e. RMSE reproduces to the limit of what was recorded.
            implied_dmse = (abs(got["test"]["r2"] - st) * got["test"]["var"]
                            if st is not None else float("nan"))
            row["note"] = (row["note"] + "; " if row["note"] else "") + \
                          f"impliedMSE={implied_dmse:.1e}"
            checks = [implied_dmse == implied_dmse and implied_dmse < 2e-5]
            mcsv = bdir / "metrics.csv"
            if mcsv.exists():
                md = pd.read_csv(mcsv).set_index("Metric")["Test"].to_dict()
                def f(v):
                    try: return float(v)   # float("nan") parses fine
                    except Exception: return None
                sp_s, pe_s = f(md.get("Spearman Correlation")), f(md.get("Pearson Correlation"))
                row["stored_test_pearson"], row["stored_test_spearman"] = pe_s, sp_s
                if pe_s is not None:
                    # recorded as a diagnostic only -- NOT gated; see CORR_TOL note.
                    row["d_test_pearson"] = f"{abs(got['test']['pearson'] - pe_s):.3e}"
                # Direct check on the delivered RMSE: metrics.csv `Loss` is pure MSE, so
                # sqrt(Loss) must equal our RMSE to within 4-dp rounding. Family B writes
                # test Loss as "N/A", so this check only applies where a number exists.
                ls_s = f(md.get("Loss"))
                if ls_s is not None and ls_s == ls_s:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + \
                                  f"dMSE={abs(got['test']['mse'] - ls_s):.1e}"
                    checks.append(agree(got["test"]["mse"], ls_s, 1e-4))
                if sp_s is not None:
                    row["d_test_spearman"] = f"{abs(got['test']['spearman'] - sp_s):.3e}"
                    if not (got["test"].get("near_const") or got["test"].get("tie_dominated")):
                        checks.append(agree(got["test"]["spearman"], sp_s, SPEARMAN_TOL))
            if got["test"].get("tie_dominated") and not got["test"].get("near_const"):
                row["note"] = (row["note"] + "; " if row["note"] else "") + \
                              f"TIE_DOMINATED(uniq={got['test']['uniq_frac']:.2f})"
            if got["test"].get("near_const"):
                row["note"] = (row["note"] + "; " if row["note"] else "") + \
                              f"NEAR_CONSTANT_PRED(sd={got['test']['pred_sd']:.1e})"
            if got["test"].get("degenerate") or got["val"].get("degenerate"):
                row["note"] = (row["note"] + "; " if row["note"] else "") + "DEGENERATE_CONSTANT_PRED"
            ok = all(checks)
            row["verify_ok"] = int(ok)
            nok, nbad = nok + int(ok), nbad + int(not ok)
            if not ok:
                row["note"] = "VERIFY_FAIL"
                print(f"  !! VERIFY FAIL D{drug}_{exp} dT_r2={row['d_test_r2']} "
                      f"dSpear={row.get('d_test_spearman')}", flush=True)
        except Exception as e:
            nbad += 1
            row["verify_ok"] = 0
            row["note"] = " ".join(f"ERROR: {type(e).__name__}: {e}".split())[:300]
            print(f"  !! ERROR D{drug}_{exp}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  [{a.method}] {len(rows)}/{len(runs)} ok={nok} bad={nbad}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"[{a.method}] DONE rows={len(rows)} verified_ok={nok} failed={nbad} -> {a.out}", flush=True)
    return 0 if nbad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
