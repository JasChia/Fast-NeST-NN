"""Recompute Spearman + RMSE (and verify Pearson/R2) for the BEST model of every
(drug, experiment) run of NeST-VNN. No retraining.

NeST-VNN differs from the other 9 architectures: the model is built as
`DrugCellNN(model_wrapper)` where `model_wrapper` is a duck-typed stand-in for
`TrainingDataWrapper` assembled by `OptunaNestVNNTrainer._create_model_wrapper()`,
and the tuner uses its OWN `DataWrapperForHparam._load_data()` (not
`util.load_train_data` / `util.load_pred_data`) to read the three split files.
This script reuses the tuner's own classes so the reconstruction is the original
code path, not a re-implementation of it.

Procedure per run (exactly what the tuner does at the end of a trial):
  rebuild architecture -> load_state_dict(model_best.pt, strict=True) -> eval()
  -> forward over val and test -> metrics.

Self-verifying, in three layers:
  1. `load_state_dict(..., strict=True)` -- any wrong wrapper field that changes the
     parameter set (ontology, gene2ind, genotype_hiddens, feature_dim) throws.
  2. Our predictions AND labels are compared element-wise against the original run's
     OWN saved `trials/trial_<N>/{val,test}_{predictions,true_labels}.txt`.
     `best_model/` is a byte-identical copy of `trials/trial_<N>/` (md5-verified),
     so those files are the original forward pass for exactly these weights. This
     catches a wrong split, wrong standardisation or wrong forward path directly,
     and stays well conditioned even when a model collapses to a constant output.
  3. Agreement with the recorded statistics, gated on scales that are actually
     comparable across drugs (gate shared with `score_models.py`):
       * R2 -- compared on the RESIDUAL scale as impliedMSE = |dR2| * var, where
         var = SStot/n of the test labels, gated at 2e-5. A fixed absolute R2
         tolerance is ~20x stricter on D201 (test label var 7.4e-4) than on every
         other drug (~1.5e-2), which manufactures failures on that drug alone.
       * MSE -- |recomputed test MSE - stored `Loss`| < 1e-4. `Loss` is pure
         nn.MSELoss(), so this directly validates the reported RMSE.
       * Spearman -- tolerance 1e-3, and SKIPPED when the prediction is
         near-constant (sd < 1e-4). Spearman is tie-sensitive while Pearson/R2 are
         not, and float32 GPU predictions can form ties this hardware does not
         reproduce.
       * Pearson -- RECORDED but NOT gated: torchmetrics accumulates it in float32
         with a streaming formula, so where the two disagree the stored value is the
         less accurate one.

Runs whose model collapsed to an exactly constant prediction have a genuinely
undefined correlation (metrics.csv literally stores "nan"); they are reproduced as
undefined, tagged DEGENERATE_CONSTANT_PRED, and counted separately.
Layers 1 and 2 are required of every run with no exceptions.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, traceback, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_paths import data_root_from  # noqa: E402

import numpy as np
import pandas as pd
import torch
import torch.utils.data as du
from scipy import stats

# ---- `src/util.py` was written for torch<1.13 (`from torch._six import inf`).
# Shim the module so the ORIGINAL util.py imports unmodified under torch 2.x.
if "torch._six" not in sys.modules:
    _six = types.ModuleType("torch._six")
    _six.inf = float("inf")
    _six.nan = float("nan")
    _six.string_classes = (str, bytes)
    sys.modules["torch._six"] = _six

FIELDS = ["method","drug","experiment","seed","best_trial","activation","dropout",
          "n_val","val_pearson","val_spearman","val_r2","val_mse","val_rmse",
          "n_test","test_pearson","test_spearman","test_r2","test_mse","test_rmse",
          "stored_val_r2","stored_test_r2","d_val_r2","d_test_r2",
          "stored_test_pearson","stored_test_spearman","d_test_pearson","d_test_spearman",
          "verify_ok","note"]

METHOD = "NeST-VNN"


def metrics(pred, y) -> dict:
    """Canonical metrics in float64 from the model's float32 predictions."""
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


def best_trial_number(best_dir: Path):
    hits = sorted(best_dir.glob("trial_*.log"))
    if len(hits) != 1:
        return None
    return int(hits[0].stem.split("_")[1])


class Args:
    """Stand-in for the tuner's argparse.Namespace (only the fields it reads)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class DrugData:
    """One `DataWrapperForHparam` per drug (ontology + transcriptomic + cell2id are
    per-drug constants); split files are swapped per experiment by re-running the
    tuner's own `_load_data()`."""

    def __init__(self, T, drug: int, data_root: Path, onto: Path, gene2id: Path,
                 zscore_method: str, cuda_id: int, seed: int):
        cl = data_root / f"Drug{drug}" / f"D{drug}_CL"
        sp0 = cl / "train_test_splits" / "experiment_0"
        self.cl = cl
        self.args = Args(
            cuda=cuda_id, seed=seed,
            cell2id=str(cl / f"D{drug}_cell2ind.txt"),
            gene2id=str(gene2id),
            onto=str(onto),
            transcriptomic=str(cl / f"D{drug}_GE_Data.txt"),
            zscore_method=zscore_method,
            train_file=str(sp0 / "true_training_data.txt"),
            val_file=str(sp0 / "validation_data.txt"),
            test_file=str(sp0 / "test_data.txt"),
        )
        self.dw = T.DataWrapperForHparam(self.args)
        self._loaded_exp = 0

    def load_experiment(self, exp: int):
        sp = self.cl / "train_test_splits" / f"experiment_{exp}"
        for f in ("true_training_data.txt", "validation_data.txt", "test_data.txt"):
            if not (sp / f).exists():
                raise FileNotFoundError(str(sp / f))
        self.args.train_file = str(sp / "true_training_data.txt")
        self.args.val_file = str(sp / "validation_data.txt")
        self.args.test_file = str(sp / "test_data.txt")
        if exp != self._loaded_exp:
            self.dw._load_data()
            self._loaded_exp = exp
        return self.dw


def build_model(T, DrugCellNN, dw, hp, trial_seed, genotype_hiddens=4):
    """Reproduce `OptunaNestVNNTrainer._create_model_wrapper()` + `DrugCellNN(...)`.

    The tuner sets exactly these nine attributes on the duck-typed wrapper; every
    other `TrainingDataWrapper` field (lr/wd/alpha/epochs/batchsize/cuda/...) is
    never touched by `DrugCellNN.__init__`. `activation` is a tuned hyperparameter
    but DrugCellNN hard-codes `torch.tanh`, so it does not enter the architecture.
    """
    class ModelWrapper:
        pass
    w = ModelWrapper()
    w.root = dw.root
    w.num_hiddens_genotype = genotype_hiddens          # fixed at 4 in setup_trials
    w.term_direct_gene_map = dw.term_direct_gene_map
    w.min_dropout_layer = int(hp["min_dropout_layer"])  # per-trial hyperparameter
    w.dropout_fraction = float(hp["dropout_fraction"])  # per-trial hyperparameter
    w.term_size_map = dw.term_size_map
    w.gene_id_mapping = dw.gene_id_mapping
    w.cell_features = dw.cell_features
    w.dG = dw.dG
    T.set_seed(trial_seed)   # trial_seed = trial.number + args.seed (fidelity only;
                             # the state_dict overwrites every initialised weight)
    return DrugCellNN(w)


@torch.no_grad()
def predict(T, model, feature, label, cell_features, dev, batch_size=2000):
    """Mirror of the tuner's `evaluate_model_metrics` forward pass (same batch size,
    same loader, shuffle=False, `aux_out_map['final']`)."""
    loader = du.DataLoader(du.TensorDataset(feature, label),
                           batch_size=batch_size, shuffle=False)
    preds = []
    for inputdata, _labels in loader:
        feats = T.util.build_input_vector(inputdata, cell_features).to(dev)
        aux_out_map, _ = model(feats)
        preds.append(aux_out_map["final"].detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    # Defaults resolve inside this clone: <repo>/ArchitecturePerformanceExperiments/NeST-VNN
    # and <repo>/Data (or $FAST_NEST_DATA_ROOT). See ../data_paths.py.
    ape = Path(__file__).resolve().parent.parent
    data = data_root_from(__file__)
    ap = argparse.ArgumentParser(
        description="Recompute test/validation metrics for every saved NeST-VNN best model")
    ap.add_argument("--results-root", default=str(ape / "NeST-VNN"),
                    help="directory holding results/D<drug>/D<drug>_<exp>/")
    ap.add_argument("--code-dir", default=str(ape / "NeST-VNN"),
                    help="directory holding the NeST-VNN src/ package")
    ap.add_argument("--data-root", default=str(data / "nest_shuffle_data" / "CombatLog2TPM"))
    ap.add_argument("--onto", default=str(data / "red_ontology.txt"))
    ap.add_argument("--gene2id", default=str(data / "red_gene2ind.txt"))
    ap.add_argument("--zscore-method", default="auc",
                    help="tuner default; jobs never pass -zscore_method")
    ap.add_argument("--out", default="scores/NeST-VNN_scores.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drugs", default="", help="comma-separated subset, e.g. 127,151")
    ap.add_argument("--experiments", default="", help="comma-separated subset, e.g. 0,16")
    a = ap.parse_args()

    code = Path(a.code_dir).resolve()
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "src"))
    T = __import__("nest_vnn_hparam_tuner")
    from drugcell_nn import DrugCellNN

    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[{METHOD}] device={dev} torch={torch.__version__}", flush=True)

    want = {int(x) for x in a.drugs.split(",") if x.strip()} if a.drugs.strip() else None
    wexp = {int(x) for x in a.experiments.split(",") if x.strip()} if a.experiments.strip() else None
    runs = []
    for ddir in sorted((Path(a.results_root) / "results").glob("D*")):
        if not ddir.is_dir() or not ddir.name[1:].isdigit():
            continue
        drug = int(ddir.name[1:])
        if want is not None and drug not in want:
            continue
        for rdir in sorted(ddir.glob(f"{ddir.name}_*"), key=lambda p: int(p.name.split("_")[1])):
            e = int(rdir.name.split("_")[1])
            if wexp is not None and e not in wexp:
                continue
            runs.append((drug, e, rdir))
    runs.sort(key=lambda r: (r[0], r[1]))
    if a.limit:
        runs = runs[: a.limit]
    print(f"[{METHOD}] {len(runs)} runs found", flush=True)

    data_root, onto, g2i = Path(a.data_root), Path(a.onto), Path(a.gene2id)
    drug_cache: dict[int, DrugData] = {}
    rows, nok, nbad, ndegen, nnear = [], 0, 0, 0, 0

    for drug, exp, rdir in runs:
        row = {k: "" for k in FIELDS}
        row.update(method=METHOD, drug=drug, experiment=exp)
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

            if drug not in drug_cache:
                drug_cache = {}  # one drug resident at a time (transcriptomic is large)
                drug_cache[drug] = DrugData(T, drug, data_root, onto, g2i,
                                            a.zscore_method, 0, seed)
            dw = drug_cache[drug].load_experiment(exp)

            model = build_model(T, DrugCellNN, dw, hp, trial_seed=bt + seed)
            sd = torch.load(bdir / "model_best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(sd, strict=True)
            model = model.to(dev).eval()

            got, mine = {}, {}
            for split, feat, lab in (("val", dw.val_feature, dw.val_label),
                                     ("test", dw.test_feature, dw.test_label)):
                if feat.size(0) == 0:
                    raise RuntimeError(f"{split} split is empty")
                p = predict(T, model, feat, lab, dw.cell_features, dev)
                mine[split] = (np.asarray(p, dtype=np.float64).ravel(),
                               np.asarray(lab.numpy(), dtype=np.float64).ravel())
                got[split] = metrics(p, lab.numpy())
            del model

            # ---- PRIMARY check: our predictions vs the ORIGINAL RUN'S OWN saved
            # predictions. `best_model/` is a byte-identical copy of
            # `trials/trial_<N>/` (md5-verified), so that directory holds the
            # original forward pass for exactly these weights, dumped at '%.6f'.
            # This validates the wrapper fields, the split, the label
            # standardisation and the forward pass directly, and -- unlike a
            # correlation -- it stays well conditioned when a model collapses to
            # a near-constant output.
            tdir = rdir / "trials" / f"trial_{bt}"
            dpred = {}
            for split in ("val", "test"):
                pf, lf = tdir / f"{split}_predictions.txt", tdir / f"{split}_true_labels.txt"
                if not (pf.exists() and lf.exists()):
                    raise RuntimeError(f"missing original {split} predictions in {tdir}")
                op = np.atleast_1d(np.loadtxt(pf, dtype=np.float64)).ravel()
                ol = np.atleast_1d(np.loadtxt(lf, dtype=np.float64)).ravel()
                mp, ml = mine[split]
                if op.size != mp.size:
                    raise RuntimeError(f"{split}: n mismatch ours={mp.size} orig={op.size}")
                dpred[split] = (float(np.max(np.abs(mp - op))),
                                float(np.max(np.abs(ml - ol))))
            # '%.6f' text rounding alone allows 5e-7; 1e-5 is ~2 orders looser than
            # that and still ~3 orders tighter than any reconstruction error.
            PRED_TOL = 1e-5
            pred_ok = all(dp < PRED_TOL and dl < PRED_TOL for dp, dl in dpred.values())
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
            notes = []

            # (3a) R2 gate on the RESIDUAL scale, not the R2 scale.
            # R2 = 1 - MSE/var, so a fixed absolute R2 tolerance is implicitly
            # var-dependent. D201's test label variance is ~7.4e-4 against ~1.5e-2
            # for every other drug, which makes an absolute R2 gate ~20x stricter
            # on that one drug and produces failures that are purely an artefact of
            # the gate. Convert the R2 gap back to MSE before comparing.
            ty = mine["test"][1]
            var = float(np.sum((ty - ty.mean()) ** 2) / ty.size)
            implied = dt * var
            notes.append(f"impliedMSE={implied:.3e}")
            checks = [implied < 2e-5]

            mcsv = bdir / "metrics.csv"
            if mcsv.exists():
                md = pd.read_csv(mcsv).set_index("Metric")["Test"].to_dict()
                def f(v):
                    try: return float(v)
                    except Exception: return None
                pe_s = f(md.get("Pearson Correlation"))
                sp_s = f(md.get("Spearman Correlation"))
                lo_s = f(md.get("Loss"))
                row["stored_test_pearson"], row["stored_test_spearman"] = pe_s, sp_s

                # (3b) `Loss` is pure nn.MSELoss() with no regularisation, so this is
                # a DIRECT check on the recomputed RMSE (RMSE = sqrt(MSE)).
                if lo_s is not None and not math.isnan(lo_s):
                    dmse = abs(got["test"]["mse"] - lo_s)
                    notes.append(f"dMSE={dmse:.3e}")
                    checks.append(dmse < 1e-4)

                # (3c) Pearson: RECORDED but NOT gated. torchmetrics' pearson_corrcoef
                # accumulates in float32 with a streaming formula, so where the two
                # disagree the STORED value is the less accurate one; a disagreement
                # is not evidence against this reconstruction.
                if pe_s is not None:
                    rec_p = got["test"]["pearson"]
                    row["d_test_pearson"] = ("nan" if (math.isnan(pe_s) or math.isnan(rec_p))
                                             else f"{abs(rec_p - pe_s):.3e}")

                # (3d) Spearman: tolerance 1e-3, and skipped entirely when the
                # prediction is near-constant. Spearman is tie-sensitive while
                # Pearson/R2 are not, and float32 GPU predictions can form ties that
                # this hardware does not reproduce.
                sd_pred = float(mine["test"][0].std())
                near_const = sd_pred < 1e-4
                if near_const:
                    notes.append(f"NEAR_CONSTANT_PRED(sd={sd_pred:.2e})")
                    nnear += 1
                if sp_s is not None:
                    rec_s = got["test"]["spearman"]
                    if math.isnan(sp_s) and math.isnan(rec_s):
                        # The original collapsed to an exactly constant prediction, so
                        # its own scipy call returned nan and metrics.csv stores "nan".
                        # We reproduce the same degeneracy. Genuinely undefined.
                        row["d_test_spearman"] = "nan"
                        notes.append("DEGENERATE_CONSTANT_PRED")
                        ndegen += 1
                    elif math.isnan(sp_s) or math.isnan(rec_s):
                        row["d_test_spearman"] = "nan"
                        notes.append("SPEARMAN_DEFINEDNESS_MISMATCH")
                        if not near_const:
                            checks.append(False)
                    else:
                        d = abs(rec_s - sp_s)
                        row["d_test_spearman"] = f"{d:.3e}"
                        if not near_const:
                            checks.append(d < 1e-3)
            else:
                checks.append(False)
                notes.append("no metrics.csv")

            # (2) Hard requirement for EVERY run: our predictions and labels must
            # match the original run's own saved ones element-wise. This is the
            # check that actually catches a wrong wrapper field, split or forward
            # path, and unlike a correlation it stays well conditioned when a model
            # collapses to a constant output.
            notes.append(f"maxdpred={max(d[0] for d in dpred.values()):.2e}")
            notes.append(f"maxdlabel={max(d[1] for d in dpred.values()):.2e}")
            checks.append(pred_ok)
            if not pred_ok:
                notes.append("PREDICTIONS DISAGREE WITH ORIGINAL")
            row["note"] = "; ".join(notes)
            ok = all(checks)
            row["verify_ok"] = int(ok)
            nok, nbad = nok + int(ok), nbad + int(not ok)
            if not ok:
                row["note"] = (row["note"] + "; " if row["note"] else "") + "VERIFY_FAIL"
                print(f"  !! VERIFY FAIL D{drug}_{exp} :: {row['note']}", flush=True)
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
    print(f"[{METHOD}] DONE rows={len(rows)} verified_ok={nok} failed={nbad} "
          f"degenerate(constant pred, nan correlation)={ndegen} "
          f"near_constant_pred(sd<1e-4, Spearman check skipped)={nnear} "
          f"-> {a.out}", flush=True)

    good = [r for r in rows if r["verify_ok"] == 1]
    if good:
        def col(name, rs=good):
            return np.array([float(r[name]) for r in rs], dtype=np.float64)
        ts, tr = col("test_spearman"), col("test_rmse")
        fin = np.isfinite(ts)
        print(f"[{METHOD}] test Spearman over {int(fin.sum())}/{ts.size} verified runs "
              f"with a DEFINED correlation: mean={ts[fin].mean():.6f} "
              f"sd={ts[fin].std(ddof=1):.6f}")
        print(f"[{METHOD}] test RMSE over all {tr.size} verified runs: "
              f"mean={tr.mean():.6f} sd={tr.std(ddof=1):.6f}")

    def note_num(r, key):
        for part in r["note"].split("; "):
            if part.startswith(key + "="):
                try: return float(part.split("=", 1)[1])
                except ValueError: return float("nan")
        return float("nan")

    for key in ("impliedMSE", "dMSE", "maxdpred", "maxdlabel"):
        arr = np.array([note_num(r, key) for r in rows])
        arr = arr[np.isfinite(arr)]
        if arr.size:
            print(f"[{METHOD}] max {key}={arr.max():.3e} (over {arr.size} runs)")

    near = {(r["drug"], r["experiment"]) for r in rows
            if "NEAR_CONSTANT_PRED" in r["note"]}
    dsp = np.array([float(r["d_test_spearman"]) for r in rows
                    if r["d_test_spearman"] and r["d_test_spearman"] != "nan"
                    and (r["drug"], r["experiment"]) not in near])
    if dsp.size:
        print(f"[{METHOD}] max d_test_spearman={dsp.max():.3e} "
              f"(over {dsp.size} runs, excluding near-constant)")
    dvr = np.array([float(r["d_val_r2"]) for r in rows if r["d_val_r2"]])
    dvr = dvr[np.isfinite(dvr)]
    if dvr.size:
        print(f"[{METHOD}] max d_val_r2={dvr.max():.3e}")
    print(f"[{METHOD}] degenerate(constant pred, nan correlation)={ndegen}/{len(rows)}")
    print(f"[{METHOD}] near_constant_pred(sd<1e-4)={nnear}/{len(rows)}")
    return 0 if nbad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
