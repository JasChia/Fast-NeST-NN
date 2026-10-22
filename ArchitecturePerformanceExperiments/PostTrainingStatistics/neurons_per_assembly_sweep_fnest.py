"""Sensitivity of fNeST-NN performance to neurons-per-assembly (1..10).

Same design as `sens_neurons.py` (NeST-VNN), so the two sweeps are directly comparable:
NO retuning -- hyperparameters are PINNED to the winning configuration each
(drug, experiment) already found, and only `nodes_per_assembly` is varied.
Changing it changes the architecture, so these models must be TRAINED.

Fidelity: drives the tuner's own `OptunaFNeSTNNTrainer.train_model()` through a stand-in
trial whose `suggest_*` calls replay the pinned values, so the training loop, seeding,
early stopping, loss and evaluation are the original code path. Hooks:
  * `setup_trials` wrapped to override `genotype_hiddens` after it runs (it hardcodes 4
    at fnest_nn_hparam_tuner.py:250).
  * `eNest` wrapped to record the width the model was ACTUALLY built with plus its
    parameter count -- the tuner's own log prints h=4 before our override, so the log
    cannot be trusted.
  * `save_metrics_to_csv` spied for FULL-PRECISION metrics (the file it writes is 4 dp).
`trial.number` is the ORIGINAL winning trial number so `trial_seed = trial.number +
args.seed` matches; k=4 therefore reproduces the stored run and acts as a control.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time, types
from pathlib import Path
import numpy as np
from scipy import stats

# Resolve inside this clone: this file sits in
# <repo>/ArchitecturePerformanceExperiments/PostTrainingStatistics/
APE = Path(__file__).resolve().parent.parent
MDIR = APE / "fNeST-NN"
sys.path.insert(0, str(APE))
from data_paths import data_root_from, nest_combat_root  # noqa: E402
DATA = data_root_from(__file__)   # <repo>/Data, or $FAST_NEST_DATA_ROOT
CL = nest_combat_root(__file__)
sys.path.insert(0, str(MDIR)); sys.path.insert(0, str(APE))

FIELDS = ["method","drug","experiment","k","k_used","n_params","orig_best_trial","seed",
          "trial_seed","status",
          "n_val","val_pearson","val_spearman","val_r2","val_mse","val_rmse",
          "n_test","test_pearson","test_spearman","test_r2","test_mse","test_rmse",
          "train_seconds","stored_test_r2_k4","d_vs_stored_k4","note"]


class PinnedTrial:
    """Stand-in optuna Trial replaying one fixed configuration."""
    def __init__(self, number, params):
        self.number = int(number); self.params = dict(params)
        self.user_attrs = {}; self._missing = []
    def _get(self, name, default):
        if name in self.params:
            return self.params[name]
        self._missing.append(name); return default
    def suggest_categorical(self, name, choices): return self._get(name, choices[0])
    def suggest_float(self, name, low, high, **kw): return float(self._get(name, low))
    def suggest_int(self, name, low, high, **kw): return int(self._get(name, low))
    def report(self, *a, **kw): return None
    def should_prune(self): return False          # measuring, not searching
    def set_user_attr(self, k, v): self.user_attrs[k] = v


def metrics_from(md):
    p = np.asarray(md.get("predictions"), dtype=np.float64).ravel()
    t = np.asarray(md.get("true_labels"), dtype=np.float64).ravel()
    if p.size == 0 or p.size != t.size:
        return None
    resid = p - t
    mse = float(np.mean(resid ** 2)); ss_tot = float(np.sum((t - t.mean()) ** 2))
    degen = (p.std() == 0.0) or (t.std() == 0.0); nan = float("nan")
    return {"n": int(t.size),
            "pearson": nan if degen else float(stats.pearsonr(p, t)[0]),
            "spearman": nan if degen else float(stats.spearmanr(p, t)[0]),
            "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else nan,
            "mse": mse, "rmse": float(math.sqrt(mse)), "degen": bool(degen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", type=int, required=True)
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/sens_fnest")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.experiments.split("-")) if "-" in a.experiments \
             else (int(a.experiments), int(a.experiments))

    import fnest_nn_hparam_tuner as T

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader(); fh.flush()

    for exp in range(lo, hi + 1):
        run = MDIR / f"results/D{a.drug}/D{a.drug}_{exp}"
        try:
            fr = json.loads((run / "final_results.json").read_text())
        except Exception as e:
            print(f"!! D{a.drug}_{exp}: {e}", flush=True); continue
        hp = fr["best_hyperparameters"]; seed = int(fr.get("seed", 0))
        logs = sorted((run / "best_model").glob("trial_*.log"))
        bt = int(logs[0].stem.split("_")[1]) if len(logs) == 1 else 0
        stored = fr.get("best_test_r2")
        d = CL / f"Drug{a.drug}/D{a.drug}_CL"; sp = d / f"train_test_splits/experiment_{exp}"
        rundir = Path(a.scratch) / f"D{a.drug}_{exp}"

        args = argparse.Namespace(
            cuda=a.cuda, drug=a.drug, n_trials=1, max_epochs=a.max_epochs, seed=seed,
            output_dir=str(rundir),
            train_file=str(sp / "true_training_data.txt"),
            val_file=str(sp / "validation_data.txt"),
            test_file=str(sp / "test_data.txt"),
            cell2id=str(d / f"D{a.drug}_cell2ind.txt"),
            ge_data=str(d / f"D{a.drug}_GE_Data.txt"))

        trainer = None
        for k in range(a.kmin, a.kmax + 1):
            row = {f: "" for f in FIELDS}
            row.update(method="fNeST-NN", drug=a.drug, experiment=exp, k=k,
                       orig_best_trial=bt, seed=seed, trial_seed=bt + seed,
                       stored_test_r2_k4=stored)
            t0 = time.time()
            try:
                if trainer is None:
                    trainer = T.OptunaFNeSTNNTrainer(args)
                trainer.best, trainer.best_test = float("-inf"), 0.0
                trainer.save_dir = rundir / f"k{k}"
                trainer.save_dir.mkdir(parents=True, exist_ok=True)
                trainer.best_model_path = trainer.save_dir / "best_model"

                orig_setup = T.OptunaFNeSTNNTrainer.setup_trials
                def setup(self, trial, _k=k, _o=orig_setup):
                    _o(self, trial)
                    self.genotype_hiddens = _k        # the ONLY thing that varies
                trainer.setup_trials = types.MethodType(setup, trainer)

                cap = {}
                orig_save, orig_enest = T.save_metrics_to_csv, T.eNest
                def spy_save(vm, tm, sd, _c=cap, _o=orig_save):
                    _c["val"], _c["test"] = vm, tm
                    return _o(vm, tm, sd)
                def spy_enest(*ar, _c=cap, _o=orig_enest, **kw):
                    m = _o(*ar, **kw)
                    _c["k_used"] = int(kw.get("nodes_per_assembly", ar[0] if ar else -1))
                    _c["n_params"] = int(sum(q.numel() for q in m.parameters()))
                    return m
                T.save_metrics_to_csv, T.eNest = spy_save, spy_enest
                try:
                    trial = PinnedTrial(bt, hp)
                    trainer.train_model(trial)
                finally:
                    T.save_metrics_to_csv, T.eNest = orig_save, orig_enest

                if "test" not in cap:
                    raise RuntimeError("training produced no metrics")
                row["k_used"] = cap.get("k_used", ""); row["n_params"] = cap.get("n_params", "")
                if cap.get("k_used") != k:
                    raise RuntimeError(f"k override failed: built with {cap.get('k_used')}, wanted {k}")
                for tag in ("val", "test"):
                    m = metrics_from(cap[tag])
                    if m is None:
                        raise RuntimeError(f"{tag} metrics empty")
                    row[f"n_{tag}"] = m["n"]
                    for key in ("pearson", "spearman", "r2", "mse", "rmse"):
                        row[f"{tag}_{key}"] = f"{m[key]:.12g}"
                    if m["degen"]:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + f"DEGENERATE_{tag}"
                if trial._missing:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + \
                                  "defaulted:" + ",".join(sorted(set(trial._missing)))
                if k == 4 and stored is not None:
                    row["d_vs_stored_k4"] = f"{abs(float(row['test_r2']) - float(stored)):.6f}"
                row["status"] = "ok"
            except Exception as e:
                row["status"] = "ERROR"
                row["note"] = " ".join(f"{type(e).__name__}: {e}".split())[:300]
                print(f"!! D{a.drug}_{exp} k={k}: {row['note']}", flush=True)
            row["train_seconds"] = f"{time.time()-t0:.1f}"
            w.writerow(row); fh.flush()
            print(f"[fNeST D{a.drug}_{exp} k={k}] {row['status']} "
                  f"test_r2={row.get('test_r2','')} test_spearman={row.get('test_spearman','')} "
                  f"{row['train_seconds']}s", flush=True)
            import shutil; shutil.rmtree(trainer.save_dir, ignore_errors=True)
    fh.close(); print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
