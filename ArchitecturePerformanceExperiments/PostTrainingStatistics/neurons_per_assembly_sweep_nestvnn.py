"""Sensitivity of NeST-VNN performance to neurons-per-assembly (1..10).

NO retuning. For each (drug, experiment) the hyperparameters are PINNED to the winning
configuration that run already found (`final_results.json["best_hyperparameters"]`), and
only `genotype_hiddens` (neurons per assembly) is varied. Changing it changes the
architecture, so these models must be TRAINED -- they cannot be re-scored from the saved
checkpoints like the accuracy work.

Fidelity: rather than re-implement training, this drives the tuner's own
`OptunaNestVNNTrainer.train_model()` through a stand-in trial object whose `suggest_*`
calls return the pinned values. The training loop, seeding, early stopping, loss and
evaluation are therefore byte-for-byte the original code path. Two surgical hooks:
  * `setup_trials` is wrapped to override `genotype_hiddens` after it runs (it hardcodes
    4 at nest_vnn_hparam_tuner.py:538).
  * `save_metrics_to_csv` is spied on to capture FULL-PRECISION metric dicts, since the
    file it writes is rounded to 4 dp.
`trial.number` is set to the ORIGINAL winning trial number so that
`trial_seed = trial.number + args.seed` reproduces the original seed exactly. k=4
therefore reproduces the stored run and acts as a built-in control.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time, types
from pathlib import Path
import numpy as np
from scipy import stats

# Resolve inside this clone: this file sits in
# <repo>/ArchitecturePerformanceExperiments/PostTrainingStatistics/
APE = Path(__file__).resolve().parent.parent
MDIR = APE / "NeST-VNN"
sys.path.insert(0, str(APE))
from data_paths import data_root_from, nest_combat_root  # noqa: E402
DATA = data_root_from(__file__)   # <repo>/Data, or $FAST_NEST_DATA_ROOT
CL = nest_combat_root(__file__)
sys.path.insert(0, str(MDIR)); sys.path.insert(0, str(MDIR / "src"))

FIELDS = ["drug","experiment","k","k_used","n_params","orig_best_trial","seed","trial_seed","status",
          "n_val","val_pearson","val_spearman","val_r2","val_mse","val_rmse",
          "n_test","test_pearson","test_spearman","test_r2","test_mse","test_rmse",
          "epochs_ran","train_seconds","stored_test_r2_k4","d_vs_stored_k4","note"]


class PinnedTrial:
    """Stand-in for an optuna Trial that replays one fixed configuration."""
    def __init__(self, number: int, params: dict):
        self.number = int(number)
        self.params = dict(params)
        self.user_attrs = {}
        self._missing = []
        self.last_epoch = -1      # train_model calls report(value, epoch) every epoch
    def _get(self, name, default=None):
        if name in self.params:
            return self.params[name]
        self._missing.append(name)
        return default
    def suggest_categorical(self, name, choices):
        v = self._get(name, choices[0])
        return v
    def suggest_float(self, name, low, high, **kw):
        return float(self._get(name, low))
    def suggest_int(self, name, low, high, **kw):
        return int(self._get(name, low))
    def report(self, value, step=None, *a, **kw):
        if step is not None:
            self.last_epoch = max(self.last_epoch, int(step))
        return None
    def should_prune(self):
        return False          # never prune: we are measuring, not searching
    def set_user_attr(self, k, v):
        self.user_attrs[k] = v


def metrics_from(md, n):
    """Full-precision metrics out of the tuner's own metric dict."""
    p = np.asarray(md.get("predictions"), dtype=np.float64).ravel()
    t = np.asarray(md.get("true_labels"), dtype=np.float64).ravel()
    if p.size == 0 or p.size != t.size:
        return None
    resid = p - t
    mse = float(np.mean(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    degen = (p.std() == 0.0) or (t.std() == 0.0)
    nan = float("nan")
    return {"n": int(t.size),
            "pearson": nan if degen else float(stats.pearsonr(p, t)[0]),
            "spearman": nan if degen else float(stats.spearmanr(p, t)[0]),
            "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else nan,
            "mse": mse, "rmse": float(math.sqrt(mse)), "degen": bool(degen)}


def build_args(drug, exp, outdir, max_epochs, cuda):
    d = CL / f"Drug{drug}/D{drug}_CL"
    sp = d / f"train_test_splits/experiment_{exp}"
    fr = json.loads((MDIR / f"results/D{drug}/D{drug}_{exp}/final_results.json").read_text())
    return argparse.Namespace(
        cuda=cuda, drug=drug, n_trials=1, max_epochs=max_epochs,
        seed=int(fr.get("seed", 0)), output_dir=str(outdir),
        train_file=str(sp / "true_training_data.txt"),
        val_file=str(sp / "validation_data.txt"),
        test_file=str(sp / "test_data.txt"),
        onto=str(DATA / "red_ontology.txt"),
        cell2id=str(d / f"D{drug}_cell2ind.txt"),
        gene2id=str(DATA / "red_gene2ind.txt"),
        transcriptomic=str(d / f"D{drug}_GE_Data.txt"),
        zscore_method="auc",
    ), fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", type=int, required=True)
    ap.add_argument("--experiments", required=True, help="e.g. 0-49")
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/sens_work")
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.experiments.split("-")) if "-" in a.experiments \
             else (int(a.experiments), int(a.experiments))

    import nest_vnn_hparam_tuner as T

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader(); fh.flush()

    for exp in range(lo, hi + 1):
        rundir = Path(a.scratch) / f"D{a.drug}_{exp}"
        try:
            args, fr = build_args(a.drug, exp, rundir, a.max_epochs, a.cuda)
        except Exception as e:
            print(f"!! D{a.drug}_{exp} setup failed: {e}", flush=True); continue
        hp = fr["best_hyperparameters"]
        bdir = MDIR / f"results/D{a.drug}/D{a.drug}_{exp}/best_model"
        logs = sorted(bdir.glob("trial_*.log"))
        bt = int(logs[0].stem.split("_")[1]) if len(logs) == 1 else 0
        stored_k4 = fr.get("best_test_r2")

        trainer = None
        for k in range(a.kmin, a.kmax + 1):
            row = {f: "" for f in FIELDS}
            row.update(drug=a.drug, experiment=exp, k=k, orig_best_trial=bt,
                       seed=args.seed, trial_seed=bt + args.seed,
                       stored_test_r2_k4=stored_k4)
            t0 = time.time()
            try:
                if trainer is None:                      # data loads once per experiment
                    trainer = T.OptunaNestVNNTrainer(args)
                trainer.best, trainer.best_test = float("-inf"), 0.0
                trainer.save_dir = rundir / f"k{k}"
                trainer.save_dir.mkdir(parents=True, exist_ok=True)
                trainer.best_model_path = trainer.save_dir / "best_model"

                orig_setup = T.OptunaNestVNNTrainer.setup_trials
                def setup(self, trial, _k=k, _o=orig_setup):
                    _o(self, trial)
                    self.genotype_hiddens = _k      # the ONLY thing that varies
                trainer.setup_trials = types.MethodType(setup, trainer)

                cap = {}
                orig_save = T.save_metrics_to_csv
                def spy(vm, tm, sd, _c=cap, _o=orig_save):
                    _c["val"], _c["test"] = vm, tm
                    return _o(vm, tm, sd)
                T.save_metrics_to_csv = spy
                # Hard evidence that k actually reached the architecture. The tuner's own
                # log line prints h=4 because setup_trials prints BEFORE our override
                # runs, so the log alone cannot be trusted -- record the real value the
                # model was built with, and its parameter count.
                orig_cls = T.DrugCellNN
                def spy_cls(w, _o=orig_cls, _c=cap):
                    m = _o(w)
                    _c["k_used"] = int(w.num_hiddens_genotype)
                    _c["n_params"] = int(sum(q.numel() for q in m.parameters()))
                    return m
                T.DrugCellNN = spy_cls
                try:
                    trial = PinnedTrial(bt, hp)
                    trainer.train_model(trial)
                finally:
                    T.save_metrics_to_csv = orig_save
                    T.DrugCellNN = orig_cls

                if not cap:
                    raise RuntimeError("training produced no metrics")
                row["k_used"] = cap.get("k_used", "")
                row["n_params"] = cap.get("n_params", "")
                if cap.get("k_used") != k:
                    raise RuntimeError(f"k override failed: built with {cap.get('k_used')}, wanted {k}")
                # Direct architectural proof that k actually took effect. `setup_trials`
                # hardcodes 4 and prints "h=4" BEFORE our override runs, so the trial log
                # is misleading; the state_dict is not. DrugCellNN builds
                # final_aux_linear_layer as nn.Linear(num_hiddens_genotype, 1).
                import torch as _t
                ck = trainer.save_dir / "trials" / f"trial_{bt}" / "model_best.pt"
                if ck.exists():
                    _sd = _t.load(ck, map_location="cpu", weights_only=False)
                    _w = _sd.get("final_aux_linear_layer.weight")
                    if _w is None:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + "arch_unverified"
                    elif int(_w.shape[1]) != k:
                        raise RuntimeError(f"architecture mismatch: wanted k={k}, "
                                           f"checkpoint has {int(_w.shape[1])}")
                    else:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + f"arch_ok(k={k})"
                else:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + "arch_unverified"

                for tag in ("val", "test"):
                    m = metrics_from(cap[tag], None)
                    if m is None:
                        raise RuntimeError(f"{tag} metrics empty")
                    row[f"n_{tag}"] = m["n"]
                    for key in ("pearson", "spearman", "r2", "mse", "rmse"):
                        row[f"{tag}_{key}"] = f"{m[key]:.12g}"
                    if m["degen"]:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + f"DEGENERATE_{tag}"
                row["epochs_ran"] = trial.last_epoch + 1 if trial.last_epoch >= 0 else ""
                if trial._missing:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + \
                                  "defaulted:" + ",".join(sorted(set(trial._missing)))
                if k == 4 and stored_k4 is not None:
                    row["d_vs_stored_k4"] = f"{abs(float(row['test_r2']) - float(stored_k4)):.3e}"
                row["status"] = "ok"
            except Exception as e:
                row["status"] = "ERROR"
                row["note"] = " ".join(f"{type(e).__name__}: {e}".split())[:300]
                print(f"!! D{a.drug}_{exp} k={k}: {row['note']}", flush=True)
            row["train_seconds"] = f"{time.time()-t0:.1f}"
            w.writerow(row); fh.flush()
            print(f"[D{a.drug}_{exp} k={k}] {row['status']} "
                  f"test_r2={row.get('test_r2','')} test_spearman={row.get('test_spearman','')} "
                  f"{row['train_seconds']}s", flush=True)
            import shutil
            shutil.rmtree(rundir / f"k{k}", ignore_errors=True)   # keep inode use flat
    fh.close()
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
