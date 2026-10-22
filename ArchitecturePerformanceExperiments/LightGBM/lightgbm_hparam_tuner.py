"""LightGBM Optuna training driver for the architecture performance comparison.

Protocol is deliberately identical to the neural tuners in this directory:
  * Optuna TPE, `-n_trials` trials, direction="maximize" on VALIDATION R^2.
  * MedianPruner, fed the validation l2 of each boosting round.
  * Boosting rounds capped by `-max_epochs`; the round is selected by
    early stopping on validation l2 (the analogue of checkpointing the best
    epoch on validation MSE).  On a fixed validation set R^2 = 1 - MSE/Var(y),
    so stopping on l2 and scoring on R^2 rank candidates identically.
  * The best trial is chosen by validation R^2 only.  Test is evaluated and
    recorded per trial but never enters a selection decision.
  * No refit on train+val -- the best trial's booster is kept as-is.

Data loading replicates the loaders used by the other nine methods in this
directory (raw AUC, no standardization).  Note this differs from
NeSTVNNShuffleAnalysis/nest_vnn/src/util.py, which applies a z-score step;
for the single-dataset (CTRP) data here that step is numerically the identity,
but it is a different code path and is not used.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from lightgbm_model import suggest_params


# ---------------------- data loading (matches the peer methods) ---------------------- #
def load_mapping(mapping_file: str, mapping_type: str) -> Dict[str, int]:
    mapping = {}
    with open(mapping_file) as fh:
        for line in fh:
            parts = line.rstrip().split()
            if len(parts) >= 2:
                mapping[parts[1]] = int(parts[0])
    print(f"Total number of {mapping_type} = {len(mapping)}")
    return mapping


def load_train_data(train_file: str, val_file: str) -> Tuple[List, List, List, List]:
    """Train/val files are index-encoded: column 0 = cell index, column 1 = AUC."""
    # Column 2 is the drug id (e.g. Drug_298), not a SMILES string. Only the 4-column
    # files (training_data.txt, test_data.txt) have all four; true_training_data.txt and
    # validation_data.txt hold just (cell_index, auc) and pandas pads the rest with NaN.
    train_df = pd.read_csv(train_file, sep="\t", header=None, names=["cell_line", "drug", "auc", "dataset"])
    val_df = pd.read_csv(val_file, sep="\t", header=None, names=["cell_line", "drug", "auc", "dataset"])
    train_features = [[row[0]] for row in train_df.values]
    train_labels = [[float(row[1])] for row in train_df.values]
    val_features = [[row[0]] for row in val_df.values]
    val_labels = [[float(row[1])] for row in val_df.values]
    return train_features, val_features, train_labels, val_labels


def load_pred_data(test_file: str, cell2id: Dict[str, int]) -> Tuple[List, List]:
    """Test file is name-encoded: column 0 = cell line name, column 2 = AUC."""
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=["cell_line", "drug", "auc", "dataset"])
    feature = [[cell2id[row[0]]] for row in test_df.values]
    label = [[float(row[2])] for row in test_df.values]
    return feature, label


def build_input_matrix(index_rows: List, ge_data: np.ndarray) -> np.ndarray:
    idx = np.asarray([int(r[0]) for r in index_rows], dtype=int)
    return ge_data[idx]


def evaluate_model_metrics(pred: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    loss = float(mean_squared_error(labels, pred))
    return {
        "loss": loss,
        "rmse": loss ** 0.5,
        "pearson_corr": float(stats.pearsonr(pred, labels)[0]),
        "spearman_corr": float(stats.spearmanr(pred, labels)[0]),
        "r2_score": float(r2_score(labels, pred)),
    }


def save_metrics_to_csv(val_metrics: Dict[str, float], test_metrics: Dict[str, float], save_dir: Path) -> None:
    metrics_data = [
        ["Pearson Correlation", f"{val_metrics['pearson_corr']:.4f}", f"{test_metrics['pearson_corr']:.4f}" if test_metrics else "N/A"],
        ["Spearman Correlation", f"{val_metrics['spearman_corr']:.4f}", f"{test_metrics['spearman_corr']:.4f}" if test_metrics else "N/A"],
        ["R² Score", f"{val_metrics['r2_score']:.4f}", f"{test_metrics['r2_score']:.4f}" if test_metrics else "N/A"],
        ["Loss", f"{val_metrics['loss']:.4f}", f"{test_metrics['loss']:.4f}" if test_metrics else "N/A"],
        ["RMSE", f"{val_metrics['rmse']:.4f}", f"{test_metrics['rmse']:.4f}" if test_metrics else "N/A"],
    ]
    df = pd.DataFrame(metrics_data, columns=["Metric", "Validation", "Test"])
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "metrics.csv", index=False)


def _pruning_callback(trial: Trial):
    """Report validation l2 each round; mirrors trial.report(val_loss, epoch)."""
    def callback(env) -> None:
        for name, metric, value, _ in env.evaluation_result_list:
            if name == "valid_0" and metric == "l2":
                # negated: direction is "maximize", so the pruner needs a value that rises as loss falls
                trial.report(-value, env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    callback.order = 10
    return callback


class OptunaLightGBMTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.save_dir = Path(args.output_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.save_dir / "best_model"
        self.best = -float("inf")
        self.best_test = None
        self.num_threads = max(1, int(args.num_threads))
        self._load_data()

    def _load_data(self) -> None:
        for required in ("cell2id", "ge_data", "train_file", "val_file", "test_file"):
            if getattr(self.args, required, None) is None:
                raise ValueError(f"-{required} is required")
        cell2id = load_mapping(self.args.cell2id, "cell lines")
        ge_data = pd.read_csv(self.args.ge_data, sep=",", header=None).values
        tr_f, va_f, tr_l, va_l = load_train_data(self.args.train_file, self.args.val_file)
        te_f, te_l = load_pred_data(self.args.test_file, cell2id)
        self.X_train = build_input_matrix(tr_f, ge_data)
        self.X_val = build_input_matrix(va_f, ge_data)
        self.X_test = build_input_matrix(te_f, ge_data)
        self.y_train = np.asarray(tr_l, dtype=float).ravel()
        self.y_val = np.asarray(va_l, dtype=float).ravel()
        self.y_test = np.asarray(te_l, dtype=float).ravel()
        for name, X, y in (("train", self.X_train, self.y_train), ("val", self.X_val, self.y_val), ("test", self.X_test, self.y_test)):
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"non-finite values in {name} fold -- check the pandas version pin (<2)")
        print(f"train={self.X_train.shape} val={self.X_val.shape} test={self.X_test.shape}")

    def train_model(self, trial: Trial) -> float:
        trial_dir = self.save_dir / "trials" / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        # Match the neural tuners: each trial gets its own seed so the search
        # explores sampling randomness as well as hyperparameters.
        trial_seed = trial.number + self.args.seed
        params = suggest_params(trial, trial_seed, self.num_threads)
        dtrain = lgb.Dataset(self.X_train, label=self.y_train, free_raw_data=False)
        dval = lgb.Dataset(self.X_val, label=self.y_val, reference=dtrain, free_raw_data=False)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.args.max_epochs,
            valid_sets=[dval],
            valid_names=["valid_0"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
                _pruning_callback(trial),
            ],
        )
        best_iter = booster.best_iteration or booster.current_iteration()
        val_pred = booster.predict(self.X_val, num_iteration=best_iter)
        test_pred = booster.predict(self.X_test, num_iteration=best_iter)
        val_metrics = evaluate_model_metrics(val_pred, self.y_val)
        test_metrics = evaluate_model_metrics(test_pred, self.y_test)
        save_metrics_to_csv(val_metrics, test_metrics, trial_dir)
        trial.set_user_attr("best_iteration", int(best_iter))
        final_val_r2 = val_metrics["r2_score"]
        final_test_r2 = test_metrics["r2_score"]
        print(f"Trial {trial.number}: rounds={best_iter} Val R²={final_val_r2:.4f} Test R²={final_test_r2:.4f}")
        if final_val_r2 > self.best:
            self.best = final_val_r2
            self.best_test = final_test_r2
            booster.save_model(str(trial_dir / "model_best.txt"), num_iteration=best_iter)
            np.savetxt(trial_dir / "test_predictions.txt", test_pred, fmt="%.4e")
            np.savetxt(trial_dir / "test_true_labels.txt", self.y_test, fmt="%.4e")
            self.copy_best_model(trial, trial_dir, self.best_model_path)
        sys.stdout.flush()
        return final_val_r2

    def exec_study(self) -> Dict:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(self.save_dir / "lightgbm_HTune.log"))
        )
        self.study = optuna.create_study(
            study_name="lightgbm",
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.args.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15, n_warmup_steps=0, interval_steps=1
            ),
            load_if_exists=True,
        )
        self.study.optimize(self.train_model, n_trials=self.args.n_trials)
        best_params = dict(self.study.best_trial.params)
        best_params.update(self.study.best_trial.user_attrs)
        return best_params

    def copy_best_model(self, trial: Trial, src: Path, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        best_params = dict(trial.params)
        best_params.update(trial.user_attrs)
        with open(dest / "hyperparameters.json", "w") as fh:
            json.dump(best_params, fh, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM (architecture comparison baseline)")
    parser.add_argument("-cuda", type=int, default=0, help="Accepted for CLI parity; LightGBM runs on CPU")
    parser.add_argument("-drug", type=int, default=-1, help="Drug ID")
    parser.add_argument("-n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("-max_epochs", type=int, default=500, help="Max boosting rounds per Optuna trial")
    parser.add_argument("-train_file", type=str, help="Training data file")
    parser.add_argument("-val_file", type=str, help="Validation data file")
    parser.add_argument("-test_file", type=str, help="Test data file")
    parser.add_argument("-cell2id", type=str, help="Cell line to ID mapping file")
    parser.add_argument("-ge_data", type=str, help="Gene expression data file")
    parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-num_threads", type=int, default=2, help="LightGBM threads")
    parser.add_argument("-output_dir", type=str, default="./results", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trainer = OptunaLightGBMTrainer(args)
    best_params = trainer.exec_study()
    final_results = {
        "best_validation_r2": trainer.best,
        "best_test_r2": trainer.best_test,
        "best_hyperparameters": best_params,
        "total_trials": len(trainer.study.trials),
        "drug_id": args.drug,
        "seed": args.seed,
    }
    with open(trainer.save_dir / "final_results.json", "w") as fh:
        json.dump(final_results, fh, indent=2, default=str)
    with open(trainer.save_dir / "best_model_results.csv", "w") as fh:
        fh.write(f"Test R2, Val R2\n{trainer.best_test}, {trainer.best}")
    print("ALL COMPLETE")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"Best Validation R²: {trainer.best:.6f}")
    print(f"Best Test R²: {trainer.best_test:.6f}")


if __name__ == "__main__":
    main()
