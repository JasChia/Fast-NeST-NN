"""Hyperparameter tuning for fNeST-NN (fast NeST neural network, linear output head)."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import json
import shutil
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.trial import Trial, TrialState
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import pearson_corrcoef, r2_score, spearman_corrcoef

from fnest_nn import eNest

# Limit host thread usage
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

DEFAULT_PATIENCE = 20
MAX_EPOCHS = 500


# ---------------------- Utility functions ---------------------- #
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_mapping(mapping_file: str, mapping_type: str) -> Dict[str, int]:
    """Load cell line to index mapping."""
    mapping: Dict[str, int] = {}
    with open(mapping_file) as fh:
        for line in fh:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])
    print(f"Total number of {mapping_type} = {len(mapping)}")
    return mapping


def load_train_data(train_file: str, val_file: str, cell2id: Dict[str, int]) -> Tuple[List, List, List, List]:
    """Load training and validation data from individual files."""
    train_df = pd.read_csv(train_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"])
    val_df = pd.read_csv(val_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"])

    train_features = [[row[0]] for row in train_df.values]
    train_labels = [[float(row[1])] for row in train_df.values]
    val_features = [[row[0]] for row in val_df.values]
    val_labels = [[float(row[1])] for row in val_df.values]
    return train_features, val_features, train_labels, val_labels


def load_pred_data(test_file: str, cell2id: Dict[str, int]) -> Tuple[List, List]:
    """Load and preprocess test data."""
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=["cell_line", "smiles", "auc", "dataset"])
    feature = [[[cell2id[row[0]]]] for row in test_df.values]
    label = [[float(row[2])] for row in test_df.values]
    return feature, label


def build_input_vector(input_data: torch.Tensor, cell_features: np.ndarray) -> torch.Tensor:
    """Build input vector from cell line IDs by looking up gene expression data."""
    num_genes = cell_features.shape[1]
    batch_size = input_data.size(0)
    feature = np.zeros((batch_size, num_genes), dtype=np.float32)
    for i in range(batch_size):
        cell_id = int(input_data[i, 0])
        feature[i] = cell_features[cell_id]
    return torch.from_numpy(feature)


def evaluate_model_metrics(
    model: nn.Module, data: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    with torch.no_grad():
        predictions, _ = model(data.to(device))
        labels_dev = labels.to(device)
        if labels_dev.dim() == 1:
            labels_dev = labels_dev.unsqueeze(1)
        loss = nn.MSELoss()(predictions, labels_dev)
        pearson_corr = pearson_corrcoef(predictions.flatten(), labels_dev.flatten()).item()
        spearman_corr = spearman_corrcoef(predictions.flatten(), labels_dev.flatten()).item()
        r2 = r2_score(predictions, labels_dev).item()
        return {
            "predictions": predictions.cpu().numpy(),
            "true_labels": labels_dev.cpu().numpy(),
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            "r2_score": r2,
            "loss": loss.item(),
        }


def save_metrics_to_csv(val_metrics: Dict[str, float], test_metrics: Dict[str, float], save_dir: Path) -> None:
    """Save metrics to CSV in the specified format."""
    metrics_data = [
        ["Pearson Correlation", f"{val_metrics['pearson_corr']:.4f}", f"{test_metrics['pearson_corr']:.4f}" if test_metrics else "N/A"],
        ["Spearman Correlation", f"{val_metrics['spearman_corr']:.4f}", f"{test_metrics['spearman_corr']:.4f}" if test_metrics else "N/A"],
        ["R² Score", f"{val_metrics['r2_score']:.4f}", f"{test_metrics['r2_score']:.4f}" if test_metrics else "N/A"],
        ["Loss", f"{val_metrics['loss']:.4f}", f"{test_metrics['loss']:.4f}" if test_metrics else "N/A"],
    ]
    df = pd.DataFrame(metrics_data, columns=["Metric", "Validation", "Test"])
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "metrics.csv", index=False)
    print(f"Metrics saved to {save_dir}/metrics.csv")


class OptunaeNestTrainer:
    """Hyperparameter tuning trainer for fNeST-NN using Optuna."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._init_device(args.cuda)
        set_seed(args.seed)

        self.save_dir = Path(args.output_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {self.save_dir}")

        self.train_input: torch.Tensor
        self.val_input: torch.Tensor
        self.test_input: torch.Tensor
        self.train_labels: torch.Tensor
        self.val_labels: torch.Tensor
        self.test_labels: torch.Tensor
        self._load_data()

        self.patience = DEFAULT_PATIENCE
        self.best = float('-inf')
        self.best_test = 0.0
        self.best_model_path = self.save_dir / "best_model"

    @staticmethod
    def _init_device(cuda_idx: int) -> torch.device:
        """Initialize and return the computation device."""
        if torch.cuda.is_available() and cuda_idx < torch.cuda.device_count():
            torch.cuda.set_device(cuda_idx)
            device = torch.device(f"cuda:{cuda_idx}")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device

    def _load_data(self) -> None:
        """Load and preprocess all data."""
        print("=== DATA LOADING DEBUG ===")
        print("Initializing data loading...")

        required_args = ["cell2id", "ge_data", "train_file", "val_file", "test_file"]
        for arg in required_args:
            if getattr(self.args, arg) is None:
                raise ValueError(f"Required argument -{arg} not provided")

        cell2id = load_mapping(self.args.cell2id, "cell lines")
        ge_data = pd.read_csv(self.args.ge_data, sep=",", header=None).values

        train_features, val_features, train_labels, val_labels = load_train_data(
            self.args.train_file, self.args.val_file, cell2id
        )
        test_features, test_labels = load_pred_data(self.args.test_file, cell2id)

        train_data = torch.tensor(train_features, dtype=torch.float32)
        val_data = torch.tensor(val_features, dtype=torch.float32)
        test_data = torch.tensor(test_features, dtype=torch.float32)

        self.train_labels = torch.tensor(train_labels, dtype=torch.float32).squeeze()
        self.val_labels = torch.tensor(val_labels, dtype=torch.float32).squeeze()
        self.test_labels = torch.tensor(test_labels, dtype=torch.float32).squeeze()

        self.train_input = build_input_vector(train_data, ge_data)
        self.val_input = build_input_vector(val_data, ge_data)
        self.test_input = build_input_vector(test_data, ge_data)

    # ---------------------- Optuna hooks ---------------------- #
    def exec_study(self) -> Dict[str, float]:
        """Execute the Optuna hyperparameter study."""
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(self.save_dir / "fnest_nn_HTune.log"))
        )
        median_pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0, interval_steps=1)
        sampler = optuna.samplers.TPESampler(seed=self.args.seed)
        study = optuna.create_study(
            study_name="fnest_nn",
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=median_pruner,
            load_if_exists=True,
        )

        study.optimize(self.train_model, n_trials=self.args.n_trials)
        self.study = study

        # Print study results and return best parameters
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics:")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len(pruned_trials)}")
        print(f"Number of complete trials: {len(complete_trials)}")

        print("Best trial:")
        best_trial = study.best_trial
        print(f"Value: {best_trial.value}")

        best_params = {}
        print("\nBEST PARAMS:")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")
            best_params[key] = value
        for key, value in best_trial.user_attrs.items():
            print(f"{key}: {value}")
            best_params[key] = value

        return best_params

    def setup_trials(self, trial: Trial) -> None:
        """Setup hyperparameters for a trial."""
        self.wd_bool = trial.suggest_categorical("wd_bool", [True, False])
        self.wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True) if self.wd_bool else 0.0
        
        self.l1_bool = trial.suggest_categorical("l1_bool", [True, False])
        self.l1 = trial.suggest_float("l1", 1e-5, 1e-2, log=True) if self.l1_bool else 0.0

        self.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        self.dropout = trial.suggest_float("dropout_fraction", 0.0, 0.7, log=False, step=0.1)
        self.genotype_hiddens = 4
        self.batch_size_power = trial.suggest_int("batch_size_power", 2, 5, step=1)

        # Activation function selection
        activation_choice = trial.suggest_categorical("activation", ["Tanh"])
        if activation_choice == "Tanh":
            self.activation = nn.Tanh
        elif activation_choice == "ReLU":
            self.activation = nn.ReLU
        
        # NEW: Output method selection (currently only linear)
        self.output_method = trial.suggest_categorical("output_method", ["linear"])
        
        self.epochs = MAX_EPOCHS

        print(f"Trial {trial.number}: h={self.genotype_hiddens}, lr={self.lr:.2e}, drop={self.dropout:.2f}, "
              f"act={activation_choice}, batch_size={int(2 ** self.batch_size_power)}, "
              f"output={self.output_method}", end="")
        sys.stdout.flush()

    def train_model(self, trial: Trial) -> float:
        """Train a model for a single Optuna trial."""
        self.setup_trials(trial)
        
        trial_dir = self.save_dir / "trials" / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = trial_dir / f"trial_{trial.number}.log"
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout
        
        final_val_r2 = None
        final_test_r2 = None

        try:
            cl_train_data = self.train_input.to(self.device)
            cl_train_labels = self.train_labels.to(self.device)
            cl_val_data = self.val_input.to(self.device)
            cl_val_labels = self.val_labels.to(self.device)
            cl_test_data = self.test_input.to(self.device)
            cl_test_labels = self.test_labels.to(self.device)

            if cl_train_labels.dim() == 1:
                cl_train_labels = cl_train_labels.unsqueeze(1)
            if cl_val_labels.dim() == 1:
                cl_val_labels = cl_val_labels.unsqueeze(1)
            if cl_test_labels.dim() == 1:
                cl_test_labels = cl_test_labels.unsqueeze(1)

            # Set trial-specific seed for unique but deterministic randomness per trial
            trial_seed = trial.number + self.args.seed
            set_seed(trial_seed)
            
            batch_size = int(2 ** self.batch_size_power)
            dataloader = DataLoader(
                TensorDataset(cl_train_data, cl_train_labels), batch_size=batch_size, shuffle=True, drop_last=True
            )
            
            try:
                model = eNest(
                    nodes_per_assembly=self.genotype_hiddens,
                    dropout=self.dropout,
                    activation=self.activation,
                    output_method=self.output_method,
                    verbosity=-1
                ).to(self.device)
                model.register_grad_hooks()
            except RuntimeError as e:
                print(f"Warning: Trial {trial.number} failed to create eNest model.")
                print(f"  Error details: {str(e)}")
                print(f"  Skipping this trial and continuing with the next one.")
                raise e
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=self.wd)
            loss_fn = nn.MSELoss()

            min_loss: Optional[torch.Tensor] = None
            counter = 0
            val_loss = torch.tensor(float('inf'), device=self.device)

            training_start_time = time.time()
            epoch_times = []

            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                for input_data, labels in dataloader:
                    model.train()
                    optimizer.zero_grad()
                    
                    if not input_data.is_cuda:
                        input_data = input_data.to(self.device)
                    if not labels.is_cuda:
                        labels = labels.to(self.device)

                    # Model returns (final_output, hidden_asm_Y)
                    pred, _ = model(input_data)
                    
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)

                    mse_loss = loss_fn(pred, labels)
                    l1_loss = sum(p.abs().sum() for p in model.parameters())
                    
                    total_loss = mse_loss + l1_loss * self.l1

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                with torch.no_grad():
                    model.eval()
                    cl_val_pred, _ = model(cl_val_data)
                    val_r2 = r2_score(cl_val_pred, cl_val_labels).item()

                    if cl_val_labels.dim() == 1:
                        cl_val_labels = cl_val_labels.unsqueeze(1)
                    new_val_loss = loss_fn(cl_val_pred, cl_val_labels)
                    
                    trial.report(new_val_loss.item(), epoch)

                    # Save best model based on validation loss (delta threshold 1e-4).
                    # First epoch always saves; subsequent epochs save only on a strict improvement.
                    if min_loss is None or (min_loss - new_val_loss).item() > 1e-4:
                        min_loss = new_val_loss
                        torch.save(model.state_dict(), trial_dir / "model_best.pt")
                        print(f"Model saved at epoch {epoch}")

                    if (val_loss - new_val_loss).item() < 1e-4:
                        counter += 1
                        if counter >= self.patience:
                            break
                    else:
                        val_loss = new_val_loss
                        counter = 0

                    if trial.should_prune():
                        print(f"Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.exceptions.TrialPruned()
            
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                epoch_times.append(epoch_duration)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    total_time = time.time() - training_start_time
                    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

                    print(
                        f"Epoch {epoch + 1}/{self.epochs} | "
                        f"Epoch time: {epoch_duration:.2f}s | "
                        f"Avg epoch time: {avg_epoch_time:.2f}s | "
                        f"Total time: {total_time:.1f}s | "
                        f"Val R²: {val_r2:.4f} | "
                        f"Val loss: {new_val_loss.item():.6f}"
                    )
                    sys.stdout.flush()

            total_training_time = time.time() - training_start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
            print(f"\n=== Training Complete ===")
            print(f"Total epochs: {len(epoch_times)}")
            print(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
            print(f"Average time per epoch: {avg_epoch_time:.2f}s")
            if len(epoch_times) > 1:
                print(f"Min epoch time: {min(epoch_times):.2f}s")
                print(f"Max epoch time: {max(epoch_times):.2f}s")
            sys.stdout.flush()

            print("\nLoading best model from saved checkpoint...")
            model.load_state_dict(torch.load(trial_dir / "model_best.pt", map_location=self.device))

            val_metrics = evaluate_model_metrics(model, cl_val_data, cl_val_labels, self.device)
            test_metrics = evaluate_model_metrics(model, cl_test_data, cl_test_labels, self.device)
            
            save_metrics_to_csv(val_metrics, test_metrics, trial_dir)
            
            final_val_r2 = val_metrics["r2_score"]
            final_test_r2 = test_metrics["r2_score"]
            print(f"Cell Line Test -- Loss {test_metrics['loss']:.6f}, R²: {final_test_r2:.4f}")
            print(f"Cell Line Validation -- Loss {val_metrics['loss']:.6f}, R²: {final_val_r2:.4f}")
            print(f"---------- Trial {trial.number} complete after {epoch} epochs ----------")
            sys.stdout.flush()

            if final_val_r2 > self.best:
                self.best = final_val_r2
                self.best_test = final_test_r2
                self.copy_best_model(trial, final_val_r2, trial_dir, self.best_model_path)

            print(f" | Val R²: {final_val_r2:.4f}, Test R²: {final_test_r2:.4f}")

            return final_val_r2

        except optuna.exceptions.TrialPruned:
            print(" | PRUNED")
            raise
            
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def copy_best_model(self, trial: Trial, trial_value: float, src: Path, dest: Path) -> None:
        """Copy the best trial directory to the best_model location."""
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

        best_params = {}
        for key, value in trial.params.items():
            best_params[key] = value
        for key, value in trial.user_attrs.items():
            best_params[key] = value

        with open(dest / "hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=2, default=str)


# ---------------------- CLI entrypoint ---------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fNeST-NN (fast NeST-NN)")
    parser.add_argument("-cuda", type=int, default=0, help="Specify GPU index")
    parser.add_argument("-drug", type=int, default=-1, help="Drug ID")
    parser.add_argument("-n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("-train_file", type=str, help="Training data file")
    parser.add_argument("-val_file", type=str, help="Validation data file")
    parser.add_argument("-test_file", type=str, help="Test data file")
    parser.add_argument("-cell2id", type=str, help="Cell line to ID mapping file")
    parser.add_argument("-ge_data", type=str, help="Gene expression data file")
    parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-output_dir", type=str, default="./results", help="Output directory for saving models and results")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    hparam_trainer = OptunaeNestTrainer(args)
    best_params = hparam_trainer.exec_study()
    
    final_results = {
        "best_validation_r2": hparam_trainer.best,
        "best_test_r2": hparam_trainer.best_test,
        "best_hyperparameters": best_params,
        "total_trials": len(hparam_trainer.study.trials),
        "drug_id": args.drug,
        "seed": args.seed,
    }
    
    with open(hparam_trainer.save_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    with open(hparam_trainer.save_dir / "best_model_results.csv", "w") as f:
        f.write(f"Test R2, Val R2\n{hparam_trainer.best_test}, {hparam_trainer.best}")
    
    print("ALL COMPLETE")
    print(f"Results saved to: {hparam_trainer.save_dir}")
    print(f"Best Validation R²: {hparam_trainer.best:.6f}")
    print(f"Best Test R²: {hparam_trainer.best_test:.6f}")


if __name__ == "__main__":
    main()
