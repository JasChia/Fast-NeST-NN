# Hyperparameter tuning for FC_NN with Direct Input and Layer Predictions
# This model uses a fully connected neural network with concatenated inputs to each layer.
# Each hidden layer produces an intermediary prediction, and all predictions are aggregated
# through a final linear layer with sigmoid to produce a single output.
# Training uses: final_loss + L1_loss
# python fc_nn_di_layer_pred_hparam_tuner.py > fc_nn_di_layer_pred_hparam.log 2>&1 &
# nohup python -u fc_nn_di_layer_pred_hparam_tuner.py -cuda 0 -drug 188 > Shuffle_Input/CombatLog2TPM/D188/HTune_logs.txt 2>&1 &
# nohup python -u fc_nn_di_layer_pred_hparam_tuner.py -cuda 3 -drug 143 > eNest_New_CL/Log2TPM/D143/HTune_logs.txt 2>&1 &

import math
import time
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState

import torch.nn as nn
import pandas as pd
import torch
from fc_nn_di_layer_pred import FC_NN_DI_Layer_Pred
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
import sys
import numpy as np
import os
import argparse
import shutil
from copy import deepcopy
from scipy import stats

# Data preprocessing functions adapted from util.py

def load_mapping(mapping_file, mapping_type):
    """Load cell line to index mapping"""
    mapping = {}
    file_handle = open(mapping_file)
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])
    file_handle.close()
    print('Total number of {} = {}'.format(mapping_type, len(mapping)))
    return mapping

def load_train_data(train_file, val_file, cell2id):
    """Load training and validation data from individual files"""
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])
    val_df = pd.read_csv(val_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])

    train_features = []
    train_labels = []
    for row in train_df.values:
        train_features.append([row[0]]) #The difference is due to different file format
        train_labels.append([float(row[1])])  # Use raw AUC values

    val_features = []
    val_labels = []
    for row in val_df.values:
        val_features.append([row[0]]) #The difference is due to different file format
        val_labels.append([float(row[1])])  # Use raw AUC values

    return train_features, val_features, train_labels, val_labels

def load_pred_data(test_file, cell2id):
    """Load and preprocess test data"""
    test_df = pd.read_csv(test_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc', 'dataset'])

    feature = []
    label = []
    for row in test_df.values:
        feature.append([[cell2id[row[0]]]]) #Original nest format
        label.append([float(row[2])])  # Use raw AUC values
    return feature, label

def build_input_vector(input_data, cell_features):
    """Build input vector from cell line IDs by looking up gene expression data"""
    # cell_features is 2D: [num_cells, num_genes]
    num_genes = cell_features.shape[1]
    batch_size = input_data.size()[0]
    
    # Create output tensor: [batch_size, num_genes]
    feature = np.zeros((batch_size, num_genes))

    for i in range(batch_size):
        cell_id = int(input_data[i, 0])
        feature[i] = cell_features[cell_id]

    feature = torch.from_numpy(feature).float()
    return feature

def pearson_correlation(pred, true):
    """Calculate Pearson correlation coefficient"""
    pred_np = pred.cpu().numpy().flatten()
    true_np = true.cpu().numpy().flatten()
    
    # Check for NaN or infinite values
    if np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np)):
        print(f"Warning: Pearson correlation - predictions contain NaN/inf values")
        return 0.0  # Return 0 correlation for invalid predictions
    
    if np.any(np.isnan(true_np)) or np.any(np.isinf(true_np)):
        print(f"Warning: Pearson correlation - true labels contain NaN/inf values")
        return 0.0  # Return 0 correlation for invalid labels
    
    try:
        return stats.pearsonr(pred_np, true_np)[0]
    except ValueError as e:
        print(f"Warning: Pearson correlation failed: {e}")
        return 0.0

def spearman_correlation(pred, true):
    """Calculate Spearman correlation coefficient"""
    pred_np = pred.cpu().numpy().flatten()
    true_np = true.cpu().numpy().flatten()
    
    # Check for NaN or infinite values
    if np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np)):
        print(f"Warning: Spearman correlation - predictions contain NaN/inf values")
        return 0.0  # Return 0 correlation for invalid predictions
    
    if np.any(np.isnan(true_np)) or np.any(np.isinf(true_np)):
        print(f"Warning: Spearman correlation - true labels contain NaN/inf values")
        return 0.0  # Return 0 correlation for invalid labels
    
    try:
        return stats.spearmanr(pred_np, true_np)[0]
    except ValueError as e:
        print(f"Warning: Spearman correlation failed: {e}")
        return 0.0

def evaluate_model_metrics(model, data, labels, device):
    """Evaluate model and return metrics - uses final aggregated output"""
    model.eval()
    with torch.no_grad():
        final_pred = model(data.to(device))
        labels_dev = labels.to(device)
        if labels_dev.dim() == 1:
            labels_dev = labels_dev.unsqueeze(1)
        # Use final aggregated prediction for evaluation metrics
        loss = nn.MSELoss()(final_pred, labels_dev)
        
        # Calculate metrics using final prediction
        pearson_corr = pearson_correlation(final_pred, labels_dev)
        spearman_corr = spearman_correlation(final_pred, labels_dev)
        r2 = r2_score(labels_dev.cpu().numpy().flatten(), final_pred.cpu().numpy().flatten())
        
        return {
            'predictions': final_pred.cpu().numpy(),
            'true_labels': labels_dev.cpu().numpy(),
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'r2_score': r2,
            'loss': loss.item()
        }

def save_metrics_to_csv(val_metrics, test_metrics, save_dir):
    """Save metrics to CSV in the specified format"""
    metrics_data = []
    
    if val_metrics:
        metrics_data.append(['Pearson Correlation', f"{val_metrics['pearson_corr']:.4f}", 
                            f"{test_metrics['pearson_corr']:.4f}" if test_metrics else "N/A"])
        metrics_data.append(['Spearman Correlation', f"{val_metrics['spearman_corr']:.4f}", 
                            f"{test_metrics['spearman_corr']:.4f}" if test_metrics else "N/A"])
        metrics_data.append(['R² Score', f"{val_metrics['r2_score']:.4f}", 
                            f"{test_metrics['r2_score']:.4f}" if test_metrics else "N/A"])
        metrics_data.append(['Loss', f"{val_metrics['loss']:.4f}", "N/A"])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(metrics_data, columns=['Metric', 'Validation', 'Test'])
    df.to_csv(f"{save_dir}/metrics.csv", index=False)
    print(f"Metrics saved to {save_dir}/metrics.csv")

parser = argparse.ArgumentParser(description = 'Train FC_NN_DI_Layer_Pred (Fully Connected NN with Direct Input and Layer Predictions)')
parser.add_argument('-cuda', help = 'Specify GPU', type = int, default = 0)
parser.add_argument('-drug', help = 'Drug ID', type = int, default = -1)
parser.add_argument('-n_trials', help = 'Number of Optuna trials', type=int, default=100)
# Data preprocessing arguments
parser.add_argument('-train_file', help = 'Training data file', type=str)
parser.add_argument('-val_file', help = 'Validation data file', type=str)
parser.add_argument('-test_file', help = 'Test data file', type=str)
parser.add_argument('-cell2id', help = 'Cell line to ID mapping file', type=str)
parser.add_argument('-ge_data', help = 'Gene expression data file', type=str)
parser.add_argument('-seed', help = 'Random seed for reproducibility', type=int, default=42)
parser.add_argument('-output_dir', help = 'Output directory for saving models and results', type=str, default='./results')
args = parser.parse_args()

# Set random seeds for reproducibility
def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

# Set the seed
set_seed(args.seed)

#Helper functions
def get_r2_score(torch_pred, torch_labels):
	pred = torch_pred.cpu().numpy()
	labels = torch_labels.cpu().numpy()
	
	# Check for NaN or infinite values
	if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
		print(f"Warning: Predictions contain NaN or inf values. NaN count: {np.sum(np.isnan(pred))}, Inf count: {np.sum(np.isinf(pred))}")
		return -1000.0  # Return very low R² for invalid predictions
	
	if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
		print(f"Warning: Labels contain NaN or inf values. NaN count: {np.sum(np.isnan(labels))}, Inf count: {np.sum(np.isinf(labels))}")
		return -1000.0  # Return very low R² for invalid labels
	
	r2 = r2_score(labels, pred)
	return r2

def auroc_score(y_pred, y_true):
	pred = y_pred.detach().cpu().numpy()
	true = y_true.detach().cpu().numpy()
	score = roc_auc_score(true, pred)
	return score


cuda_dev_idx = args.cuda

# Add comprehensive device validation and debugging
if torch.cuda.is_available() and cuda_dev_idx < torch.cuda.device_count():
    device = f'cuda:{cuda_dev_idx}'
    torch.cuda.set_device(cuda_dev_idx)
    print(f"Using device: {device}")
else:
    device = 'cpu'
    print("Using device: cpu")

class OptunaFCNNLayerPredTrainer():

    def __init__(self):
        self.drug = args.drug
        self.save_dir = args.output_dir
        self.load_data(self.drug)
        if (os.path.exists(self.save_dir) == False):
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Created output directory: {self.save_dir}")
        self.patience = 20
        self.best = -999.99
        self.best_test = 0.0
        

    def exec_study(self):
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(f"{self.save_dir}/FC_NN_DI_Layer_Pred_HTune.log")
        )
        median_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=15, n_warmup_steps=0, interval_steps=1
        )
        study_name = f"FC_NN_DI_Layer_Pred_HTune"
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", sampler=sampler, pruner=median_pruner, load_if_exists=True)

        study.optimize(self.train_model, n_trials=args.n_trials)
        self.study = study  # Store study for later access
        return self.print_result(study)


    def setup_trials(self, trial):
        # Hyperparameter suggestions for FC_NN with Direct Input and Layer Predictions
        self.wd_bool = trial.suggest_categorical("wd_bool", [True, False])
        if (self.wd_bool):
            self.wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
        else:
            self.wd = 0
        
        self.l1_bool = trial.suggest_categorical("l1_bool", [True, False])
        if (self.l1_bool):
            self.l1 = trial.suggest_float("l1", 1e-5, 1e-2, log=True)
        else:
            self.l1 = 0

        self.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        self.dropout = trial.suggest_float("dropout_fraction", 0.0, 0.7, log=False, step=0.1)
        self.genotype_hiddens = 4 #trial.suggest_int("genotype_hiddens", 2, 8, step=1)
        # no max_attempts for fully connected network
        self.batch_size_power = trial.suggest_int("batch_size_power", 2, 5, step=1)
        # Activation function selection
        activation_choice = trial.suggest_categorical("activation", ["Tanh", "ReLU"])
        if activation_choice == "Tanh":
            self.activation = nn.Tanh
        elif activation_choice == "ReLU":
            self.activation = nn.ReLU
        
        self.epochs = 500
        
        print(f"Trial {trial.number}: h={self.genotype_hiddens}, activation={activation_choice}, lr={self.lr:.2e}, drop={self.dropout:.2f}, batch_size={int(2 ** self.batch_size_power)}")
        sys.stdout.flush()

    def load_data(self, drug=-1):
        print("=== DATA LOADING DEBUG ===")
        print("Initializing data loading...")
        
        # Check required arguments
        required_args = ['cell2id', 'ge_data', 'train_file', 'val_file', 'test_file']
        for arg in required_args:
            if not hasattr(args, arg) or getattr(args, arg) is None:
                raise ValueError(f"Required argument -{arg} not provided")
        
        # Load cell line to ID mapping
        cell2id = load_mapping(args.cell2id, 'cell lines')
        
        # Load gene expression data
        ge_data = pd.read_csv(args.ge_data, sep=',', header=None).values
        
        # Load training and validation data
        train_features, val_features, train_labels, val_labels = load_train_data(
            args.train_file, args.val_file, cell2id
        )
        
        # Load test data
        test_features, test_labels = load_pred_data(
            args.test_file, cell2id
        )
        
        # Convert to tensors and build input vectors
        train_data = torch.Tensor(train_features)
        val_data = torch.Tensor(val_features)
        test_data = torch.Tensor(test_features)
        
        # Ensure labels are properly shaped (squeeze extra dimensions)
        train_labels = torch.FloatTensor(train_labels).squeeze()
        val_labels = torch.FloatTensor(val_labels).squeeze()
        test_labels = torch.FloatTensor(test_labels).squeeze()
        
        # Build input vectors using gene expression data
        self.train_input = build_input_vector(train_data, ge_data)
        self.val_input = build_input_vector(val_data, ge_data)
        self.test_input = build_input_vector(test_data, ge_data)
        
        # Store labels
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels


    def train_model(self, trial):
        self.setup_trials(trial)
        
        # Create trial-specific directory
        trial_dir = f"{self.save_dir}/trials/trial_{trial.number}"
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir, exist_ok=True)
        
        # Set up logging for this trial
        log_file = f"{trial_dir}/trial_{trial.number}.log"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Redirect stdout and stderr to log file
            sys.stdout = open(log_file, 'w')
            sys.stderr = sys.stdout
        
            #If data is small enough, then it is faster to load everything onto GPU immediately
            # Get data from single split
            cl_train_data = self.train_input.to(device)
            cl_train_labels = self.train_labels.to(device)
            if cl_train_labels.dim() == 1:
                cl_train_labels = cl_train_labels.unsqueeze(1)
            cl_val_data = self.val_input.to(device)
            cl_val_labels = self.val_labels.to(device)
            if cl_val_labels.dim() == 1:
                cl_val_labels = cl_val_labels.unsqueeze(1)
            cl_test_data = self.test_input.to(device)
            cl_test_labels = self.test_labels.to(device)
            if cl_test_labels.dim() == 1:
                cl_test_labels = cl_test_labels.unsqueeze(1)

            epochs = self.epochs
            cl_dataloader = DataLoader(TensorDataset(cl_train_data, cl_train_labels), batch_size=int(2 ** self.batch_size_power), shuffle=True, drop_last=True)

            # Create FC_NN_DI_Layer_Pred model with hyperparameters
            # This model has concatenated inputs to each layer and produces intermediary predictions
            # Use trial number to create unique seed for each trial while maintaining reproducibility
            trial_seed = trial.number + args.seed
            # Model creation
            
            try:
                model = FC_NN_DI_Layer_Pred(
                    dropout_fraction=self.dropout,
                    activation=self.activation,
                    genotype_hiddens=self.genotype_hiddens,
                    seed=trial_seed  # each trial number has same seed
                ).to(device)
            except RuntimeError as e:
                raise e
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.wd)

            min_loss = None
            best_model_state = None
            counter = 0
            best_val_r2 = -100
            best_val_r2_epoch = -1
            val_loss = 100  # Initialize val_loss

            for epoch in range(epochs):
                for i, (input_data, labels) in enumerate(cl_dataloader):
                    model.train()
                    optimizer.zero_grad()
                    
                    # Move data to GPU
                    input_data = input_data.to(device)
                    labels = labels.to(device)

                    # FC_NN_DI_Layer_Pred returns single aggregated output
                    final_pred = model(input_data)
                    
                    # Check for NaN/inf in predictions
                    if torch.any(torch.isnan(final_pred)) or torch.any(torch.isinf(final_pred)):
                        print(f"Warning: Model predictions contain NaN/inf values. Skipping batch.")
                        continue

                    loss_fn = nn.MSELoss()

                    # Ensure labels have the same shape as final prediction
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)  # Add dimension to match pred shape
                    
                    # Compute loss for final aggregated prediction
                    final_loss = loss_fn(final_pred, labels)
                    
                    # L1 regularization
                    l1_loss = sum(p.abs().sum() for p in model.parameters())
                    
                    # Check for NaN/inf in losses
                    if torch.isnan(final_loss) or torch.isinf(final_loss):
                        print(f"Warning: Final loss is NaN/inf. Skipping batch.")
                        continue
                    
                    if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                        print(f"Warning: L1 loss is NaN/inf. Skipping batch.")
                        continue
                    
                    total_loss = final_loss + l1_loss * self.l1
                    
                    # Check total loss
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Total loss is NaN/inf. Skipping batch.")
                        print(f"Final loss: {final_loss.item():.6f}, L1 loss: {l1_loss.item():.6f}")
                        continue

                    total_loss.backward()
                    
                    # Check for NaN/inf gradients
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                                print(f"Warning: Gradients contain NaN/inf. Skipping batch.")
                                optimizer.zero_grad()
                                continue
                            grad_norm += p.grad.norm().item() ** 2
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                with torch.no_grad():
                    model.eval()
                    final_val_pred = model(cl_val_data.to(device))
                    
                    # Check for NaN/inf in validation predictions
                    if torch.any(torch.isnan(final_val_pred)) or torch.any(torch.isinf(final_val_pred)):
                        print(f"Warning: Validation predictions contain NaN/inf values.")
                        val_r2 = -1000.0  # Very low R² for invalid predictions
                    else:
                        val_r2 = get_r2_score(final_val_pred, cl_val_labels)
                    
                    # Compute validation loss using final aggregated prediction
                    if cl_val_labels.dim() == 1:
                        cl_val_labels = cl_val_labels.unsqueeze(1)
                    new_val_loss = loss_fn(final_val_pred, cl_val_labels.to(device))
                    
                    # Check validation loss
                    if torch.isnan(new_val_loss) or torch.isinf(new_val_loss):
                        print(f"Warning: Validation loss is NaN/inf. Setting to large value.")
                        new_val_loss = torch.tensor(1000.0, device=device)
                    
                    # Report validation loss to Optuna every epoch
                    trial.report(new_val_loss.item(), epoch)
                    
                    # Save best model based on validation loss
                    if min_loss is None:
                        min_loss = new_val_loss
                        best_model_state = deepcopy(model.state_dict())
                        torch.save(model.state_dict(), f"{trial_dir}/model_best.pt")
                        print(f"Model saved at epoch {epoch}")
                    elif min_loss - new_val_loss > 0.0001:  # delta threshold
                        min_loss = new_val_loss
                        best_model_state = deepcopy(model.state_dict())
                        torch.save(model.state_dict(), f"{trial_dir}/model_best.pt")
                        print(f"Model saved at epoch {epoch}")
                    
                    if (val_r2 > best_val_r2):
                        best_val_r2 = val_r2
                        best_val_r2_epoch = epoch
                    
                    if (val_loss - new_val_loss < 0.0001):
                        counter += 1
                        if (counter >= self.patience):
                            break
                    else:
                        val_loss = new_val_loss
                        counter = 0
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        print(f"Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.exceptions.TrialPruned()
            
            # Load the best model for final evaluation
            if best_model_state is not None:
                print("\nLoading best model for final evaluation...")
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), f"{trial_dir}/model_final.pt")
                print("Best model saved as model_final.pt")
            else:
                print("\nNo best model found, using final model...")
            
            # Evaluate best model on validation and test data
            print("\nEvaluating best model on validation data...")
            val_metrics = evaluate_model_metrics(model, cl_val_data, cl_val_labels, device)
            
            print("\nEvaluating best model on test data...")
            test_metrics = evaluate_model_metrics(model, cl_test_data, cl_test_labels, device)
            
            # Save metrics to CSV
            save_metrics_to_csv(val_metrics, test_metrics, trial_dir)
            
            final_val_r2, final_test_r2 = self.print_model_statistics(f"{trial_dir}/model_final.pt", cl_train_data, cl_train_labels, cl_val_data, cl_val_labels, cl_test_data, cl_test_labels, trial.number)
            print(f"Best Val R2: {best_val_r2}=={final_val_r2} from Epoch {best_val_r2_epoch} | Test R2: {final_test_r2}")
            print(f"---------- Trial {trial.number} complete after {epoch} epochs ----------")
            sys.stdout.flush()

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            print(f"Val R2: {final_val_r2} Test R2: {final_test_r2}")
            if (final_val_r2 > self.best):
                self.best = final_val_r2
                self.best_test = final_test_r2
                self.copy_best_model(trial, final_val_r2, trial_dir, f"{self.save_dir}/best_model")

            # Save trial results to output directory
            self.save_trial_results(trial, final_val_r2, final_test_r2, val_metrics, test_metrics, trial_dir)

            return final_val_r2
            
        finally:
            # Restore stdout and stderr
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def save_trial_results(self, trial, val_r2, test_r2, val_metrics, test_metrics, trial_dir):
        """Save comprehensive trial results to output directory"""
        
        # Save trial summary
        trial_summary = {
            'trial_number': trial.number,
            'validation_r2': val_r2,
            'test_r2': test_r2,
            'hyperparameters': trial.params,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        # Save as JSON
        import json
        with open(f"{trial_dir}/trial_summary.json", 'w') as f:
            json.dump(trial_summary, f, indent=2, default=str)
        
        # Save detailed metrics
        if val_metrics and test_metrics:
            detailed_metrics = {
                'validation': {
                    'pearson_correlation': val_metrics['pearson_corr'],
                    'spearman_correlation': val_metrics['spearman_corr'],
                    'r2_score': val_metrics['r2_score'],
                    'loss': val_metrics['loss']
                },
                'test': {
                    'pearson_correlation': test_metrics['pearson_corr'],
                    'spearman_correlation': test_metrics['spearman_corr'],
                    'r2_score': test_metrics['r2_score'],
                    'loss': test_metrics['loss']
                }
            }
            
            with open(f"{trial_dir}/detailed_metrics.json", 'w') as f:
                json.dump(detailed_metrics, f, indent=2)
        
        # Save hyperparameters
        with open(f"{trial_dir}/hyperparameters.txt", 'w') as f:
            f.write("Trial Hyperparameters\n")
            f.write("=" * 30 + "\n")
            for key, value in trial.params.items():
                f.write(f"{key}: {value}\n")
        
        # Save predictions if available
        if val_metrics and 'predictions' in val_metrics:
            np.savetxt(f"{trial_dir}/val_predictions.txt", val_metrics['predictions'], '%.6f')
            np.savetxt(f"{trial_dir}/val_true_labels.txt", val_metrics['true_labels'], '%.6f')
        
        if test_metrics and 'predictions' in test_metrics:
            np.savetxt(f"{trial_dir}/test_predictions.txt", test_metrics['predictions'], '%.6f')
            np.savetxt(f"{trial_dir}/test_true_labels.txt", test_metrics['true_labels'], '%.6f')

    def print_model_statistics(self, modeldir, cl_train_data, cl_train_labels, cl_val_data, cl_val_labels, cl_test_data, cl_test_labels, trial_number):
        total_val_r2 = 0.0
        total_train_r2 = 0.0
        total_train_loss = 0.0
        total_val_loss = 0.0
        cl_val_pred = None
        cl_train_pred = None

        # Recreate model with same hyperparameters
        # Note: For evaluation, we use the base seed since we're loading a saved model
        # The architecture was already determined during training with trial-specific seed
        try:
            model = FC_NN_DI_Layer_Pred(
                dropout_fraction=self.dropout,
                activation=self.activation,
                genotype_hiddens=self.genotype_hiddens,
                seed=args.seed  # Use base seed for evaluation
            ).to(device)
        except RuntimeError as e:
            if "Failed to generate valid network" in str(e):
                print(f"Failed to generate valid sparse network for model evaluation after {self.max_attempts} attempts")
                print("Skipping model statistics evaluation")
                return 0.0, 0.0
            else:
                raise e
        model.load_state_dict(torch.load(f"{modeldir}"))
        model.eval()
        model = model.to(device)  # Move model to GPU
        with torch.no_grad():
            loss_fn = nn.MSELoss()
            #Else statements shouldn't run, artifact of ensembling models
            if (cl_val_pred == None):
                final_val_pred = model(cl_val_data.to(device))
                cl_val_pred = final_val_pred
            else:
                temp_final_val_pred = model(cl_val_data.to(device))
                cl_val_pred += temp_final_val_pred
            # Ensure labels have the same shape as predictions
            if cl_val_labels.dim() == 1:
                cl_val_labels = cl_val_labels.unsqueeze(1)
            total_val_loss += loss_fn(cl_val_pred, cl_val_labels.to(device))
        
            if (cl_train_pred == None):
                final_train_pred = model(cl_train_data.to(device))
                cl_train_pred = final_train_pred
            else:
                temp_final_train_pred = model(cl_train_data.to(device))
                cl_train_pred += temp_final_train_pred
            # Ensure training labels have the same shape as predictions
            if cl_train_labels.dim() == 1:
                cl_train_labels = cl_train_labels.unsqueeze(1)
            total_train_loss += loss_fn(cl_train_pred, cl_train_labels.to(device))

            final_test_pred = model(cl_test_data.to(device))
            cl_test_pred = final_test_pred
            # Ensure test labels have the same shape as predictions
            if cl_test_labels.dim() == 1:
                cl_test_labels = cl_test_labels.unsqueeze(1)
            total_test_loss = loss_fn(cl_test_pred, cl_test_labels.to(device))
            
        test_r2 = get_r2_score(cl_test_pred, cl_test_labels)
        val_r2 = get_r2_score(cl_val_pred, cl_val_labels)
        train_r2 = get_r2_score(cl_train_pred, cl_train_labels)

        print(f"CL Test -- Loss {total_test_loss}, R2: {test_r2}")
        print(f"Cell Line Validation -- Raw Loss: {total_val_loss}, R2: {val_r2}")
        print(f"Cell Line Train -- Raw Loss: {total_train_loss}, R2: {train_r2}")
        sys.stdout.flush()
        return val_r2, test_r2

    def copy_best_model(self, trial, trial_value, src, dest):
        if (os.path.exists(dest)):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        with open(f"{dest}/save.log", "w") as f:
            f.write(f"Saving current best trial, Trial: {trial.number} Started at {trial.datetime_start}\n")
            f.write("Best trial:\n")
            f.write(f"Value: {trial_value}\n")

            best_params = {}
            f.write("\nBEST PARAMS:")
            for key, value in trial.params.items():
                f.write("{}: {}\n".format(key, value))
                best_params[key] = value
            for key, value in trial.user_attrs.items():
                f.write("{}: {}\n".format(key, value))
                best_params[key] = value

    def print_result(self, study):
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics:")
        print("Number of finished trials:", len(study.trials))
        print("Number of pruned trials:", len(pruned_trials))
        print("Number of complete trials:", len(complete_trials))

        print("Best trial:")
        best_trial = study.best_trial

        print("Value: ", best_trial.value)

        best_params = {}
        print("\nBEST PARAMS:")
        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))
            best_params[key] = value
        for key, value in best_trial.user_attrs.items():
            print("{}: {}".format(key, value))
            best_params[key] = value

        return best_params


if __name__ == "__main__":
    hparam_trainer = OptunaFCNNLayerPredTrainer()
    best_params = hparam_trainer.exec_study()
    
    # Save final results summary
    final_results = {
        'best_validation_r2': hparam_trainer.best,
        'best_test_r2': hparam_trainer.best_test,
        'best_hyperparameters': best_params,
        'total_trials': len(hparam_trainer.study.trials),
        'drug_id': args.drug,
        'seed': args.seed
    }
    
    # Save final results as JSON
    import json
    with open(f"{hparam_trainer.save_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save CSV summary
    with open(f"{hparam_trainer.save_dir}/best_model_results.csv", "w") as f:
        f.write(f"Test R2, Val R2\n{hparam_trainer.best_test}, {hparam_trainer.best}")
    
    # Save study summary
    with open(f"{hparam_trainer.save_dir}/study_summary.txt", 'w') as f:
        f.write("Hyperparameter Tuning Study Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Drug ID: {args.drug}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Best Validation R²: {hparam_trainer.best:.6f}\n")
        f.write(f"Best Test R²: {hparam_trainer.best_test:.6f}\n")
        f.write(f"Total Trials: {len(hparam_trainer.study.trials)}\n")
        f.write("\nBest Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print("ALL COMPLETE")
    print(f"Results saved to: {hparam_trainer.save_dir}")
    print(f"Best Validation R²: {hparam_trainer.best:.6f}")
    print(f"Best Test R²: {hparam_trainer.best_test:.6f}")
