#D80_25 Last run
"""Hyperparameter tuning for NeST VNN (Neural Network with Visible Structure)

Uses Optuna for hyperparameter optimization with ranges from Old_eNest.
Logging style and saving format follows ERK_SNN scheduler patterns.
Original nest_vnn implementation is preserved in src/ directory.

Usage:
    python -u nest_vnn_hparam_tuner.py \
        -drug 188 \
        -onto /path/to/ontology.txt \
        -train_file /path/to/train.txt \
        -val_file /path/to/val.txt \
        -test_file /path/to/test.txt \
        -cell2id /path/to/cell2ind.txt \
        -gene2id /path/to/gene2ind.txt \
        -transcriptomic /path/to/ge_data.txt \
        -n_trials 100 \
        -seed 42 \
        -output_dir results/D188/D188_0 \
        -cuda 0
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

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
import torch.utils.data as du
from optuna.trial import Trial, TrialState
from torch.autograd import Variable
from torch.nn import MSELoss
from scipy import stats
from sklearn.metrics import r2_score

# Add src directory to path for importing original nest_vnn modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from drugcell_nn import DrugCellNN
from training_data_wrapper import TrainingDataWrapper
import util

# Limit host thread usage
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

DEFAULT_PATIENCE = 20
VAL_IMPROVEMENT_EPS = 1e-4
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


def pearson_correlation(pred, true):
    """Calculate Pearson correlation coefficient"""
    pred_np = pred.cpu().numpy().flatten()
    true_np = true.cpu().numpy().flatten()
    
    if np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np)):
        return 0.0
    if np.any(np.isnan(true_np)) or np.any(np.isinf(true_np)):
        return 0.0
    
    try:
        return stats.pearsonr(pred_np, true_np)[0]
    except ValueError:
        return 0.0


def spearman_correlation(pred, true):
    """Calculate Spearman correlation coefficient"""
    pred_np = pred.cpu().numpy().flatten()
    true_np = true.cpu().numpy().flatten()
    
    if np.any(np.isnan(pred_np)) or np.any(np.isinf(pred_np)):
        return 0.0
    if np.any(np.isnan(true_np)) or np.any(np.isinf(true_np)):
        return 0.0
    
    try:
        return stats.spearmanr(pred_np, true_np)[0]
    except ValueError:
        return 0.0


def get_r2_score(torch_pred, torch_labels):
    """Calculate R2 score"""
    pred = torch_pred.cpu().numpy().flatten()
    labels = torch_labels.cpu().numpy().flatten()
    
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        return -1000.0
    if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
        return -1000.0
    
    return r2_score(labels, pred)


def evaluate_model_metrics(
    model: nn.Module, 
    val_feature: torch.Tensor, 
    val_label: torch.Tensor, 
    cell_features: np.ndarray,
    cuda_id: int,
    batch_size: int = 2000
) -> Dict[str, float]:
    """Evaluate model and return metrics (matches original validation code exactly)."""
    # Check if data is empty before processing
    if val_feature.size(0) == 0 or val_label.size(0) == 0:
        print(f"Warning: Empty data provided to evaluate_model_metrics (features: {val_feature.size(0)}, labels: {val_label.size(0)})")
        return {
            "predictions": np.array([]),
            "true_labels": np.array([]),
            "pearson_corr": 0.0,
            "spearman_corr": 0.0,
            "r2_score": -1000.0,
            "loss": 0.0,
        }
    
    val_label_gpu = val_label.cuda(cuda_id)
    
    model.eval()
    val_loader = du.DataLoader(
        du.TensorDataset(val_feature, val_label), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Match original validation code exactly
    val_predict = torch.zeros(0, 0).cuda(cuda_id)
    val_loss = torch.tensor(0.0).cuda(cuda_id)  # Initialize as tensor
    
    with torch.no_grad():
        for i, (inputdata, labels) in enumerate(val_loader):
            features = util.build_input_vector(inputdata, cell_features)
            cuda_features = Variable(features.cuda(cuda_id))
            cuda_labels = Variable(labels.cuda(cuda_id))
            
            aux_out_map, _ = model(cuda_features)
            
            if val_predict.size()[0] == 0:
                val_predict = aux_out_map['final'].data
            else:
                val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
            
            # Calculate loss (matches original)
            for name, output in aux_out_map.items():
                loss = MSELoss()
                if name == 'final':
                    val_loss += loss(output, cuda_labels)
    
    # Handle empty predictions (matches original check)
    if val_predict.size()[0] == 0:
        print(f"Warning: No predictions generated in evaluate_model_metrics")
        return {
            "predictions": np.array([]),
            "true_labels": val_label_gpu.cpu().numpy(),
            "pearson_corr": 0.0,
            "spearman_corr": 0.0,
            "r2_score": -1000.0,
            "loss": val_loss.item() if torch.is_tensor(val_loss) else float(val_loss),
        }
    
    pearson_corr = pearson_correlation(val_predict, val_label_gpu)
    spearman_corr = spearman_correlation(val_predict, val_label_gpu)
    r2 = get_r2_score(val_predict, val_label_gpu)
    
    return {
        "predictions": val_predict.cpu().numpy(),
        "true_labels": val_label_gpu.cpu().numpy(),
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "r2_score": r2,
        "loss": val_loss.item() if torch.is_tensor(val_loss) else float(val_loss),
    }


def save_metrics_to_csv(val_metrics: Dict, test_metrics: Dict, save_dir: Path) -> None:
    """Save metrics to CSV in the specified format."""
    metrics_data = [
        ["Pearson Correlation", f"{val_metrics['pearson_corr']:.4f}", 
         f"{test_metrics['pearson_corr']:.4f}" if test_metrics else "N/A"],
        ["Spearman Correlation", f"{val_metrics['spearman_corr']:.4f}", 
         f"{test_metrics['spearman_corr']:.4f}" if test_metrics else "N/A"],
        ["R² Score", f"{val_metrics['r2_score']:.4f}", 
         f"{test_metrics['r2_score']:.4f}" if test_metrics else "N/A"],
        ["Loss", f"{val_metrics['loss']:.4f}", 
         f"{test_metrics['loss']:.4f}" if test_metrics else "N/A"],
    ]
    df = pd.DataFrame(metrics_data, columns=["Metric", "Validation", "Test"])
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "metrics.csv", index=False)
    print(f"Metrics saved to {save_dir}/metrics.csv")


class DataWrapperForHparam:
    """Wrapper class to hold data and hyperparameters for hyperparameter tuning."""
    
    def __init__(self, args):
        self.args = args
        self.cuda = args.cuda
        self.seed = args.seed
        
        # Load mappings
        self.cell_id_mapping = util.load_mapping(args.cell2id, 'cell lines')
        self.gene_id_mapping = util.load_mapping(args.gene2id, 'genes')
        
        # Load ontology
        self._load_ontology(args.onto)
        
        # Load transcriptomic data
        self.transcriptomic_data = np.genfromtxt(args.transcriptomic, delimiter=',')
        self.cell_features = self.transcriptomic_data.reshape(
            self.transcriptomic_data.shape[0], 
            self.transcriptomic_data.shape[1], 
            1
        )
        
        # Load training, validation, and test data
        self._load_data()
    
    def _load_ontology(self, file_name):
        """Load ontology file and build graph structure."""
        import networkx as nx
        import networkx.algorithms.components.connected as nxacc
        import networkx.algorithms.dag as nxadag
        
        dG = nx.DiGraph()
        term_direct_gene_map = {}
        term_size_map = {}
        
        with open(file_name) as fh:
            for line in fh:
                line = line.rstrip().split()
                if line[2] == 'default':
                    dG.add_edge(line[0], line[1])
                else:
                    if line[1] not in self.gene_id_mapping:
                        continue
                    if line[0] not in term_direct_gene_map:
                        term_direct_gene_map[line[0]] = set()
                    term_direct_gene_map[line[0]].add(self.gene_id_mapping[line[1]])
        
        for term in dG.nodes():
            term_gene_set = set()
            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]
            deslist = nxadag.descendants(dG, term)
            for child in deslist:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]
            if len(term_gene_set) == 0:
                print('There is empty terms, please delete term:', term)
                sys.exit(1)
            else:
                term_size_map[term] = len(term_gene_set)
        
        roots = [n for n in dG.nodes if dG.in_degree(n) == 0]
        
        print('There are', len(roots), 'roots:', roots[0])
        print('There are', len(dG.nodes()), 'terms')
        
        self.dG = dG
        self.root = roots[0]
        self.term_size_map = term_size_map
        self.term_direct_gene_map = term_direct_gene_map
    
    def _load_data(self):
        """Load training, validation, and test data from separate files."""
        args = self.args
        
        # Load training data (files have 2 columns: cell_line, auc)
        # But NeST VNN expects 4 columns, so we'll add dummy columns
        train_df_raw = pd.read_csv(args.train_file, sep='\t', header=None)
        
        # Check if file has 2 or 4 columns
        if train_df_raw.shape[1] == 2:
            # Format: cell_line, auc (no smiles or dataset)
            train_df = pd.DataFrame({
                'cell_line': train_df_raw[0],
                'smiles': [''] * len(train_df_raw),  # Dummy smiles column
                'auc': train_df_raw[1],
                'dataset': ['default'] * len(train_df_raw)  # Default dataset for standardization
            })
        else:
            # Format: cell_line, smiles, auc, dataset (4 columns)
            train_df = pd.read_csv(args.train_file, sep='\t', header=None,
                                   names=['cell_line', 'smiles', 'auc', 'dataset'])
        
        # Load validation data
        val_df_raw = pd.read_csv(args.val_file, sep='\t', header=None)
        if val_df_raw.shape[1] == 2:
            val_df = pd.DataFrame({
                'cell_line': val_df_raw[0],
                'smiles': [''] * len(val_df_raw),
                'auc': val_df_raw[1],
                'dataset': ['default'] * len(val_df_raw)
            })
        else:
            val_df = pd.read_csv(args.val_file, sep='\t', header=None,
                                names=['cell_line', 'smiles', 'auc', 'dataset'])
        
        # Load test data
        # Test data format: cell_line_name (string), smiles, auc, dataset (4 columns)
        test_df_raw = pd.read_csv(args.test_file, sep='\t', header=None)
        if test_df_raw.shape[1] == 2:
            # If 2 columns, treat as cell_line_id (int), auc
            test_df = pd.DataFrame({
                'cell_line': test_df_raw[0],
                'smiles': [''] * len(test_df_raw),
                'auc': test_df_raw[1],
                'dataset': ['default'] * len(test_df_raw)
            })
        else:
            # If 4 columns: cell_line (name), smiles, auc, dataset
            test_df = pd.read_csv(args.test_file, sep='\t', header=None,
                                 names=['cell_line', 'smiles', 'auc', 'dataset'])
        
        # ---------------- Standardization (ignore dataset column, treat all as one dataset) ----------------
        # Compute single center/scale statistics from training data (ignoring dataset column)
        train_center, train_scale = self._calc_std_vals(train_df, args.zscore_method)
        
        # Standardize all data using the same training statistics
        train_df = self._standardize_data(train_df, train_center, train_scale)
        val_df = self._standardize_data(val_df, train_center, train_scale)
        test_df = self._standardize_data(test_df, train_center, train_scale)
        
        # Convert to features and labels
        # Training/validation: cell_line is already an integer ID
        # Test: cell_line is a name (string) that needs mapping via cell2id
        def get_cell_id(cell_line_value):
            """Get cell line ID from either an integer ID or a string name."""
            # If it's already an integer, use it directly (training/validation data)
            if isinstance(cell_line_value, (int, np.integer)):
                return int(cell_line_value)
            # Try to convert to int (in case it's a string representation of an int)
            try:
                return int(cell_line_value)
            except (ValueError, TypeError):
                # If conversion fails, treat it as a name and look it up in cell2id mapping
                # cell_id_mapping format: {cell_line_name: integer_id}
                cell_id = self.cell_id_mapping.get(cell_line_value)
                if cell_id is None:
                    raise ValueError(f"Cell line '{cell_line_value}' not found in cell2id mapping")
                return int(cell_id)
        
        # Process training data (cell_line is already integer ID)
        self.train_feature = torch.Tensor([[get_cell_id(row[0])] for row in train_df.values])
        self.train_label = torch.FloatTensor([[float(row[2])] for row in train_df.values])
        print(f"Loaded training data: {len(self.train_feature)} samples")
        
        # Validate that training data exists (required for model training)
        if len(self.train_feature) == 0:
            raise ValueError(
                f"Training dataset is empty! This is required for model training. "
                f"Please check that the training file '{args.train_file}' contains valid data."
            )
        
        # Process validation data (cell_line is already integer ID)
        self.val_feature = torch.Tensor([[get_cell_id(row[0])] for row in val_df.values])
        self.val_label = torch.FloatTensor([[float(row[2])] for row in val_df.values])
        print(f"Loaded validation data: {len(self.val_feature)} samples")
        
        # Process test data (cell_line is a name that needs mapping via cell2id)
        # After standardization, test_df has columns: ['cell_line', 'smiles', 'z']
        # row[0] = cell_line name (string), row[2] = standardized AUC (z)
        test_features_list = []
        test_labels_list = []
        for row in test_df.values:
            cell_line_name = row[0]  # Cell line name (string like "CCL_839")
            standardized_auc = row[2]  # Standardized AUC value (z)
            
            # Map cell line name to integer ID using cell2id mapping
            if cell_line_name not in self.cell_id_mapping:
                print(f"Warning: Cell line '{cell_line_name}' not found in cell2id mapping, skipping")
                continue
            
            cell_id = self.cell_id_mapping[cell_line_name]
            test_features_list.append([cell_id])
            test_labels_list.append([float(standardized_auc)])
        
        if len(test_features_list) == 0:
            print(f"ERROR: Test data is empty after processing! Original test file had {len(test_df_raw)} rows.")
            print(f"Test dataframe shape after standardization: {test_df.shape}")
            if len(test_df) > 0:
                print(f"First few test rows: {test_df.head()}")
                print(f"Sample cell_line values: {test_df['cell_line'].head(10).tolist()}")
            self.test_feature = torch.Tensor([]).reshape(0, 1)
            self.test_label = torch.FloatTensor([]).reshape(0, 1)
        else:
            self.test_feature = torch.Tensor(test_features_list)
            self.test_label = torch.FloatTensor(test_labels_list)
        
        print(f"Loaded test data: {len(self.test_feature)} samples")
    
    def _calc_std_vals(self, df, zscore_method):
        """Calculate standardization values (ignoring dataset column, treating all as one dataset)."""
        # Compute single center/scale for all data, ignoring dataset column
        if zscore_method == 'zscore':
            center = df['auc'].mean()
            scale = df['auc'].std()
        elif zscore_method == 'robustz':
            center = df['auc'].median()
            scale = df['auc'].quantile(0.75) - df['auc'].quantile(0.25)
        else:
            center = 0.0
            scale = 1.0
        
        if pd.isna(scale) or scale == 0.0:
            scale = 1.0
        
        return center, scale
    
    def _standardize_data(self, df, center, scale):
        """Standardize data using single center/scale values (ignoring dataset column)."""
        # Apply standardization: z = (auc - center) / scale
        df = df.copy()
        df['z'] = (df['auc'] - center) / scale
        df = df[['cell_line', 'smiles', 'z']]
        return df


class OptunaNestVNNTrainer:
    """Hyperparameter tuning trainer for NeST VNN using Optuna."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._init_device(args.cuda)
        set_seed(args.seed)

        self.save_dir = Path(args.output_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {self.save_dir}")

        # Load data
        self.data_wrapper = DataWrapperForHparam(args)
        
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

    def exec_study(self) -> Dict[str, float]:
        """Execute the Optuna hyperparameter study."""
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(self.save_dir / "nest_vnn_HTune.log"))
        )
        median_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=15, n_warmup_steps=0, interval_steps=1
        )
        sampler = optuna.samplers.TPESampler(seed=self.args.seed)
        study = optuna.create_study(
            study_name="nest_vnn_HTune",
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
        """Setup hyperparameters for a trial using Old_eNest ranges."""
        # Weight decay
        self.wd_bool = trial.suggest_categorical("wd_bool", [True, False])
        self.wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True) if self.wd_bool else 0.0

        # L1 regularization
        self.l1_bool = trial.suggest_categorical("l1_bool", [True, False])
        self.l1 = trial.suggest_float("l1", 1e-5, 1e-2, log=True) if self.l1_bool else 0.0

        # Learning rate
        self.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
        # Dropout
        self.dropout = trial.suggest_float("dropout_fraction", 0.0, 0.7, log=False, step=0.1)
        
        # NeST VNN specific hyperparameters
        self.genotype_hiddens = 4  # Fixed as in original
        
        # Alpha for auxiliary loss weighting
        self.alpha = trial.suggest_float("alpha", 0.0, 1.0, log=False)
        
        # Batch size (minimum 16, so batch_size_power starts at 4: 2^7=128)
        self.batch_size_power = trial.suggest_int("batch_size_power", 7, 10, step=1)
        
        # Activation function
        activation_choice = trial.suggest_categorical("activation", ["Tanh", "ReLU"])
        self.activation = activation_choice
        
        # Min dropout layer
        self.min_dropout_layer = trial.suggest_int("min_dropout_layer", 1, 4, step=1)
        
        self.epochs = MAX_EPOCHS

        print(f"\nTrial {trial.number}: h={self.genotype_hiddens}, lr={self.lr:.2e}, "
              f"drop={self.dropout:.2f}, alpha={self.alpha:.2f}, "
              f"batch_size={int(2 ** self.batch_size_power)}")
        sys.stdout.flush()

    def _create_model_wrapper(self):
        """Create a wrapper object that mimics TrainingDataWrapper for DrugCellNN."""
        class ModelWrapper:
            pass
        
        wrapper = ModelWrapper()
        wrapper.root = self.data_wrapper.root
        wrapper.num_hiddens_genotype = self.genotype_hiddens
        wrapper.term_direct_gene_map = self.data_wrapper.term_direct_gene_map
        wrapper.min_dropout_layer = self.min_dropout_layer
        wrapper.dropout_fraction = self.dropout
        wrapper.term_size_map = self.data_wrapper.term_size_map
        wrapper.gene_id_mapping = self.data_wrapper.gene_id_mapping
        wrapper.cell_features = self.data_wrapper.cell_features
        wrapper.dG = self.data_wrapper.dG
        
        return wrapper

    def train_model(self, trial: Trial) -> float:
        """Train a model for a single Optuna trial.
        
        This training loop matches the original NeST VNN implementation exactly,
        including:
        - Same weight initialization (param * 0.1 with masking for direct gene layers)
        - Same loss calculation (MSELoss with alpha-weighted auxiliary outputs)
        - Same gradient masking for direct gene layers
        - Same optimizer settings (AdamW with betas=(0.9, 0.99), eps=1e-05)
        - Same early stopping logic based on validation loss improvement
        """
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
            # Set seed for reproducibility (matches original: util.set_seeds)
            trial_seed = trial.number + self.args.seed
            set_seed(trial_seed)
            
            # Create model wrapper and model (matches original DrugCellNN initialization)
            model_wrapper = self._create_model_wrapper()
            model = DrugCellNN(model_wrapper)
            model.cuda(self.args.cuda)
            
            # Initialize weights exactly as original:
            # term_mask_map = util.create_term_mask(...)
            # param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1 for gene layers
            # param.data = param.data * 0.1 for all other layers
            term_mask_map = util.create_term_mask(
                model.term_direct_gene_map, 
                model.gene_dim, 
                self.args.cuda
            )
            for name, param in model.named_parameters():
                term_name = name.split('_')[0]
                if '_direct_gene_layer.weight' in name:
                    param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
                else:
                    param.data = param.data * 0.1
            
            # Set seeds again for reproducible data loading (matches original)
            set_seed(trial_seed)
            
            # Create data loaders (matches original)
            batch_size = int(2 ** self.batch_size_power)
            
            # Check if batch size is too large for the dataset
            train_dataset_size = len(self.data_wrapper.train_feature)
            if batch_size > train_dataset_size:
                print(f"Warning: Batch size {batch_size} is larger than training dataset size {train_dataset_size}. "
                      f"Using batch size {train_dataset_size} instead.")
                batch_size = max(1, train_dataset_size)  # Ensure at least batch size 1
            
            train_loader = du.DataLoader(
                du.TensorDataset(self.data_wrapper.train_feature, self.data_wrapper.train_label),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loader = du.DataLoader(
                du.TensorDataset(self.data_wrapper.val_feature, self.data_wrapper.val_label),
                batch_size=batch_size,
                shuffle=False
            )
            
            # Check if loaders are empty (shouldn't happen if data loading validation passed, but safety check)
            if len(train_loader) == 0:
                raise ValueError(
                    f"Training data loader is empty. This should not happen if training data was loaded correctly. "
                    f"Dataset size: {train_dataset_size}, Batch size: {batch_size}, drop_last: True. "
                    f"This can occur if dataset_size < batch_size with drop_last=True, but batch_size should have been adjusted."
                )
            
            # Optimizer (matches original: AdamW with same hyperparameters)
            # Note: Original uses weight_decay=self.data_wrapper.lr, we use tunable wd
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.lr, 
                betas=(0.9, 0.99), 
                eps=1e-05, 
                weight_decay=self.wd
            )
            optimizer.zero_grad()

            min_loss = None
            best_model_state = None
            counter = 0
            best_val_r2 = float('-inf')
            best_val_r2_epoch = -1
            delta = 0.001  # Matches original args.delta default

            epoch_start_time = time.time()
            training_start_time = time.time()
            
            # Print header matching original format
            print("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\tgrad_norm\telapsed_time")

            for epoch in range(self.epochs):
                # ============ TRAINING (matches original exactly) ============
                model.train()
                train_predict = torch.zeros(0, 0).cuda(self.args.cuda)
                train_label_gpu = None  # Initialize to avoid UnboundLocalError
                total_loss = torch.tensor(0.0).cuda(self.args.cuda)  # Initialize total_loss
                
                for i, (inputdata, labels) in enumerate(train_loader):
                    # Convert torch tensor to Variable (matches original)
                    features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
                    cuda_features = Variable(features.cuda(self.args.cuda))
                    cuda_labels = Variable(labels.cuda(self.args.cuda))
                    
                    # Forward + Backward + Optimize (matches original)
                    optimizer.zero_grad()
                    
                    aux_out_map, _ = model(cuda_features)
                    
                    # Accumulate predictions (matches original)
                    if train_predict.size()[0] == 0:
                        train_predict = aux_out_map['final'].data
                        train_label_gpu = cuda_labels
                    else:
                        train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
                        train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)
                    
                    # Calculate loss with auxiliary outputs (matches original exactly)
                    batch_loss = torch.tensor(0.0).cuda(self.args.cuda)
                    for name, output in aux_out_map.items():
                        loss = MSELoss()
                        if name == 'final':
                            batch_loss += loss(output, cuda_labels)
                        else:
                            batch_loss += self.alpha * loss(output, cuda_labels)
                    
                    # L1 regularization (additional hyperparameter for tuning)
                    if self.l1 > 0:
                        # Use torch operations to ensure tensor type is preserved
                        l1_loss = torch.sum(torch.stack([p.abs().sum() for p in model.parameters()]))
                        batch_loss += self.l1 * l1_loss
                    
                    # Accumulate total loss for epoch (for printing)
                    total_loss += batch_loss
                    
                    batch_loss.backward()
                    
                    # Apply gradient mask for direct gene layers (matches original exactly)
                    for name, param in model.named_parameters():
                        if '_direct_gene_layer.weight' not in name:
                            continue
                        term_name = name.split('_')[0]
                        param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
                    
                    optimizer.step()
                
                # Calculate training metrics (matches original)
                # Handle case where train_loader is empty
                if train_label_gpu is None or train_predict.size()[0] == 0:
                    # No training data processed, skip this epoch
                    print(f"Warning: No training data in epoch {epoch}, skipping...")
                    continue
                
                train_corr = util.pearson_corr(train_predict, train_label_gpu)
                
                # ============ VALIDATION (matches original exactly) ============
                model.eval()
                val_predict = torch.zeros(0, 0).cuda(self.args.cuda)
                val_label_gpu = None  # Initialize to avoid UnboundLocalError
                val_loss = torch.tensor(0.0).cuda(self.args.cuda)  # Initialize as tensor
                
                for i, (inputdata, labels) in enumerate(val_loader):
                    # Convert torch tensor to Variable (matches original)
                    features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
                    cuda_features = Variable(features.cuda(self.args.cuda))
                    cuda_labels = Variable(labels.cuda(self.args.cuda))
                    
                    aux_out_map, _ = model(cuda_features)
                    
                    if val_predict.size()[0] == 0:
                        val_predict = aux_out_map['final'].data
                        val_label_gpu = cuda_labels
                    else:
                        val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
                        val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)
                    
                    # Only count final loss for validation (matches original)
                    for name, output in aux_out_map.items():
                        loss = MSELoss()
                        if name == 'final':
                            val_loss += loss(output, cuda_labels)
                
                # Handle case where val_loader is empty
                if val_label_gpu is None:
                    print(f"Warning: No validation data in epoch {epoch}, skipping validation...")
                    val_corr = 0.0
                    val_r2 = -1000.0
                else:
                    val_corr = util.pearson_corr(val_predict, val_label_gpu)
                    val_r2 = get_r2_score(val_predict, val_label_gpu)
                
                # Print epoch stats (matches original format)
                epoch_end_time = time.time()
                true_auc = torch.mean(train_label_gpu).item() if torch.is_tensor(train_label_gpu) else float(torch.mean(train_label_gpu))
                pred_auc = torch.mean(train_predict).item() if torch.is_tensor(train_predict) else float(torch.mean(train_predict))
                total_loss_val = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
                val_loss_val = val_loss.item() if torch.is_tensor(val_loss) else float(val_loss)
                print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
                    epoch, train_corr, total_loss_val, true_auc, pred_auc, val_corr, val_loss_val, 
                    epoch_end_time - epoch_start_time))
                epoch_start_time = epoch_end_time
                
                # Report to Optuna
                trial.report(val_loss.item() if torch.is_tensor(val_loss) else val_loss, epoch)
                
                # Save best model (matches original logic with delta threshold)
                # Original NeST VNN saves when: min_loss - val_loss > delta
                if min_loss is None:
                    min_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    torch.save(model.state_dict(), trial_dir / "model_best.pt")
                    print(f"Model saved at epoch {epoch}")
                    counter = 0  # Reset counter on improvement
                elif (min_loss - val_loss).item() > delta:
                    min_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    torch.save(model.state_dict(), trial_dir / "model_best.pt")
                    print(f"Model saved at epoch {epoch}")
                    counter = 0  # Reset counter on improvement
                else:
                    # Early stopping counter (added for efficiency in hyperparameter tuning)
                    # Note: Original NeST VNN doesn't have explicit early stopping
                    counter += 1
                    if counter >= self.patience:
                        print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                        break
                
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_val_r2_epoch = epoch
                
                # Check if trial should be pruned
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at epoch {epoch}\n")
                    raise optuna.exceptions.TrialPruned()

            total_training_time = time.time() - training_start_time
            print(f"\n=== Training Complete ===")
            print(f"Total epochs: {epoch + 1}")
            print(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
            sys.stdout.flush()

            # Load best model for final evaluation (matches original)
            if best_model_state is not None:
                print("\nLoading best model for final evaluation...")
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), trial_dir / "model_final.pt")
                print("Best model saved as model_final.pt")
            else:
                print("\nNo best model found, using final model...")

            # Evaluate on validation data (matches original)
            print("\nEvaluating best model on validation data...")
            val_metrics = evaluate_model_metrics(
                model,
                self.data_wrapper.val_feature,
                self.data_wrapper.val_label,
                self.data_wrapper.cell_features,
                self.args.cuda
            )
            
            # Evaluate on test data (matches original)
            print("\nEvaluating best model on test data...")
            print(f"Test data shape: features={self.data_wrapper.test_feature.shape}, labels={self.data_wrapper.test_label.shape}")
            test_metrics = evaluate_model_metrics(
                model,
                self.data_wrapper.test_feature,
                self.data_wrapper.test_label,
                self.data_wrapper.cell_features,
                self.args.cuda
            )

            save_metrics_to_csv(val_metrics, test_metrics, trial_dir)

            final_val_r2 = val_metrics["r2_score"]
            final_test_r2 = test_metrics["r2_score"]
            
            # Print final metrics (matches original format)
            print("=" * 60)
            print("VALIDATION METRICS:")
            print(f"Validation Pearson Correlation: {val_metrics['pearson_corr']:.4f}")
            print(f"Validation Spearman Correlation: {val_metrics['spearman_corr']:.4f}")
            print(f"Validation R² Score: {val_metrics['r2_score']:.4f}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print("=" * 60)
            print("TEST METRICS:")
            print(f"Test Pearson Correlation: {test_metrics['pearson_corr']:.4f}")
            print(f"Test Spearman Correlation: {test_metrics['spearman_corr']:.4f}")
            print(f"Test R² Score: {test_metrics['r2_score']:.4f}")
            print("=" * 60)
            
            print(f"Best Val R2: {best_val_r2:.4f}=={final_val_r2:.4f} from Epoch {best_val_r2_epoch} | Test R2: {final_test_r2:.4f}")
            print(f"---------- Trial {trial.number} complete after {epoch + 1} epochs ----------")
            print(f"Val R²: {final_val_r2:.4f}, Test R²: {final_test_r2:.4f}\n")
            sys.stdout.flush()

            if final_val_r2 > self.best:
                self.best = final_val_r2
                self.best_test = final_test_r2
                self.copy_best_model(trial, final_val_r2, trial_dir, self.best_model_path)
            
            # Save trial results
            self._save_trial_results(trial, final_val_r2, final_test_r2, val_metrics, test_metrics, trial_dir)
            
            return final_val_r2

        except optuna.exceptions.TrialPruned:
            print(" | PRUNED\n")
            raise
        
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def _save_trial_results(self, trial, val_r2, test_r2, val_metrics, test_metrics, trial_dir):
        """Save comprehensive trial results to output directory."""
        # Save hyperparameters
        with open(trial_dir / "hyperparameters.json", "w") as f:
            json.dump(trial.params, f, indent=2, default=str)
        
        # Save predictions
        if val_metrics and 'predictions' in val_metrics:
            np.savetxt(trial_dir / "val_predictions.txt", val_metrics['predictions'], '%.6f')
            np.savetxt(trial_dir / "val_true_labels.txt", val_metrics['true_labels'], '%.6f')
        
        if test_metrics and 'predictions' in test_metrics:
            np.savetxt(trial_dir / "test_predictions.txt", test_metrics['predictions'], '%.6f')
            np.savetxt(trial_dir / "test_true_labels.txt", test_metrics['true_labels'], '%.6f')

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
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for NeST VNN")
    
    # GPU and basic settings
    parser.add_argument("-cuda", type=int, default=0, help="Specify GPU index")
    parser.add_argument("-drug", type=int, default=-1, help="Drug ID")
    parser.add_argument("-n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-output_dir", type=str, default="./results", 
                       help="Output directory for saving models and results")
    
    # Data files
    parser.add_argument("-train_file", type=str, required=True, help="Training data file")
    parser.add_argument("-val_file", type=str, required=True, help="Validation data file")
    parser.add_argument("-test_file", type=str, required=True, help="Test data file")
    
    # NeST VNN specific arguments
    parser.add_argument("-onto", type=str, required=True, 
                       help="Ontology file used to guide the neural network")
    parser.add_argument("-cell2id", type=str, required=True, help="Cell line to ID mapping file")
    parser.add_argument("-gene2id", type=str, required=True, help="Gene to ID mapping file")
    parser.add_argument("-transcriptomic", type=str, required=True, 
                       help="Transcriptomic data (gene expression) for cell lines")
    parser.add_argument("-zscore_method", type=str, default='auc',
                       help="Zscore method (zscore/robustz/auc)")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    hparam_trainer = OptunaNestVNNTrainer(args)
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

