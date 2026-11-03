# Layer Pruned Fully Connected Neural Network (LayerPrunedFC_NN) - Complete Documentation

**Shared conventions:** **[`../README.md`](../README.md)** (data, jobs).

## Running bundled jobs (this repository)

### Environment and CLI (current code)

Use the repository root **`environment.yml`** (see root **`README.md`**; `conda activate fast-nest-nn`).

**`layer_prune_nn_hparam_tuner.py`**: **`-n_trials`**, **`-max_epochs`** (default **500**), standard data paths as in **`jobs/`**.

**Expected outputs:** Optuna journal (**`LayerPrunedFC_NN_HTune.log`**), **`trials/trial_*/model_best.pt`** (when target sparsity is met), **`best_model/`** with baked-in weights, **`final_results.json`**. The tuner saves a single portable **`model_best.pt`**.

---

## Overview

This directory contains a complete system for hyperparameter tuning of Layer Pruned Fully Connected Neural Networks (LayerPrunedFC_NN) for drug response prediction. The system uses iterative L1-norm pruning during training to create sparse neural networks that maintain performance while reducing model complexity.

### Key Features
- **Iterative Pruning**: Prunes weights during training at configurable intervals
- **Target Sparsity Validation**: Ensures models achieve target sparsity levels before saving
- **Optuna-based Hyperparameter Optimization**: Automatically searches optimal hyperparameters
- **Bundled job lists**: `jobs/layer_prune_nn_jobs.txt` for batch execution on your cluster
- **Comprehensive Metrics**: Tracks R², Pearson correlation, Spearman correlation, and loss

---

## Methodology

### Neural Network Architecture

The `LayerPrunedFC_NN` model implements a fully connected neural network with the following structure:

#### Layer Configuration
- **Input Layer**: 689 nodes (gene expression features)
- **Hidden Layers**: 8 layers with decreasing width
  - Layer 1: `76 * genotype_hiddens` nodes
  - Layer 2: `32 * genotype_hiddens` nodes
  - Layer 3: `13 * genotype_hiddens` nodes
  - Layer 4: `4 * genotype_hiddens` nodes
  - Layer 5: `2 * genotype_hiddens` nodes
  - Layer 6: `2 * genotype_hiddens` nodes
  - Layer 7: `1 * genotype_hiddens` nodes
  - Layer 8: `1 * genotype_hiddens` nodes
- **Output Layer**: 1 node (AUC prediction) with Sigmoid activation

#### Architecture Details (from `layer_prune_fc_nn.py`)
```python
# Each hidden layer consists of:
- Linear transformation
- Activation function (Tanh or ReLU, tunable)
- BatchNorm1d (optional, tunable)
- Dropout (tunable fraction)
```

### Pruning Methodology

The pruning strategy uses **iterative L1-norm unstructured pruning**:

1. **Initial State**: Network starts as fully connected
2. **Pruning Schedule**: Prunes at regular intervals during training (configurable frequency)
3. **Pruning Strategy**: 
   - Uses L1-norm to identify least important weights
   - Prunes minimum of:
     - Weights needed to reach target sparsity (`nest_edges_by_layer`)
     - Weights specified by pruning percentage
4. **Layer-Specific Targets**: Each layer has a minimum number of edges that must remain (`nest_edges_by_layer`)

#### Pruning Logic (from `layer_prune_fc_nn.py:82-115`)
```python
def prune(self, sparsity_level=0.5):
    # For each layer:
    # 1. Count remaining unpruned weights
    # 2. Calculate edges needed to prune to reach target
    # 3. Calculate edges that sparsity_level would prune
    # 4. Prune minimum of these two, ensuring >= 0
    edges_to_prune = max(0, min(
        remaining_edges - target_edges,
        int(sparsity_level * remaining_edges)
    ))
```

**Key Properties**:
- **Iterative**: Can be called multiple times during training
- **Safe**: Never prunes below minimum required edges per layer
- **Adaptive**: Adjusts pruning amount based on current state

#### Target Sparsity Validation

The model includes a validation function `is_fully_pruned_to_target()` that checks if all layers have exactly the target number of nonzero edges specified in `nest_edges_by_layer`. This function:

- Counts nonzero weights in each layer (using pruning masks if present)
- Compares against target edges for each layer
- Returns `True` only if all layers match their targets exactly

**Usage in Training**:
- Models are only saved during training if `is_fully_pruned_to_target()` returns `True`
- After training, if the model hasn't achieved target sparsity, the trial returns `-1000.0` to Optuna to signal a failed trial
- Before saving the final model, pruning masks are removed using `prune.remove()` to create a clean model state

### Training Process

1. **Data Loading**: Loads gene expression data, cell line mappings, and drug response labels
2. **Model Initialization**: Creates fully connected network
3. **Training Loop**:
   - Forward pass through network
   - Compute MSE loss + L1 regularization
   - Backward pass and gradient clipping
   - Optimizer step (AdamW)
   - **Periodic Pruning**: Every `prune_frequency` epochs, prune model
4. **Validation**: After each epoch, evaluate on validation set
5. **Model Saving**: 
   - Only saves `model_best.pt` if `is_fully_pruned_to_target()` returns `True`
   - Tracks best validation loss among fully-pruned models
6. **Early Stopping**: Patience-based early stopping on validation loss
7. **Final Model Preparation**:
   - Loads best saved model (or the current model if it is at target sparsity)
   - Removes pruning masks using `prune.remove()` for all linear layers
   - Overwrites `model_best.pt` with the mask-free weights
   - If target sparsity not achieved, returns `-1000.0` to Optuna

---

## File Structure and Coordination

### Core Files

#### 1. `layer_prune_fc_nn.py`
**Purpose**: Neural network model definition

**Key Components**:
- `LayerPrunedFC_NN` class: Main model class
- `prune()` method: Implements iterative L1-norm pruning
- `is_fully_pruned_to_target()` method: Validates that all layers have achieved target sparsity
- Architecture definition: Layer sizes, activation functions, BatchNorm

**Key Methods**:
- `prune(sparsity_level)`: Prunes weights using L1-norm, ensuring minimum edges per layer are preserved
- `is_fully_pruned_to_target()`: Returns `True` if all layers have exactly `nest_edges_by_layer[i]` nonzero edges
- `forward(X)`: Standard forward pass through the network

**Dependencies**: None (standalone model)

**Used By**: `layer_prune_nn_hparam_tuner.py`

---

#### 2. `layer_prune_nn_hparam_tuner.py`
**Purpose**: Hyperparameter optimization using Optuna

**Key Components**:
- `OptunaLayerPruneNNTrainer` class: Manages hyperparameter tuning
- `setup_trials()`: Defines hyperparameter search space
- `train_model()`: Trains model for each trial with pruning
- `load_data()`: Loads and preprocesses drug response data

**Hyperparameters Tuned**:
- Learning rate: `1e-5` to `1e-2` (log scale)
- Weight decay: `1e-5` to `1e-2` (log scale, optional)
- L1 regularization: `1e-5` to `1e-2` (log scale, optional)
- Dropout: `0.0` to `0.7` (step 0.1)
- Batch size: `2^2` to `2^5` (4 to 32)
- Activation: Tanh or ReLU
- **Prune frequency**: 1 to 50 epochs
- **Prune percentage**: 0.05 to 0.5 (step 0.05)
- **BatchNorm**: True or False

**Dependencies**: 
- `layer_prune_fc_nn.py` (imports `LayerPrunedFC_NN`)
- Optuna, PyTorch, scikit-learn, pandas

**Used By**: the bundled `jobs/*.txt` lines

---

#### 3. `cleanup_failed_experiment.py`
**Purpose**: Clean up failed experiments to allow rerunning

**Features**:
- Deletes `results/D{drug}/{job_id}/`
- Dry-run mode to preview deletions

**Usage Examples**:
```bash
# Clean up a single experiment
python cleanup_failed_experiment.py --job-id D5_0

# Clean up multiple experiments
python cleanup_failed_experiment.py --job-id D5_0 D5_1 D5_2

# Dry run (preview without deleting)
python cleanup_failed_experiment.py --job-id D5_0 --dry-run
```

**What Gets Deleted**:
1. Experiment directory: `results/D{drug_id}/{job_id}/`

**Dependencies**: None (standalone)

**Used By**: Users (run manually to clean up failed experiments)

---

## Usage Guide

Bundled commands are in **`jobs/layer_prune_nn_jobs.txt`**. From **`ArchitecturePerformanceExperiments/LP-NN/`**:

```bash
cd ArchitecturePerformanceExperiments/LP-NN
grep -v '^#' jobs/layer_prune_nn_jobs.txt | grep -v '^$' | bash
```

Use your cluster scheduler for parallel GPU runs. See **“Running bundled jobs”** at the top of this README.

### Monitor and check results

Results are organized by drug and experiment:
```
results/
├── D5/
│   ├── D5_0/
│   │   ├── best_model/
│   │   │   ├── metrics.csv
│   │   │   ├── model_best.pt
│   │   │   └── save.log
│   │   ├── trials/
│   │   │   ├── trial_0/
│   │   │   │   ├── metrics.csv
│   │   │   │   ├── hyperparameters.txt
│   │   │   │   └── model_best.pt (mask-free after training)
│   │   │   └── ...
│   │   ├── final_results.json
│   │   └── study_summary.txt
│   └── D5_1/
│       └── ...
└── D80/
    └── ...
```

---

## Hyperparameter Tuning

### Optuna Study Configuration

**Study Name**: `LayerPrunedFC_NN_HTune`

**Storage**: Journal file storage (`LayerPrunedFC_NN_HTune.log`)

**Sampler**: TPE (Tree-structured Parzen Estimator) with seed

**Pruner**: MedianPruner
- `n_startup_trials=15`: Wait 15 trials before pruning
- `n_warmup_steps=0`: No warmup steps
- `interval_steps=1`: Check every epoch

**Direction**: Maximize (validation R²)

### Hyperparameter Search Space

| Hyperparameter | Type | Range | Notes |
|---------------|------|-------|-------|
| `lr` | float | 1e-5 to 1e-2 | Log scale |
| `wd_bool` | categorical | [True, False] | Enable weight decay |
| `wd` | float | 1e-5 to 1e-2 | If wd_bool=True, log scale |
| `l1_bool` | categorical | [True, False] | Enable L1 regularization |
| `l1` | float | 1e-5 to 1e-2 | If l1_bool=True, log scale |
| `dropout_fraction` | float | 0.0 to 0.7 | Step 0.1 |
| `batch_size_power` | int | 2 to 5 | Batch size = 2^power (4-32) |
| `activation` | categorical | ["Tanh", "ReLU"] | Activation function |
| `prune_frequency` | int | 1 to 50 | Prune every N epochs |
| `prune_percentage` | float | 0.05 to 0.5 | Step 0.05 (minimum 5% to ensure pruning occurs) |
| `use_batchnorm` | categorical | [True, False] | Use BatchNorm layers |

### Training Configuration

- **Epochs**: 500 maximum
- **Early Stopping**: Patience of 20 epochs (no improvement in validation loss)
- **Optimizer**: AdamW
  - Betas: (0.9, 0.99)
  - Eps: 1e-05
  - Weight decay: Tunable
- **Loss Function**: MSE + L1 regularization
- **Gradient Clipping**: Max norm 1.0

### Pruning Schedule

Pruning occurs during training:
- **Frequency**: Every `prune_frequency` epochs
- **Method**: L1-norm unstructured pruning
- **Amount**: Minimum of:
  - Edges needed to reach target sparsity (`nest_edges_by_layer`)
  - `prune_percentage * remaining_edges`
- **Validation**: Models are only saved if they achieve target sparsity (`is_fully_pruned_to_target()`)
- **Final Model**: Pruning masks are removed and `model_best.pt` is rewritten with plain weights

Example: If `prune_frequency=10` and `prune_percentage=0.1`:
- Epoch 10: Prune up to 10% of remaining weights (or amount needed to reach target)
- Epoch 20: Prune up to 10% of remaining weights (or amount needed to reach target)
- Epoch 30: Prune up to 10% of remaining weights (or amount needed to reach target)
- ... continues until target sparsity is achieved or training completes

**Important**: If a model fails to achieve target sparsity after training completes, the trial returns `-1000.0` to Optuna, signaling a failed trial.

---

## Summary

This system provides a complete pipeline for hyperparameter tuning of Layer Pruned Fully Connected Neural Networks:

1. **Model**: `LayerPrunedFC_NN` with iterative L1-norm pruning and target sparsity validation
2. **Optimization**: Optuna-based hyperparameter search with trial failure handling
3. **Execution**: one command per `(drug, experiment)` in `jobs/`

The methodology combines:
- **Architecture Search**: Finding optimal network configuration (activation, BatchNorm, dropout)
- **Pruning Strategy**: Iterative weight removal during training with target sparsity enforcement
- **Hyperparameter Optimization**: Automated search for best settings (learning rate, regularization, pruning schedule)
- **Quality Control**: Only saves models that achieve target sparsity, returns low scores for failed trials

### Key Implementation Details

- **Model Saving**: `model_best.pt` is saved during training only when `is_fully_pruned_to_target()` returns `True`
- **Final Model**: Pruning masks are removed with `prune.remove()` and `model_best.pt` is rewritten with plain weights
- **Trial Failure**: Trials that don't achieve target sparsity return `-1000.0` to Optuna
- **Pruning Range**: `prune_percentage` ranges from 0.05 to 0.5 (minimum 5% to ensure pruning occurs)
