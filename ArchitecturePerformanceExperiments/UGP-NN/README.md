# Relaxed Global Pruned Fully Connected Neural Network (RelaxedGlobalPrunedFC_NN) — UGP-NN

**Shared conventions:** **[`../README.md`](../README.md)** (data, jobs).

## Running bundled jobs (this repository)

### Environment and CLI (current code)

Use the repository root **`environment.yml`** (see root **`README.md`**; `conda activate fast-nest-nn`).

**`relaxed_global_prune_nn_hparam_tuner.py`**: **`-n_trials`**, **`-max_epochs`** (default **500**), standard data paths as in **`jobs/`**.

**Expected outputs:** Optuna journal, **`trials/trial_*/model_best.pt`**, **`best_model/`** with baked-in weights after training, **`final_results.json`**. As in **GP-NN**, a single **`model_best.pt`** is kept on disk.

---

## Overview

This directory contains a complete system for hyperparameter tuning of Global Pruned Fully Connected Neural Networks (RelaxedGlobalPrunedFC_NN) for drug response prediction. The system uses **global unstructured pruning** during training to create sparse neural networks that maintain performance while reducing model complexity.

### Key Features
- **Global Pruning**: Prunes weights across all layers simultaneously using global unstructured pruning
- **Optuna-based Hyperparameter Optimization**: Automatically searches optimal hyperparameters
- **Bundled job lists**: `jobs/relaxed_global_prune_nn_warmup_jobs.txt` for batch execution on your cluster
- **Comprehensive Metrics**: Tracks R², Pearson correlation, Spearman correlation, and loss

---

## Methodology

### Neural Network Architecture

The `RelaxedGlobalPrunedFC_NN` model implements a fully connected neural network with the following structure:

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

#### Architecture Details (from `relaxed_global_prune_fc_nn.py`)
```python
# Each hidden layer consists of:
- Linear transformation
- Activation function (Tanh or ReLU, tunable)
- BatchNorm1d (optional, tunable)
- Dropout (tunable fraction)
```

### Pruning Methodology: Global vs. Layer-wise

**This is the key methodological difference from layer-wise pruning.**

#### Global Unstructured Pruning

The `RelaxedGlobalPrunedFC_NN` uses **global unstructured pruning**, which is fundamentally different from layer-wise pruning:

1. **Global Perspective**: Instead of pruning each layer independently, global pruning considers **all weights across all layers simultaneously**
2. **Unified Ranking**: All weights in the network are ranked together by their L1-norm magnitude
3. **Cross-Layer Selection**: The globally least important weights are removed, regardless of which layer they belong to

#### Comparison: Global vs. Layer-wise Pruning

| Aspect | Global Pruning (RelaxedGlobalPrunedFC_NN) | Layer-wise Pruning (LayerPrunedFC_NN) |
|--------|-----------------------------------|--------------------------------------|
| **Scope** | All layers simultaneously | Each layer independently |
| **Weight Ranking** | Global ranking across entire network | Per-layer ranking |
| **Pruning Method** | `prune.global_unstructured()` | `prune.l1_unstructured()` per layer |
| **Weight Selection** | Globally least important weights | Least important weights per layer |
| **Layer Balance** | May prune more from some layers than others | Each layer pruned proportionally |
| **Computational Cost** | Single global operation | Multiple per-layer operations |
| **Flexibility** | More flexible weight distribution | More controlled per-layer sparsity |

#### Pruning Logic (from `relaxed_global_prune_fc_nn.py`)

```python
def prune(self, sparsity_level=0.5):
    # self.parameters_to_prune: all linear layers except the output layer (built in __init__)
    prune.global_unstructured(
        self.parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity_level
    )
```

**Key Properties**:
- **Global**: Prunes across all layers in a single operation
- **Iterative**: Can be called multiple times during training
- **Efficient**: Single global operation vs. multiple per-layer operations

#### How Global Pruning Works

1. **Weight Collection**: All weights from all layers (except output layer) are collected into a single set
2. **Global Ranking**: All weights are ranked by their L1-norm magnitude across the entire network
3. **Global Selection**: The `sparsity_level` fraction of remaining weights with the smallest magnitude is selected globally
4. **Pruning Application**: Selected weights are pruned from their respective layers

**Example**:
- Network has 3 layers with weights: Layer1 (1000 weights), Layer2 (500 weights), Layer3 (200 weights)
- Total: 1700 weights
- If `sparsity_level=0.1`, prune 170 weights globally
- The 170 globally least important weights are removed, which may come from any combination of layers
- Result: Some layers may lose more weights than others

#### Why Global Pruning?

**Advantages**:
1. **Optimal Weight Selection**: Considers the entire network when selecting which weights to remove
2. **Flexibility**: Allows the network to naturally determine which layers need more/less pruning
3. **Efficiency**: Single global operation is computationally efficient
4. **Better Performance**: Often achieves better performance at the same sparsity level compared to layer-wise pruning

**Trade-offs**:
1. **Less Control**: Less direct control over per-layer sparsity compared to layer-wise pruning
2. **Layer Imbalance**: Some layers may end up with different sparsity levels than others

### Training Process

1. **Data Loading**: Loads gene expression data, cell line mappings, and drug response labels
2. **Model Initialization**: Creates fully connected network
3. **Training Loop**:
   - Forward pass through network
   - Compute MSE loss + L1 regularization
   - Backward pass and gradient clipping
   - Optimizer step (AdamW)
   - **Periodic Global Pruning**: Every `prune_frequency` epochs after `warmup` epochs, prune model globally
4. **Validation**: After each epoch, evaluate on validation set
5. **Model Saving**: 
   - Saves `model_best.pt` on validation-loss improvement once `warmup + prune_frequency` epochs have passed
6. **Early Stopping**: Patience-based early stopping on validation loss (counted from the same point)
7. **Final Model Preparation**:
   - Loads best saved model (or current model if no best model saved)
   - Removes pruning masks using `prune.remove()` for all linear layers
   - Overwrites `model_best.pt` with the mask-free weights

---

## File Structure and Coordination

### Core Files

#### 1. `relaxed_global_prune_fc_nn.py`
**Purpose**: Neural network model definition with global pruning

**Key Components**:
- `RelaxedGlobalPrunedFC_NN` class: Main model class
- `prune()` method: Implements global unstructured pruning using `prune.global_unstructured()`
- Architecture definition: Layer sizes, activation functions, BatchNorm

**Key Methods**:
- `prune(sparsity_level)`: Prunes `sparsity_level` of the remaining weights globally using L1-norm
- `forward(X)`: Standard forward pass through the network

**Key Attributes**:
- `parameters_to_prune`: The linear layers (excluding the output layer) passed to `prune.global_unstructured()`

**Dependencies**: None (standalone model)

**Used By**: `relaxed_global_prune_nn_hparam_tuner.py`

---

#### 2. `relaxed_global_prune_nn_hparam_tuner.py`
**Purpose**: Hyperparameter optimization using Optuna

**Key Components**:
- `OptunaRelaxedGlobalPruneNNTrainer` class: Manages hyperparameter tuning
- `setup_trials()`: Defines hyperparameter search space
- `train_model()`: Trains model for each trial with global pruning
- `load_data()`: Loads and preprocesses drug response data

**Hyperparameters Tuned**:
- Learning rate: `1e-5` to `1e-2` (log scale)
- Weight decay: `1e-5` to `1e-2` (log scale, optional)
- L1 regularization: `1e-5` to `1e-2` (log scale, optional)
- Dropout: `0.0` to `0.7` (step 0.1)
- Batch size: `2^2` to `2^5` (4 to 32)
- Activation: Tanh or ReLU
- **Prune frequency**: 1 to 10 epochs
- **Prune percentage**: 0.01 to 0.5 (step 0.01)
- **Warmup**: 0 to 200 epochs before pruning starts
- **BatchNorm**: True or False

**Dependencies**: 
- `relaxed_global_prune_fc_nn.py` (imports `RelaxedGlobalPrunedFC_NN`)
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

Bundled commands are in **`jobs/relaxed_global_prune_nn_warmup_jobs.txt`**. From **`ArchitecturePerformanceExperiments/UGP-NN/`**:

```bash
cd ArchitecturePerformanceExperiments/UGP-NN
grep -v '^#' jobs/relaxed_global_prune_nn_warmup_jobs.txt | grep -v '^$' | bash
```

Use your cluster scheduler for parallel GPU runs. See the **“Running bundled jobs”** section at the top of this README.

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

**Study Name**: `RelaxedGlobalPrunedFC_NN_HTune`

**Storage**: Journal file storage (`RelaxedGlobalPrunedFC_NN_HTune.log`)

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
| `prune_frequency` | int | 1 to 10 | Prune every N epochs |
| `prune_percentage` | float | 0.01 to 0.5 | Step 0.01 |
| `warmup` | int | 0 to 200 | Epochs before pruning starts |
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
- **Frequency**: Every `prune_frequency` epochs after `warmup` epochs
- **Method**: Global unstructured L1-norm pruning (`prune.global_unstructured()`)
- **Amount**: `prune_percentage` of the remaining weights, with no lower bound on the edge count
- **Final Model**: Pruning masks are removed and `model_best.pt` is rewritten with plain weights

Example: If `warmup=0`, `prune_frequency=10` and `prune_percentage=0.1`:
- Epoch 10: Prune 10% of current nonzero weights globally across all layers
- Epoch 20: Prune 10% of remaining nonzero weights globally
- Epoch 30: Prune 10% of remaining nonzero weights globally
- ... continues until training completes

---

## Summary

This system provides a complete pipeline for hyperparameter tuning of Global Pruned Fully Connected Neural Networks:

1. **Model**: `RelaxedGlobalPrunedFC_NN` with iterative global unstructured pruning
2. **Optimization**: Optuna-based hyperparameter search
3. **Execution**: one command per `(drug, experiment)` in `jobs/`

The methodology combines:
- **Architecture Search**: Finding optimal network configuration (activation, BatchNorm, dropout)
- **Global Pruning Strategy**: Iterative weight removal across all layers simultaneously
- **Hyperparameter Optimization**: Automated search for best settings (learning rate, regularization, pruning schedule)

### Key Implementation Details

- **Global Pruning**: Uses `prune.global_unstructured()` to prune across all layers simultaneously, selecting globally least important weights
- **Model Saving**: `model_best.pt` is saved on validation-loss improvement after `warmup + prune_frequency` epochs
- **Final Model**: Pruning masks are removed with `prune.remove()` and `model_best.pt` is rewritten with plain weights
- **Pruning Range**: `prune_percentage` ranges from 0.01 to 0.5

### Methodological Advantages

**Global Pruning Benefits**:
- **Optimal Weight Selection**: Considers entire network when selecting weights to remove
- **Natural Layer Balance**: Allows network to determine optimal per-layer sparsity
- **Computational Efficiency**: Single global operation vs. multiple per-layer operations
- **Better Performance**: Often achieves better performance at same sparsity level
