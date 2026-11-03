# Sparse Neural Network Hyperparameter Tuning System

## Running bundled jobs (this repository)

### Environment and CLI (current code)

Activate the Conda env from root **`environment.yml`** (`conda activate fast-nest-nn`).

**`r_sparse_nn_hparam_tuner.py`** supports **`-n_trials`**, **`-max_epochs`** (default **500**; use **`-max_epochs 10`** for a quick run), **`-cuda`**, **`-train_file`**, **`-val_file`**, **`-test_file`**, **`-cell2id`**, **`-ge_data`**, **`-output_dir`**.

**Expected outputs:** **`Sparse_NN_HTune.log`**, **`trials/trial_*/model_best.pt`**, **`best_model/`**, **`final_results.json`**, per-trial **`metrics.csv`**. (The tuner saves a single **`model_best.pt`**.)

## Overview

This directory contains a complete system for hyperparameter tuning of sparse neural networks (Sparse_NN) for drug response prediction. The system uses Optuna for automated hyperparameter optimization.

## Methodology

### Experimental Design

The system follows a rigorous experimental methodology:

1. **10 Drugs**: The system processes 10 different drugs: `[5, 80, 99, 127, 151, 188, 244, 273, 298, 380]`

2. **50 Experiments Per Drug**: Each drug undergoes **exactly 50 independent experiments**, where each experiment uses:
   - A different train/validation/test split (experiment_0 through experiment_49)
   - A unique random seed (`seed = experiment_id * 1000`)
   - The same sparse neural network architecture configuration

3. **100 Hyperparameter Trials Per Experiment**: For each of the 50 experiments per drug, Optuna performs 100 hyperparameter optimization trials, exploring different combinations of:
   - Learning rate (1e-5 to 1e-2, log scale)
   - Weight decay (1e-5 to 1e-2, log scale, optional)
   - L1 regularization (1e-5 to 1e-2, log scale, optional)
   - Dropout fraction (0.0 to 0.7, step 0.1)
   - Batch size (2^2 to 2^5, i.e., 4, 8, 16, 32)
   - Activation function (Tanh or ReLU)

4. **Total Scale**: 
   - 10 drugs × 50 experiments = **500 total experiments**
   - 500 experiments × 100 trials = **50,000 hyperparameter optimization trials**

### Sparse Neural Network Architecture

The `Sparse_NN` model uses a hierarchical sparse architecture inspired by Nest VNN:

- **Input Layer**: 689 nodes (gene expression features)
- **Hidden Layers**: 8 layers with decreasing node counts
  - Layer 1: 76 × `genotype_hiddens` nodes (default: 304 nodes)
  - Layer 2: 32 × `genotype_hiddens` nodes (default: 128 nodes)
  - Layer 3: 13 × `genotype_hiddens` nodes (default: 52 nodes)
  - Layer 4: 4 × `genotype_hiddens` nodes (default: 16 nodes)
  - Layer 5: 2 × `genotype_hiddens` nodes (default: 8 nodes)
  - Layer 6: 2 × `genotype_hiddens` nodes (default: 8 nodes)
  - Layer 7: 1 × `genotype_hiddens` nodes (default: 4 nodes)
  - Layer 8: 1 × `genotype_hiddens` nodes (default: 4 nodes)
- **Output Layer**: 1 node (drug response prediction)

**Sparsity**: Each layer has a predefined number of edges (connections) that is much smaller than a fully connected network:
- Layer 0: 1321 × `genotype_hiddens` edges (default: 5,284 edges)
- Layer 1: 92 × `genotype_hiddens`² edges (default: 1,472 edges)
- Layer 2: 36 × `genotype_hiddens`² edges (default: 576 edges)
- And so on...

The sparse connectivity is enforced using `torch.nn.utils.prune`, which:
1. Randomly prunes connections to achieve the target edge count
2. Ensures each node has at least one input and one output connection ("alive" nodes)
3. Regenerates the network if nodes become "dead" (up to 10,000 attempts)
4. Maintains sparsity during training through pruning masks

### Reproducibility

Each experiment maintains reproducibility through:
- **Experiment-level seed**: `seed = experiment_id * 1000` (e.g., experiment 0 uses seed 0, experiment 1 uses seed 1000)
- **Trial-level seed**: `trial_seed = trial.number + experiment_seed` ensures each Optuna trial within an experiment has a unique but reproducible seed
- **Sparse network generation**: The seed is used when generating the random sparse connectivity masks

## System Architecture

### Workflow

```
jobs/r_sparse_nn_jobs.txt  (10 drugs × 50 experiments)
        │  one line per job
        ▼
r_sparse_nn_hparam_tuner.py  (Optuna, 100 trials per experiment)
        ▼
results/D{drug}/D{drug}_{i}/
```

## What this repository contains (RSNN/)

**Local-only (created when you run experiments; gitignored):** `results/` and the per-job `D{drug}_{i}.log` files written by the job lines.

## Usage (current)

1. `cd <path/to/Fast-NeST-NN>/ArchitecturePerformanceExperiments/RSNN`
2. Ensure **`Data/`** at the repo root matches paths in **`jobs/`** (see root **`README.md`**), e.g. `Data/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/...`.
3. Run lines from **`jobs/r_sparse_nn_jobs.txt`** (or submit them to your cluster).

## File structure (as shipped)

```text
ArchitecturePerformanceExperiments/RSNN/
├── README.md
├── r_sparse_nn_hparam_tuner.py
├── sparse_nn.py
├── cleanup_failed_experiment.py
├── test_sparsity_training.ipynb
└── jobs/
    └── r_sparse_nn_jobs.txt
```

After runs, **`results/`** and the per-job logs appear locally (ignored by Git).

---

## Output Structure

### Per-Experiment Directory

Each experiment (e.g., `results/D5/D5_0/`) contains:

1. **`Sparse_NN_HTune.log`**: Optuna study log
2. **`study_summary.txt`**: Summary of best hyperparameters and results
3. **`final_results.json`**, **`best_model_results.csv`**: Best validation and test R²
4. **`best_model/`**: Copy of the best trial's directory
5. **`trials/`**: Directory containing individual trial results
   - `trial_{n}/trial_{n}.log`: Trial-specific log
   - `trial_{n}/model_best.pt`: Best model checkpoint (lowest validation loss)
   - `trial_{n}/metrics.csv`: Validation and test metrics of the best checkpoint

### Study Summary Format

The `study_summary.txt` file contains:
- Drug ID and seed
- Best validation R² and best test R²
- Total number of trials
- Best hyperparameters (learning rate, dropout, activation, etc.)

## Technical Details

### Sparse Network Generation

The sparse network is generated using `torch.nn.utils.prune`:

1. **Initial Pruning**: Uses `prune.random_unstructured()` to randomly prune connections
2. **Mask Extraction**: Extracts pruning masks from temporary Linear layers
3. **Node Aliveness Check**: Verifies each node has at least one input and one output connection
4. **Regeneration**: If nodes are dead, samples a new network (up to 10,000 attempts)
5. **Mask Application**: Uses `prune.custom_from_mask()` to apply masks to actual network layers

### Hyperparameter Search Space

Optuna explores the following hyperparameter space:

| Hyperparameter | Range/Options | Type |
|---------------|---------------|------|
| Learning Rate | 1e-5 to 1e-2 | Log-uniform float |
| Weight Decay | 1e-5 to 1e-2 (optional) | Log-uniform float |
| L1 Regularization | 1e-5 to 1e-2 (optional) | Log-uniform float |
| Dropout Fraction | 0.0 to 0.7 | Uniform float (step 0.1) |
| Batch Size | 4, 8, 16, 32 | Integer (2^2 to 2^5) |
| Activation | Tanh, ReLU | Categorical |
| Genotype Hiddens | 4 (fixed) | Integer |

**Note**: `genotype_hiddens` is currently fixed at 4 but can be modified in the code.

### Training Details

- **Optimizer**: AdamW with betas=(0.9, 0.99), eps=1e-05
- **Loss Function**: MSE Loss + L1 Regularization (if enabled)
- **Epochs**: 500 maximum per trial
- **Early stopping**: patience 20 epochs on validation loss
- **Gradient Clipping**: Max norm of 1.0 to prevent gradient explosion
- **Device**: CUDA (GPU) if available, CPU otherwise

## Expected structure

### Experiment Structure

Each experiment has:
- A unique seed: `experiment_id * 1000`
- A unique data split: `experiment_{experiment_id}/`
- A unique output directory: `results/D{drug}/D{drug}_{experiment_id}/`

### Sparse Network

The sparse network has:
- Exact edge counts matching `nest_edges_by_layer`
- All nodes "alive" (at least one input and one output connection)
- Pruning masks maintained during training (via `torch.prune`)

## References

- **Optuna**: Hyperparameter optimization framework (https://optuna.org/)
- **PyTorch Pruning**: `torch.nn.utils.prune` for maintaining sparsity
- **Nest VNN**: Original architecture inspiration with hierarchical sparse connectivity

## Loading a trained model

When you load a sparse architecture, the initial seed doesn't matter before loading. All parameters will be replaced.
