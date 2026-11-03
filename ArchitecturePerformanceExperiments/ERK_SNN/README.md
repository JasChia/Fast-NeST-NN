# ERK_SNN (Erdos-Renyi Kernel Sparse Neural Network) Hyperparameter Tuning System

**Shared conventions:** **[`../README.md`](../README.md)** (data, jobs).

## Running bundled jobs (this repository)

Use **`jobs/ERK_SNN_jobs.txt`** from **`ArchitecturePerformanceExperiments/ERK_SNN/`** (one shell command per line; skip `#` comments). Paths use **`../../Data/...`** relative to this directory. Run lines with your cluster scheduler (Slurm, GNU Parallel, etc.).

### Environment and CLI (current code)

Use the root **`environment.yml`** (see repo **`README.md`**; `conda activate fast-nest-nn`).

**`ERK_SNN_hparam_tuner.py`**: **`-n_trials`**, **`-max_epochs`** (default **500**), **`-cuda`**, standard data arguments as in **`jobs/ERK_SNN_jobs.txt`**.

**Expected outputs:** **`ERK_SNN_HTune.log`**, **`trials/trial_*/model_best.pt`**, **`best_model/`**, **`final_results.json`**.

## Overview

This directory contains a complete system for hyperparameter tuning of ERK_SNN (Erdos-Renyi Kernel Sparse Neural Network) for drug response prediction. The system uses Optuna for automated hyperparameter optimization

**ERK Initialization**: The network uses Erdos-Renyi Kernel (ERK) initialization, which assigns layer-wise edge probabilities based on layer dimensions. This creates a more structured sparse topology compared to uniform random initialization, with larger layers naturally having higher sparsity.

## Methodology

### Experimental Design

The system follows a rigorous experimental methodology:

1. **10 Drugs**: The system processes 10 different drugs: `[5, 80, 99, 127, 151, 188, 244, 273, 298, 380]`

2. **50 Experiments Per Drug**: Each drug undergoes **exactly 50 independent experiments**, where each experiment uses a different train/validation/test split and a different random seed.

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

### ERK_SNN Architecture

The `ERK_SNN` model uses a hierarchical sparse architecture inspired by Nest VNN with ERK initialization:

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

**Sparsity**: ERK densities are scaled to a target overall sparsity equal to the NeST network's, `1 - sum(nest_edges_by_layer) / total fully connected edges`; ERK decides how the edges are spread across layers.

**ERK Initialization**: The sparse connectivity is initialized using Erdos-Renyi Kernel (ERK) probabilities:
1. **ERK Density Calculation**: For each layer, ERK density is calculated as `d_l = (n_{l-1} + n_l) / (n_{l-1} * n_l)`, where `n_l` is the number of nodes in layer `l`
2. **Scaling**: ERK densities are scaled (capped at 1.0) toward the target overall sparsity (derived from NeST edge counts)
3. **Edge Sampling**: For each layer, `max(1, int(d'_l · n_{l-1} · n_l))` edges are chosen uniformly at random, where `d'_l` is the scaled, capped density
4. **Node Aliveness**: Ensures each node has at least one input and one output connection ("alive" nodes)
5. **Regeneration**: Regenerates the network if nodes become "dead" (up to 10,000 attempts)
6. **Sparsity Maintenance**: Maintains sparsity during training through pruning masks

### Reproducibility

Each experiment maintains reproducibility through:
- **Experiment-level seed**: `seed = experiment_id * 1000` (e.g., experiment 0 uses seed 0, experiment 1 uses seed 1000)
- **Trial-level seed**: `trial_seed = trial.number + experiment_seed` ensures each Optuna trial within an experiment has a unique but reproducible seed
- **Sparse network generation**: The seed is used when generating the ERK-initialized sparse connectivity masks

## System Architecture

### Workflow

Bundled commands live in **`jobs/ERK_SNN_jobs.txt`**. Each line invokes **`ERK_SNN_hparam_tuner.py`** with paths under **`../../Data/`** (see repository root `README.md`). Run those lines with your cluster scheduler (Slurm, GNU Parallel, etc.). Outputs land under **`results/`**.

## Usage Instructions

### Prerequisites

1. Work from this directory:
   ```bash
   cd ArchitecturePerformanceExperiments/ERK_SNN
   ```

2. Ensure drug bundles exist under **`Data/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/`** at the repo root (from **`Data_archives/`**).

### Step 1: Run bundled jobs

```bash
grep -v '^#' jobs/ERK_SNN_jobs.txt | grep -v '^$' | bash
```

(Prefer submitting each line as a separate GPU job on a real cluster.)

### Step 2: Monitor Progress

```bash
tail -f D*_*.log
```

### Step 3: Check Results

Results are organized by drug and experiment:

```bash
# View results for a specific drug/experiment
ls -la results/D5/D5_0/

# View final results
cat results/D5/D5_0/final_results.json

# Check trial results
ls results/D5/D5_0/trials/
```

## File Structure

```
ERK_SNN/
├── README.md                       # This file
├── ERK_SNN_hparam_tuner.py         # Hyperparameter tuning script
├── ERK_SNN.py                      # ERK_SNN model definition
├── jobs/                           # Bundled job command lists
│   └── ERK_SNN_jobs.txt
├── D{drug}_{experiment}.log        # Per-job stdout (when you run experiments)
└── results/                        # Experiment results
    └── D{drug}/                    # Per-drug directory
        └── D{drug}_{experiment}/    # Per-experiment directory
            ├── ERK_SNN_HTune.log
            ├── final_results.json
            ├── best_model_results.csv
            ├── best_model/
            └── trials/              # Optuna trial results
                └── trial_{n}/
                    ├── trial_{n}.log
                    ├── model_best.pt
                    └── metrics.csv
```

## Output Structure

### Per-Experiment Directory

Each experiment (e.g., `results/D5/D5_0/`) contains:

1. **`ERK_SNN_HTune.log`**: Optuna study log
2. **`final_results.json`**: Best validation and test R², best hyperparameters, trial count, drug ID and seed
3. **`best_model_results.csv`**: Best test and validation R²
4. **`best_model/`**: Copy of the best trial's directory plus `hyperparameters.json`
5. **`trials/`**: Directory containing individual trial results
   - `trial_{n}/trial_{n}.log`: Trial-specific log
   - `trial_{n}/model_best.pt`: Best model checkpoint (lowest validation loss)
   - `trial_{n}/metrics.csv`: Validation and test metrics of the best checkpoint

## Technical Details

### ERK Network Generation

The sparse network is generated using ERK (Erdos-Renyi Kernel) initialization:

1. **ERK Density Calculation**: For each layer, calculates ERK density based on layer dimensions: `d_l = (n_{l-1} + n_l) / (n_{l-1} * n_l)`
2. **Sparsity Scaling**: Scales ERK densities toward the target overall sparsity (derived from NeST edge counts)
3. **Edge Sampling**: For each layer, `max(1, int(d'_l · n_{l-1} · n_l))` edges are chosen uniformly at random, where `d'_l` is the scaled, capped density
4. **Node Aliveness Check**: Verifies each node has at least one input and one output connection
5. **Regeneration**: If nodes are "dead", samples a new network (up to 10,000 attempts)
6. **Mask Application**: Uses `torch.nn.utils.prune.custom_from_mask()` to apply masks to Linear layers

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
- Per-layer edge counts set by the scaled ERK densities
- All nodes "alive" (at least one input and one output connection)
- Pruning masks maintained during training (via `torch.prune`)

## References

- **Optuna**: Hyperparameter optimization framework (https://optuna.org/)
- **PyTorch Pruning**: `torch.nn.utils.prune` for maintaining sparsity
- **Nest VNN**: Original architecture inspiration with hierarchical sparse connectivity

## Loading a trained model

When you load a sparse architecture, the initial seed doesn't matter before loading. All parameters will be replaced.
