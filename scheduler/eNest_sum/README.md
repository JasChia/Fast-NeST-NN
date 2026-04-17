# eNest Hyperparameter Tuning Documentation

## Table of Contents
1. [Overview](#overview)
2. [Methodology](#methodology)
3. [File Structure](#file-structure)
4. [Environment Setup](#environment-setup)
5. [Input Requirements](#input-requirements)
6. [Usage Guide](#usage-guide)
7. [Directory Structure](#directory-structure)
8. [Output Files](#output-files)
9. [Hyperparameters](#hyperparameters)

---

## Overview

The **eNest** (Efficient NeST) model is a variant neural network architecture based on the NeST (Nested Structured Topology) framework. It is identical, but runs many times more efficiently by utilizing pytorch hooks and more vectorized operations. Unlike fully connected networks, eNest uses a hierarchical assembly structure derived from biological ontology, where assemblies represent functional groupings of genes. The model processes gene expression data (689 genes) through a structured network that respects biological relationships encoded in an ontology graph.

### Key Features

- **Hierarchical Assembly Structure**: The network architecture follows a biological ontology where genes are organized into assemblies (functional groups)
- **Dual Output System**: The model produces both a final prediction and intermediary assembly-level predictions
- **Multi-Task Learning**: Uses a weighted loss combining final prediction loss and intermediary assembly prediction losses
- **Masked Connections**: Network connections are constrained by the ontology structure, ensuring biological interpretability

---

## Methodology

### Model Architecture

The eNest model consists of:

1. **Input Layer**: 689 gene expression features (fixed input dimension)
2. **Assembly Layers**: Multiple layers where each layer processes assemblies according to the ontology depth
3. **Hidden Assembly Processing**: Intermediate assemblies are processed to produce hidden assembly predictions
4. **Final Output**: The root assembly (NEST) produces the final drug response prediction

### Forward Pass

The forward pass follows this structure:

```
Input (689 genes) 
  → Layer 1 (with masking)
  → Layer 2 (with masking)
  → ...
  → Final Layer (NEST assembly)
```

At each layer (except the final one), the model:
- Applies dropout
- Performs linear transformation
- Applies batch normalization
- Applies activation function (Tanh or ReLU)
- Combines forwarded values with transformed values using masks
- Extracts final assembly states for that layer

The final layer produces:
- **Final Output (Y)**: Shape `[batch_size, 1]` - Single prediction from the root assembly
- **Hidden Assembly Output (hidden_asm_Y)**: Shape `[batch_size, hidden_assemblies]` - Predictions from all non-root assemblies

### Loss Function Methodology

The eNest model uses a **dual-loss training strategy** that is central to its methodology:

```
Training Loss = α × Loss(intermediary_assemblies, labels) + Loss(final_assembly, labels) + L1_regularization
```

Where:
- **α (alpha)**: Hyperparameter (0.0 to 1.0) that controls the weight of intermediary assembly supervision
- **Loss(intermediary_assemblies, labels)**: MSE loss between all hidden assembly predictions and the true label
- **Loss(final_assembly, labels)**: MSE loss between the final prediction and the true label
- **L1_regularization**: Optional L1 regularization on model parameters

#### Why This Loss Function?

1. **Hierarchical Supervision**: By supervising intermediary assemblies, the model learns meaningful representations at each level of the biological hierarchy
2. **Multi-Scale Learning**: The model learns to predict drug response at multiple scales (individual assemblies and final output)
3. **Improved Generalization**: Intermediate supervision acts as a form of regularization, encouraging the network to learn biologically meaningful features
4. **Tunable Balance**: The `alpha` hyperparameter allows balancing between:
   - `α = 0.0`: Only final prediction is supervised (standard approach)
   - `α = 1.0`: Equal weight to all assembly predictions and final prediction
   - Intermediate values: Gradual weighting between the two extremes

#### Evaluation Methodology

**Important**: During validation and evaluation, **only the final prediction loss is used** for model selection and reporting. This ensures:
- Fair comparison across different alpha values
- Focus on the primary task (final drug response prediction)
- Consistency in evaluation metrics

The intermediary assembly predictions are used **only during training** to guide learning, but the model is ultimately evaluated on its final prediction quality.

### Activation Function Constraint

The model requires activation functions that return 0 for input 0. This is critical because:
- Masked connections are set to zero
- The model relies on activation(0) = 0 to maintain proper masking behavior
- Currently supported: `nn.Tanh` and `nn.ReLU` (both satisfy this property)

---

## File Structure

### Core Files

#### `eNest.py`
The main model definition file containing:
- **`eNest` class**: The neural network model
  - `__init__()`: Initializes the model with ontology-based architecture
  - `forward()`: Performs forward pass, returns `(final_pred, hidden_asm_pred)`
  - `register_grad_hooks()`: Registers gradient masking hooks to maintain sparsity
- **`setup_Nest_Masks` class**: Static methods for building the network structure
  - `load_ontology()`: Loads the ontology graph and gene mappings
  - `calculate_depths()`: Computes topological depth of each node
  - `make_layer_masks()`: Creates connection masks for each layer
  - `node_to_idx_dict()`: Maps ontology nodes to network indices

**Key Parameters:**
- `nodes_per_assembly`: Number of neurons per assembly (currently fixed at 4 via `genotype_hiddens`)
- `dropout`: Dropout rate (0.0-0.7)
- `activation`: Activation function (`nn.Tanh` or `nn.ReLU`)
- `verbosity`: Verbosity level for debugging

#### `eNest_hparam_tuner.py`
Optuna-based hyperparameter tuning script that:
- Loads and preprocesses drug response data
- Runs hyperparameter optimization trials
- Trains models with different hyperparameter combinations
- Evaluates and saves best models
- Generates comprehensive metrics and reports

**Key Classes:**
- `OptunaENestTrainer`: Main trainer class
  - `setup_trials()`: Defines hyperparameter search space
  - `train_model()`: Trains a single trial
  - `load_data()`: Loads and preprocesses data
  - `evaluate_model_metrics()`: Computes evaluation metrics

**Key Functions:**
- `load_mapping()`: Loads cell line to index mappings
- `load_train_data()`: Loads training and validation data
- `build_input_vector()`: Converts cell IDs to gene expression vectors
- `evaluate_model_metrics()`: Computes Pearson, Spearman, R², and MSE

#### `generate_jobs.py`
Job generation script that creates batch job files for GPU queue execution.

**Functions:**
- `generate_enest_jobs()`: Creates `jobs/eNest_jobs.txt` (simple text format)
- `generate_enest_jobs_json()`: Creates `jobs/eNest_advanced_jobs.json` (JSON format with metadata)

#### `advanced_gpu_queue.py`
GPU queue manager that:
- Monitors GPU availability
- Distributes jobs across available GPUs
- Manages job execution and logging
- Handles job completion and failures
- Saves execution results

---

## Environment Setup

### Required Environment

All scripts should be run in the **`cuda11_env`** conda environment:

```bash
conda activate cuda11_env
```

### Required Python Packages

The following packages are required (install via conda/pip):
- `torch` (PyTorch with CUDA support)
- `optuna` (hyperparameter optimization)
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `networkx`

### Required Data Files

The model expects the following data files in `/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData`:
- `red_ontology.txt`: Ontology graph defining assembly relationships
- `red_gene2ind.txt`: Gene to index mapping file

---

## Input Requirements

### Command Line Arguments for `eNest_hparam_tuner.py`

**Required Arguments:**
- `-train_file`: Path to training data file (tab-separated, format: `cell_line\tSMILES\tAUC\tdataset`)
- `-val_file`: Path to validation data file (same format)
- `-test_file`: Path to test data file (same format)
- `-cell2id`: Path to cell line to index mapping file
- `-ge_data`: Path to gene expression data file (CSV format, no header)

**Optional Arguments:**
- `-cuda`: GPU device ID (default: 0)
- `-drug`: Drug ID for logging/identification (default: -1)
- `-n_trials`: Number of Optuna trials (default: 100)
- `-seed`: Random seed for reproducibility (default: 42)
- `-output_dir`: Output directory for results (default: `./results`)

### Data File Formats

#### Training/Validation/Test Files
Tab-separated format with columns: `cell_line`, `SMILES`, `AUC`, `dataset`
- Column 0: Cell line identifier (string)
- Column 1: SMILES string (for training/val) or AUC value (for test)
- Column 2: AUC value (for training/val) or dataset (for test)
- Column 3: Dataset identifier

**Important Note on Data Format:**
- **Training/Validation files**: The code reads `row[1]` as the AUC label (despite the column being named 'smiles' in the code)
- **Test files**: The code reads `row[2]` as the AUC label (column named 'auc')
- This appears to be a legacy format difference. Ensure your data files match this format.

#### Cell Line Mapping File
Format: `index\tcell_line_name`
- Maps cell line names to integer indices

#### Gene Expression Data File
CSV format (comma-separated), no header
- Rows: Cell lines (one per row)
- Columns: Gene expression values (689 genes)
- Each row corresponds to a cell line index from the mapping file

---

## Usage Guide

### Step 1: Generate Job Files

Navigate to the eNest directory and generate job files:

```bash
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/eNest
conda activate cuda11_env
python generate_jobs.py
```

This creates:
- `jobs/eNest_jobs.txt`: Simple text format job list
- `jobs/eNest_advanced_jobs.json`: JSON format with metadata

### Step 2: Run Hyperparameter Tuning (Single Job)

For a single experiment:

```bash
conda activate cuda11_env
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/eNest

python -u eNest_hparam_tuner.py \
    -drug 5 \
    -train_file /path/to/train_data.txt \
    -val_file /path/to/val_data.txt \
    -test_file /path/to/test_data.txt \
    -cell2id /path/to/cell2ind.txt \
    -ge_data /path/to/GE_Data.txt \
    -n_trials 100 \
    -seed 42 \
    -output_dir results/D5/D5_0 \
    -cuda 0
```

### Step 3: Run Batch Jobs with GPU Queue Manager

For running multiple experiments in parallel:

```bash
conda activate cuda11_env
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/eNest

# Run in background with nohup
nohup python advanced_gpu_queue.py \
    jobs/eNest_advanced_jobs.json \
    --max-gpus 16 \
    --log-file logs/gpu_queue_manager.log \
    > logs/queue_output.log 2>&1 &

# Monitor progress
tail -f logs/gpu_queue_manager.log
tail -f logs/queue_output.log
```

### Step 4: Monitor Results

Check individual experiment results:

```bash
# View experiment log
tail -f results/D5/D5_0/D5_0.log

# View hyperparameter tuning log
tail -f results/D5/D5_0/eNest_HTune.log

# View trial results
ls results/D5/D5_0/trials/
cat results/D5/D5_0/trials/trial_0/trial_0.log
```

---

## Directory Structure

After running experiments, the directory structure will be:

```
scheduler/eNest/
├── eNest.py                    # Model definition
├── eNest_hparam_tuner.py       # Hyperparameter tuning script
├── generate_jobs.py            # Job generation script
├── advanced_gpu_queue.py       # GPU queue manager
├── jobs/
│   ├── eNest_jobs.txt          # Simple job list
│   └── eNest_advanced_jobs.json # JSON job file with metadata
├── logs/
│   ├── gpu_queue_manager.log   # Queue manager log
│   ├── queue_output.log         # Queue output
│   └── queue_results.json      # Job execution results
└── results/
    └── D{drug_id}/
        └── D{drug_id}_{experiment_id}/
            ├── D{drug_id}_{experiment_id}.log      # Experiment log
            ├── eNest_HTune.log                      # Optuna study log
            ├── best_model_results.csv               # Best model metrics
            ├── final_results.json                   # Final results summary
            ├── study_summary.txt                    # Study summary
            ├── best_model/                          # Best model directory
            │   ├── model_best.pt                    # Best model weights
            │   ├── metrics.csv                      # Best model metrics
            │   ├── trial_{trial_num}.log            # Trial log
            │   └── save.log                         # Save information
            └── trials/
                └── trial_{trial_num}/
                    ├── trial_{trial_num}.log        # Trial log
                    ├── model_best.pt                 # Best model for this trial
                    ├── model_final.pt                # Final model state
                    ├── trial_summary.json            # Trial summary
                    ├── detailed_metrics.json         # Detailed metrics
                    ├── hyperparameters.txt           # Hyperparameters
                    ├── metrics.csv                   # Metrics CSV
                    ├── val_predictions.txt           # Validation predictions
                    ├── val_true_labels.txt           # Validation true labels
                    ├── test_predictions.txt          # Test predictions
                    └── test_true_labels.txt          # Test true labels
```

---

## Output Files

### Main Results Files

#### `final_results.json`
Final summary of the hyperparameter tuning study:
```json
{
  "best_validation_r2": 0.85,
  "best_test_r2": 0.82,
  "best_hyperparameters": {...},
  "total_trials": 100,
  "drug_id": 5,
  "seed": 42
}
```

#### `best_model_results.csv`
CSV file with best model performance:
```csv
Test R2, Val R2
0.82, 0.85
```

#### `study_summary.txt`
Human-readable summary of the study with best hyperparameters.

#### `eNest_HTune.log`
Optuna journal file containing the complete study history (can be loaded to resume or analyze).

### Trial-Specific Files

Each trial directory contains:
- **Model weights**: `model_best.pt`, `model_final.pt`
- **Metrics**: `metrics.csv`, `detailed_metrics.json`
- **Predictions**: `val_predictions.txt`, `test_predictions.txt`
- **Hyperparameters**: `hyperparameters.txt`, `trial_summary.json`

---

## Hyperparameters

### Tunable Hyperparameters

The hyperparameter tuning script searches over:

1. **`alpha`** (0.0-1.0, continuous)
   - Weight for intermediary assembly loss
   - Critical for the dual-loss methodology
   - 0.0 = only final prediction supervised
   - 1.0 = equal weight to all assemblies
   - **Note**: Currently sampled continuously (not stepped), despite documentation suggesting step=0.1

3. **`dropout_fraction`** (0.0-0.7, step=0.1)
   - Dropout rate for regularization
   - Applied at each layer

4. **`lr`** (1e-5 to 1e-2, log scale)
   - Learning rate for AdamW optimizer

5. **`wd`** (1e-5 to 1e-2, log scale) or `False`
   - Weight decay for L2 regularization
   - Optional (can be disabled)

6. **`l1`** (1e-5 to 1e-2, log scale) or `False`
   - L1 regularization coefficient
   - Optional (can be disabled)

7. **`activation`** (`Tanh` or `ReLU`)
   - Activation function
   - Must return 0 for input 0 (enforced by assert)

8. **`batch_size_power`** (2-5, step=1)
   - Batch size = 2^batch_size_power
   - Results in batch sizes: 4, 8, 16, 32

### Fixed Hyperparameters

- **`nodes_per_assembly`**: 4 (fixed via `genotype_hiddens`, not currently tunable)
- **`epochs`**: 500 (maximum training epochs)
- **`patience`**: 20 (early stopping patience)
- **Optimizer**: AdamW with betas=(0.9, 0.99), eps=1e-05
- **Gradient clipping**: max_norm=1.0

### Model Selection

The best model is selected based on **validation R² score** (not validation loss). This ensures:
- Focus on explained variance rather than absolute error
- Better alignment with biological interpretation
- Scale-independent evaluation

---

## Methodology Details

### Assembly Processing

The model processes assemblies in topological order based on the ontology:

1. **Gene Layer (Depth 0)**: Input genes (689 genes)
2. **Intermediate Layers (Depth 1 to max_depth-1)**: Hidden assemblies
3. **Root Layer (max_depth)**: Final NEST assembly

At each layer, assemblies:
- Receive inputs from their parent assemblies (or genes)
- Perform linear transformation with masked connections
- Apply batch normalization and activation
- Forward values to child assemblies or produce final output

### Masking Strategy

Three types of masks control information flow:

1. **Layer Masks (`layer_masks`)**: Connections allowed by ontology structure
2. **Forward Layer Masks (`forward_layer_masks`)**: Identity connections for forwarding values
3. **Final Assembly Layer Masks (`final_asm_layer_masks`)**: Marks where assemblies have accumulated all inputs

### Gradient Masking

Gradient hooks ensure that masked connections remain zero during backpropagation:
- Weights are masked after each optimizer step
- Gradients are masked during backpropagation
- Maintains sparsity throughout training

---

## Example Workflow

### Complete Example: Running Drug 5 Experiments

```bash
# 1. Activate environment
conda activate cuda11_env

# 2. Navigate to eNest directory
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/eNest

# 3. Generate job files (if not already done)
python generate_jobs.py

# 4. Run single experiment (for testing)
python -u eNest_hparam_tuner.py \
    -drug 5 \
    -train_file /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/true_training_data.txt \
    -val_file /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/validation_data.txt \
    -test_file /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/test_data.txt \
    -cell2id /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/D5_cell2ind.txt \
    -ge_data /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/D5_GE_Data.txt \
    -n_trials 100 \
    -seed 42 \
    -output_dir results/D5/D5_0 \
    -cuda 0

# 5. Or run batch jobs
nohup python advanced_gpu_queue.py jobs/eNest_advanced_jobs.json --max-gpus 16 --log-file logs/gpu_queue_manager.log > logs/queue_output.log 2>&1 &

# 6. Monitor progress
tail -f logs/gpu_queue_manager.log
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `cuda11_env` is activated and all packages are installed
2. **CUDA Out of Memory**: Reduce batch size or `nodes_per_assembly` (currently fixed at 4)
3. **Assertion Error on Activation**: Only use `nn.Tanh` or `nn.ReLU`
4. **File Not Found**: Check that data files exist and paths are correct
5. **GPU Not Available**: Check `nvidia-smi` and ensure CUDA is properly configured

### Debugging

- Set `verbosity > 0` in model initialization for debug output
- Check individual trial logs in `trials/trial_X/trial_X.log`
- Review `queue_results.json` for job execution status
- Check `eNest_HTune.log` for Optuna study details

---

## Notes

- The model architecture is fixed based on the ontology (131 assemblies, 689 genes)
- All paths are relative to the `scheduler/eNest` directory when running scripts
- The GPU queue manager automatically handles CUDA_VISIBLE_DEVICES to limit PyTorch to 16 GPUs
- Results are saved incrementally, so interrupted jobs can be resumed
- Model selection is based on validation R², not validation loss

