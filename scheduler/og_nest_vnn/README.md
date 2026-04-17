# NeST VNN Hyperparameter Tuning

This directory contains the hyperparameter tuning framework for NeST VNN (Neural Network with Visible Structure).

## Overview

This implementation:
- **Keeps the original NeST VNN implementation intact** in the `src/` directory
- **Adds Optuna-based hyperparameter tuning** with ranges from `Old_eNest`
- **Uses the distributed scheduler** from `ERK_SNN` for multi-node execution
- **Follows the command structure** from `Old_eNest`
- **Training loop matches the original exactly** (same loss calculation, gradient masking, weight initialization)

## Directory Structure

```
og_nest_vnn/
├── README.md                    # This file
├── nest_vnn_hparam_tuner.py     # Main hyperparameter tuning script
├── generate_jobs.py             # Job generation script
├── distributed_gpu_queue.py     # Distributed GPU queue manager for multi-node
├── src/                         # Original NeST VNN source files (unchanged)
│   ├── drugcell_nn.py           # Neural network model (DrugCellNN)
│   ├── training_data_wrapper.py # Data loading utilities
│   ├── util.py                  # Helper functions
│   └── vnn_trainer.py           # Original VNN trainer
├── shared/                      # Shared filesystem for distributed locking
│   ├── locks/                   # Job lock files
│   └── status/                  # Job status files
├── jobs/                        # Generated job files
│   ├── nest_vnn_jobs.txt        # Simple text job file
│   └── nest_vnn_advanced_jobs.json  # Advanced JSON job file
├── logs/                        # Log files
│   └── distributed_queue_*.log  # Per-node queue logs
└── results/                     # Experiment results
    └── D{drug}/D{drug}_{exp}/   # Per-experiment results
```

## Hyperparameter Ranges

The following hyperparameters are tuned (ranges from `Old_eNest`):

| Hyperparameter | Range | Scale |
|----------------|-------|-------|
| Learning Rate (lr) | 1e-5 to 1e-2 | log |
| Weight Decay (wd) | 0 or 1e-5 to 1e-2 | log |
| L1 Regularization (l1) | 0 or 1e-5 to 1e-2 | log |
| Dropout Fraction | 0.0 to 0.7 | step=0.1 |
| Alpha (aux loss weight) | 0.0 to 1.0 | linear |
| Batch Size Power | 4 to 5 (batch=16 to 32) | int |
| Activation | Tanh, ReLU | categorical |
| Min Dropout Layer | 1 to 4 | int |

**Fixed parameters (matching original NeST VNN):**
- `genotype_hiddens`: 4
- `Max epochs`: 500
- `Early stopping patience`: 20
- `Delta (improvement threshold)`: 0.001

## Training Loop Verification

The training loop in `nest_vnn_hparam_tuner.py` matches the original `vnn_trainer.py` exactly:

1. **Weight Initialization**: `param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1` for direct gene layers, `param.data * 0.1` for others
2. **Loss Calculation**: MSELoss for final output + alpha-weighted MSELoss for auxiliary outputs
3. **Gradient Masking**: Gradients masked for `_direct_gene_layer.weight` parameters
4. **Optimizer**: AdamW with `betas=(0.9, 0.99)`, `eps=1e-05`
5. **Early Stopping**: Based on validation loss improvement with delta threshold

## Usage

### Step 1: Generate Jobs

```bash
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/og_nest_vnn
python generate_jobs.py
```

This creates:
- `jobs/nest_vnn_jobs.txt` - Simple text file with job commands
- `jobs/nest_vnn_advanced_jobs.json` - JSON file with job metadata

### Step 2: Run the Distributed Queue Manager

Run on each compute node (adjust `--max-gpus` and `--node-id` for each node):

```bash
# On L5 (example with 19 GPUs):
nohup python -u distributed_gpu_queue.py jobs/nest_vnn_advanced_jobs.json \
    --max-gpus 19 \
    --node-id l5 \
    --lock-dir shared/locks \
    --status-dir shared/status \
    --log-file logs/distributed_queue_l5.log \
    > logs/distributed_queue_l5.out 2>&1 &

# On L4 (example with 8 GPUs):
nohup python -u distributed_gpu_queue.py jobs/nest_vnn_advanced_jobs.json \
    --max-gpus 8 \
    --node-id l4 \
    --lock-dir shared/locks \
    --status-dir shared/status \
    --log-file logs/distributed_queue_l4.log \
    > logs/distributed_queue_l4.out 2>&1 &

# On L6 (example with 4 GPUs):
nohup python -u distributed_gpu_queue.py jobs/nest_vnn_advanced_jobs.json \
    --max-gpus 4 \
    --node-id l6 \
    --lock-dir shared/locks \
    --status-dir shared/status \
    --log-file logs/distributed_queue_l6.log \
    > logs/distributed_queue_l6.out 2>&1 &
```

### Step 3: Monitor Progress

```bash
# Watch a specific node's log:
tail -f logs/distributed_queue_l5.log

# Check job status:
ls -la shared/status/ | wc -l

# Check completed jobs:
grep -l '"status": "completed"' shared/status/*.json | wc -l
```

### Step 4: Run a Single Experiment (Manual)

```bash
python -u nest_vnn_hparam_tuner.py \
    -drug 188 \
    -onto /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/D188_ontology.txt \
    -train_file /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/train_test_splits/experiment_0/true_training_data.txt \
    -val_file /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/train_test_splits/experiment_0/validation_data.txt \
    -test_file /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/train_test_splits/experiment_0/test_data.txt \
    -cell2id /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/D188_cell2ind.txt \
    -gene2id /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/D188_gene2ind.txt \
    -transcriptomic /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug188/D188_CL/D188_GE_Data.txt \
    -n_trials 100 \
    -seed 42 \
    -output_dir results/D188/D188_0 \
    -cuda 0
```

## Output Format

Each experiment saves:
- `metrics.csv` - Validation and test metrics (Pearson, Spearman, R², Loss)
- `hyperparameters.json` - Best hyperparameters
- `final_results.json` - Complete results summary
- `best_model_results.csv` - Best model R² scores
- `nest_vnn_HTune.log` - Optuna study log
- `best_model/` - Best model checkpoint and hyperparameters
- `trials/trial_N/` - Individual trial results with:
  - `trial_N.log` - Training log (same format as original NeST VNN)
  - `model_best.pt` - Best model weights for this trial
  - `metrics.csv` - Trial metrics
  - `val_predictions.txt`, `test_predictions.txt` - Predictions
  - `val_true_labels.txt`, `test_true_labels.txt` - True labels

## Log Format

The per-trial training log matches the original NeST VNN format:

```
epoch	train_corr	train_loss	true_auc	pred_auc	val_corr	val_loss	grad_norm	elapsed_time
0	0.1234	0.5678	0.4567	0.4321	0.2345	0.6789	12.3456	1.2345
...
```

## Dependencies

- Python 3.8+
- PyTorch
- Optuna
- NumPy
- Pandas
- NetworkX
- scikit-learn
- scipy

## Source Files (from original NeST VNN)

The following files in `src/` are copied from the original implementation and remain unchanged:

- `drugcell_nn.py` - DrugCellNN model with ontology-guided architecture
- `training_data_wrapper.py` - Data loading and preprocessing
- `util.py` - Helper functions (pearson_corr, spearman_corr, build_input_vector, etc.)
- `vnn_trainer.py` - Original VNN trainer (for reference)

## Notes

- The distributed queue manager uses file-based locking to ensure no job runs twice across nodes
- Jobs are automatically skipped if already completed or running
- The queue periodically reloads the job file to pick up new jobs
- Each GPU on each node runs one job at a time
- CUDA_VISIBLE_DEVICES is set per-job to isolate GPU usage
