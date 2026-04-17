# Sparse Neural Network Hyperparameter Tuning System

## Overview

This directory contains a complete system for hyperparameter tuning of sparse neural networks (Sparse_NN) for drug response prediction. The system uses Optuna for automated hyperparameter optimization and manages GPU resources to run hundreds of experiments in parallel.

## Methodology

### Experimental Design

The system follows a rigorous experimental methodology:

1. **12 Drugs**: The system processes 12 different drugs: `[5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380]`

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
   - 12 drugs × 50 experiments = **600 total experiments**
   - 600 experiments × 100 trials = **60,000 hyperparameter optimization trials**

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

### Component Overview

The system consists of three main components:

1. **`generate_jobs.py`**: Generates job files for all experiments
2. **`advanced_gpu_queue.py`**: Manages GPU resources and executes jobs in parallel
3. **`r_sparse_nn_hparam_tuner.py`**: Performs hyperparameter optimization for a single experiment
4. **`sparse_nn.py`**: Defines the Sparse_NN model architecture

### Workflow

```
┌─────────────────┐
│ generate_jobs.py│  Creates 600 job configurations
│                 │  (12 drugs × 50 experiments)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ jobs/               │
│ - r_sparse_nn_      │
│   advanced_jobs.json│
│ - r_sparse_nn_      │
│   jobs.txt          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ advanced_gpu_queue  │  Manages GPU allocation
│ .py                 │  Executes jobs in parallel
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ r_sparse_nn_        │  Runs Optuna optimization
│ hparam_tuner.py     │  (100 trials per experiment)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ results/            │  Stores outputs:
│ D{drug}/D{drug}_{i}/│  - Best models
│   - trials/         │  - Metrics
│   - study_summary   │  - Logs
└─────────────────────┘
```

## Usage Instructions

### Prerequisites

1. Ensure you're in the correct directory:
   ```bash
   cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/r_sparse_nn
   ```

2. Verify data files exist at:
   ```
   /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/
   ```

### Step 1: Generate Job Files

Generate job configuration files for all experiments:

```bash
python generate_jobs.py
```

This creates:
- `jobs/r_sparse_nn_jobs.txt`: Simple text file with commands
- `jobs/r_sparse_nn_advanced_jobs.json`: JSON file with detailed job metadata

**Output**: 600 jobs total (12 drugs × 50 experiments)

### Step 2: Start GPU Queue Manager

Start the GPU queue manager to execute jobs:

```bash
nohup python advanced_gpu_queue.py jobs/r_sparse_nn_advanced_jobs.json --max-gpus 16 --log-file logs/gpu_queue_manager.log > logs/queue_output.log 2>&1 &
```

**Parameters**:
- `--max-gpus 16`: Maximum number of GPUs to use simultaneously
- `--log-file logs/gpu_queue_manager.log`: Path for queue manager logs
- `> logs/queue_output.log 2>&1 &`: Redirects output and runs in background

### Step 3: Monitor Progress

Monitor the queue manager:

```bash
# View queue manager log
tail -f logs/gpu_queue_manager.log

# View queue output
tail -f logs/queue_output.log

# Check running jobs
ps aux | grep advanced_gpu_queue
```

### Step 4: Check Results

Results are organized by drug and experiment:

```bash
# View results for a specific drug/experiment
ls -la results/D5/D5_0/

# View study summary
cat results/D5/D5_0/study_summary.txt

# Check trial results
ls results/D5/D5_0/trials/
```

## File Structure

```
r_sparse_nn/
├── README.md                      # This file
├── generate_jobs.py                # Job generator script
├── advanced_gpu_queue.py           # GPU queue manager
├── r_sparse_nn_hparam_tuner.py     # Hyperparameter tuning script
├── sparse_nn.py                    # Sparse_NN model definition
├── test_sparsity_training.ipynb    # Testing notebook
├── jobs/                           # Generated job files
│   ├── r_sparse_nn_jobs.txt
│   └── r_sparse_nn_advanced_jobs.json
├── logs/                           # Log files
│   ├── gpu_queue_manager.log
│   └── queue_output.log
└── results/                        # Experiment results
    └── D{drug}/                    # Per-drug directory
        └── D{drug}_{experiment}/    # Per-experiment directory
            ├── D{drug}_{experiment}.log
            ├── Sparse_NN_HTune.log
            ├── study_summary.txt
            └── trials/              # Optuna trial results
                └── trial_{n}/
                    ├── trial_{n}.log
                    ├── model_best.pt
                    ├── model_final.pt
                    └── metrics.csv
```

## Output Structure

### Per-Experiment Directory

Each experiment (e.g., `results/D5/D5_0/`) contains:

1. **`D{drug}_{experiment}.log`**: Main experiment log file
2. **`Sparse_NN_HTune.log`**: Optuna study log
3. **`study_summary.txt`**: Summary of best hyperparameters and results
4. **`trials/`**: Directory containing individual trial results
   - `trial_{n}/trial_{n}.log`: Trial-specific log
   - `trial_{n}/model_best.pt`: Best model checkpoint (by validation R²)
   - `trial_{n}/model_final.pt`: Final model checkpoint (after all epochs)
   - `trial_{n}/metrics.csv`: Training metrics per epoch

### Study Summary Format

The `study_summary.txt` file contains:
- Best trial number
- Best hyperparameters (learning rate, dropout, activation, etc.)
- Best validation R² score
- Test set metrics (R², Pearson correlation, RMSE)
- Number of completed trials

## Technical Details

### Sparse Network Generation

The sparse network is generated using `torch.nn.utils.prune`:

1. **Initial Pruning**: Uses `prune.random_unstructured()` to randomly prune connections
2. **Mask Extraction**: Extracts pruning masks from temporary Linear layers
3. **Node Aliveness Check**: Verifies each node has at least one input and one output connection
4. **Regeneration**: If nodes are "dead", regenerates the network with a new seed (up to 10,000 attempts)
5. **Mask Application**: Uses `prune.custom_from_mask()` to apply masks to actual network layers
6. **Bias Pruning**: Biases are zeroed if all incoming connections to that neuron are pruned

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
- **Epochs**: 500 per trial
- **Early Stopping**: None (all epochs are completed)
- **Gradient Clipping**: Max norm of 1.0 to prevent gradient explosion
- **Device**: CUDA (GPU) if available, CPU otherwise

### GPU Queue Manager Features

The `advanced_gpu_queue.py` script provides:

1. **Automatic GPU Detection**: Finds available GPUs using `nvidia-smi`
2. **GPU Allocation**: Assigns jobs to available GPUs
3. **Process Management**: Tracks running jobs and handles failures
4. **Retry Logic**: Automatically retries failed jobs (up to `max_retries`)
5. **Priority Queue**: Jobs can be prioritized (currently all set to priority 5)
6. **Logging**: Comprehensive logging of job status and execution
7. **Graceful Shutdown**: Handles SIGINT/SIGTERM signals to clean up running jobs

## Verification

### Verify Job Generation

After running `generate_jobs.py`, verify:

```bash
# Check job count
grep -c "^python" jobs/r_sparse_nn_jobs.txt  # Should be 600

# Check JSON structure
python -c "import json; data=json.load(open('jobs/r_sparse_nn_advanced_jobs.json')); print(f'Total jobs: {data[\"total_jobs\"]}')"
```

### Verify Experiment Structure

Each experiment should have:
- Unique seed: `experiment_id * 1000`
- Unique data split: `experiment_{experiment_id}/`
- Unique output directory: `results/D{drug}/D{drug}_{experiment_id}/`

### Verify Sparse Network

The sparse network should have:
- Exact edge counts matching `nest_edges_by_layer`
- All nodes "alive" (at least one input and one output connection)
- Pruning masks maintained during training (via `torch.prune`)

## Troubleshooting

### Common Issues

1. **GPU Not Available**: Check `nvidia-smi` and ensure CUDA is properly configured
2. **Out of Memory**: Reduce `--max-gpus` or batch size in hyperparameter space
3. **Job Failures**: Check individual experiment logs in `results/D{drug}/D{drug}_{experiment}/`
4. **Queue Manager Stops**: Check `logs/gpu_queue_manager.log` for errors

### Checking Job Status

```bash
# View queue manager status
tail -f logs/gpu_queue_manager.log

# Check running processes
ps aux | grep r_sparse_nn_hparam_tuner

# Check GPU usage
nvidia-smi

# View recent experiment logs
tail -f results/D5/D5_0/D5_0.log
```

## Code Verification

The following code snippets verify key aspects of the methodology:

### 50 Experiments Per Drug

From `generate_jobs.py`:
```python
for drug in drugs:  # 12 drugs
    for i in range(50):  # 50 experiments per drug
        seed = i * 1000  # Unique seed per experiment
        # ... creates job configuration
```

### Same Sparse Network Configuration

From `r_sparse_nn_hparam_tuner.py`:
```python
# genotype_hiddens is fixed at 4 for all experiments
self.genotype_hiddens = 4  # Fixed, not tuned
```

### Unique Seeds Per Experiment

From `generate_jobs.py`:
```python
seed = i * 1000  # experiment 0: seed 0, experiment 1: seed 1000, etc.
```

From `r_sparse_nn_hparam_tuner.py`:
```python
trial_seed = trial.number + args.seed  # Each trial within experiment has unique seed
```

### Sparse Network Edge Counts

From `sparse_nn.py`:
```python
self.nest_edges_by_layer = {
    0: 1321 * self.genotype_hiddens,  # 5,284 edges (when genotype_hiddens=4)
    1: 92 * self.genotype_hiddens * self.genotype_hiddens,  # 1,472 edges
    # ... etc
}
```

## References

- **Optuna**: Hyperparameter optimization framework (https://optuna.org/)
- **PyTorch Pruning**: `torch.nn.utils.prune` for maintaining sparsity
- **Nest VNN**: Original architecture inspiration with hierarchical sparse connectivity

## Contact

For questions or issues, refer to the code comments or examine the log files in the `logs/` and `results/` directories.
Note that when you load a sparse architecture initial seed doesn't matter before loading. All parameters will be replaced. Additionally, model_best and model_final should become the same in every trial cases.
