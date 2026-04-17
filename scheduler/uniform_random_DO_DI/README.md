# Uniform Random DO+DI (Direct Output + Direct Input) Sparse Neural Network

This directory contains a sparse neural network implementation that uses uniform random pruning for initialization with two key architectural features:

## Architecture Features

### 1. Direct Input (DI) Connections
Each hidden layer receives **both** the original input (gene expression) AND the previous layer's output. The gene-to-layer connections are specified separately for each layer with their own edge counts, allowing the network to access raw input features at every depth.

**Gene → Layer Connection Counts (multiplied by genotype_hiddens):**
- Gene → Layer 1: 637 edges
- Gene → Layer 2: 253 edges
- Gene → Layer 3: 112 edges
- Gene → Layer 4: 92 edges
- Gene → Layer 5: 39 edges
- Gene → Layer 6: 35 edges
- Gene → Layer 7: 44 edges
- Gene → Layer 8: 109 edges

### 2. Direct Output (DO) Connections
Nodes in each intermediary layer are split into non-overlapping sets of size `genotype_hiddens` (default: 4). Each set attempts to predict the final output through an additional linear layer + activation, which provides auxiliary loss signals during training.

## Network Architecture

```
Input (689 genes)
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      │
    Layer 1 (76 × genotype_hiddens) ──► Direct Output         │
       │                                                      │
       ├──────────────────────────────────────────────────┬───┤
       ▼                                                  │   │
    Layer 2 (32 × genotype_hiddens) ──► Direct Output     │   │
       │                                                  │   │
       ├──────────────────────────────────────────────┬───┼───┤
       ▼                                              │   │   │
    Layer 3 (13 × genotype_hiddens) ──► Direct Output │   │   │
       │                                              │   │   │
       ... (continues for all 8 layers)               │   │   │
       │                                              │   │   │
       ▼                                              ▼   ▼   ▼
    Layer 8 (1 × genotype_hiddens) ────────────────► Main Output
```

**Legend:**
- Horizontal arrows from Input: Direct Input connections (gene → each layer)
- Vertical arrows: Layer-to-Layer connections (previous layer → current layer)
- Side arrows to "Direct Output": Auxiliary predictions for intermediate supervision

## Files

| File | Description |
|------|-------------|
| `uniform_random_do_di_snn.py` | Main SNN model implementation with DO+DI architecture |
| `uniform_random_do_di_snn_hparam_tuner.py` | Optuna-based hyperparameter tuning for the DO+DI model |
| `generate_jobs.py` | Job generator for distributed training experiments |
| `distributed_gpu_queue.py` | Multi-GPU distributed job queue manager |
| `monitor_distributed_jobs.py` | Job status monitoring utility |
| `cleanup_failed_experiment.py` | Utility to clean up failed experiments |
| `advanced_gpu_queue.py` | Alternative GPU queue manager |

## Usage

### 1. Generate Jobs
```bash
cd /nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/uniform_random_DO_DI
python generate_jobs.py
```

### 2. Run Distributed Training
```bash
python distributed_gpu_queue.py jobs/uniform_random_do_di_snn_advanced_jobs.json \
    --max-gpus 16 \
    --node-id node1 \
    --lock-dir shared/locks \
    --status-dir shared/status
```

### 3. Monitor Jobs
```bash
tail -f logs/distributed_queue.log
python monitor_distributed_jobs.py
```

## Testing the Model

```bash
python uniform_random_do_di_snn.py
```

This will run built-in tests for:
1. Reproducibility (same seed = same network)
2. Different seeds produce different results
3. Forward pass with auxiliary outputs
4. Connectivity verification (correct number of edges)
