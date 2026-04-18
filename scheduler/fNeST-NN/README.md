# fNeST-NN (fast NeST neural network)

This folder contains training and job-scheduling code for **fNeST-NN**, the optimized variant of NeST-VNN described in *Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction* (ICIBM 2026). Compared with the original NeST-VNN stack, fNeST-NN parallelizes assembly computation by depth and **aggregates auxiliary predictions** into the final drug-response output.

## Paper mapping

- **Methods:** §2.1 Fast NeST Neural Network (fNeST-NN); performance and runtime comparisons in §2–3.
- **Outputs:** Per-drug regression with 50 CV trials × 100 Optuna trials (see paper / supplementary for hyperparameter ranges).

## Layout

| File | Role |
|------|------|
| `fnest_nn.py` | Core `eNest` PyTorch module (linear readout head). |
| `fnest_nn_hparam_tuner.py` | Optuna-driven training; writes `results/...` and `fnest_nn_HTune.log` per run. |
| `jobs/fnest_nn_jobs.txt` | Bundled training commands (paths use `../../Data/...` from this folder). |
| `jobs/fnest_nn_advanced_jobs.json` | Same jobs as JSON (optional). |
| `cleanup_failed_experiment.py` | Ops helper for partial runs. |
| `result_agg.py` | Optional R² summary over `results/D*/`. |

## Data paths

Restore drug bundles from **`Data_archives/`** into **`Data/`** at the repo root (see root `README.md`). Job files reference **`../../Data/`**; to use another root, set **`FAST_NEST_DATA_ROOT`** or replace that prefix in the job lists.

## Typical workflow

```bash
cd scheduler/fNeST-NN
grep -v '^#' jobs/fnest_nn_jobs.txt | grep -v '^$' | bash
```

For large sweeps, use your cluster scheduler (Slurm, GNU Parallel, etc.) instead of a bare shell loop. Run artifacts live under `results/`, `logs/`, and `shared/` (gitignored at repo level).

## Environment

Use the Conda **`environment.yml`** at the repository root (see root **`README.md`**) or an equivalent env (for example **`cuda11_env`**) with PyTorch, Optuna, pandas, and **`torchmetrics`**.

## Running `fnest_nn_hparam_tuner.py` manually

Required arguments: **`-train_file`**, **`-val_file`**, **`-test_file`**, **`-cell2id`**, **`-ge_data`**. Common options: **`-n_trials`**, **`-max_epochs`** (default **500**; use **`-max_epochs 10`** for a quick sanity check), **`-cuda`**, **`-seed`**, **`-output_dir`**.

Example (nested **`nest_shuffle_data/...`** paths; use **`Data/D{N}_CL/...`** if your checkout uses the flat layout):

```bash
conda activate fast-nest-nn
cd scheduler/fNeST-NN
python -u fnest_nn_hparam_tuner.py \
  -cuda 0 -drug 298 \
  -train_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/train_test_splits/experiment_0/true_training_data.txt \
  -val_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/train_test_splits/experiment_0/validation_data.txt \
  -test_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/train_test_splits/experiment_0/test_data.txt \
  -cell2id ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/D298_cell2ind.txt \
  -ge_data ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/D298_GE_Data.txt \
  -n_trials 100 -max_epochs 500 -seed 0 -output_dir results/D298/D298_0
```

**Automated smoke test:** from the repo root, **`bash scripts/verify_repo.sh`** runs **`-n_trials 2 -max_epochs 10`** and writes under **`results/VERIFY_SMOKE/`**.

## Expected outputs (under `-output_dir`)

| Artifact | Description |
|----------|-------------|
| `fnest_nn_HTune.log` | Optuna journal file for the study |
| `trials/trial_<k>/model_best.pt` | Best validation checkpoint for that trial |
| `trials/trial_<k>/metrics.csv` | Validation/test metrics for that trial |
| `best_model/` | Copy of the best trial by validation R² |
| `final_results.json` | Study summary |
| `best_model_results.csv` | One-line best validation/test R² |

## What to change on your system

- **`FAST_NEST_DATA_ROOT`** or edit paths if data are not under `<repo>/Data`.
- **`../../Data/`** prefix in **`jobs/fnest_nn_jobs.txt`** if your relative layout differs.
