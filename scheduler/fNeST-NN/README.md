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
