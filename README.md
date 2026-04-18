# Fast-NeST-NN

This repository contains code, data artifacts, and analysis workflows associated with:

**"Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction"**

This work complements a submission to **ICIBM 2026**.

## Python environment

A **minimal Conda spec** (PyTorch + CUDA 11.x stack, Optuna, NumPy/Pandas/SciPy, `torchmetrics`, Profiling deps) lives in **`environment.yml`** at the repo root. It is intended to mirror a typical **`cuda11_env`**-style setup on Linux without pinning every transitive package.

```bash
cd /path/to/Fast-NeST-NN
conda env create -f environment.yml    # first time
conda activate fast-nest-nn
```

If you already use **`cuda11_env`** (or another env) and prefer to keep it, ensure it includes at least: **PyTorch**, **Optuna**, **pandas**, **numpy**, **scipy**, **scikit-learn**, **matplotlib**, **statsmodels**, **networkx**, **tqdm**, and **`torchmetrics`** (used by several tuners).

**CPU-only:** install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for your platform, then `conda install` the remaining packages from `environment.yml` (omit the `pytorch-cuda` line).

---

## Data layout (required on disk)

Large inputs live under **`Data/`** at the repository root (gitignored). After extracting **`Data_archives/`**, you should have, for each drug, a cell-line bundle directory. Two layouts are used in this repo:

| Layout | Example path to `D298_cell2ind.txt` |
|--------|-------------------------------------|
| **Nested** (paths in many `jobs/*.txt` files) | `Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/D298_cell2ind.txt` |
| **Flat** (some local checkouts) | `Data/D298_CL/D298_cell2ind.txt` |

Per-drug folders contain **`D{drug}_cell2ind.txt`**, **`D{drug}_GE_Data.txt`**, and **`train_test_splits/experiment_*/`** with `true_training_data.txt`, `validation_data.txt`, and `test_data.txt`.

**Ontology files** used by NeST-VNN and Profiling: **`Data/red_ontology.txt`** and **`Data/red_gene2ind.txt`** (or set **`FAST_NEST_DATA_ROOT`** / **`FNEST_ONTOLOGY_DIR`** as documented in `scheduler/data_paths.py` and `Profiling/README.md`).

---

## Verification (smoke test)

From the repo root, with **`Data/`** populated for at least one drug (default check uses **drug 298** and the **flat** `Data/D298_CL/...` layout):

```bash
conda activate fast-nest-nn   # or cuda11_env
bash scripts/verify_repo.sh
```

This script:

1. **`python -m py_compile`** on all tracked `scheduler/**/*_hparam_tuner.py` files.
2. Imports **pandas / scipy / statsmodels** (needed by `scheduler/compute_metrics_and_comparisons.py`).
3. Runs **fNeST-NN** hyperparameter tuning with **`-n_trials 2`** and **`-max_epochs 10`**, writing under **`scheduler/fNeST-NN/results/VERIFY_SMOKE/`** (gitignored).  
   **Expected artifacts:** `final_results.json`, `fnest_nn_HTune.log`, `trials/trial_*/model_best.pt`, `trials/trial_*/metrics.csv`, and **`best_model/`** for the best trial.
4. Runs **`Profiling/run_profiling.sh`** with **`--cpu --num-runs 2`**, writing **`Profiling/VERIFY_SMOKE_results/npa_4/summary_statistics.csv`** and **`speedup_fnest_vs_nest_vnn_bar.png`**.

**Overrides:**

| Variable | Meaning |
|----------|---------|
| `VERIFY_DRUG_ID` | Drug id for the fNeST smoke run (default `298`). |
| `VERIFY_DATA_PREFIX` | Absolute path to the `D{drug}_CL` directory (default `<repo>/Data/D${VERIFY_DRUG_ID}_CL`). |
| `SKIP_COMPILE=1` | Skip `py_compile` only. |
| `SKIP_FNEST_SMOKE=1` | Skip the fNeST-NN run. |
| `SKIP_PROFILING=1` | Skip Profiling. |

---

## Repository Guide

### `Data/` (local / not in Git)

The `Data/` directory at the repo root is listed in `.gitignore` so large local inputs stay on disk only. Restore comparable content from **`Data_archives/`** (per-condition compressed archives tracked with Git LFS) or copy from your own backup.

### `Data_archives/` (in Git, LFS)

Per-dataset archives are stored as `*.tar.gz` files (Git LFS). Extract at the repository root:

```bash
tar -xzf Data_archives/<name>.tar.gz
```

### `scheduler/` (in Git — code and docs; outputs excluded)

The **`scheduler/`** tree is tracked as part of this repository (nested `.git` metadata was removed so everything lives on `main`). Each experiment subfolder keeps scripts, configs, and READMEs.

**Training and tuning outputs** live under folders such as **`results/`**, **`long_results/`**, **`logs/`**, and **`shared/`** inside those subfolders (names vary by pipeline). On a full machine those trees can be **terabytes** in total, so they are **gitignored** (`scheduler/**/results/`, `scheduler/**/long_results/`, etc.) and are **not** pushed to GitHub. Your local or cluster copy keeps the full output trees as produced by the pipelines.

See **[scheduler/README.md](scheduler/README.md)** for the **paper model ↔ folder map**, what is tracked vs local-only, and optional archiving of `results/`.

### `NeSTVNNShuffleAnalysis/` (in Git — code and docs; large outputs excluded)

The **NeST-VNN gene-order shuffle** pipeline (training jobs, metric aggregation, statistical tests for the manuscript) lives under **`NeSTVNNShuffleAnalysis/`** in this repository. Training code under **`nest_vnn/src/`** and job lists are tracked; **`nest_vnn/nest_vnn_logs/`** and similar run outputs remain **local-only** (see **`NeSTVNNShuffleAnalysis/.gitignore`**).

Inputs still use a sibling **`Data/`** tree at the repo root (or **`NeSTVNNShuffleAnalysis/Data/`** as documented there).

See **[NeSTVNNShuffleAnalysis/README.md](NeSTVNNShuffleAnalysis/README.md)** for data layout, run instructions, expected outputs, and troubleshooting.

### `Profiling/`

Runtime benchmarking and speedup analysis for efficient NeST variants versus baseline NeST-VNN/DrugCellNN implementations.

See **[Profiling/README.md](Profiling/README.md)** for detailed setup, required inputs, commands, and expected outputs.

Please contact the corresponding author of the paper or owner of this repo for more information or help replicating experiments.