# Architecture performance experiments

This directory holds **drug-response training code and bundled job lists** for *Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction* (ICIBM 2026). Each subfolder is one **architecture / method** evaluated in the paper.

**Related:** the **NeST-VNN gene-order shuffle** study lives at repository root in **`NeSTVNNShuffleAnalysis/`** (not under this folder). Runtime benchmarking is under **`Profiling/`**.

---

## Model index

| Folder | Hyperparameter tuner | Optuna trainer class (in `*_hparam_tuner.py`) | Network module (typical) |
|--------|----------------------|-----------------------------------------------|---------------------------|
| `NeST-VNN/` | `nest_vnn_hparam_tuner.py` | `OptunaNestVNNTrainer` | `DrugCellNN` in `src/` |
| `fNeST-NN/` | `fnest_nn_hparam_tuner.py` | `OptunaFNeSTNNTrainer` | `eNest` in `fnest_nn.py` |
| `FCNN/` | `fc_nn_hparam_tuner.py` | `OptunaFCNNTrainer` | `FC_NN` in `fc_nn.py` |
| `Dense-fNeST/` | `dense_fnest_hparam_tuner.py` | `OptunaDenseFNeSTTrainer` | `Dense_fNeST` in `dense_fnest_fc_nn.py` |
| `RSNN/` | `r_sparse_nn_hparam_tuner.py` | `OptunaRSNNTrainer` | `Sparse_NN` in `sparse_nn.py` |
| `ERK_SNN/` | `ERK_SNN_hparam_tuner.py` | `OptunaERKSNNTrainer` | `ERK_SNN` in `ERK_SNN.py` |
| `RP-fNeST/` | `uniform_random_do_di_snn_hparam_tuner.py` | `OptunaRPfNeSTTrainer` | `UniformRandomDODISNN` |
| `GP-NN/` | `global_prune_nn_hparam_tuner.py` | `OptunaGlobalPruneNNTrainer` | `GlobalPrunedFC_NN` |
| `LP-NN/` | `layer_prune_nn_hparam_tuner.py` | `OptunaLayerPruneNNTrainer` | `LayerPrunedFC_NN` |
| `UGP-NN/` | `relaxed_global_prune_nn_hparam_tuner.py` | `OptunaRelaxedGlobalPruneNNTrainer` | `RelaxedGlobalPrunedFC_NN` |

Each method’s **`README.md`** describes paper-specific methodology, paths, and any legacy notes. **Shared** instructions (data layout and how to run jobs) are **here** so individual READMEs stay short and consistent.

---

## Shared: data and paths

- **Tracked:** Python sources, READMEs, and **`jobs/*_jobs.txt`** / **`jobs/*_advanced_jobs.json`** (paths relative to `<repo>/Data/` via **`../../Data/...`** from each method directory).
- **Not tracked:** **`results/`**, **`long_results/`**, **`logs/`**, **`shared/`**, and other local output or scratch folders under this tree (see root **`.gitignore`**).

Clone the repo and extract **`Data_archives/`** into **`Data/`** at the repo root (see root **`README.md`**). To point job lists elsewhere, set **`FAST_NEST_DATA_ROOT`** or bulk-replace the **`../../Data`** prefix.

**Layouts:** many jobs use **`Data/nest_shuffle_data/CombatLog2TPM/Drug{N}/D{N}_CL/...`**. A flat **`Data/D{N}_CL/...`** tree also works if you pass consistent paths into the tuner.

**Ontology / gene maps:** shared helpers are in **`data_paths.py`**.

---

## Shared: how to run bundled jobs

1. `cd ArchitecturePerformanceExperiments/<MethodFolder>/`
2. Run lines from **`jobs/*_jobs.txt`** (one `python ...` command per line; skip `#` comments), or submit them to your **cluster job runner** (Slurm, GNU Parallel, etc.). This repo does **not** ship in-tree GPU queue drivers.

Each **`*_hparam_tuner.py`** accepts **`-n_trials`** and **`-max_epochs`** (default **500** epochs per Optuna trial).

---

## Shared: optional archiving

To tarball a local **`results/`** tree (does not affect Git):

```bash
cd ArchitecturePerformanceExperiments/<MethodFolder>
tar -I 'gzip -1' -cf results.tar.gz results
```

---

## Aggregation / downstream statistics

Cross-method CSV aggregation (paired tests, Benjamini–Hochberg, etc.) is **not** implemented by any script **committed in this repository** (for example there is no `compute_metrics_and_comparisons.py` in the tree). After experiments finish, use your own analysis code or restore scripts from backups. Typical dependencies (**pandas**, **scipy**, **statsmodels**) are available from the root **`environment.yml`**.
