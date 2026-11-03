# Fast-NeST-NN

This repository contains code, data artifacts, and analysis workflows associated with:

**"Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction"**

This work complements a submission to **ICIBM 2026**.

## Python environment

A **minimal Conda spec** lives in **`environment.yml`** at the repo root.

```bash
cd /path/to/Fast-NeST-NN
conda env create -f environment.yml    # first time
conda activate fast-nest-nn
```

---

## Data layout (required on disk)

Large inputs live under **`Data/`** at the repository root (gitignored, apart from its README and the two
graph files below). Every bundled job command uses one layout:

    Data/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/

Per-drug folders contain **`D{drug}_cell2ind.txt`**, **`D{drug}_GE_Data.txt`**, and **`train_test_splits/experiment_*/`** with `true_training_data.txt`, `validation_data.txt`, and `test_data.txt`.

**Ontology files** used by NeST-VNN and Profiling: **`Data/red_ontology.txt`** and **`Data/red_gene2ind.txt`**, directly under `Data/` and **tracked in Git** --- see [Data/README.md](Data/README.md) for their contents and provenance. To point the drug bundles elsewhere set **`FAST_NEST_DATA_ROOT`** (see `ArchitecturePerformanceExperiments/data_paths.py`); `Profiling/` uses its own variables, documented in `Profiling/README.md`.

---

## Repository Guide

### `Data/` (the graph files are in Git; everything else is local)

The contents of `Data/` are listed in `.gitignore` so large local inputs stay on disk only; its README and **`red_ontology.txt`** / **`red_gene2ind.txt`**, the NeST graph the published models were built from, are tracked. Restore the rest from **`Data_archives/`** (per-condition compressed archives tracked with Git LFS) or copy from your own backup. See **[Data/README.md](Data/README.md)**.

### `Data_archives/` (in Git, LFS)

Per-drug dataset archives are stored as `*.tar.gz` files (Git LFS). This is the exact data
utilized for all experiments. Each archive unpacks to `D{drug}_CL/` at its own root, so it
must be extracted into the matching per-drug directory — see
**[Data_archives/README.md](Data_archives/README.md)** for the one-line loop that does all ten.

### `ArchitecturePerformanceExperiments/` (in Git — code and docs; outputs excluded)

Contains all training scripts to replicate each method's performance shown in the paper in combination with the data.

See **[ArchitecturePerformanceExperiments/README.md](ArchitecturePerformanceExperiments/README.md)** for the model index, shared data path conventions, and how to run bundled jobs.

### `NeSTVNNShuffleAnalysis/`

The **NeST-VNN gene-order shuffle** pipeline (training code and **`nest_vnn_jobs/jobs.txt`**) lives under **`NeSTVNNShuffleAnalysis/`**. Training code under **`nest_vnn/src/`** and job lists are tracked; **`nest_vnn/nest_vnn_logs/`** is **gitignored** and only appears locally after runs (see **`NeSTVNNShuffleAnalysis/.gitignore`**).

Inputs come from the repo-root **`Data/`** tree; **`NeSTVNNShuffleAnalysis/Data`** is a link to it.

See **[NeSTVNNShuffleAnalysis/README.md](NeSTVNNShuffleAnalysis/README.md)** for data layout, run instructions, and expected outputs.

### `Profiling/`

Runtime benchmarking and speedup analysis for efficient NeST variants versus baseline NeST-VNN/DrugCellNN implementations.

See **[Profiling/README.md](Profiling/README.md)** for detailed setup, required inputs, commands, and expected outputs.

If you have questions, please contact the corresponding author of the paper or owner of this repo for more information or help replicating experiments.
