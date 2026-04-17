# Fast-NeST-NN

This repository contains code, data artifacts, and analysis workflows associated with:

**"Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction"**

This work complements a submission to **ICIBM 2026**.

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

### `repo_archives/` (in Git, LFS) — `NeSTVNNShuffleAnalysis` only

The **`NeSTVNNShuffleAnalysis/`** workspace is large; it is still shipped as **split tar parts** under `repo_archives/NeSTVNNShuffleAnalysis.tar.part-*` (each part under GitHub’s 2 GiB LFS object limit). The raw folder remains gitignored at the repo root.

**Reconstruct and extract**

```bash
cat repo_archives/NeSTVNNShuffleAnalysis.tar.part-* > NeSTVNNShuffleAnalysis_restored.tar
tar -xf NeSTVNNShuffleAnalysis_restored.tar
rm NeSTVNNShuffleAnalysis_restored.tar
```

### `NeSTVNNShuffleAnalysis/` (after extraction)

Reproducible NeST-VNN gene-order permutation/shuffle experiment workspace, including training jobs, logs, metric aggregation, and statistical testing outputs used for the manuscript analyses.

See **[NeSTVNNShuffleAnalysis/README.md](NeSTVNNShuffleAnalysis/README.md)** for full data layout, run instructions, expected outputs, and troubleshooting.

### `Profiling/`

Runtime benchmarking and speedup analysis for efficient NeST variants versus baseline NeST-VNN/DrugCellNN implementations.

See **[Profiling/README.md](Profiling/README.md)** for detailed setup, required inputs, commands, and expected outputs.
