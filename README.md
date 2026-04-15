# Fast-NeST-NN

This repository contains code, data artifacts, and analysis workflows associated with:

**"Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction"**

This work complements a submission to **ICIBM 2026**.

## Repository Guide

### `Data/`

Core input/reference data used across experiments (for example, ontology and gene index mappings used by NeST-based models and profiling scripts). Keep this folder at the repository root so downstream scripts can resolve paths consistently.

### `NeSTVNNShuffleAnalysis/`

Reproducible NeST-VNN gene-order permutation/shuffle experiment workspace, including training jobs, logs, metric aggregation, and statistical testing outputs used for the manuscript analyses.

See **[NeSTVNNShuffleAnalysis/README.md](NeSTVNNShuffleAnalysis/README.md)** for full data layout, run instructions, expected outputs, and troubleshooting.

### `Profiling/`

Runtime benchmarking and speedup analysis for efficient NeST variants versus baseline NeST-VNN/DrugCellNN implementations.

See **[Profiling/README.md](Profiling/README.md)** for detailed setup, required inputs, commands, and expected outputs.

## Note on Upcoming Additions

An additional project folder will be added and documented here once setup is complete.
