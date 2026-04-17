# Scheduler workspace

This directory holds **drug-response experiments** for *Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction* (ICIBM 2026). Each subfolder corresponds to a **named model or analysis** in the paper.

## Model ↔ folder (paper notation)

| Paper name | Typical folder |
|------------|------------------|
| NeST-VNN | `NeST-VNN/` |
| fNeST-NN | `fNeST-NN/` |
| FCNN | `FCNN/` |
| Dense-fNeST | `Dense-fNeST/` |
| RSNN | `RSNN/` |
| ERK-SNN | `ERK_SNN/` |
| RP-fNeST | `RP-fNeST/` |
| LP-NN | `LP-NN/` |
| GP-NN | `GP-NN/`, `GP-NN_no_warmup/` (ablation) |
| UGP-NN | `UGP-NN/` |
| Uniform random direct-output SNN (baseline) | `uniform_random_direct_output_SNN/` |
| Direct-input FC residual (ablation / pathway) | `fc_nn_direct_input/` |
| Gene-order shuffle / NeST-sum experiments | `eNest_sum/` |
| Pooled metrics, Wilcoxon, BH, **Pearson supplements** | `metrics_analysis/`, `sparse_wd/` |

See each folder’s **`README.md`** for commands. Folders **not** listed here may be legacy or in flux; treat as non-paper unless documented.

## What is in Git vs only on your machine

- **Tracked:** Python sources, small CSV/JSON summaries, READMEs, and **bundled job command lists** under each model’s `jobs/` (paths relative to `<repo>/Data/`).
- **Not tracked:** `**/results/`, `**/long_results/`, `**/logs/`, `**/shared/` (see root `.gitignore`).

Clone the repo, extract **`Data_archives/`** into **`Data/`** at the repo root (see root `README.md`). Job files use `../../Data/...` from `scheduler/<Model>/`; to point elsewhere, set **`FAST_NEST_DATA_ROOT`** in the environment *or* bulk-replace that prefix in the job lists. **Run the shell commands** in each model’s `jobs/*_jobs.txt` from that model’s directory (skip `#` comment lines), or feed the file to your cluster scheduler (Slurm, GNU Parallel, etc.); this repo no longer ships in-tree GPU queue drivers.

## Optional archiving

To tarball a local `results/` tree for backup (does not change Git):

```bash
cd scheduler/<ModelFolder>
tar -I 'gzip -1' -cf results.tar.gz results
```

Very large trees can be split (e.g. `tar -cf - results | split -b 1800m - results.tar.part-`) for off-site storage; see each model README for conventions.

## Root-level helpers

- `compute_metrics_and_comparisons.py` / `compute_metrics_and_comparisons_median_mad.py` — aggregate statistics across methods (paired tests, multiple comparison adjustment); inputs are CSVs produced under each model’s `results/` or `metrics_analysis/`.
- `extract_non_significant_pvalues.py` — supplementary filtering helper for reported tables.
