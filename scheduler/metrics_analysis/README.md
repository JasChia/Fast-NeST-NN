# metrics_analysis

Post-hoc **statistical comparison** of methods: paired Wilcoxon tests, **Benjamini–Hochberg** adjustment, and **Pearson** / Spearman / R² summaries reported in the paper and **Supplementary Tables**.

## Contents

- `pearson_*`, `r2_*`, `spearman_*` CSVs — test statistics vs reference (e.g. eNest_sum) with BH and unadjusted variants.
- `median_mad/` — robust (median/MAD) variants of the same comparisons.
- `combined_metrics_all_methods.csv` — merged summary for plotting.
- `binarization_evaluation/` — auxiliary binarization study (see folder README).
- `plot_r2_comparison.py`, `csv_to_latex.py` — figure/table export helpers.

## Inputs

These scripts expect per-method metrics exported from each model’s `results/` (gitignored on your machine). After runs complete, point loaders at your local result roots or copy aggregated CSVs into this directory.

## Usage

Run individual scripts with paths edited for your checkout, e.g.:

```bash
cd scheduler/metrics_analysis
python plot_r2_comparison.py   # after configuring paths inside the script
```
