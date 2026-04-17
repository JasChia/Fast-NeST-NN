# sparse_wd

Supporting analyses for **sparse / weight-decay** comparisons and **multiple-testing** tables: BH-adjusted p-value tables, ER (Erdős–Rényi) sparsity visualizations, and merged R² summaries used in discussion of **ERK-SNN**, **UGP-NN**, and related baselines.

## Files

- `compute_bh_adjusted_table.py` — Benjamini–Hochberg adjustments for tabulated tests.
- `bh_adjusted_tables/` — CSV outputs for manuscript tables.
- `ER_NeST_sparsity_analysis.py`, `ER_NeST_sparsity_comparison_*.png` — sparsity pattern figures.
- `merge_test_r2_to_aggregated.py`, `test_r2_*.csv` — aggregated test R² for comparisons.

Inputs are CSVs produced from scheduler experiments; paths inside scripts may need pointing at your local `results/` exports.
