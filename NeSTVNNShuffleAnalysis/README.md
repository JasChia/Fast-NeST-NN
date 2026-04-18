# NeST-VNN shuffle experiment (reproducible pipeline)

This repository contains the **NeST-VNN** gene-order shuffle study: for each of **12 drugs**, **50** paired runs (**shuffled** vs **unshuffled** input gene order), with metrics and statistical summaries used in the paper tables.

Repository layout:

```text
NeSTVNNShuffleAnalysis/
├── Data/                              # Local copy of inputs (see “Data layout” below)
├── README.md                          # This file
├── analysis_results/                  # Shuffled vs unshuffled tests on combined_model_metrics.csv
│   ├── analyze_metrics.py
│   ├── create_results_table.py
│   ├── detailed_statistical_results.csv
│   ├── statistical_summary_table.csv
│   └── clean_statistical_results.csv
└── nest_vnn/                          # Training code, logs, job list, Wilcoxon summaries
    ├── src/                           # PyTorch training (train.py, VNN, DrugCell NN, …)
    ├── nest_vnn_logs/                 # One directory per run (see “Expected outputs”)
    ├── nest_vnn_jobs/
    │   ├── jobs.txt                   # All 1200 training commands (relative paths)
    │   └── run_all_jobs.sh            # Optional: run every line in jobs.txt sequentially
    ├── combine_metrics.py             # Build combined_model_metrics.csv from nest_vnn_logs
    ├── combined_model_metrics.csv     # Long-form metrics (all runs × metrics)
    ├── regenerate_test_summary.py     # Wilcoxon + BH → test_*_summary_wilcoxon.csv
    ├── test_r2_summary_wilcoxon.csv # Table-ready R² (mean ± SD) + BH p-values
    ├── test_pearson_summary_wilcoxon.csv
    ├── shuffle_assignment_analysis/ # Gene → assembly QA after shuffle
    │   ├── analyze_shuffle_assignments.py
    │   ├── run_shuffle_analysis.sh
    │   └── outputs/                   # Gitignored; large JSON + logs
    └── scripts/
        └── rewrite_jobs_portable_paths.py  # One-off: migrate old absolute ANL paths
```

All **paths inside `nest_vnn_jobs/jobs.txt`** are **relative to the `nest_vnn/` directory**:

- Training data and ontology: **`../Data/...`** (sibling `Data/` folder at repo root).
- Run outputs: **`-modeldir nest_vnn_logs/D{drug}_{shuffled|unshuffled}_exp_{id}`** so each job writes under `nest_vnn/nest_vnn_logs/`.

---

## Requirements

- **Python 3** with **PyTorch** (CUDA build recommended), **pandas**, **numpy**, **scipy**, **scikit-learn** (see `nest_vnn/src/` imports).
- **Recommended:** the Conda **`environment.yml`** at the **Fast-NeST-NN** repository root (see parent **`README.md`**), or any equivalent env (e.g. **`cuda11_env`**) with the same packages.
- **GPU** with enough memory for DrugCell-style VNN training (same settings as `jobs.txt`: batch size 8, 250 epochs, early stopping inside trainer).
- **Environment variable** `CUDA_VISIBLE_DEVICES` (or `-cuda` in `train.py`) set appropriately when launching jobs.

Optional: maintain a Conda env YAML under `nest_vnn/conda-envs/` if you use one; the training code does not hardcode an environment path.

**Note:** end-to-end **`scripts/verify_repo.sh`** in the parent repo exercises **fNeST-NN + Profiling**, not this shuffle workspace—follow the sections below for NeST-VNN shuffle runs.

---

## Data layout (`Data/`)

Place inputs under **`NeSTVNNShuffleAnalysis/Data/`** so they mirror the former shared lab tree (paths under `../Data/` from `nest_vnn/`):

```text
Data/
├── red_ontology.txt
├── red_gene2ind.txt
└── nest_shuffle_data/
    └── CombatLog2TPM/
        └── Drug{ID}/
            └── D{ID}_CL/
                ├── D{ID}_cell2ind.txt
                ├── D{ID}_GE_Data.txt
                └── train_test_splits/
                    ├── experiment_0/
                    │   ├── training_data.txt
                    │   ├── test_data.txt
                    │   └── shuffled_gene_expression.txt   # only for shuffled runs
                    ├── experiment_1/
                    │   └── …
                    └── …
                    └── experiment_49/
                        └── …
```

**Drug IDs** in this study: `5, 57, 80, 99, 127, 151, 188, 201, 244, 273, 298, 380`.

If your data live elsewhere, either symlink `Data` → that location or set **`NEST_VNN_DATA`** to an absolute directory when running shuffle QA (see below). Retraining from `jobs.txt` expects the **`../Data/`** layout unless you regenerate `jobs.txt` with different prefixes.

---

## Replicating training (all commands in `jobs.txt`)

1. **Clone** the repository and populate **`Data/`** as above.

2. **Working directory must be `nest_vnn/`** so relative paths resolve:

   ```bash
   cd NeSTVNNShuffleAnalysis/nest_vnn
   ```

3. **Run jobs** (choose one):

   - **Sequential (simplest, slow):**

     ```bash
     ./nest_vnn_jobs/run_all_jobs.sh
     ```

   - **Manual / cluster:** submit each non-comment line of `nest_vnn_jobs/jobs.txt` as its own job. Every line is a full `python -u src/train.py ...` invocation; keep **`cwd=nest_vnn`**.

4. **Header in `jobs.txt`:** 12 drugs × 50 experiments × 2 shuffle conditions = **1200** jobs. Training uses **epoch=250**, **batchsize=8**, **seed=42**, **genotype_hiddens=4** (see file header).

5. **`train.py`** creates the output directory for `-modeldir` if it does not exist (`os.makedirs(..., exist_ok=True)`).

---

## Expected output structure (per run)

Each training run writes to:

```text
nest_vnn/nest_vnn_logs/D{drug}_{shuffled|unshuffled}_exp_{k}/
├── model_metrics.csv       # Validation/Test Pearson, Spearman, R², loss (used downstream)
├── train.log               # Text log
├── val_predictions.txt
├── test_predictions.txt
├── val_true_labels.txt
├── test_true_labels.txt
├── val_metrics.txt
├── test_metrics.txt
└── std.txt                 # If produced by training
```

Stdout from each job is redirected to **`D{drug}_{shuffled|unshuffled}_exp_{k}.log`** in `nest_vnn/` (same line as in `jobs.txt`).

---

## `nest_vnn_logs/` contents and archive usage

`nest_vnn/nest_vnn_logs/` is the main run-output directory for model training. It contains one subdirectory per job:

- `D{drug}_{shuffled|unshuffled}_exp_{k}/` (for example, `D5_shuffled_exp_0/`)
- Each run directory stores metrics, prediction outputs, and logs described in the "Expected output structure" section above.

To package all run outputs into a single compressed archive:

```bash
tar -czf "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/Fast-NeST-NN/NeSTVNNShuffleAnalysis/nest_vnn/nest_vnn_logs.tar.gz" -C "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/Fast-NeST-NN/NeSTVNNShuffleAnalysis/nest_vnn" "nest_vnn_logs"
```

To extract that archive back into `nest_vnn/`:

```bash
tar -xzf "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/Fast-NeST-NN/NeSTVNNShuffleAnalysis/nest_vnn/nest_vnn_logs.tar.gz" -C "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/Fast-NeST-NN/NeSTVNNShuffleAnalysis/nest_vnn"
```

Quick verification after extraction:

```bash
ls "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/Fast-NeST-NN/NeSTVNNShuffleAnalysis/nest_vnn/nest_vnn_logs" | head
```

---

## After training: aggregate metrics and paper table

From **`nest_vnn/`**:

```bash
python combine_metrics.py
```

Writes **`nest_vnn/combined_model_metrics.csv`** (all `model_metrics.csv` merged with `Drug_ID`, `Shuffle_Status`, `Experiment_Number`).

**Wilcoxon + Benjamini–Hochberg** summaries (same logic as the paper’s R² table; excludes drugs **57** and **201** from the BH set):

```bash
python regenerate_test_summary.py
```

Produces:

- `nest_vnn/test_r2_summary_wilcoxon.csv`
- `nest_vnn/test_pearson_summary_wilcoxon.csv`

**Additional shuffled vs unshuffled tests** (independent *t*-test and Mann–Whitney on validation/test columns):

```bash
cd ../analysis_results
python analyze_metrics.py
python create_results_table.py
```

These scripts read **`../nest_vnn/combined_model_metrics.csv`** and write CSVs in `analysis_results/` (paths are resolved from each script’s location; you do **not** need a particular shell `cwd` beyond `analysis_results/` for those two commands).

---

## Shuffle assignment QA (optional)

Checks that shuffled gene-to-ontology assembly assignments behave as expected (separate from predictive R²).

```bash
cd NeSTVNNShuffleAnalysis/nest_vnn
./shuffle_assignment_analysis/run_shuffle_analysis.sh
```

Uses **`Data/`** by default (`../Data` from `nest_vnn`). Override data root:

```bash
export NEST_VNN_DATA=/absolute/path/to/your/Data
./shuffle_assignment_analysis/run_shuffle_analysis.sh
```

Large JSON outputs go to **`shuffle_assignment_analysis/outputs/`** (gitignored).

---

## Environment variables (summary)

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Select GPU(s) for `train.py`. |
| `NEST_VNN_DATA` | Optional absolute override for shuffle QA data root (default: `<repo>/Data`). |

---

## Migrating old absolute job paths

If you still have a `jobs.txt` that points at `/nfs/.../ANL_Drug_CData`, run once (after backing up the file):

```bash
cd nest_vnn
python scripts/rewrite_jobs_portable_paths.py
```

Then apply the same **`-modeldir` / `_exp_` / `nest_vnn_logs/`** transforms as in the current `jobs.txt`, or keep this repository’s checked-in `jobs.txt` as the source of truth.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `FileNotFoundError` for `../Data/...` | Run training from **`nest_vnn/`**; confirm **`Data/`** exists next to `nest_vnn/`. |
| `combine_metrics` finds 0 rows | Confirm folders match **`D(\d+)_(shuffled\|unshuffled)_exp_(\d+)`** under `nest_vnn_logs/`. |
| Wilcoxon script errors | Each drug needs paired experiment IDs in **both** shuffle states; incomplete runs are skipped for that drug. |
| CUDA OOM | Lower batch size in `jobs.txt` (requires editing all lines consistently) or use a larger GPU. |

---

## License / citation

Include the same license and citation policy as your parent project; training code follows the DrugCell / NeST-VNN lineage used in the original experiments.
