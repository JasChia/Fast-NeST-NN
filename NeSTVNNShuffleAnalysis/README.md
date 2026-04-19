# NeST-VNN shuffle experiment (reproducible pipeline)

This directory is part of the **Fast-NeST-NN** monorepo at **`NeSTVNNShuffleAnalysis/`** (tracked on `main`). You do **not** need to extract a separate archive; clone the repository once and work from this folder.

This workspace implements the **NeST-VNN** gene-order shuffle study: for each of **10 drugs**, **50** paired runs (**shuffled** vs **unshuffled** input gene order), with metrics used in the paper tables.

### What is in this folder now

At the repository root, **`NeSTVNNShuffleAnalysis/`** contains only:

```text
NeSTVNNShuffleAnalysis/
├── .gitignore
├── README.md                          # This file
└── nest_vnn/
    ├── src/                           # PyTorch training stack
    │   ├── train.py                   # Entrypoint invoked by jobs.txt
    │   ├── drugcell_nn.py
    │   ├── vnn_trainer.py
    │   ├── training_data_wrapper.py
    │   ├── util.py
    │   ├── predict.py
    │   ├── optuna_nn_trainer.py
    │   └── ccc_loss.py
    ├── nest_vnn_jobs/
    │   ├── jobs.txt                   # 1000 training commands (paths relative to nest_vnn/)
    │   └── run_all_jobs.sh            # Optional: run every line sequentially
    ├── nest_vnn_logs/                 # One subdirectory per successful job (see below)
    ├── scripts/
    │   └── rewrite_jobs_portable_paths.py   # One-off: migrate old absolute paths in jobs
    └── shuffle_assignment_analysis/
        ├── analyze_shuffle_assignments.py
        └── run_shuffle_analysis.sh
        # shuffle_assignment_analysis/outputs/ is created at runtime (gitignored)
```

**`nest_vnn_logs/`** holds per-run outputs when training has been executed. It is **gitignored** (see **`.gitignore`**) because it is large; a fresh clone has an empty or absent tree until you run jobs or restore an archive.

There is **no** `NeSTVNNShuffleAnalysis/Data/` directory in this tree: inputs are expected at the **repository root** **`Data/`** (see **Data layout** below). There is **no** `analysis_results/` directory and **no** `combine_metrics.py` / `regenerate_test_summary.py` / combined CSVs in this snapshot—those were part of the downstream paper pipeline and are not bundled here (see **Post-training aggregation** below).

All **paths inside `nest_vnn_jobs/jobs.txt`** are **relative to the `nest_vnn/` directory**:

- Training data and ontology: **`../Data/...`** (sibling `Data/` folder at repo root).
- Run outputs: **`-modeldir nest_vnn_logs/D{drug}_{shuffled|unshuffled}_exp_{id}`** so each job writes under `nest_vnn/nest_vnn_logs/`.

---

## Requirements

- **Python 3** with **PyTorch** (CUDA build recommended), **pandas**, **numpy**, **scipy**, **scikit-learn** (see `nest_vnn/src/` imports).
- **Recommended:** the Conda **`environment.yml`** at the **Fast-NeST-NN** repository root (see parent **`README.md`**), then `conda activate fast-nest-nn`, or any other env with the same packages (PyTorch, Optuna, pandas, scikit-learn, etc.).
- **GPU** with enough memory for DrugCell-style VNN training (same settings as `jobs.txt`: batch size 8, 250 epochs, early stopping inside trainer).
- **Environment variable** `CUDA_VISIBLE_DEVICES` (or `-cuda` in `train.py`) set appropriately when launching jobs.

The training code does not hardcode a Conda path; use the root **`environment.yml`** or your own env YAML if you maintain one locally.

---

## Data layout (`Data/`)

Training reads from **`../Data/`** relative to **`nest_vnn/`**, i.e. **`<repository-root>/Data/`**. Use that path (recommended), or create **`NeSTVNNShuffleAnalysis/Data/`** with the same tree and symlink it to **`../Data`** from the repo root if you prefer a local copy beside this folder.

Expected layout:

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

**Drug IDs** in this study: `5, 80, 99, 127, 151, 188, 244, 273, 298, 380`.

If your data live elsewhere, either symlink `Data` → that location or set **`NEST_VNN_DATA`** to an absolute directory when running shuffle QA (see below). Retraining from `jobs.txt` expects the **`../Data/`** layout unless you regenerate `jobs.txt` with different prefixes.

---

## Replicating training (all commands in `jobs.txt`)

1. **Clone** the **Fast-NeST-NN** repository (this monorepo) and populate **`Data/`** at the **repository root** as described in **Data layout** (extract from **`Data_archives/`** per the root **`README.md`** if needed).

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

4. **Header in `jobs.txt`:** 10 drugs × 50 experiments × 2 shuffle conditions = **1000** jobs. Training uses **epoch=250**, **batchsize=8**, **seed=42**, **genotype_hiddens=4** (see file header).

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

To package all run outputs into a single compressed archive (from **`nest_vnn/`**):

```bash
cd NeSTVNNShuffleAnalysis/nest_vnn
tar -czf nest_vnn_logs.tar.gz -C . nest_vnn_logs
```

To extract that archive back into `nest_vnn/`:

```bash
cd NeSTVNNShuffleAnalysis/nest_vnn
tar -xzf nest_vnn_logs.tar.gz -C .
```

Quick verification after extraction:

```bash
ls nest_vnn_logs | head
```

---

## Post-training aggregation (not in this tree)

This repository snapshot **does not include** the downstream aggregation stack that was used for the paper tables:

- No **`combine_metrics.py`** or **`nest_vnn/combined_model_metrics.csv`**
- No **`regenerate_test_summary.py`** or **`test_*_summary_wilcoxon.csv`**
- No **`analysis_results/`** (no `analyze_metrics.py`, `create_results_table.py`, or precomputed statistical CSVs)

After training, each run still produces **`nest_vnn_logs/.../model_metrics.csv`** and related files under **`nest_vnn_logs/`** as described above. **This repository does not include** `combine_metrics.py`, Wilcoxon pipelines, or a **`metrics_analysis/`** folder—use your own downstream scripts if you need merged tables or BH-adjusted tests.

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

If you still have a `jobs.txt` with **absolute paths** to an old data root (not `<repo>/Data`), run once (after backing up the file):

```bash
cd nest_vnn
python scripts/rewrite_jobs_portable_paths.py
```

Then apply the same **`-modeldir` / `_exp_` / `nest_vnn_logs/`** transforms as in the current `jobs.txt`, or keep this repository’s checked-in `jobs.txt` as the source of truth.