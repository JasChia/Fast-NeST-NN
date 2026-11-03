# NeST-VNN shuffle experiment (reproducible pipeline)

This directory is part of the **Fast-NeST-NN** monorepo at **`NeSTVNNShuffleAnalysis/`** (tracked on `main`); clone the repository once and work from this folder.

This workspace implements the **NeST-VNN** gene-order shuffle study: for each of **10 drugs**, **50** paired runs (**shuffled** vs **unshuffled** input gene order), with metrics used in the paper tables.

### Contents

At the repository root, **`NeSTVNNShuffleAnalysis/`** contains only:

```text
NeSTVNNShuffleAnalysis/
├── .gitignore
├── README.md                          # This file
├── Data -> ../Data                    # link to the repo-root Data/
├── create_train_test_splits.py        # Generated the splits and shuffled gene orders
├── nest_vnn/
│   ├── src/                           # PyTorch training stack
│   │   ├── train.py                   # Entrypoint invoked by jobs.txt
│   │   ├── drugcell_nn.py
│   │   ├── vnn_trainer.py
│   │   ├── training_data_wrapper.py
│   │   ├── util.py
│   │   ├── predict.py
│   │   ├── optuna_nn_trainer.py
│   │   └── ccc_loss.py
│   ├── nest_vnn_jobs/
│   │   ├── jobs.txt                   # 1000 training commands (paths relative to nest_vnn/)
│   │   └── run_all_jobs.sh            # Optional: run every line sequentially
│   ├── nest_vnn_logs/                 # One subdirectory per successful job (see below)
│   └── shuffle_assignment_analysis/
│       ├── analyze_shuffle_assignments.py
│       ├── run_shuffle_analysis.sh
│       └── shuffle_performance_table.py     # Table 3 / Supp Tables S1-S3: gene-permuted vs unshuffled
│       # shuffle_assignment_analysis/outputs/ is created at runtime (gitignored)
├── degree_preserving_shuffle/         # Third arm: degree-preserving first-layer shuffle
│   ├── README.md
│   ├── make_degree_shuffled_ontology.py
│   ├── rewiring_stats.py              # how many connections the swaps move
│   ├── degree_preserving_tables.py    # Table 3 / Supp Tables S1-S3: degree-preserving vs unshuffled
│   └── ontologies/                    # red_ontology_degshuf_exp{0..49}.txt
└── interpretability/                  # Assembly / gene importance (SI / RLIPP)
    ├── README.md
    ├── METHODOLOGY.md
    ├── rlipp_calculator.py            # vendored verbatim from idekerlab/nest_vnn
    ├── rlipp_helper.py                # vendored verbatim
    ├── calc_scores_serial.py
    ├── collect_importance.py
    ├── analyze_rankings.py
    ├── run_importance.sh              # predict + score one run of any arm
    ├── build_supp_tables.py           # Table 4: same-split importance agreement
    └── verify_equivalence.py
```

Two companion studies sit alongside `nest_vnn/`:

- **[`degree_preserving_shuffle/`](degree_preserving_shuffle/)** — a **third arm** next to
  unshuffled and shuffled. Rather than permuting the gene expression columns, it randomizes
  which **first-hidden-layer** assembly each gene feeds, **within groups of equal-degree
  assemblies**, leaving every assembly's input degree and every gene's output degree
  unchanged. Everything else, including the expression matrix, is identical to the
  unshuffled arm, so the only variable is the gene-to-assembly assignment.
- **[`interpretability/`](interpretability/)** — **SI / RLIPP** assembly and gene importance
  computed from the trained models in `nest_vnn_logs/`, and how importance rankings change
  between arms.

**`nest_vnn_logs/`** holds per-run outputs when training has been executed. It is **gitignored** (see **`.gitignore`**) because it is large.

Inputs are expected at the **repository root** **`Data/`** (see **Data layout** below). The paper's tables are rebuilt from the per-run outputs by the scripts listed under **Post-training aggregation** below.

All **paths inside `nest_vnn_jobs/jobs.txt`** are **relative to the `nest_vnn/` directory**:

- Training data and ontology: **`../Data/...`** (`NeSTVNNShuffleAnalysis/Data` links to the repo-root `Data/`).
- Run outputs: **`-modeldir nest_vnn_logs/D{drug}_{shuffled|unshuffled}_exp_{id}`** so each job writes under `nest_vnn/nest_vnn_logs/`.

---

## Requirements

- **Python 3** with **PyTorch** (CUDA build recommended), **pandas**, **numpy**, **scipy**, **scikit-learn** (see `nest_vnn/src/` imports).
- **Recommended:** the Conda **`environment.yml`** at the **Fast-NeST-NN** repository root (see parent **`README.md`**), then `conda activate fast-nest-nn`, or any other env with the same packages (PyTorch, Optuna, pandas, scikit-learn, etc.).
- **GPU** with enough memory for DrugCell-style VNN training (same settings as `jobs.txt`: batch size 8, 250 epochs; `vnn_trainer.py` trains all epochs and keeps the last checkpoint that improved validation loss by more than `-delta`, default 1e-3 — there is no early stopping).
- **Environment variable** `CUDA_VISIBLE_DEVICES` (or `-cuda` in `train.py`) set appropriately when launching jobs.

The training code does not hardcode a Conda path; use the root **`environment.yml`** or your own env YAML if you maintain one locally.

---

## Data layout (`Data/`)

Training reads from **`../Data/`** relative to **`nest_vnn/`**; **`NeSTVNNShuffleAnalysis/Data`** is a link to **`<repository-root>/Data/`**.

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

The splits and shuffled matrices were generated by `create_train_test_splits.py` (copy it into
`Data/nest_shuffle_data/` and run it there; it writes `experiment_0`–`experiment_99`; experiments 0–49 are used). Experiment `i` uses seed
`42 + i` for both its 80/20 train/test split and its gene-column permutation. The 64/16
train/validation split is drawn later, inside `util.load_train_data`.

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

## Post-training aggregation

`shuffle_assignment_analysis/shuffle_performance_table.py` builds the paper's shuffled vs
unshuffled comparison (the Gene Permuted column of main Table 3 for R² and of Supplementary
Tables S1–S3 for Pearson, Spearman and RMSE) from every run's `test_metrics.txt`, with RMSE
computed from `test_predictions.txt` / `test_true_labels.txt`. The Degree-Preserving column comes
from `degree_preserving_shuffle/degree_preserving_tables.py`:

```bash
cd NeSTVNNShuffleAnalysis/nest_vnn
python shuffle_assignment_analysis/shuffle_performance_table.py   # -> shuffle_assignment_analysis/outputs/
cd ..
python degree_preserving_shuffle/degree_preserving_tables.py      # -> degree_preserving_shuffle/tables/
```

Mean ± SD (ddof=1) over the 50 experiments, paired Wilcoxon signed-rank (R², Pearson and Spearman
on the 4-dp values in `test_metrics.txt`, RMSE from the saved predictions; normal approximation, no continuity correction), Benjamini–Hochberg across the 10
drugs.

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
