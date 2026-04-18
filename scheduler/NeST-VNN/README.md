# NeST-VNN (Nested Systems in Tumors Visual Neural Network)

Training and scheduling code for the **original NeST-VNN** implementation (Zhao et al.), as used in *Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction* (ICIBM 2026) for permutation tests and performance baselines.

## Paper mapping

- **Gene-order permutation** (§2.3, §3.1): compare shuffled vs unshuffled input genes; supplementary Pearson tables.
- **Performance / runtime** (§2–3): NeST-VNN vs fNeST-NN and other architectures.

## Layout

| File | Role |
|------|------|
| `nest_vnn_hparam_tuner.py` | Optuna HPO; uses `src/` for the published NeST-VNN trainer stack. |
| `jobs/nest_vnn_jobs.txt` | Bundled training commands (paths use `../../Data/...` from this folder). |
| `jobs/nest_vnn_advanced_jobs.json` | Same jobs as JSON (optional). |
| `cleanup_failed_experiment.py` | Operations helper. |

## Data

Populate **`Data/`** at the repository root from **`Data_archives/`** (see root `README.md`). Job files reference ontology and drug bundles under `../../Data/`.

## Workflow

```bash
cd scheduler/NeST-VNN
grep -v '^#' jobs/nest_vnn_jobs.txt | grep -v '^$' | bash
```

For clusters, submit each command line with Slurm or GNU Parallel instead of a sequential shell loop.

Outputs: `results/`, `logs/`, `shared/` (gitignored).

## Environment

Use the Conda **`environment.yml`** at the repository root (see root **`README.md`**) or **`cuda11_env`** with PyTorch, Optuna, pandas, **`torchmetrics`**, and the NeST-VNN **`src/`** dependencies.

## Manual run (`nest_vnn_hparam_tuner.py`)

In addition to drug split files (**`-train_file`**, **`-val_file`**, **`-test_file`**, **`-cell2id`**), NeST-VNN requires ontology and transcriptomic inputs: **`-onto`** (`red_ontology.txt`), **`-gene2id`** (`red_gene2ind.txt`), **`-transcriptomic`** (cell-line × gene matrix for the drug bundle).

Common flags: **`-n_trials`**, **`-max_epochs`** (default **500**), **`-cuda`**, **`-zscore_method`**, **`-output_dir`**.

```bash
conda activate fast-nest-nn
cd scheduler/NeST-VNN
python -u nest_vnn_hparam_tuner.py \
  -cuda 0 -drug 298 \
  -train_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/train_test_splits/experiment_0/true_training_data.txt \
  -val_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/train_test_splits/experiment_0/validation_data.txt \
  -test_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/train_test_splits/experiment_0/test_data.txt \
  -cell2id ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/D298_cell2ind.txt \
  -onto ../../Data/red_ontology.txt \
  -gene2id ../../Data/red_gene2ind.txt \
  -transcriptomic ../../Data/nest_shuffle_data/CombatLog2TPM/Drug298/D298_CL/D298_GE_Data.txt \
  -n_trials 2 -max_epochs 10 -seed 0 -output_dir results/D298/D298_0
```

Adjust paths if your **`Data/`** tree uses the flat **`D{N}_CL/`** layout.

## Expected outputs

Same pattern as other tuners: **`nest_vnn_HTune.log`**, **`trials/trial_*/model_best.pt`**, **`best_model/`**, **`final_results.json`**, per-trial **`metrics.csv`**.
