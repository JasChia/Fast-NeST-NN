# FCNN

**Fully Connected Neural Network** — dense-layer baseline with matched width/depth to the NeST graph-sized models (§2.4, Table 2).

**Shared setup (data paths and job-running conventions):** see **[`../README.md`](../README.md)** in **`ArchitecturePerformanceExperiments/`**.

## Files

- `fc_nn.py` — model definition.
- `fc_nn_hparam_tuner.py` — Optuna training driver.
- `jobs/fc_nn_jobs.txt` — bundled training commands (paths relative to `<repo>/Data/` via `../../Data/...`).
- `jobs/fc_nn_advanced_jobs.json` — same jobs as JSON (optional).

## Usage

```bash
cd ArchitecturePerformanceExperiments/FCNN
grep -v '^#' jobs/fc_nn_jobs.txt | grep -v '^$' | bash
```

Use your cluster’s scheduler for parallel runs. Ensure **`Data/`** exists at the repo root after extracting `Data_archives/`. Override data location by setting **`FAST_NEST_DATA_ROOT`** and replacing the `../../Data` prefix in the job file if needed. Outputs under `results/` (gitignored).

## Environment and CLI

Activate the Conda env from root **`environment.yml`** (`conda activate fast-nest-nn` after `conda env create -f environment.yml`).

**`fc_nn_hparam_tuner.py`** accepts **`-n_trials`**, **`-max_epochs`** (default **500**), **`-cuda`**, paths **`-train_file`**, **`-val_file`**, **`-test_file`**, **`-cell2id`**, **`-ge_data`**, **`-output_dir`**.

**Expected outputs:** `FC_NN_HTune.log`, `trials/trial_*/model_best.pt`, `best_model/`, `final_results.json`, per-trial `metrics.csv`.
