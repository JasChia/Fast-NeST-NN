# LightGBM

The non-neural baseline in the architecture comparison: same drugs, same splits, same
Optuna protocol and same metrics as the neural methods, on CPU.

Shared conventions (data layout, jobs): [`../README.md`](../README.md).
Environment: root `environment.yml`, `conda activate fast-nest-nn` (includes `lightgbm`).

## Run it

`jobs/lightgbm_jobs.txt` holds one command per experiment, 500 lines, with paths relative
to this directory. Submit the lines to your scheduler, or run them directly:

```bash
cd ArchitecturePerformanceExperiments/LightGBM
grep -v '^#' jobs/lightgbm_jobs.txt | grep -v '^$' | bash
```

A single experiment looks like:

```bash
python -u lightgbm_hparam_tuner.py -drug 5 \
  -train_file ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/train_test_splits/experiment_0/true_training_data.txt \
  -val_file   .../validation_data.txt \
  -test_file  .../test_data.txt \
  -cell2id    ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/D5_cell2ind.txt \
  -ge_data    ../../Data/nest_shuffle_data/CombatLog2TPM/Drug5/D5_CL/D5_GE_Data.txt \
  -n_trials 100 -seed 0 -output_dir results/D5/D5_0
```

Then collect across experiments:

```bash
python collect_results.py     # -> lightgbm_results.csv
```

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `-drug` | `-1` | Drug ID |
| `-train_file`, `-val_file`, `-test_file` | — | Split files for one experiment |
| `-cell2id` | — | Cell line → index mapping |
| `-ge_data` | — | Gene expression matrix |
| `-output_dir` | `./results` | Per-experiment output directory |
| `-n_trials` | `100` | Optuna trials |
| `-max_epochs` | `500` | Max boosting rounds per trial |
| `-seed` | `42` | Experiment seed (`experiment_id * 1000` in the bundled jobs) |
| `-num_threads` | `2` | LightGBM threads |
| `-cuda` | `0` | Ignored; accepted for CLI parity with the neural methods |

## Outputs

```
results/D{drug}/D{drug}_{experiment}/
├── final_results.json
├── best_model_results.csv
├── best_model/          # hyperparameters.json, metrics.csv
└── trials/trial_{n}/    # metrics.csv
```

`metrics.csv` carries Pearson Correlation, Spearman Correlation, R² Score, Loss (MSE) and
RMSE for validation and test — the same five rows every other method writes.

To recompute these from saved models without retraining, or to run cross-method paired
statistics, see [`../PostTrainingStatistics/`](../PostTrainingStatistics/).

## Protocol

Matched to the neural methods so comparisons are paired:

- 10 drugs: `[5, 80, 99, 127, 151, 188, 244, 273, 298, 380]`
- 50 experiments per drug, each a different split and seed (`seed = experiment_id * 1000`)
- 100 Optuna trials per experiment, TPE seeded with the experiment seed
- 500 experiments × 100 trials = 50,000 trials

The trial with the highest validation R² is copied to `best_model/`, as in the neural
methods.

## Code

`lightgbm_model.py` holds the parameter block and search space — the role `fc_nn.py` and
`fnest_nn.py` play for the neural methods. `lightgbm_hparam_tuner.py` drives Optuna and
the train/val/test protocol.

| Hyperparameter | Range | Type |
|---|---|---|
| `learning_rate` | 1e-3 – 3e-1 | log-uniform float |
| `num_leaves` | 7 – 255 | log-uniform int |
| `max_depth` | -1, 3, 5, 8, 12 | categorical |
| `min_child_samples` | 5 – 100 | int |
| `feature_fraction` | 0.3 – 1.0 | float |
| `bagging_fraction` | 0.5 – 1.0 | float |
| `bagging_freq` | 1 – 7 | int |
| `lambda_l1`, `lambda_l2` | 1e-8 – 10.0 | log-uniform float |

`FIXED_PARAMS` sets `deterministic: True` and `force_row_wise: True`. Both are needed:
without them LightGBM's histogram construction depends on thread count, so the same seed
and the same `-num_threads` still produce different boosters across machines.

`seed`, `bagging_seed` and `feature_fraction_seed` are set to `trial.number + -seed`, so
each trial is independently seeded.
