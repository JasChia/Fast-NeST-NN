# Post-Training Statistics

Extra statistics computed from models a training run already saved. These read
`results/D{drug}/D{drug}_{experiment}/` trees and write CSVs.

Shared conventions (data layout, jobs): [`../README.md`](../README.md).
Environment: root `environment.yml`, `conda activate fast-nest-nn`.

| Script | What it does |
|---|---|
| `score_models.py` | Re-score the 9 neural methods |
| `score_nestvnn.py` | Re-score NeST-VNN |
| `score_lightgbm.py` | Re-score LightGBM |
| `aggregate_scores.py` | Merge + paired Wilcoxon across methods |
| `compare_lightgbm_fnest.py` | LightGBM vs fNeST-NN head-to-head |
| `neurons_per_assembly_sweep_nestvnn.py` | Vary neurons-per-assembly, k=1…10 |
| `neurons_per_assembly_sweep_fnest.py` | Same, for fNeST-NN |
| `neurons_per_assembly_aggregate.py` | Summarise the sweep |

---

## Re-scoring saved models

A finished run saves `best_model/metrics.csv` at 4 decimal places, and the tuners differ
in what they record. These scripts reload the saved checkpoint and recompute Pearson,
Spearman, R², MSE and RMSE in float64, in one schema across every method, keyed by
`(method, drug, experiment, seed, best_trial)` so methods can be paired.

### Run it

```bash
cd ArchitecturePerformanceExperiments/PostTrainingStatistics

# one of: fNeST-NN ERK_SNN RP-fNeST Dense-fNeST FCNN GP-NN LP-NN RSNN UGP-NN
python score_models.py --method ERK_SNN \
  --results-root ../ERK_SNN --code-dir ../ERK_SNN \
  --data-root ../../Data/nest_shuffle_data/CombatLog2TPM \
  --out scores/ERK_SNN_scores.csv --device cuda

# NeST-VNN defaults to the right paths inside a clone
python score_nestvnn.py --out scores/NeST-VNN_scores.csv --device cuda

python score_lightgbm.py \
  --results-root ../LightGBM --code-dir ../LightGBM \
  --data-root ../../Data/nest_shuffle_data/CombatLog2TPM \
  --out scores/LightGBM_scores.csv

python aggregate_scores.py --scores-dir scores
```

The nine `score_models.py` invocations are independent — run them in parallel.

### Arguments

`score_models.py` — `--method`, `--results-root`, `--code-dir`, `--data-root`, `--out`
required; `--device` (`cuda`), `--limit` (0 = all), `--loader-from` (module providing
`load_*` helpers, if not the tuner).

`score_lightgbm.py` — `--results-root`, `--code-dir`, `--data-root`, `--out` required;
`--limit`.

`score_nestvnn.py` — everything defaults into the clone (`../NeST-VNN`, and `<repo>/Data`
or `$FAST_NEST_DATA_ROOT`). Override with `--results-root`, `--code-dir`, `--data-root`,
`--onto`, `--gene2id`, `--out`, `--device`, `--limit`, `--drugs 127,151`,
`--experiments 0,16`. `--zscore-method` defaults to `auc`, matching the tuner — the
bundled jobs never pass `-zscore_method`.

`aggregate_scores.py` — `--scores-dir` (default `./scores`), summaries written back into
it; `--drugs` (default: the reported 10) restricts which drugs are summarised, `--drugs
all` keeps every drug found.

### Check the output before using it

Every row carries `verify_ok` and `note`. Each script rebuilds the architecture from the
run's own saved hyperparameters and calls `load_state_dict(..., strict=True)` — the same
path the tuner used — then compares its numbers against what the run stored. Wrong mask
seed, wrong activation or wrong split cannot pass silently.

```bash
awk -F, 'NR>1 && $(NF-1)!=1' scores/ERK_SNN_scores.csv   # should print nothing
```

Three comparisons are deliberately loose, for reasons measured and documented at the top
of `score_models.py`: the originals computed Pearson in float32 via torchmetrics; Spearman
is tie-sensitive at 4 dp; and runs with near-constant or tie-dominated predictions are
flagged (`NEAR_CONSTANT_PRED`, `TIE_DOMINATED`) instead of compared. The binding check is
an implied-MSE gate against full-precision R² from `final_results.json`.

### Output columns

29 columns. Identity: `method, drug, experiment, seed, best_trial, activation, dropout`.
Recomputed: `n_val`, `val_{pearson,spearman,r2,mse,rmse}`, `n_test`,
`test_{pearson,spearman,r2,mse,rmse}`. Verification: `stored_*`, their deltas,
`verify_ok`, `note`.

`aggregate_scores.py` writes `all_methods_scores.csv`, `summary_by_method.csv`,
`summary_by_method_drug.csv`, and `paired_wilcoxon[_druglevel]_vs_<reference>.csv`.
By default everything is restricted to the reported 10 drugs. Within that, pairing uses
the intersection of `(drug, experiment)` keys, since a method may not cover them all.

### `compare_lightgbm_fnest.py`

A focused drug-level comparison that reads each method's native outputs rather than the
re-scored CSVs. Arguments: `--lightgbm-dir` (`../LightGBM`, needs `lightgbm_results.csv`
from `collect_results.py`), `--fnest-results` (`../fNeST-NN/results`), `--out`.

Each drug is collapsed to its mean paired difference before Wilcoxon, because the 50
experiments within a drug are overlapping 80/20 resamples of one cell-line pool, not
independent replicates. That leaves n=10 drugs, so the smallest attainable two-sided
p is 2/2¹⁰ = 0.00195.

---

## Neurons-per-assembly sweep

How much does performance depend on neurons per assembly (`genotype_hiddens` for NeST-VNN,
`nodes_per_assembly` for fNeST-NN), which both methods fix at 4?

Hyperparameters are **not** re-tuned. For each `(drug, experiment)` the script reads that
run's selected hyperparameters from `final_results.json` and replays them through a
`PinnedTrial` that answers every Optuna `suggest_*` with the stored value. Only k varies,
so k-values pair exactly within a run.

### Run it

```bash
# one unit of work = one (drug, experiment) = 10 trainings
python neurons_per_assembly_sweep_nestvnn.py \
  --drug 127 --experiments 0-0 --kmin 1 --kmax 10 --cuda 0 \
  --out sweep/sweep_D127_0.csv --scratch /tmp/sweep_D127_0
rm -rf /tmp/sweep_D127_0

python neurons_per_assembly_aggregate.py --sweep-dir sweep --out-dir scores
```

Fan out across `(drug, experiment)`:

```bash
for d in 5 80 99 127 151 188 244 273 298 380; do
  for e in $(seq 0 49); do
    echo "python neurons_per_assembly_sweep_fnest.py --drug $d --experiments $e-$e \
          --out sweep/sweep_fnest_D${d}_${e}.csv --scratch /tmp/sw_${d}_${e}"
  done
done | xargs -P 4 -I{} bash -c '{}'
```

An experiment whose output CSV already has all 10 rows is skipped, so an interrupted sweep
can be rerun.

**Scale.** Both sweeps cover the reported 10 drugs
`[5, 80, 99, 127, 151, 188, 244, 273, 298, 380]` × 50 experiments × 10 k = 5,000
trainings each.

### Arguments (both sweep scripts)

`--drug` and `--experiments` (e.g. `0-49` or `7`) required, plus `--out`. Optional:
`--kmin`/`--kmax` (1/10), `--max-epochs` (500, matching the tuners), `--cuda` (0),
`--scratch` (checkpoint scratch dir — delete it afterwards).

`neurons_per_assembly_aggregate.py` — `--sweep-dir` (`./sweep`), `--out-dir` (`./scores`).
Reports performance vs k, a drug × k table, and each k paired against the k=4 control.
Because the 50 experiments per drug are overlapping resamples, the headline test collapses
each drug to its mean and runs Wilcoxon across drugs.

### Check that k took effect

The tuner's log line prints `h=4` whatever k is, because `setup_trials` prints before the
override lands. Ignore it. Two other checks run instead: the model class is wrapped to
record the width it was actually built with (`k_used`, plus `n_params`), and the saved
checkpoint's `final_aux_linear_layer.weight` shape is asserted against k. Both land in the
output — treat a row without `note=arch_ok(k=N)` as suspect.
