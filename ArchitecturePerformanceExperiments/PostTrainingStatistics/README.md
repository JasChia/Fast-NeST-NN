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
| `performance_tables.py` | Per-drug tables: main Table 5, Supp Tables S4–S6 |
| `neurons_per_assembly_sweep_nestvnn.py` | Vary neurons-per-assembly, k=1…10 |
| `neurons_per_assembly_sweep_fnest.py` | Same, for fNeST-NN |
| `neurons_per_assembly_aggregate.py` | Summarise the sweep |
| `neurons_per_assembly_tests.py` | Per-drug paired tests of each k against k=4 |

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
all` keeps every drug found; `--degenerate` (`zero`, the default, or `drop`) sets how
runs that collapsed to a constant prediction are scored — see below.

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
the intersection of `(drug, experiment)` keys.

### Runs that collapse to a constant prediction

A run whose predictions are all one value has no defined Pearson or Spearman. Letting the
mean skip those rows is available-case analysis under informative missingness: the value
is absent *because* the model failed, so the surviving mean flatters whichever method
fails most. `--degenerate zero` (the default) scores them 0, the limiting correlation of a
constant predictor; `--degenerate drop` reproduces the skip-the-NaN behaviour. R2 and RMSE
are unaffected either way — both are defined for a constant predictor and already
penalise it.

`summary_by_method.csv` carries `n_degenerate` per method, so the rate is reportable next
to the metric rather than buried in it. Run both modes if you want the sensitivity.

### `compare_lightgbm_fnest.py`

A focused drug-level comparison that reads each method's native outputs rather than the
re-scored CSVs. Arguments: `--lightgbm-dir` (`../LightGBM`, needs `lightgbm_results.csv`
from `collect_results.py`), `--fnest-results` (`../fNeST-NN/results`), `--out`.

Each drug is collapsed to its mean paired difference before Wilcoxon, because the 50
experiments within a drug are overlapping 80/20 resamples of one cell-line pool, not
independent replicates. That leaves n=10 drugs, so the smallest attainable two-sided
p is 2/2¹⁰ = 0.00195.

### `performance_tables.py`

Builds the paper's per-drug tables from the re-scored CSVs: every method, LightGBM included,
against fNeST-NN (main Table 5 for R², Supplementary Tables S4 Pearson, S5 RMSE and S6 Spearman).

```bash
python performance_tables.py --scores-dir scores   # -> perf_<metric>_all_methods.csv, perf_lightgbm_vs_fnest.csv
```

Mean ± SD (ddof=1) per drug over its 50 experiments; a paired Wilcoxon signed-rank against
fNeST-NN *within* each drug, pairing by experiment (zero_method `wilcox`, no continuity
correction, normal approximation); Benjamini–Hochberg within each method and metric across
the 10 drugs. With the defaults (full-precision re-scored values, constant-prediction runs
scored 0) this reproduces every printed cell of Table 5 and S4–S6. `--degenerate` works as in
`aggregate_scores.py`.

`--values stored --degenerate drop` uses the 4-dp values each tuner stored and drops the
constant-prediction runs.

---

## Neurons-per-assembly sweep

How much does performance depend on neurons per assembly (`genotype_hiddens` for NeST-VNN,
`nodes_per_assembly` for fNeST-NN), which both methods fix at 4?

Hyperparameters are **not** re-tuned. For each `(drug, experiment)` the script reads that
run's selected hyperparameters from `final_results.json` and replays them through a
`PinnedTrial` that answers every Optuna `suggest_*` with the stored value. Only k varies,
so k-values pair exactly within a run.

### Run it

Keep the two methods in separate sweep directories: the aggregator pools whatever its
`--pattern` matches, so run it once per method.

```bash
# one unit of work = one (drug, experiment) = 10 trainings
python neurons_per_assembly_sweep_nestvnn.py \
  --drug 127 --experiments 0-0 --kmin 1 --kmax 10 --cuda 0 \
  --out sweep_nestvnn/sweep_D127_0.csv --scratch /tmp/sweep_D127_0
rm -rf /tmp/sweep_D127_0

python neurons_per_assembly_aggregate.py --sweep-dir sweep_nestvnn --out-dir scores
```

Fan out across `(drug, experiment)`:

```bash
for d in 5 80 99 127 151 188 244 273 298 380; do
  for e in $(seq 0 49); do
    echo "python neurons_per_assembly_sweep_fnest.py --drug $d --experiments $e-$e \
          --out sweep_fnest/sweep_D${d}_${e}.csv --scratch /tmp/sw_${d}_${e}"
  done
done | xargs -P 4 -I{} bash -c '{}'

python neurons_per_assembly_aggregate.py --sweep-dir sweep_fnest --out-dir scores_fnest
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

`neurons_per_assembly_aggregate.py` — `--sweep-dir` (`./sweep`), `--out-dir`
(`./scores`), `--pattern` (`sweep_D*_*.csv`).
Reports performance vs k, a drug × k table, and each k paired against the k=4 control.
Its headline test collapses each drug to its mean and runs Wilcoxon across drugs (n=10).

### Per-drug paired tests

`neurons_per_assembly_tests.py` keeps all 50 experiments per drug instead of collapsing
them, pairing width k with k=4 **by cross-validation split**: experiment e at width k is
compared with experiment e at k=4, which shares its train/validation/test partition and
its pinned hyperparameters. It tests each drug separately (n=50 pairs) and also writes one
pooled test over all 500 pairs per (metric, k).

Wilcoxon signed-rank with `zero_method="wilcox"`, no continuity correction, normal
approximation — the variant the published n=50 tables used — then Benjamini–Hochberg
across the nine k comparisons within each drug and metric. Metrics: R², Pearson,
Spearman, RMSE. Runs that collapsed to a constant prediction have undefined correlations
and are scored 0 by default, as in `aggregate_scores.py`.

```bash
# after the sweep and the aggregator (which writes sens_all_runs.csv)
python neurons_per_assembly_tests.py --scores-dir scores_fnest
python neurons_per_assembly_tests.py --scores-dir scores        # NeST-VNN
```

`--scores-dir` (`./scores`, must contain `sens_all_runs.csv`), `--out-dir` (defaults to
`--scores-dir`), `--degenerate` (`zero` or `drop`). Writes three CSVs beside the input:

| Output | Contents |
|---|---|
| `sens_tests_by_drug.csv` | one row per (drug, metric, k): `n_pairs`, `mean_k`, `mean_k4`, `delta_vs_k4`, `p`, `q_BH` |
| `sens_tests_summary.csv` | per (metric, k): how many drugs are significantly lower or higher than at k=4 |
| `sens_tests_pooled.csv` | per (metric, k): one test over all 500 pairs, BH across the nine k |

Supplementary Table S7 reports the pooled test: the mean ± SD of each k over all 500 runs
(from `sens_all_runs.csv`) and the BH-adjusted pooled p-value against k=4.

### Check that k took effect

Two checks run: the model class is wrapped to
record the width it was actually built with (`k_used`, plus `n_params`), and the saved
checkpoint's `final_aux_linear_layer.weight` shape is asserted against k. Both land in the
output. The NeST-VNN sweep also writes `note=arch_ok(k=N)`; treat a row without it as
suspect. The fNeST-NN sweep records `k_used` only — check it equals the requested k.
