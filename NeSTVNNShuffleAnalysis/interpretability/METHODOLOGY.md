# Interpretability scoring — exact methodology

Importance is the **NeST-VNN system importance (SI) / RLIPP** score, taken from the
authors' own implementation. No alternative or custom importance measure is used
anywhere in this directory.

**Source:** `rlipp_calculator.py` and `rlipp_helper.py` are vendored **verbatim**
from [idekerlab/nest_vnn](https://github.com/idekerlab/nest_vnn) @ `ef1b739d91cf9c91b40fb8ee4f739737cc4d0210`
(verified identical by `diff`). Definition in Zhao et al., *Cancer Discovery*
2024;14:508–23, Methods → "SI Score".

---

## 1. Definition

For a trained model and its test cells:

| Level | Score | Definition |
|---|---|---|
| Assembly | **P_rho** (= the paper's **SI**) | Spearman(ridge prediction from that assembly's neurons, **model-predicted AUC**) |
| Assembly | **C_rho** | same, from the concatenated neurons of all its ontology children |
| Assembly | **RLIPP** | `P_rho / C_rho` |
| Gene | **Rho** | Spearman(gene's 1-D embedding, **model-predicted AUC**) |

Two properties matter and are easy to misread:

1. **The regression target is the model's own prediction, not the measured AUC.**
   SI quantifies how much of *the model's decision* an assembly explains. This is
   what makes the score well-defined and comparable for a permuted model that
   predicts poorly.
2. **SI is the parent term alone.** The paper uses a "variation of RLIPP" that drops
   the child normalisation; `RLIPP = P_rho/C_rho` is also emitted by the code.
   The paper calls assemblies with **SI ≥ 0.5** important.

Three details are in the code but not the paper: PCA to `n_components =
genotype_hiddens` (=4) is fit on the design matrix before ridge; `RidgeCV(cv=5)`
uses CV only to select α and the reported ρ is **in-sample**; and "children" means
all ontology children — child assemblies (4 columns each) **and** directly
annotated genes (1 column each).

## 2. Per-run pipeline

For each of the 1,200 trained models (12 drugs × {unshuffled, shuffled} × 50 experiments):

```
src/predict.py            # repo's own, transcriptomic input variant
  -load   <run>/model_best.pt
  -predict <exp>/test_data.txt      -std <run>/std.txt   -zscore_method auc
  -transcriptomic  D{N}_GE_Data.txt            (unshuffled arm)
                or <exp>/shuffled_gene_expression.txt  (shuffled arm)
  -> predict.txt                      model-predicted AUC per test cell
  -> hidden/<element>.hidden          131 assemblies (157×4) + 689 genes (157×1)

calc_scores_serial.py     # drives the VERBATIM RLIPPCalculator methods
  -> rlipp.out      Term  P_rho  P_pval  C_rho  C_pval  RLIPP
  -> gene_rho.out   Gene  Rho    P_val
```

The transcriptomic matrix always matches the one that model was **trained** on;
test split and `std.txt` are the run's own. Verified: re-deriving predictions for
`D5_unshuffled_exp_0` reproduces test Pearson **0.5709**, the value that run
recorded at training time.

## 3. The one deviation, and why it is safe

`RLIPPCalculator.calc_scores()` dispatches per-term and per-gene work through
`joblib.Parallel(backend="multiprocessing", n_jobs=cpu_count)`, forking after the
class has already used `multiprocessing.Pool`. On this cluster that wrapper hangs:
**>43 min without emitting a row**, while the same work takes ~10 s.

`calc_scores_serial.py` therefore changes **only the executor**. Every number comes
from the unmodified `exec_lm`, `calc_term_rlipp` and `calc_gene_rho` methods, and
the iteration order, drug ordering, headers and format strings replicate
`calc_scores()`. `verify_equivalence.py` asserts the result is **byte-identical**
to upstream `calc_scores()` run single-threaded — confirmed on D5 exp 0 for both
`rlipp.out` and `gene_rho.out`.

## 4. Scale and storage

`predict.py` writes 1,640 per-element files per run; across 1,200 runs that is
~2M files, which will exhaust a typical shared-filesystem inode quota. Write them
to node-local scratch and delete them after scoring, keeping only `rlipp.out`,
`gene_rho.out` and `predict.txt` (3 files × 1,200 runs). See
[`README.md`](README.md) for the per-run command.

Cost: ~137 s per run (predict 127 s + scoring 10 s) on one mid-range GPU; 1,200
runs at 16-way concurrency is roughly 3 h.

## 5. Scope note

Trained models exist for **12** drugs (5, 57, 80, 99, 127, 151, 188, 201, 244, 273,
298, 380) and all 12 are scored here. The shuffle study's grid is complete and
uniform over those 12: 12 drugs x {unshuffled, shuffled} x 50 experiments = 1,200
runs, 100 saved models per drug. The architecture comparison covers the same 12 for
every method except fNeST-NN and LightGBM, which cover 10 (no D57, no D201).

**Reporting uses 10 drugs**, the set in `Data_archives/` and the
`ArchitecturePerformanceExperiments/*/jobs/` lists:

    5, 80, 99, 127, 151, 188, 244, 273, 298, 380

D57 and D201 were trained but are filtered out before reporting. Both studies report
this same 10; `analyze_rankings.py` applies the filter, which is why
`assembly_importance.csv` / `gene_importance.csv` carry 12 drugs while every
`ranking_*` table carries 10.

One stale artifact to ignore: `nest_vnn_jobs/jobs.txt` lists D201 and omits D380.
It predates the filtering and is neither what was run (12 drugs) nor what is
reported (the 10 above). Do not take a drug set from it.

## 6. Files

| File | Role |
|---|---|
| `rlipp_calculator.py`, `rlipp_helper.py` | upstream, **verbatim** |
| `calc_scores_serial.py` | executor-only driver (§3) |
| `verify_equivalence.py` | byte-identity gate vs upstream |
| `run_importance.sh` | one run: predict → score → clean up |
| `collect_importance.py` | aggregate all runs into tidy CSVs |
| `results/D{N}_{arm}_exp_{e}/` | `rlipp.out`, `gene_rho.out`, `predict.txt` |
