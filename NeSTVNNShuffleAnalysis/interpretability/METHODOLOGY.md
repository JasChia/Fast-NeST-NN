# Interpretability scoring — exact methodology

Importance is the **NeST-VNN system importance (SI) / RLIPP** score, taken from the
authors' own implementation. No alternative or custom importance measure is used
anywhere in this directory.

**Source:** `rlipp_calculator.py` and `rlipp_helper.py` are vendored **verbatim**
from [idekerlab/nest_vnn](https://github.com/idekerlab/nest_vnn) @ `ef1b739d91cf9c91b40fb8ee4f739737cc4d0210`.
Definition in Zhao et al., *Cancer Discovery*
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

Two properties matter:

1. **The regression target is the model's own prediction, not the measured AUC.**
   SI quantifies how much of *the model's decision* an assembly explains. This is
   what makes the score well-defined and comparable for a permuted model that
   predicts poorly.
2. **SI is the parent term alone.** The paper uses a "variation of RLIPP" that drops
   the child normalisation; `RLIPP = P_rho/C_rho` is also emitted by the code.
   The paper calls assemblies with **SI ≥ 0.5** important.

Three implementation details: PCA to `n_components =
genotype_hiddens` (=4) is fit on the design matrix before ridge; `RidgeCV(cv=5)`
uses CV only to select α and the reported ρ is **in-sample**; and "children" means
all ontology children — child assemblies (4 columns each) **and** directly
annotated genes (1 column each).

## 2. Per-run pipeline

For each trained model (10 drugs × {unshuffled, shuffled, degshuf} × 50 experiments = 1,500):

```
src/predict.py            # repo's own, transcriptomic input variant
  -load   <run>/model_best.pt
  -predict <exp>/test_data.txt      -std <run>/std.txt   -zscore_method auc
  -transcriptomic  D{N}_GE_Data.txt            (unshuffled arm)
                or <exp>/shuffled_gene_expression.txt  (shuffled arm)
  -> predict.txt                      model-predicted AUC per test cell
  -> hidden/<element>.hidden          131 assemblies (n_test×4) + 689 genes (n_test×1)

calc_scores_serial.py     # drives the VERBATIM RLIPPCalculator methods
  -> rlipp.out      Term  P_rho  P_pval  C_rho  C_pval  RLIPP
  -> gene_rho.out   Gene  Rho    P_val
```

`degshuf` arm: `-load` from `degree_preserving_shuffle/nest_vnn_logs/…`, `-transcriptomic D{N}_GE_Data.txt`; the scorer's `-ontology` is `degree_preserving_shuffle/ontologies/red_ontology_degshuf_exp{E}.txt`.

The transcriptomic matrix always matches the one that model was **trained** on;
test split and `std.txt` are the run's own.

## 3. The one deviation from upstream

`RLIPPCalculator.calc_scores()` dispatches per-term and per-gene work through
`joblib.Parallel(backend="multiprocessing", n_jobs=cpu_count)`.

`calc_scores_serial.py` changes **only the executor**. Every number comes
from the unmodified `exec_lm`, `calc_term_rlipp` and `calc_gene_rho` methods, and
the iteration order, drug ordering, headers and format strings replicate
`calc_scores()`. `verify_equivalence.py` asserts the result is **byte-identical**
to upstream `calc_scores()` run single-threaded.

## 4. Scale and storage

`predict.py` writes 1,640 per-element files per run. Write them
to node-local scratch and delete them after scoring, keeping only `rlipp.out`,
`gene_rho.out` and `predict.txt` (3 files × 1,500 runs). See
[`README.md`](README.md) for the per-run command.

Cost: ~137 s per run (predict 127 s + scoring 10 s) on one mid-range GPU; 1,500
runs at 16-way concurrency is roughly 3.5 h.

## 5. Scope note

The 10 reported drugs are 5, 80, 99, 127, 151, 188, 244, 273, 298, 380 (the set in `Data_archives/`).

## 6. Files

| File | Role |
|---|---|
| `rlipp_calculator.py`, `rlipp_helper.py` | upstream, **verbatim** |
| `calc_scores_serial.py` | executor-only driver (§3) |
| `verify_equivalence.py` | byte-identity gate vs upstream |
| `run_importance.sh` | one run: predict → score → clean up |
| `collect_importance.py` | aggregate all runs into tidy CSVs |
| `analyze_rankings.py` | shuffled vs unshuffled importance-ranking comparison (SI / Rho) |
| `build_supp_tables.py` | unshuffled vs perturbed importance agreement on the same split (`table4_*.csv`) |
| `results/D{N}_{arm}_exp_{e}/` | `rlipp.out`, `gene_rho.out`, `predict.txt` |
