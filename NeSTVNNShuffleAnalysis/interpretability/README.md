# Interpretability — assembly and gene importance

NeST-VNN system importance (SI) / RLIPP scores computed from models already trained by the
shuffle study in [`../nest_vnn/`](../nest_vnn/), plus how importance rankings change
between arms.

`rlipp_calculator.py` and `rlipp_helper.py` are vendored verbatim from
[idekerlab/nest_vnn](https://github.com/idekerlab/nest_vnn) at commit
`ef1b739d91cf9c91b40fb8ee4f739737cc4d0210`. No custom importance measure is used.

**[`METHODOLOGY.md`](METHODOLOGY.md) defines SI, RLIPP and gene rho.** Read it before
interpreting any output — in particular, the regression target is the model's own
prediction, not measured AUC.

## Score one run

Two steps per `(drug, arm, experiment)`: dump hidden states with the repo's `src/predict.py`,
then score them. `run_importance.sh <drug> <arm> <experiment> [cpu_count]` does both for any
of the three arms (`unshuffled`, `shuffled`, `degshuf`); each run is independent. The steps for the unshuffled and shuffled arms are:

```bash
D=127; ARM=unshuffled; E=0; NCPU=8
ROOT=$(cd ../.. && pwd)
NV=$ROOT/NeSTVNNShuffleAnalysis/nest_vnn
I=$ROOT/NeSTVNNShuffleAnalysis/interpretability
CL=$ROOT/Data/nest_shuffle_data/CombatLog2TPM/Drug$D/D${D}_CL
EX=$CL/train_test_splits/experiment_$E
RUN=$NV/nest_vnn_logs/D${D}_${ARM}_exp_${E}
OUT=$I/results/D${D}_${ARM}_exp_${E}

# must be the matrix this arm was TRAINED on
if [ "$ARM" = shuffled ]; then GE=$EX/shuffled_gene_expression.txt
                         else GE=$CL/D${D}_GE_Data.txt; fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/hidden" "$OUT"

cd "$NV"
python -u src/predict.py -predict "$EX/test_data.txt" -batchsize 1000 \
  -gene2id "$ROOT/Data/red_gene2ind.txt" -cell2id "$CL/D${D}_cell2ind.txt" \
  -load "$RUN/model_best.pt" -hidden "$TMP/hidden" -result "$TMP/predict" -cuda 0 \
  -transcriptomic "$GE" -zscore_method auc -std "$RUN/std.txt"

python -u "$I/calc_scores_serial.py" \
  -hidden "$TMP/hidden" -ontology "$ROOT/Data/red_ontology.txt" \
  -test "$EX/test_data.txt" -predicted "$TMP/predict.txt" \
  -gene2idfile "$ROOT/Data/red_gene2ind.txt" -cell2idfile "$CL/D${D}_cell2ind.txt" \
  -sys_output "$OUT/rlipp.out" -gene_output "$OUT/gene_rho.out" \
  -cpu_count "$NCPU" -drug_count 0 -genotype_hiddens 4

cp "$TMP/predict.txt" "$OUT/predict.txt"
```

Four things in there matter:

- **`-std "$RUN/std.txt"`** must be that run's own file.
- **`-transcriptomic`** must match what the arm trained on.
- **`degshuf` runs** live in `../degree_preserving_shuffle/nest_vnn_logs/` and train on the
  unpermuted matrix. The scorer's `-ontology` must be that experiment's rewired file,
  `../degree_preserving_shuffle/ontologies/red_ontology_degshuf_exp{E}.txt`; `predict.py`
  needs none, because the pickled model already carries its wiring.
- **Hidden dumps go to scratch and are deleted.** `predict.py` writes ~1,640 files per
  run; only `rlipp.out`, `gene_rho.out` and `predict.txt` are kept.

About 137 s per run on one mid-range GPU (predict ~127 s, scoring ~10 s).

## Collect and analyse

```bash
python collect_importance.py    # -> assembly_importance.csv, gene_importance.csv
python analyze_rankings.py > ranking_report.txt   # -> ranking_*.csv
python build_supp_tables.py                       # gene order permutation -> table4_{assembly,overlap,gene}.csv
python build_supp_tables.py --perturbed degshuf   # degree-preserving -> table4_*_degshuf.csv
python verify_equivalence.py <dir with hidden/ and predict.txt> <drug> <experiment>   # checks the vendored calculator
```

`collect_importance.py`, `analyze_rankings.py` and `build_supp_tables.py` resolve paths from their own location; `build_supp_tables.py` also takes `--perturbed`, `--importance-dir` and `--out-dir`.

Gene retention between arms is 51.7% identity-aligned
and 3.4% slot-aligned. Those answer different questions — does the same *gene* stay
important, or the same *position*.
`analyze_rankings.py` writes both (`ranking_gene_by_drug_identity.csv`,
`ranking_gene_by_drug_slot.csv`). Those figures use signed gene rho and compare models
across different splits.

**Main Table 4** comes from `build_supp_tables.py`, which compares the unshuffled model with
the shuffled (default) or degree-preserving (`--perturbed degshuf`) model of the *same* split.
Its Assembly SI, Gene SI and Important Assemblies Shared columns are the mean/SD in
`table4_assembly*.csv`, `table4_gene*.csv` and the `si_ge_0.5` column of `table4_overlap*.csv`;
the Mean row is the unweighted mean of the 10 drug means. Gene scores are aligned by identity, which for
`degshuf` is the input position, and use |rho|. Each per-drug p-value tests the same-split agreement against a resampling reference
(unshuffled models of different splits), not against zero; the script's docstring gives the
exact definitions.

## Files

| Script | Role |
|---|---|
| `run_importance.sh` | Predict + score one run of any arm → `results/D{N}_{arm}_exp_{E}/` |
| `calc_scores_serial.py` | Score one run → `rlipp.out`, `gene_rho.out` |
| `collect_importance.py` | Gather runs into the two importance CSVs |
| `analyze_rankings.py` | Ranking comparisons between arms |
| `build_supp_tables.py` | Same-split agreement for either perturbation, main Table 4 |
| `verify_equivalence.py` | Check the vendored calculator |
| `rlipp_calculator.py`, `rlipp_helper.py` | Vendored upstream — do not edit |

Outputs (`results/`, the importance CSVs, `ranking_*`, `table4_*`) are not tracked.
