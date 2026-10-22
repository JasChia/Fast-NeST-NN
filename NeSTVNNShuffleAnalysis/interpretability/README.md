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
then score them. No scheduler script is included; each run is independent.

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

Three things in there matter:

- **`-std "$RUN/std.txt"`** must be that run's own file. A different arm's silently
  rescales the inputs.
- **`-transcriptomic`** must match what the arm trained on. Getting it wrong gives
  plausible but meaningless scores.
- **Hidden dumps go to scratch and are deleted.** `predict.py` writes ~1,640 files per
  run; 1,200 runs is ~2M files and will exhaust a typical inode quota. Only `rlipp.out`,
  `gene_rho.out` and `predict.txt` are kept.

About 137 s per run on one mid-range GPU (predict ~127 s, scoring ~10 s).

## Collect and analyse

```bash
python collect_importance.py    # -> assembly_importance.csv, gene_importance.csv
python analyze_rankings.py      # -> ranking_*.csv, ranking_report.txt
python verify_equivalence.py    # checks the vendored calculator
```

These three resolve the repo root from their own location and take no path arguments.

**Say which alignment you mean.** Gene retention between arms is 51.7% identity-aligned
and 3.4% slot-aligned. Those answer different questions — does the same *gene* stay
important, or the same *position* — and support opposite conclusions.
`analyze_rankings.py` writes both (`ranking_gene_by_drug_identity.csv`,
`ranking_gene_by_drug_slot.csv`).

## Files

| Script | Role |
|---|---|
| `calc_scores_serial.py` | Score one run → `rlipp.out`, `gene_rho.out` |
| `collect_importance.py` | Gather runs into the two importance CSVs |
| `analyze_rankings.py` | Ranking comparisons between arms |
| `verify_equivalence.py` | Check the vendored calculator |
| `rlipp_calculator.py`, `rlipp_helper.py` | Vendored upstream — do not edit |

Outputs (`results/`, the importance CSVs, `ranking_*`) are not tracked.
