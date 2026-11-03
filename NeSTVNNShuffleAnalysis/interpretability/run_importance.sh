#!/bin/bash
# Compute NeST-VNN importance (SI/RLIPP + gene rho) for ONE trained run.
#   run_importance.sh <drug> <arm: unshuffled|shuffled|degshuf> <experiment> [cpu_count]
# Run from anywhere; paths resolve from this script's location (repo-relative).
# Uses the repo's src/predict.py and the vendored idekerlab/nest_vnn rlipp_calculator.py.
# Per arm, three inputs differ -- each must be what that arm was TRAINED with:
#   arm         model dir                                             ontology (scorer only)                         expression
#   unshuffled  nest_vnn/nest_vnn_logs/D{N}_unshuffled_exp_{E}         Data/red_ontology.txt                          D{N}_GE_Data.txt
#   shuffled    nest_vnn/nest_vnn_logs/D{N}_shuffled_exp_{E}           Data/red_ontology.txt                          experiment_{E}/shuffled_gene_expression.txt
#   degshuf     degree_preserving_shuffle/nest_vnn_logs/D{N}_degshuf_exp_{E}
#                                                                     degree_preserving_shuffle/ontologies/red_ontology_degshuf_exp{E}.txt
#                                                                                                                    D{N}_GE_Data.txt
# predict.py takes no -onto (torch.load restores the pickled, rewired model); only the scorer
# needs the ontology, and for degshuf it must be that experiment's file.
# The ~1,640 per-element hidden files go to a temp dir and are deleted after scoring.
set -uo pipefail
D=$1; ARM=$2; E=$3; NCPU=${4:-2}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NV=$ROOT/NeSTVNNShuffleAnalysis/nest_vnn
I=$ROOT/NeSTVNNShuffleAnalysis/interpretability
DS=$ROOT/NeSTVNNShuffleAnalysis/degree_preserving_shuffle
CL=$ROOT/Data/nest_shuffle_data/CombatLog2TPM/Drug$D/D${D}_CL
EX=$CL/train_test_splits/experiment_$E
OUT=$I/results/D${D}_${ARM}_exp_${E}
case "$ARM" in
  unshuffled) RUN=$NV/nest_vnn_logs/D${D}_${ARM}_exp_${E}; ONTO=$ROOT/Data/red_ontology.txt; GE=$CL/D${D}_GE_Data.txt ;;
  shuffled)   RUN=$NV/nest_vnn_logs/D${D}_${ARM}_exp_${E}; ONTO=$ROOT/Data/red_ontology.txt; GE=$EX/shuffled_gene_expression.txt ;;
  degshuf)    RUN=$DS/nest_vnn_logs/D${D}_${ARM}_exp_${E}; ONTO=$DS/ontologies/red_ontology_degshuf_exp${E}.txt; GE=$CL/D${D}_GE_Data.txt ;;
  *) echo "arm must be unshuffled|shuffled|degshuf"; exit 1 ;;
esac
for f in "$RUN/model_best.pt" "$RUN/std.txt" "$ONTO" "$GE" "$EX/test_data.txt"; do
  [ -s "$f" ] || { echo "MISSING: $f"; exit 1; }
done

TMP=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rlipp_${D}_${ARM}_${E}_XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/hidden" "$OUT"

cd "$NV"
python -u src/predict.py -predict "$EX/test_data.txt" -batchsize 1000 \
  -gene2id "$ROOT/Data/red_gene2ind.txt" -cell2id "$CL/D${D}_cell2ind.txt" \
  -load "$RUN/model_best.pt" -hidden "$TMP/hidden" -result "$TMP/predict" -cuda 0 \
  -transcriptomic "$GE" -zscore_method auc -std "$RUN/std.txt" > "$TMP/predict.log" 2>&1 \
  || { echo "PREDICT FAILED D$D $ARM exp$E"; tail -5 "$TMP/predict.log"; exit 1; }

python -u "$I/calc_scores_serial.py" -hidden "$TMP/hidden" -ontology "$ONTO" \
  -test "$EX/test_data.txt" -predicted "$TMP/predict.txt" \
  -gene2idfile "$ROOT/Data/red_gene2ind.txt" -cell2idfile "$CL/D${D}_cell2ind.txt" \
  -sys_output "$OUT/rlipp.out" -gene_output "$OUT/gene_rho.out" \
  -cpu_count "$NCPU" -drug_count 0 -genotype_hiddens 4 > "$TMP/rlipp.log" 2>&1 \
  || { echo "RLIPP FAILED D$D $ARM exp$E"; tail -5 "$TMP/rlipp.log"; exit 1; }
cp "$TMP/predict.txt" "$OUT/predict.txt"
echo "OK D$D $ARM exp$E"
