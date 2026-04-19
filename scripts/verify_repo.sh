#!/usr/bin/env bash
# Local/private checks only. All writes must stay under paths ignored by the root .gitignore.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARCH="$ROOT/ArchitecturePerformanceExperiments"

echo "=========================================="
echo "Fast-NeST-NN local checks"
echo "Repo: $ROOT"
echo "=========================================="

if [[ "${SKIP_COMPILE:-0}" != "1" ]]; then
  echo ""
  echo "== Python syntax: ArchitecturePerformanceExperiments *_hparam_tuner.py =="
  python -m py_compile \
    "$ARCH/FCNN/fc_nn_hparam_tuner.py" \
    "$ARCH/RSNN/r_sparse_nn_hparam_tuner.py" \
    "$ARCH/Dense-fNeST/dense_fnest_hparam_tuner.py" \
    "$ARCH/NeST-VNN/nest_vnn_hparam_tuner.py" \
    "$ARCH/fNeST-NN/fnest_nn_hparam_tuner.py" \
    "$ARCH/RP-fNeST/uniform_random_do_di_snn_hparam_tuner.py" \
    "$ARCH/ERK_SNN/ERK_SNN_hparam_tuner.py" \
    "$ARCH/GP-NN/global_prune_nn_hparam_tuner.py" \
    "$ARCH/LP-NN/layer_prune_nn_hparam_tuner.py" \
    "$ARCH/UGP-NN/relaxed_global_prune_nn_hparam_tuner.py"
  echo "OK: py_compile tuners"
fi

if [[ "${SKIP_NEST_SHUFFLE_COMPILE:-0}" != "1" ]]; then
  echo ""
  echo "== Python syntax: NeSTVNNShuffleAnalysis/nest_vnn/src (training stack) =="
  NS="$ROOT/NeSTVNNShuffleAnalysis/nest_vnn/src"
  python -m py_compile \
    "$NS/train.py" \
    "$NS/drugcell_nn.py" \
    "$NS/vnn_trainer.py" \
    "$NS/optuna_nn_trainer.py" \
    "$NS/training_data_wrapper.py" \
    "$NS/util.py" \
    "$NS/ccc_loss.py" \
    "$NS/predict.py"
  echo "OK: py_compile NeST-VNN shuffle src"
  echo ""
  echo "== Python syntax: NeSTVNNShuffleAnalysis shuffle_assignment_analysis =="
  python -m py_compile \
    "$ROOT/NeSTVNNShuffleAnalysis/nest_vnn/shuffle_assignment_analysis/analyze_shuffle_assignments.py"
  echo "OK: py_compile analyze_shuffle_assignments"
fi

# --help imports full dependency stack (torch, sklearn, …). Run only when SKIP_HELP=0
# and the conda env from environment.yml is active (or set SKIP_HELP=1 to skip).
if [[ "${SKIP_HELP:-1}" != "1" ]]; then
  echo ""
  echo "== argparse --help: each *_hparam_tuner.py (no training; stdout only) =="
  if ! python -c "import sklearn, torch" 2>/dev/null; then
    echo "SKIP: --help checks need scikit-learn + torch (activate conda env from environment.yml or set SKIP_HELP=1)."
  else
  for tuner in \
    "$ARCH/FCNN/fc_nn_hparam_tuner.py" \
    "$ARCH/RSNN/r_sparse_nn_hparam_tuner.py" \
    "$ARCH/Dense-fNeST/dense_fnest_hparam_tuner.py" \
    "$ARCH/NeST-VNN/nest_vnn_hparam_tuner.py" \
    "$ARCH/fNeST-NN/fnest_nn_hparam_tuner.py" \
    "$ARCH/RP-fNeST/uniform_random_do_di_snn_hparam_tuner.py" \
    "$ARCH/ERK_SNN/ERK_SNN_hparam_tuner.py" \
    "$ARCH/GP-NN/global_prune_nn_hparam_tuner.py" \
    "$ARCH/LP-NN/layer_prune_nn_hparam_tuner.py" \
    "$ARCH/UGP-NN/relaxed_global_prune_nn_hparam_tuner.py"
  do
    python "$tuner" --help >/dev/null
    echo "OK: $(basename "$(dirname "$tuner")")/$(basename "$tuner") --help"
  done
  fi
fi

echo ""
echo "== Import check: optional aggregation dependencies (conda env) =="
if python -c "import pandas, numpy, scipy, statsmodels; print('OK: pandas/numpy/scipy/statsmodels')" 2>/dev/null; then
  :
else
  echo "SKIP: pandas/scipy/statsmodels not importable (optional for offline aggregation; use environment.yml)."
fi

if [[ "${SKIP_FNEST_SMOKE:-0}" == "1" ]]; then
  echo "SKIP_FNEST_SMOKE=1 — skipping fNeST-NN mini tuning run."
else
  echo ""
  echo "== fNeST-NN: 2 Optuna trials, 10 epochs (writes under fNeST-NN/results/; gitignored) =="
  DRUG="${VERIFY_DRUG_ID:-298}"
  DATA_PREFIX="${VERIFY_DATA_PREFIX:-$ROOT/Data/D${DRUG}_CL}"
  if [[ ! -f "$DATA_PREFIX/D${DRUG}_cell2ind.txt" ]]; then
    echo "ERROR: Expected drug data at: $DATA_PREFIX"
    echo "Set VERIFY_DATA_PREFIX to your D\${DRUG}_CL folder or extract Data_archives (see README)."
    exit 1
  fi
  OUT="$ARCH/fNeST-NN/results/VERIFY_SMOKE/D${DRUG}_smoke"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  cd "$ARCH/fNeST-NN"
  python -u fnest_nn_hparam_tuner.py \
    -cuda 0 \
    -drug "$DRUG" \
    -train_file "$DATA_PREFIX/train_test_splits/experiment_0/true_training_data.txt" \
    -val_file "$DATA_PREFIX/train_test_splits/experiment_0/validation_data.txt" \
    -test_file "$DATA_PREFIX/train_test_splits/experiment_0/test_data.txt" \
    -cell2id "$DATA_PREFIX/D${DRUG}_cell2ind.txt" \
    -ge_data "$DATA_PREFIX/D${DRUG}_GE_Data.txt" \
    -n_trials 2 \
    -max_epochs 10 \
    -seed 0 \
    -output_dir "$OUT"
  cd "$ROOT"
  test -f "$OUT/final_results.json" || { echo "ERROR: missing $OUT/final_results.json"; exit 1; }
  test -f "$OUT/trials/trial_0/model_best.pt" || { echo "ERROR: missing model_best.pt under trial_0"; exit 1; }
  echo "OK: fNeST-NN smoke — $OUT"
fi

if [[ "${SKIP_PROFILING:-0}" == "1" ]]; then
  echo "SKIP_PROFILING=1 — skipping Profiling/"
else
  echo ""
  echo "== Profiling: CPU, 2 timing runs (output under Profiling/; gitignored) =="
  PROF_OUT="$ROOT/Profiling/VERIFY_SMOKE_results"
  rm -rf "$PROF_OUT"
  cd "$ROOT/Profiling"
  ./run_profiling.sh --cpu --num-runs 2 --nodes-per-assembly 4 --output-dir VERIFY_SMOKE_results
  test -f "$PROF_OUT/npa_4/summary_statistics.csv" || {
    echo "WARN: expected $PROF_OUT/npa_4/summary_statistics.csv — check run log."
  }
  echo "OK: Profiling smoke — $PROF_OUT"
  cd "$ROOT"
fi

echo ""
echo "Local checks finished successfully."
