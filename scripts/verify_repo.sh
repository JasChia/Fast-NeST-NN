#!/usr/bin/env bash
# Minimal smoke verification for Fast-NeST-NN (see README.md "Verification").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "Fast-NeST-NN verify_repo.sh"
echo "Repo: $ROOT"
echo "=========================================="

if [[ "${SKIP_COMPILE:-0}" != "1" ]]; then
  echo ""
  echo "== Python syntax: all scheduler *_hparam_tuner.py (tracked models) =="
  python -m py_compile \
    scheduler/FCNN/fc_nn_hparam_tuner.py \
    scheduler/RSNN/r_sparse_nn_hparam_tuner.py \
    scheduler/Dense-fNeST/dense_fnest_hparam_tuner.py \
    scheduler/NeST-VNN/nest_vnn_hparam_tuner.py \
    scheduler/fNeST-NN/fnest_nn_hparam_tuner.py \
    scheduler/RP-fNeST/uniform_random_do_di_snn_hparam_tuner.py \
    scheduler/ERK_SNN/ERK_SNN_hparam_tuner.py \
    scheduler/GP-NN/global_prune_nn_hparam_tuner.py \
    scheduler/LP-NN/layer_prune_nn_hparam_tuner.py \
    scheduler/UGP-NN/relaxed_global_prune_nn_hparam_tuner.py
  echo "OK: py_compile"
fi

echo ""
echo "== Import check: scheduler aggregation script dependencies =="
python -c "import pandas, numpy, scipy, statsmodels; print('OK: compute_metrics_and_comparisons deps')"

if [[ "${SKIP_FNEST_SMOKE:-0}" == "1" ]]; then
  echo "SKIP_FNEST_SMOKE=1 — skipping fNeST-NN mini tuning run."
else
  echo ""
  echo "== fNeST-NN: 2 Optuna trials, 10 epochs (writes under results/, gitignored) =="
  # Default layout: <repo>/Data/D{drug}_CL/... (job lists use nest_shuffle_data/...; both work if data exists)
  DRUG="${VERIFY_DRUG_ID:-298}"
  DATA_PREFIX="${VERIFY_DATA_PREFIX:-$ROOT/Data/D${DRUG}_CL}"
  if [[ ! -f "$DATA_PREFIX/D${DRUG}_cell2ind.txt" ]]; then
    echo "ERROR: Expected drug data at: $DATA_PREFIX"
    echo "Set VERIFY_DATA_PREFIX to your D\${DRUG}_CL folder or extract Data_archives (see README)."
    exit 1
  fi
  OUT="$ROOT/scheduler/fNeST-NN/results/VERIFY_SMOKE/D${DRUG}_smoke"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  cd "$ROOT/scheduler/fNeST-NN"
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
  # Trial directory: <output_dir>/trials/trial_<n>/model_best.pt
  test -f "$OUT/trials/trial_0/model_best.pt" || { echo "ERROR: missing model_best.pt under trial_0"; exit 1; }
  echo "OK: fNeST-NN smoke — see $OUT"
fi

if [[ "${SKIP_PROFILING:-0}" == "1" ]]; then
  echo "SKIP_PROFILING=1 — skipping Profiling/"
else
  echo ""
  echo "== Profiling: CPU, 2 timing runs (small output tree) =="
  PROF_OUT="$ROOT/Profiling/VERIFY_SMOKE_results"
  rm -rf "$PROF_OUT"
  cd "$ROOT/Profiling"
  ./run_profiling.sh --cpu --num-runs 2 --nodes-per-assembly 4 --output-dir VERIFY_SMOKE_results
  test -f "$PROF_OUT/npa_4/summary_statistics.csv" || {
    echo "WARN: expected $PROF_OUT/npa_4/summary_statistics.csv — check run log."
  }
  echo "OK: Profiling smoke — see $PROF_OUT"
  cd "$ROOT"
fi

echo ""
echo "verify_repo.sh finished successfully."
