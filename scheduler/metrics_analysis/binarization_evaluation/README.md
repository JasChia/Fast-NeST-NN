# AUC Binarization Scheme Evaluation for eNest_linear_fair (with Fine-tuning)

## Overview

This directory contains scripts and results for evaluating two AUC binarization schemes on the eNest_linear_fair drug response prediction model. The evaluation:
1. Loads pre-trained models
2. Fine-tunes them for up to 10 epochs on binarized labels
3. Evaluates Pearson correlation on the test set

## Motivation

Drug response prediction models typically output continuous AUC (Area Under the dose-response Curve) values. However, clinical decisions often require binary classifications (sensitive vs. resistant) or categorical classifications. This evaluation assesses how well the eNest_linear_fair model can be adapted to predict binarized drug sensitivity labels under two different binarization strategies.

## Binarization Schemes

### Scheme 1: Median-Based Binary Classification

**Method:** 
- Compute the median AUC value from the **training set** for each drug/experiment
- Apply this threshold to classify **test set** samples as:
  - **Sensitive (1):** AUC > training_median
  - **Resistant (0):** AUC ≤ training_median

**Rationale:**
- Simple, balanced split ensuring ~50% in each class (on training set)
- No samples excluded
- Threshold is computed from training data and applied to test data (realistic evaluation scenario)
- Well-suited for ranking-based evaluation

**Note on AUC interpretation:** Higher AUC values indicate greater drug effectiveness (more cell killing), thus samples with AUC > median are labeled as "sensitive" (responsive to drug).

### Scheme 2: 3-Category Classification (with Undefined Zone)

**Method:**
- Compute median (μ) and standard deviation (σ) of AUC values from the **training set** for each drug/experiment
- Apply these thresholds to classify **test set** samples as:
  - **Sensitive (1):** AUC > μ + σ (from training)
  - **Resistant (0):** AUC < μ - σ (from training)
  - **Undefined (excluded):** μ - σ ≤ AUC ≤ μ + σ

**Rationale:**
- Captures only "extreme" responders (top ~16% and bottom ~16% assuming normal distribution on training set)
- Removes ambiguous samples near the median
- May provide cleaner signal for correlation metrics
- Trade-off: smaller sample size per evaluation
- Thresholds computed from training data ensure realistic evaluation (no data leakage)

**Note:** Undefined samples are excluded from Pearson and R² calculations.

## Fine-tuning Procedure

For each experiment and binarization scheme:
1. Load the pre-trained model from `model_best.pt`
2. Binarize training and validation labels using training-set thresholds
3. Fine-tune for up to 10 epochs with early stopping (patience=3)
4. Select the model with best validation Pearson correlation
5. Evaluate on test set

**Fine-tuning parameters:**
- Learning rate: 1e-4
- Max epochs: 10
- Early stopping patience: 3
- Other hyperparameters (weight decay, L1, batch size) preserved from original training

## Metrics Computed

For both schemes, we compute **Pearson Correlation Coefficient (r)** which measures linear correlation between model predictions and binarized labels.

Metrics are computed on the **test set** using the same data folds as the original training. **Important:** Binarization thresholds (median, median ± std) are computed from the **training set** and then applied to all splits, ensuring a realistic evaluation scenario without data leakage.

## Skipping Experiments

Experiments are skipped (and counted in summary) when:
- All training or validation labels are the same (would cause NaN Pearson)
- Insufficient valid samples for categorical scheme (< 10 train or < 5 val samples)

## Data and Model Sources

### Models
- **Source:** `eNest_linear_fair/results/D{drug_id}/D{drug_id}_{exp_id}/best_model/model_best.pt`
- **Hyperparameters:** `hyperparameters.json` in same directory
- Models were trained with Optuna hyperparameter optimization (100 trials per experiment)

### Data Folds
- Data paths computed programmatically matching `eNest_linear_fair/generate_jobs.py`
- Test data: `{base_path}/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/train_test_splits/experiment_{i}/test_data.txt`
- 50 experiments per drug with different train/val/test splits
- 10 drugs total: D5, D80, D99, D127, D151, D188, D244, D273, D298, D380

### Gene Expression Features
- Cell line gene expression data from CombatLog2TPM processed files
- 689 genes mapped through the NeST ontology

## Usage

```bash
# Evaluate all drugs (default)
python evaluate_binarization_schemes.py

# Evaluate specific drugs
python evaluate_binarization_schemes.py --drugs 5,127,188

# Specify output directory and GPU
python evaluate_binarization_schemes.py --output_dir ./my_results --cuda 0

# CPU-only evaluation
python evaluate_binarization_schemes.py --cuda -1
```

## Output Files

| File | Description |
|------|-------------|
| `detailed_results.json` | Per-experiment results with all metrics |
| `drug_summaries.json` | Aggregated statistics per drug |
| `median_binarization_results.csv` | Summary table for Scheme 1 |
| `categorical_binarization_results.csv` | Summary table for Scheme 2 |
| `binarization_comparison.csv` | Side-by-side comparison of both schemes |

## Interpretation Guidelines

### Expected Behavior
- **Median scheme:** Should show moderate positive correlations since the model was trained on continuous AUC
- **Categorical scheme:** May show higher correlations (cleaner signal) but with larger variance (smaller N)

### Caveats
1. **Model trained on continuous AUC:** The eNest_linear_fair model was trained to predict continuous AUC values, not binary labels. These metrics assess transfer to binary classification.

2. **R² can be negative:** When model predictions do not align well with binary labels, R² can be negative (predictions worse than mean baseline).

3. **Sample size effects:** The categorical scheme excludes ~68% of samples (assuming normal distribution), which may increase variance in metrics.

4. **Drug-specific thresholds:** Each drug/experiment has its own median/std computed from its training set, then applied to the test set. This ensures realistic evaluation and makes cross-drug comparisons reflect both model quality and data characteristics.

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- SciPy

## References

- eNest_linear_fair model: Based on NeST (Neural Structured Learning) architecture with linear output combination
- Data: CTRP (Cancer Therapeutics Response Portal) drug response data

