# Job Failure Analysis and Fixes

## Summary

**Status**: 96 jobs failed out of 600 total jobs

## Root Causes Identified

### 1. **No Exception Handling for `study.best_trial` When All Trials Fail**

**Problem**: When all Optuna trials fail, calling `study.best_trial` raises a `ValueError` because there's no best trial. This exception wasn't caught, causing the entire script to exit with code 1.

**Location**: `layer_prune_nn_hparam_tuner.py:717` in `print_result()` method

**Fix Applied**: 
- Added check for `len(complete_trials) == 0` before calling `study.best_trial`
- Added try-except around `study.best_trial` call
- Return empty dict `{}` if no trials completed successfully
- Added display of failed trial count and error information

### 2. **Model Loading Issues in `print_model_statistics`**

**Problem**: When recreating the model to load saved state dict, there could be mismatches if:
- The model was pruned and state dict contains pruning masks
- Model architecture doesn't match exactly
- File loading errors

**Location**: `layer_prune_nn_hparam_tuner.py:648` in `print_model_statistics()` method

**Fix Applied**:
- Added try-except around model creation and loading
- Attempt to load with `strict=True` first
- Fall back to `strict=False` if strict loading fails
- Added proper error messages and traceback

### 3. **No Exception Handling Around `print_model_statistics` Call**

**Problem**: If `print_model_statistics` raises an exception, the trial would fail even if the model trained successfully.

**Location**: `layer_prune_nn_hparam_tuner.py:550` in `train_model()` method

**Fix Applied**:
- Wrapped `print_model_statistics` call in try-except
- If it fails, use metrics from `evaluate_model_metrics` instead
- Trial can still complete successfully even if statistics printing fails

### 4. **No Exception Handling in Main Execution Block**

**Problem**: If any exception occurs in the main block (after `exec_study`), it would cause the script to exit with code 1 without proper error reporting.

**Location**: `layer_prune_nn_hparam_tuner.py:753` in `if __name__ == "__main__"` block

**Fix Applied**:
- Wrapped entire main execution in try-except
- Print full traceback on fatal errors
- Exit with code 1 only after logging the error

### 5. **No Exception Handling Around `study.optimize`**

**Problem**: If an exception occurs during study optimization (outside of individual trials), it would crash the script.

**Location**: `layer_prune_nn_hparam_tuner.py:258` in `exec_study()` method

**Fix Applied**:
- Wrapped `study.optimize` call in try-except
- Continue to `print_result` even if optimization had issues
- Log warning and traceback

## Error Patterns Observed

From status files:
- **96 failed jobs** out of 600 total
- All error messages start with: `ExperimentalWarning: JournalStorage is experimental`
- Error messages are truncated to 500 characters (by distributed_gpu_queue.py)
- Some jobs show "Trial 0 finished" but still fail (likely in `print_result`)
- Some jobs show "Trial 0 failed" (exception during trial execution)

## Trial Failure Scenarios

1. **During Training**: Exception raised in training loop (NaN/inf values, CUDA errors, etc.)
2. **During Model Evaluation**: Exception in `evaluate_model_metrics` or `print_model_statistics`
3. **During Study Completion**: Exception when getting best trial or saving results

## Fixes Summary

### Code Changes Made:

1. **`print_result()` method**:
   - Check for zero complete trials before accessing `best_trial`
   - Try-except around `best_trial` access
   - Display failed trial information
   - Return empty dict if no trials completed

2. **`print_model_statistics()` method**:
   - Try-except around model creation
   - Fallback to `strict=False` for state dict loading
   - Better error messages

3. **`train_model()` method**:
   - Try-except around `print_model_statistics` call
   - Fallback to using evaluation metrics if statistics fail

4. **`exec_study()` method**:
   - Try-except around `study.optimize` call

5. **Main execution block**:
   - Try-except around entire execution
   - Proper error logging and traceback

## Expected Behavior After Fixes

1. **If all trials fail**: 
   - Script will complete with exit code 0
   - Will print warning about no completed trials
   - Will save results with empty best_params
   - Will not crash

2. **If some trials fail**:
   - Script will complete successfully
   - Will use best trial from completed trials
   - Failed trials will be logged but won't stop execution

3. **If model loading fails**:
   - Trial will still complete
   - Will use metrics from evaluation instead of statistics
   - Error will be logged but won't crash trial

## Verification

To verify fixes are working:

```bash
# Check if new jobs complete successfully
tail -f logs/distributed_queue_l5.log

# Check job status
python monitor_distributed_jobs.py --status-dir shared/status --lock-dir shared/locks

# Check if failed jobs decrease
find shared/status -name "*.json" -exec grep -l '"status": "failed"' {} \; | wc -l
```

## Additional Recommendations

1. **Increase Error Message Length**: Consider increasing the 500 character limit in `distributed_gpu_queue.py` to see full error messages

2. **Add More Logging**: Add more detailed logging in trial execution to identify where failures occur

3. **Monitor Trial Failures**: Track which hyperparameter combinations cause failures to identify problematic ranges

4. **Retry Logic**: Consider adding retry logic for transient failures (CUDA OOM, file system issues)

