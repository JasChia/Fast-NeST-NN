# fc_nn_direct_input

Exploratory / ablation stack using a **fully connected residual-style block with direct gene inputs** (naming: `fc_nn_residual_*`). Used to isolate the effect of **skip connections from input genes** compared with pathway-only models.

## Files

- `fc_nn_residual.py`, `fc_nn_residual_hparam_tuner.py` — model + Optuna (run from this directory).
- `advanced_gpu_queue.py` — optional queue driver.

Rename note: `fc_nn_residual_*` predates the folder name `fc_nn_direct_input`; behavior matches “direct input” experiments in the supplement.

Configure data paths in the tuner / queue scripts to your extracted `Data_archives/` trees.
