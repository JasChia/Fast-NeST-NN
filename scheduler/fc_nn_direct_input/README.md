# fc_nn_direct_input

Exploratory / ablation stack using a **fully connected residual-style block with direct gene inputs** (naming: `fc_nn_residual_*`). Used to isolate the effect of **skip connections from input genes** compared with pathway-only models.

## Files

- `fc_nn_residual.py`, `fc_nn_residual_hparam_tuner.py` — model + Optuna (run from this directory).

Rename note: `fc_nn_residual_*` predates the folder name `fc_nn_direct_input`; behavior matches “direct input” experiments in the supplement.

Configure data paths on the command line to your **`Data/`** tree (see repository root `README.md` and `Data_archives/`). There is no bundled `jobs/` list in this folder; construct commands similarly to `scheduler/FCNN/`.
