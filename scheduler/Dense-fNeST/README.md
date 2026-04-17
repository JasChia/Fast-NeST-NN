# Dense-fNeST

**Dense Fast NeST Neural Network** — the fully dense baseline described in the paper: same depth/neuron counts and **direct gene inputs** plus **auxiliary readouts** as fNeST-NN, but without NeST-constrained sparse connectivity (Table “model characteristics”, §2.4).

## Layout

- `dense_fnest_fc_nn.py` — model definition (`FC_NN_DI_Layer_Pred` class).
- `dense_fnest_hparam_tuner.py` — Optuna training driver (run **from this directory**).
- `jobs/dense_fnest_jobs.txt` — one training command per experiment (comments start with `#`).
- `jobs/dense_fnest_advanced_jobs.json` — same jobs as JSON (optional for tooling).
- `archived_no_linear_layer/` — archived variant without the linear readout experiment.

## Data

Populate **`Data/`** at the repository root (see root `README.md` and `Data_archives/`). Paths inside the job files use `../../Data/...` relative to this folder.

## Run

```bash
cd scheduler/Dense-fNeST
# Example: run every non-comment line (use GNU Parallel, Slurm array jobs, or a simple loop)
grep -v '^#' jobs/dense_fnest_jobs.txt | grep -v '^$' | bash
```

For large sweeps, prefer a real scheduler (e.g. Slurm `sbatch` with one task per line, or `parallel -a jobs/dense_fnest_jobs.txt`). Outputs: `results/`, `logs/` (gitignored).
