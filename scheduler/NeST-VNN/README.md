# NeST-VNN (Nested Systems in Tumors Visual Neural Network)

Training and scheduling code for the **original NeST-VNN** implementation (Zhao et al.), as used in *Sparsity, Structure, and Interpretability in Biologically Informed Neural Networks for Drug Response Prediction* (ICIBM 2026) for permutation tests and performance baselines.

## Paper mapping

- **Gene-order permutation** (§2.3, §3.1): compare shuffled vs unshuffled input genes; supplementary Pearson tables.
- **Performance / runtime** (§2–3): NeST-VNN vs fNeST-NN and other architectures.

## Layout

| File | Role |
|------|------|
| `nest_vnn_hparam_tuner.py` | Optuna HPO; uses `src/` for the published NeST-VNN trainer stack. |
| `jobs/nest_vnn_jobs.txt` | Bundled training commands (paths use `../../Data/...` from this folder). |
| `jobs/nest_vnn_advanced_jobs.json` | Same jobs as JSON (optional). |
| `cleanup_failed_experiment.py` | Operations helper. |

## Data

Populate **`Data/`** at the repository root from **`Data_archives/`** (see root `README.md`). Job files reference ontology and drug bundles under `../../Data/`.

## Workflow

```bash
cd scheduler/NeST-VNN
grep -v '^#' jobs/nest_vnn_jobs.txt | grep -v '^$' | bash
```

For clusters, submit each command line with Slurm or GNU Parallel instead of a sequential shell loop.

Outputs: `results/`, `logs/`, `shared/` (gitignored).
