# uniform_random_direct_output_SNN

**Uniform random direct-output sparse neural network** baseline: matches the experimental protocol for sparse models with **random connectivity** and **direct readout from intermediate states** (related to RSNN/RP-fNeST ablations in §2.4).

## Files

- `uniform_random_snn.py`, `uniform_random_snn_hparam_tuner.py` — model + training (if present).
- `monitor_distributed_jobs.py` — job monitoring.

Filenames use `uniform_random_snn` internally; the folder name reflects the paper description (“uniform random” + direct output SNN).

Restore training data from **`Data_archives/`** and align paths inside the Python drivers. Outputs: `results/`, `shared/` (gitignored).
