# Network profiling (eNest vs DrugCellNN)

This folder benchmarks **eNest** (efficient NeST) against the original **DrugCellNN** (NeST-VNN): forward-pass latency, full training-step latency (forward + loss + backward), across batch sizes and “nodes per assembly” (NPA).

Model code lives under **`helpers/`** (vendored snapshots) so profiling does not depend on other project trees on `PYTHONPATH`.

---

## Environment (recommended)

Use the Conda **`environment.yml`** at the **repository root** (see root **`README.md`**), which includes PyTorch (CUDA 11.x), NumPy/Pandas, matplotlib, networkx, scikit-learn, scipy, and tqdm—matching or exceeding **`Profiling/requirements.txt`**.

```bash
cd /path/to/Fast-NeST-NN
conda env create -f environment.yml
conda activate fast-nest-nn
```

Alternatively: **`pip install -r requirements.txt`** in **`Profiling/`**, then install PyTorch for your platform from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Quick start (from a fresh clone)

```bash
cd Fast-NeST-NN/Profiling
pip install -r requirements.txt   # skip if you used root environment.yml
# Install PyTorch: https://pytorch.org/get-started/locally/
./run_profiling.sh --cpu --num-runs 10 --nodes-per-assembly 4 --output-dir profiling_results
```

You need **`Data/red_ontology.txt`** and **`Data/red_gene2ind.txt`** at the repository root (sibling of `Profiling/`). If your layout differs, use **`--data-dir`** or **`FNEST_ONTOLOGY_DIR`** (see below).

---

### Main paper figure

The primary publication figure is:

**`<output-dir>/npa_<n>/speedup_fnest_vs_nest_vnn_bar.png`**

(for example `profiling_results/npa_4/speedup_fnest_vs_nest_vnn_bar.png`). It shows **Model Inference** and **Model Training** speedup of **fNeST-NN** over **NeST-VNN** vs batch size. Filename constant: `PAPER_SPEEDUP_BAR_FIGURE` in `profile_networks.py`.

---

## 1. Ontology data paths (portable defaults)

The profiler resolves a **data directory** in this order:

1. **`FNEST_ONTOLOGY_DIR`** or **`NEST_VNN_DATA_DIR`** (absolute path to the folder that contains the two files below)
2. **`<repository-root>/Data`** — repository root is the parent of the **`Profiling`** directory

Required files in that directory:

- `red_ontology.txt`
- `red_gene2ind.txt`

Override without editing code:

```bash
export FNEST_ONTOLOGY_DIR=/path/to/dir_with_ontology_files
python profile_networks.py ...
```

Or per run:

```bash
python profile_networks.py --data-dir /path/to/dir ...
```

---

## 2. What is in `helpers/`

| File | Role |
|------|------|
| `nest_data_paths.py` | Resolves `Data/` or env-based ontology directory |
| `eNest.py` | eNest module and `setup_Nest_Masks` |
| `drugcell_nn.py` | DrugCellNN `nn.Module` |
| `training_data_wrapper.py` | Imported by `drugcell_nn` |
| `util.py` | `load_mapping`, `create_term_mask`, etc. |

---

## 3. How to run

Run from **`Profiling/`** so `import helpers` works (the script also adds its directory to `sys.path`).

### 3.1 From repository root

```bash
cd /path/to/Fast-NeST-NN
python Profiling/profile_networks.py --num-runs 100 --min-npa 4 --max-npa 4 --output-dir Profiling/profiling_results
```

### 3.2 From `Profiling/`

```bash
cd /path/to/Fast-NeST-NN/Profiling
./run_profiling.sh --cuda 0 --num-runs 100 --nodes-per-assembly 4 --output-dir profiling_results
./run_profiling.sh ... --data-dir /custom/path/to/ontology   # optional
```

- `--nodes-per-assembly N` sets both `--min-npa` and `--max-npa` to `N`.
- `--cpu` runs on CPU (no `--cuda` passed through).

### 3.3 Full CLI (`profile_networks.py`)

| Argument | Meaning |
|----------|---------|
| `--cuda ID` | CUDA device index. Omit for **CPU**. |
| `--data-dir DIR` | Folder with `red_ontology.txt` and `red_gene2ind.txt` (overrides env default). |
| `--num-runs N` | Timed repeats per batch size. |
| `--min-batch K` / `--max-batch L` | Batch sizes `2**K` … `2**L`. |
| `--min-npa` / `--max-npa` | Inclusive NPA range; one `npa_<n>/` per value. |
| `--output-dir DIR` | Output root. |

Defaults: `--min-npa 4`, `--max-npa 4` if unspecified.

---

## 4. Expected outputs

Under **`--output-dir`**, only **`npa_<n>/`** subdirectories are created.

| File | Contents |
|------|----------|
| `eNest_raw_times.json` | Raw timings (seconds). |
| `DrugCellNN_raw_times.json` | Same for DrugCellNN. |
| `summary_statistics.csv` | Means/stds (ms), ratios. |
| **`speedup_fnest_vs_nest_vnn_bar.png`** | **Main paper figure** (paired bars). |
