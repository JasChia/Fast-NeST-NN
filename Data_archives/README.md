# Data archives

Per-drug cell-line bundles for the Fast-NeST-NN pipelines, shipped as **`*.tar.gz`** (Git LFS). Extract at the **repository root** so paths match job files (`Data/...`):

```bash
cd /path/to/Fast-NeST-NN
tar -xzf Data_archives/D298_CL.tar.gz
```

Repeat for each archive you need. Archives unpack to **`D{ID}_CL/`** under **`Data/`** (create **`Data/`** first if it does not exist, or extract with a prefix that matches your layout).

---

## Archives present in `Data_archives/` (this repository)

These files are what you get from a normal clone (names may evolve; list the folder to confirm):

| Archive |
|---------|
| `D5_CL.tar.gz` |
| `D80_CL.tar.gz` |
| `D99_CL.tar.gz` |
| `D127_CL.tar.gz` |
| `D151_CL.tar.gz` |
| `D188_CL.tar.gz` |
| `D244_CL.tar.gz` |
| `D273_CL.tar.gz` |
| `D298_CL.tar.gz` |
| `D380_CL.tar.gz` |

Layout should match **`Data/nest_shuffle_data/CombatLog2TPM/Drug{ID}/D{ID}_CL/...`** or the flat **`Data/D{ID}_CL/...`** convention in the root **`README.md`**. Bundled training jobs under **`ArchitecturePerformanceExperiments/`** include drug **380** among the **10** IDs (see **`ArchitecturePerformanceExperiments/*/jobs/`**).

---

## Drug ID → compound name

Paper / job commands use numeric drug IDs. Common mapping (not every ID has an archive in **`Data_archives/`**—see table above):

| ID | Name |
|----|------|
| 5 | Topotecan |
| 80 | Bendamustine |
| 99 | CD-437 |
| 127 | Clofarabine |
| 151 | Doxorubicin |
| 188 | Gemcitabine |
| 244 | Adavosertib |
| 273 | Mitomycin C |
| 298 | Nelarabine |
| 380 | Camptothecin |

See the root **`README.md`** for how **`Data/`** and **`nest_shuffle_data/`** layouts relate to **`ArchitecturePerformanceExperiments/*/jobs/`** paths.
