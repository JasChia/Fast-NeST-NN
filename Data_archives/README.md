# Data archives

Per-drug cell-line bundles for the Fast-NeST-NN pipelines, shipped as **`*.tar.gz`** (Git LFS).

Each archive unpacks to **`D{ID}_CL/`** at its own root, so it must be extracted *into* the
per-drug directory the job files expect. Create that directory first:

```bash
cd /path/to/Fast-NeST-NN
for A in Data_archives/D*_CL.tar.gz; do
  ID=$(basename "$A" _CL.tar.gz); ID=${ID#D}
  mkdir -p "Data/nest_shuffle_data/CombatLog2TPM/Drug${ID}"
  tar -xzf "$A" -C "Data/nest_shuffle_data/CombatLog2TPM/Drug${ID}/"
done
```

That produces `Data/nest_shuffle_data/CombatLog2TPM/Drug{ID}/D{ID}_CL/...`, which is the
path every bundled job command uses. Extracting at the repository root instead gives
`<repo>/D{ID}_CL/`, which no job can find.

The ontology and gene-index files (`red_ontology.txt`, `red_gene2ind.txt`) are **not** in
these archives; they are expected directly under `Data/`.

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

These ten IDs are the reported drug set, used by every bundled job list. Models were also
trained for D57 and D201, which are filtered out before reporting and have no archive here.

---

## Drug ID → compound name

Paper and job commands use numeric drug IDs:

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
