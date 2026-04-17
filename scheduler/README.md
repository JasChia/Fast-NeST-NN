# Scheduler workspace

This directory holds multiple experiment stacks (for example `fc_nn/`, `og_nest_vnn/`, `r_sparse_nn/`, and others). Each subfolder typically contains Python entrypoints, shell helpers, and a **`README.md`** describing how runs are launched.

## What is in Git vs only on your machine

- **Tracked in Git:** source code, configs, documentation, and anything **outside** `results/`.
- **Not tracked (outputs):** run artifacts live under names such as **`results/`**, **`long_results/`**, **`logs/`**, and **`shared/`** inside each experiment subfolder (exact layout varies by pipeline). These trees are the bulk of disk usage (often hundreds of GB to TB **per** experiment family). They are excluded via the root **`.gitignore`** (`scheduler/**/results/`, `scheduler/**/long_results/`, `scheduler/**/logs/`, `scheduler/**/shared/`) so `git push` stays practical.

Your local or HPC checkout should keep the full `results/` directories where you already generated them; cloning this repo alone will **not** download those outputs.

## Compressing or moving a `results/` folder (optional)

To archive or copy a single experiment’s outputs without changing tracked code, compress from **inside** the parent of `results/` (example: `scheduler/og_nest_vnn/`):

```bash
cd scheduler/og_nest_vnn
tar -I 'gzip -1' -cf results.tar.gz results
```

Restore elsewhere:

```bash
tar -xzf results.tar.gz
```

For very large trees, split an **uncompressed** tar stream so each part stays under GitHub LFS’s **2 GiB per file** limit if you ever store parts in LFS or on another host:

```bash
tar -cf - results | split -b 1800m - results.tar.part-
# Restore:
cat results.tar.part-* | tar -xf -
```

## Nested Git metadata

This tree is part of the **Fast-NeST-NN** monorepo on `main`. A historical nested `.git` under `scheduler/` was removed so experiment folders are ordinary directories here, not a separate repository.
