# `Data/`

Local inputs live here and are gitignored, with two exceptions: **`red_ontology.txt`** and
**`red_gene2ind.txt`** are tracked, so the NeST graph the published models were built from
is in the repository rather than only on our disks.

## The two tracked files

| File | Size | Contents |
|---|---|---|
| `red_ontology.txt` | 28 KB, 1,471 lines | the NeST graph: 150 assembly-to-assembly edges over 131 protein assemblies, rooted at `NEST:1`, plus 1,321 gene-to-assembly edges covering 689 genes across 129 assemblies |
| `red_gene2ind.txt` | 6.7 KB, 689 lines | `index<TAB>gene_symbol`, the input ordering the models expect |

`red_ontology.txt` is tab-separated, `parent<TAB>child<TAB>type`, where `type` is `default`
for an assembly-to-assembly edge and `gene` for a gene-to-assembly edge — the format
NeST-VNN's `-onto` argument reads.

## Provenance

These are a gene-subset reduction of `sample/ontology.txt` and `sample/gene2ind.txt` from
the original NeST-VNN repository (<https://github.com/idekerlab/nest_vnn>), restricted to
the genes present in the transcriptomic matrices used here: 689 of that file's 718 genes,
with the assembly hierarchy left intact (identical 131 assemblies and 150 assembly-to-assembly
edges; 1,321 of the original 1,366 gene edges survive). NeST itself is from Zheng et al.,
*Science* 2021. Cite the original sources for the ontology; this repository only carries the
reduced copy so that results here are reproducible.

## Everything else under `Data/`

Not in Git. Restore the per-drug bundles from `../Data_archives/` (Git LFS; see
[Data_archives/README.md](../Data_archives/README.md)) into the layout the job lists expect:

    Data/nest_shuffle_data/CombatLog2TPM/Drug{drug}/D{drug}_CL/

To keep the bundles elsewhere, set `FAST_NEST_DATA_ROOT` to a directory holding the same
layout plus these two graph files (see `ArchitecturePerformanceExperiments/data_paths.py`).
