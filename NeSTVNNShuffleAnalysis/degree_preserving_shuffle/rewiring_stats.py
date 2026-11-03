#!/usr/bin/env python3
"""How much the degree-preserving shuffle rewires.

Prints the gene-assembly connection counts, the share of connections that moved in
the 50 committed ontologies, and the share moved as a function of swap attempts per
connection.

Usage:
    cd NeSTVNNShuffleAnalysis/degree_preserving_shuffle
    python rewiring_stats.py
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from make_degree_shuffled_ontology import first_hidden_layer, load_ontology_edges, shuffle_group

HERE = Path(__file__).resolve().parent
ONTO = HERE.parents[1] / "Data" / "red_ontology.txt"
N_EXP = 50
SEED_BASE = 42


def moved_edges(gene_map, new_sets):
    return sum(len(set(genes) - set(gene_map[a])) for a, genes in new_sets.items())


def main():
    child_map, gene_map, _ = load_ontology_edges(ONTO)
    layer1 = first_hidden_layer(child_map, gene_map)
    by_degree = defaultdict(list)
    for a in layer1:
        by_degree[len(gene_map[a])].append(a)
    # same group order as make_degree_shuffled_ontology, so the random draws match
    groups = [by_degree[d] for d in sorted(by_degree) if len(by_degree[d]) > 1]
    swappable = [a for group in groups for a in group]

    n_all = sum(len(genes) for genes in gene_map.values())
    n_layer1 = sum(len(gene_map[a]) for a in layer1)
    n_swap = sum(len(gene_map[a]) for a in swappable)
    print(f"first-layer assemblies: {len(layer1)}, {len(layer1) - len(swappable)} with a unique degree, "
          f"{len(swappable)} swappable in {len(groups)} equal-degree groups")
    print(f"gene-assembly connections: {n_all} total, {n_layer1} first-layer, {n_swap} swappable")

    moved = []
    for e in range(N_EXP):
        _, new_gene, _ = load_ontology_edges(HERE / "ontologies" / f"red_ontology_degshuf_exp{e}.txt")
        moved.append(moved_edges(gene_map, new_gene))
    for name, n in [("swappable", n_swap), ("all", n_all)]:
        pct = 100 * np.array(moved) / n
        print(f"moved, % of {name} connections: {pct.mean():.3f} (SD {pct.std(ddof=1):.3f})")

    print("swap attempts per connection: mean % of swappable connections moved")
    for rounds in [1, 2, 3, 5, 10, 20, 50]:
        moved_r = []
        for e in range(N_EXP):
            rng = random.Random(SEED_BASE + e)
            new_sets = {}
            for group in groups:
                new_sets.update(shuffle_group(group, gene_map, rng, rounds))
            moved_r.append(moved_edges(gene_map, new_sets))
        print(f"  {rounds:>2}: {100 * np.mean(moved_r) / n_swap:.3f}")


if __name__ == "__main__":
    main()
