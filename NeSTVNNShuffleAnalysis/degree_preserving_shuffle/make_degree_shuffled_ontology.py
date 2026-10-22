#!/usr/bin/env python3
"""
Build a degree-preserving, first-hidden-layer-only shuffled ontology.

The existing shuffled arm permutes the gene expression COLUMNS
(``shuffled_gene_expression.txt``) and leaves the ontology untouched, which
relabels every gene at every level of the hierarchy.  This variant does the
opposite and is deliberately narrower:

  * only genes and assemblies in the FIRST HIDDEN LAYER are touched
    (first hidden layer = assemblies with no child assemblies, i.e. the leaves
    of the assembly DAG, whose entire input is directly annotated genes);
  * assemblies are grouped by input connection degree, and connections are
    randomized only WITHIN a group of equal-degree assemblies;
  * input degree of every assembly and output degree of every gene are
    unchanged, by construction (see ``shuffle_group``).

Everything else -- the gene expression matrix, gene2ind, the assembly-to-assembly
edges, and direct gene annotations on non-leaf assemblies -- is copied through
byte-for-byte, so the only thing that differs from the unshuffled arm is which
first-layer assembly each gene feeds.

Ontology format (tab separated, as in ``red_ontology.txt``):
    <parent>\t<child>\tdefault     assembly -> assembly
    <assembly>\t<gene>\tgene       assembly -> directly annotated gene

Usage:
    python make_degree_shuffled_ontology.py -onto ../../Data/red_ontology.txt \\
        -experiment 0 -output red_ontology_degshuf_exp0.txt
"""

import argparse
import random
from collections import defaultdict


def load_ontology_edges(onto_file):
    """Read the ontology into assembly->assembly and assembly->gene edge maps."""
    child_map = defaultdict(list)
    gene_map = defaultdict(list)
    order = []
    for line in open(onto_file):
        parent, child, relation = line.rstrip("\n").split("\t")
        order.append((parent, child, relation))
        if relation == "default":
            child_map[parent].append(child)
        else:
            gene_map[parent].append(child)
    return child_map, gene_map, order


def first_hidden_layer(child_map, gene_map):
    """Assemblies whose input is only genes: no child assemblies, >=1 gene."""
    assemblies = set(child_map) | set(gene_map)
    for children in child_map.values():
        assemblies.update(children)
    return sorted(a for a in assemblies if not child_map[a] and gene_map[a])


def shuffle_group(assemblies, gene_map, rng, rounds=20):
    """Randomize gene->assembly edges within one equal-degree group.

    Uses double-edge swaps: pick two edges (a1, g1) and (a2, g2) and replace
    them with (a1, g2) and (a2, g1).  A swap moves one gene off each assembly
    and one gene onto each, so every assembly keeps its input degree and every
    gene keeps its output degree.  Swaps that would duplicate an existing edge
    are rejected, so the result stays a simple graph.
    """
    edges = [(a, g) for a in assemblies for g in gene_map[a]]
    sets = {a: set(gene_map[a]) for a in assemblies}
    if len(edges) < 2:
        return sets
    for _ in range(rounds * len(edges)):
        i, j = rng.randrange(len(edges)), rng.randrange(len(edges))
        (a1, g1), (a2, g2) = edges[i], edges[j]
        if a1 == a2 or g1 == g2:
            continue
        if g2 in sets[a1] or g1 in sets[a2]:
            continue
        sets[a1].remove(g1); sets[a1].add(g2)
        sets[a2].remove(g2); sets[a2].add(g1)
        edges[i], edges[j] = (a1, g2), (a2, g1)
    return sets


def main():
    parser = argparse.ArgumentParser(description="Degree-preserving first-layer ontology shuffle")
    parser.add_argument("-onto", type=str, required=True, help="Original ontology file")
    parser.add_argument("-experiment", type=int, required=True, help="Experiment index (sets the seed)")
    parser.add_argument("-output", type=str, required=True, help="Shuffled ontology file to write")
    parser.add_argument("-seed_base", type=int, default=42, help="Seed = seed_base + experiment")
    args = parser.parse_args()

    child_map, gene_map, order = load_ontology_edges(args.onto)
    layer1 = first_hidden_layer(child_map, gene_map)

    # group first-layer assemblies by input connection degree
    by_degree = defaultdict(list)
    for a in layer1:
        by_degree[len(gene_map[a])].append(a)

    rng = random.Random(args.seed_base + args.experiment)
    shuffled = {}
    for degree in sorted(by_degree):
        group = by_degree[degree]
        if len(group) < 2:
            continue  # nothing to exchange with; left exactly as in the original
        shuffled.update(shuffle_group(group, gene_map, rng))

    # write the ontology back out, preserving original line order; only the gene
    # lines of shuffled first-layer assemblies are replaced
    emitted = set()
    with open(args.output, "w") as out:
        for parent, child, relation in order:
            if relation == "gene" and parent in shuffled:
                if parent not in emitted:
                    emitted.add(parent)
                    for gene in sorted(shuffled[parent]):
                        out.write(f"{parent}\t{gene}\tgene\n")
                continue
            out.write(f"{parent}\t{child}\t{relation}\n")

    # verification: degrees must be identical to the original, and we report how
    # much actually moved so a no-op shuffle cannot pass unnoticed
    new_child, new_gene, _ = load_ontology_edges(args.output)

    def degrees(edge_map):
        return {a: len(v) for a, v in edge_map.items() if v}

    assert degrees(child_map) == degrees(new_child), "assembly->assembly edges changed"
    assert degrees(gene_map) == degrees(new_gene), "assembly input degree changed"
    old_out, new_out = defaultdict(int), defaultdict(int)
    for a, genes in gene_map.items():
        for g in genes:
            old_out[g] += 1
    for a, genes in new_gene.items():
        for g in genes:
            new_out[g] += 1
    assert old_out == new_out, "gene output degree changed"

    moved = sum(len(set(new_gene[a]) - set(gene_map[a])) for a in shuffled)
    total = sum(len(gene_map[a]) for a in shuffled)
    print(f"experiment {args.experiment}: first-layer assemblies={len(layer1)}, "
          f"shuffled={len(shuffled)} in {sum(1 for d in by_degree if len(by_degree[d]) > 1)} degree groups")
    print(f"  edges rewired: {moved}/{total} ({100.0 * moved / total:.1f}%) "
          f"of first-layer edges in shufflable groups")
    print(f"  degrees verified unchanged; wrote {args.output}")


if __name__ == "__main__":
    main()
