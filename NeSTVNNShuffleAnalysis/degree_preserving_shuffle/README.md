# Degree-preserving gene swapping

A third input condition beside the **unshuffled** and **gene-permuted** NeST-VNN runs in
[`../nest_vnn/`](../nest_vnn/). Gene-to-assembly connections of the first hidden layer (the
76 assemblies with no child assemblies) are randomized by double-edge swaps within groups of
assemblies that have the same input degree, so every assembly keeps its input degree and every
gene keeps its output degree. Everything else is the unshuffled arm; only `-onto` differs.

| Arm | `-onto` | `-transcriptomic` |
|---|---|---|
| unshuffled | `red_ontology.txt` | `D{N}_GE_Data.txt` |
| gene-permuted (`shuffled`) | `red_ontology.txt` | `shuffled_gene_expression.txt` |
| degree-preserving | `ontologies/red_ontology_degshuf_exp{E}.txt` | `D{N}_GE_Data.txt` |

## Files

```
degree_preserving_shuffle/
├── make_degree_shuffled_ontology.py   # one rewired ontology per experiment (seed 42 + experiment)
├── ontologies/                        # the 50 ontologies used, red_ontology_degshuf_exp{0..49}.txt
├── rewiring_stats.py                  # how many connections the swaps move
└── degree_preserving_tables.py        # Degree-Preserving column of Table 3 and Supplementary Tables S1-S3
```

## Generate the ontologies

```bash
cd NeSTVNNShuffleAnalysis/degree_preserving_shuffle
for E in $(seq 0 49); do
  python make_degree_shuffled_ontology.py \
    -onto ../../Data/red_ontology.txt \
    -experiment $E \
    -output ontologies/red_ontology_degshuf_exp${E}.txt
done
```

Each run re-reads the file it wrote and asserts that assembly-to-assembly edge counts, assembly input
degrees and gene output degrees are unchanged. Regenerating reproduces the committed files byte for byte.

## Train

All 500 runs (10 drugs x 50 experiments), with the same settings as `../nest_vnn/nest_vnn_jobs/jobs.txt`:

```bash
cd NeSTVNNShuffleAnalysis/nest_vnn
for D in 5 80 99 127 151 188 244 273 298 380; do
  CL=../Data/nest_shuffle_data/CombatLog2TPM/Drug${D}/D${D}_CL
  for E in $(seq 0 49); do
    OUT=../degree_preserving_shuffle/nest_vnn_logs/D${D}_degshuf_exp_${E}
    python -u src/train.py \
      -onto ../degree_preserving_shuffle/ontologies/red_ontology_degshuf_exp${E}.txt \
      -gene2id ../Data/red_gene2ind.txt \
      -cell2id $CL/D${D}_cell2ind.txt \
      -transcriptomic $CL/D${D}_GE_Data.txt \
      -train $CL/train_test_splits/experiment_${E}/training_data.txt \
      -test $CL/train_test_splits/experiment_${E}/test_data.txt \
      -modeldir $OUT -std $OUT/std.txt \
      -genotype_hiddens 4 -epoch 250 -batchsize 8 -seed 42 > D${D}_degshuf_exp_${E}.log
  done
done
```

Outputs go to `degree_preserving_shuffle/nest_vnn_logs/D{drug}_degshuf_exp_{experiment}/`.
Importance scores for these runs: `../interpretability/run_importance.sh <drug> degshuf <experiment>`.

## Rewiring statistics

```bash
cd NeSTVNNShuffleAnalysis/degree_preserving_shuffle
python rewiring_stats.py
```

Prints the number of first-layer and swappable connections, the percentage of swappable and of
all gene-assembly connections that moved (mean and SD over the 50 ontologies), and the
percentage moved against the number of swap attempts per connection.

## Tables

```bash
cd NeSTVNNShuffleAnalysis
python degree_preserving_shuffle/degree_preserving_tables.py   # -> degree_preserving_shuffle/tables/
```

Pairs each degree-preserving run with the unshuffled run of the same drug and experiment and
writes the per-drug mean and SD of R², Pearson, Spearman and RMSE with paired Wilcoxon tests
(BH across drugs) to `degree_preserving_by_drug.csv`, and the across-drug test to
`degree_preserving_across_drugs.csv`.
