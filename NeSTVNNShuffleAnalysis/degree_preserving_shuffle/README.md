# Degree-preserving first-hidden-layer shuffle

A third arm beside the **unshuffled** and **shuffled** NeST-VNN experiments in
[`../nest_vnn/`](../nest_vnn/).

The existing shuffled arm permutes the gene expression columns
(`shuffled_gene_expression.txt`) and leaves the ontology alone. This arm changes only the assignment:

| Arm | `-onto` | `-transcriptomic` |
|---|---|---|
| unshuffled | `red_ontology.txt` | `D{N}_GE_Data.txt` |
| shuffled | `red_ontology.txt` | `shuffled_gene_expression.txt` |
| this arm | `ontologies/red_ontology_degshuf_exp{E}.txt` | `D{N}_GE_Data.txt` |

Only first-hidden-layer assemblies are touched — those with no child assemblies, whose
input is entirely directly annotated genes (76 of 131). Gene-to-assembly edges are
randomized only within groups of equal input degree, so every assembly keeps its input
degree and every gene keeps its output degree. Everything else — the expression matrix,
`red_gene2ind.txt`, assembly-to-assembly edges, gene annotations on non-leaf assemblies —
is copied through unchanged.

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

Arguments: `-onto`, `-experiment` and `-output` are required; `-seed_base` defaults to 42
and the seed is `seed_base + experiment`. Seeding by experiment only means a given
experiment uses the same shuffle across all drugs, matching how the shuffled arm seeds its
permutation.

The 50 files are committed under `ontologies/` so published results reproduce exactly;
regenerating them is byte-identical.

## Train

Same `src/train.py`, hyperparameters, splits and transcriptomic matrix as the existing
arms — only `-onto` differs. One command per `(drug, experiment)`; no scheduler script is
included.

```bash
cd NeSTVNNShuffleAnalysis/nest_vnn
D=127; E=0
CL=../Data/nest_shuffle_data/CombatLog2TPM/Drug${D}/D${D}_CL
OUT=../degree_preserving_shuffle/nest_vnn_logs/D${D}_degshuf_exp_${E}
mkdir -p "$OUT"
python -u src/train.py \
  -onto ../degree_preserving_shuffle/ontologies/red_ontology_degshuf_exp${E}.txt \
  -gene2id ../Data/red_gene2ind.txt \
  -cell2id $CL/D${D}_cell2ind.txt \
  -transcriptomic $CL/D${D}_GE_Data.txt \
  -train $CL/train_test_splits/experiment_${E}/training_data.txt \
  -test  $CL/train_test_splits/experiment_${E}/test_data.txt \
  -modeldir "$OUT" \
  -std "$OUT/std.txt" \
  -genotype_hiddens 4 -epoch 250 -batchsize 8 -seed 42
rc=$?        # capture before anything else; see below
```

Scale: the reported 10-drug set `[5, 80, 99, 127, 151, 188, 244, 273, 298, 380]` × 50
experiments = 500 runs at 250 epochs each. Outputs land in
`nest_vnn_logs/D{drug}_degshuf_exp_{experiment}/` and are not tracked.

### Two gotchas

**Pass `-std` explicitly.** It defaults to the *relative* `MODEL/std.txt` (`train.py:29`)
and `jobs.txt` never overrides it, while `train.py` only creates the `-modeldir` parent.
With a real `-modeldir`, nothing creates `MODEL/`, so `load_train_data`'s
`std_df.to_csv()` (`util.py:139`) raises
`OSError: Cannot save file into a non-existent directory: 'MODEL'`. This happens while
building `TrainingDataWrapper`, before the first epoch, so the run dies after ~2 min of
loading — but the preceding ontology output ("There are 131 terms") makes it look like a
healthy start. Pinning `-std` inside the run's own `-modeldir` also avoids a real race:
`load_train_data` writes that file and `load_pred_data` reads it, so concurrent runs
sharing a path clobber each other's standardization.

**Capture `$?` first.** `echo "$(date) exit=$?"` always reports 0 — the command
substitution resets `$?`. Put `rc=$?` on the line right after `python`, and check that
`model_best.pt` exists and is non-empty.

## How degrees are preserved

`shuffle_group()` uses double-edge swaps: take edges `(a1, g1)` and `(a2, g2)`, replace
them with `(a1, g2)` and `(a2, g1)`. Each swap moves one gene off and one gene onto both
assemblies, so degrees are invariant by construction rather than by correction afterwards.
Swaps that would duplicate an existing edge are rejected, keeping the graph simple. Degree
groups of size 1 have nothing to swap with and are left as-is.

The script then re-reads what it wrote, asserts all three degree invariants
(assembly→assembly edges, assembly input degree, gene output degree), and prints how many
first-layer edges actually moved, so a no-op shuffle can't pass unnoticed.

## Files

```
degree_preserving_shuffle/
├── make_degree_shuffled_ontology.py
└── ontologies/red_ontology_degshuf_exp{0..49}.txt
```
