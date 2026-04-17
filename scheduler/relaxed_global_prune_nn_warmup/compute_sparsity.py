"""
Compute sparsity level of each best model in relaxed_global_prune_nn_warmup/results.

Sparsity level = (number of nonzero connections in hidden linear layers)
                 / (total edges in self.fc_edges_by_layer)

model_best.pt has PyTorch pruning reparameterization (weight_orig + weight_mask).
We compute effective_weight = weight_orig * weight_mask and count nonzeros.
"""

import torch
import numpy as np
import os
import sys
import re
from collections import defaultdict

RESULTS_DIR = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler/relaxed_global_prune_nn_warmup/results"

GENOTYPE_HIDDENS = 4

FC_EDGES_BY_LAYER = {
    0: 689 * 76 * GENOTYPE_HIDDENS,           # 209456
    1: 76 * GENOTYPE_HIDDENS * 32 * GENOTYPE_HIDDENS,  # 38912
    2: 32 * GENOTYPE_HIDDENS * 13 * GENOTYPE_HIDDENS,  # 6656
    3: 13 * GENOTYPE_HIDDENS * 4 * GENOTYPE_HIDDENS,   # 832
    4: 4 * GENOTYPE_HIDDENS * 2 * GENOTYPE_HIDDENS,    # 128
    5: 2 * GENOTYPE_HIDDENS * 2 * GENOTYPE_HIDDENS,    # 64
    6: 2 * GENOTYPE_HIDDENS * 1 * GENOTYPE_HIDDENS,    # 32
    7: 1 * GENOTYPE_HIDDENS * 1 * GENOTYPE_HIDDENS,    # 16
}
TOTAL_FC_EDGES = sum(FC_EDGES_BY_LAYER.values())

NODES_PER_LAYER = {
    0: 689,
    1: 76 * GENOTYPE_HIDDENS,
    2: 32 * GENOTYPE_HIDDENS,
    3: 13 * GENOTYPE_HIDDENS,
    4: 4 * GENOTYPE_HIDDENS,
    5: 2 * GENOTYPE_HIDDENS,
    6: 2 * GENOTYPE_HIDDENS,
    7: 1 * GENOTYPE_HIDDENS,
    8: 1 * GENOTYPE_HIDDENS,
}

EXPECTED_SHAPES = []
for i in range(len(NODES_PER_LAYER) - 1):
    EXPECTED_SHAPES.append((NODES_PER_LAYER[i+1], NODES_PER_LAYER[i]))
OUTPUT_LAYER_SHAPE = (1, NODES_PER_LAYER[8])


def extract_nn_index(key):
    m = re.match(r'NN\.(\d+)\.', key)
    return int(m.group(1)) if m else None


def compute_sparsity_from_state_dict(state_dict):
    """
    Given a state_dict from model_best.pt, compute sparsity.
    Handles both pruned (weight_orig + weight_mask) and unpruned (weight) formats.
    """
    layer_weights = {}

    for key in state_dict.keys():
        idx = extract_nn_index(key)
        if idx is None:
            continue

        if key.endswith('.weight_orig'):
            tensor = state_dict[key]
            if tensor.dim() == 2:
                mask_key = 'NN.{}.weight_mask'.format(idx)
                if mask_key in state_dict:
                    effective = tensor * state_dict[mask_key]
                else:
                    effective = tensor
                layer_weights[idx] = {
                    'shape': tuple(tensor.shape),
                    'nonzero': int((effective != 0).sum().item()),
                    'total': tensor.numel(),
                }
        elif key.endswith('.weight') and 'NN.{}.weight_orig'.format(idx) not in state_dict:
            tensor = state_dict[key]
            if tensor.dim() == 2:
                layer_weights[idx] = {
                    'shape': tuple(tensor.shape),
                    'nonzero': int((tensor != 0).sum().item()),
                    'total': tensor.numel(),
                }

    sorted_indices = sorted(layer_weights.keys())

    if len(sorted_indices) < 9:
        raise ValueError(
            "Expected at least 9 linear layers, got {}. Indices: {}".format(
                len(sorted_indices), sorted_indices))

    hidden_indices = sorted_indices[:-1]
    output_idx = sorted_indices[-1]

    output_info = layer_weights[output_idx]
    if output_info['shape'] != OUTPUT_LAYER_SHAPE:
        raise ValueError(
            "Expected output layer shape {}, got {}".format(
                OUTPUT_LAYER_SHAPE, output_info['shape']))

    if len(hidden_indices) != 8:
        raise ValueError(
            "Expected 8 hidden linear layers, got {}".format(len(hidden_indices)))

    for i, idx in enumerate(hidden_indices):
        if layer_weights[idx]['shape'] != EXPECTED_SHAPES[i]:
            raise ValueError(
                "Layer {}: expected shape {}, got {}".format(
                    i, EXPECTED_SHAPES[i], layer_weights[idx]['shape']))

    total_nonzero = sum(layer_weights[idx]['nonzero'] for idx in hidden_indices)

    per_layer_info = {}
    for i, idx in enumerate(hidden_indices):
        info = layer_weights[idx]
        per_layer_info[i] = {
            'nn_idx': idx,
            'shape': info['shape'],
            'nonzero': info['nonzero'],
            'fc_edges': FC_EDGES_BY_LAYER[i],
            'layer_density': info['nonzero'] / FC_EDGES_BY_LAYER[i],
        }

    density = total_nonzero / TOTAL_FC_EDGES
    return total_nonzero, TOTAL_FC_EDGES, density, per_layer_info


def main():
    print("Total FC edges (genotype_hiddens={}): {}".format(GENOTYPE_HIDDENS, TOTAL_FC_EDGES))
    print("FC edges by layer: {}".format(FC_EDGES_BY_LAYER))
    print()
    sys.stdout.flush()

    drugs = sorted(
        [d for d in os.listdir(RESULTS_DIR) if d.startswith('D')],
        key=lambda x: int(x[1:])
    )

    all_sparsities = []
    per_drug_sparsities = defaultdict(list)
    errors = []
    model_count = 0

    for drug in drugs:
        drug_dir = os.path.join(RESULTS_DIR, drug)
        experiments = sorted(
            [e for e in os.listdir(drug_dir) if e.startswith(drug + '_')],
            key=lambda x: int(x.split('_')[1])
        )

        for exp in experiments:
            model_path = os.path.join(drug_dir, exp, "best_model", "model_best.pt")
            if not os.path.exists(model_path):
                errors.append("MISSING: {}".format(model_path))
                continue

            try:
                sd = torch.load(model_path, map_location='cpu')
                total_nonzero, total_fc, density, per_layer = compute_sparsity_from_state_dict(sd)
                all_sparsities.append(density)
                per_drug_sparsities[drug].append(density)
                model_count += 1
                if model_count % 50 == 0:
                    print("  Processed {} models...".format(model_count))
                    sys.stdout.flush()
            except Exception as e:
                import traceback
                errors.append("ERROR {}: {}\n{}".format(model_path, e, traceback.format_exc()))
                continue

        drug_arr = np.array(per_drug_sparsities[drug])
        if len(drug_arr) > 0:
            print("{}: n={}, mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
                drug, len(drug_arr), drug_arr.mean(), drug_arr.std(),
                drug_arr.min(), drug_arr.max()))
        else:
            print("{}: n=0 (all models failed)".format(drug))
        sys.stdout.flush()

    print("\n" + "=" * 70)

    if errors:
        print("\n{} errors encountered:".format(len(errors)))
        for e in errors[:10]:
            print("  {}".format(e))
        if len(errors) > 10:
            print("  ... and {} more".format(len(errors) - 10))
        print()

    all_arr = np.array(all_sparsities)
    print("\n=== OVERALL RESULTS (across all drugs) ===")
    print("Total models analyzed: {}".format(len(all_arr)))
    print("Mean sparsity (nonzero/fc_edges): {:.6f}".format(all_arr.mean()))
    print("Std deviation:                    {:.6f}".format(all_arr.std()))
    print("Min:                              {:.6f}".format(all_arr.min()))
    print("Max:                              {:.6f}".format(all_arr.max()))
    print("Median:                           {:.6f}".format(np.median(all_arr)))

    print("\n=== PER-DRUG SUMMARY ===")
    print("{:<8} {:>4} {:>10} {:>10} {:>10} {:>10}".format(
        'Drug', 'N', 'Mean', 'Std', 'Min', 'Max'))
    print("-" * 56)
    for drug in drugs:
        arr = np.array(per_drug_sparsities[drug])
        if len(arr) > 0:
            print("{:<8} {:>4} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(
                drug, len(arr), arr.mean(), arr.std(), arr.min(), arr.max()))

    # Detailed layer breakdown for one sample
    print("\n=== DETAILED LAYER BREAKDOWN (first model: {}/{}) ===".format(drugs[0], 
        sorted([e for e in os.listdir(os.path.join(RESULTS_DIR, drugs[0])) if e.startswith(drugs[0] + '_')],
               key=lambda x: int(x.split('_')[1]))[0]))
    first_drug = drugs[0]
    first_exp = sorted(
        [e for e in os.listdir(os.path.join(RESULTS_DIR, first_drug)) if e.startswith(first_drug + '_')],
        key=lambda x: int(x.split('_')[1])
    )[0]
    sample_path = os.path.join(RESULTS_DIR, first_drug, first_exp, "best_model", "model_best.pt")
    sd = torch.load(sample_path, map_location='cpu')
    _, _, _, per_layer = compute_sparsity_from_state_dict(sd)
    print("Model: {}".format(sample_path))
    print("{:<8} {:<16} {:>10} {:>10} {:>10}".format(
        'Layer', 'Shape', 'Nonzero', 'FC Edges', 'Density'))
    print("-" * 58)
    total_nz_check = 0
    for i in sorted(per_layer.keys()):
        info = per_layer[i]
        total_nz_check += info['nonzero']
        print("{:<8} {:<16} {:>10} {:>10} {:>10.6f}".format(
            i, str(info['shape']), info['nonzero'], info['fc_edges'], info['layer_density']))
    print("{:<8} {:<16} {:>10} {:>10} {:>10.6f}".format(
        'TOTAL', '', total_nz_check, TOTAL_FC_EDGES, total_nz_check / TOTAL_FC_EDGES))


if __name__ == "__main__":
    main()
