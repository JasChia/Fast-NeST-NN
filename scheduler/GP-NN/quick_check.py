"""Quick check: sample a few models from each drug to see if nonzero counts vary."""
import torch
import re
import sys
import os

RESULTS_DIR = "results"
GENOTYPE_HIDDENS = 4

NEST_EDGES = {
    0: 1321 * GENOTYPE_HIDDENS,
    1: 92 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
    2: 36 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
    3: 13 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
    4: 4 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
    5: 2 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
    6: 2 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
    7: 1 * GENOTYPE_HIDDENS * GENOTYPE_HIDDENS,
}
TOTAL_NEST_EDGES = sum(NEST_EDGES.values())
print("total_nest_edges = {}".format(TOTAL_NEST_EDGES))

FC_EDGES = {
    0: 689 * 76 * GENOTYPE_HIDDENS,
    1: 76 * GENOTYPE_HIDDENS * 32 * GENOTYPE_HIDDENS,
    2: 32 * GENOTYPE_HIDDENS * 13 * GENOTYPE_HIDDENS,
    3: 13 * GENOTYPE_HIDDENS * 4 * GENOTYPE_HIDDENS,
    4: 4 * GENOTYPE_HIDDENS * 2 * GENOTYPE_HIDDENS,
    5: 2 * GENOTYPE_HIDDENS * 2 * GENOTYPE_HIDDENS,
    6: 2 * GENOTYPE_HIDDENS * 1 * GENOTYPE_HIDDENS,
    7: 1 * GENOTYPE_HIDDENS * 1 * GENOTYPE_HIDDENS,
}
TOTAL_FC_EDGES = sum(FC_EDGES.values())
print("total_fc_edges = {}".format(TOTAL_FC_EDGES))
print()

def get_nonzero_counts(model_path):
    sd = torch.load(model_path, map_location='cpu')
    
    layer_weights = {}
    for key in sd.keys():
        m = re.match(r'NN\.(\d+)\.', key)
        if m is None:
            continue
        idx = int(m.group(1))
        
        if key.endswith('.weight_orig') and sd[key].dim() == 2:
            mask_key = 'NN.{}.weight_mask'.format(idx)
            if mask_key in sd:
                eff = sd[key] * sd[mask_key]
                mask_nz = int((sd[mask_key] != 0).sum().item())
                eff_nz = int((eff != 0).sum().item())
            else:
                mask_nz = None
                eff_nz = int((sd[key] != 0).sum().item())
            layer_weights[idx] = (tuple(sd[key].shape), mask_nz, eff_nz)
        elif key.endswith('.weight') and 'NN.{}.weight_orig'.format(idx) not in sd and sd[key].dim() == 2:
            nz = int((sd[key] != 0).sum().item())
            layer_weights[idx] = (tuple(sd[key].shape), None, nz)
    
    sorted_idx = sorted(layer_weights.keys())
    hidden_idx = sorted_idx[:-1]
    
    total_mask_nz = 0
    total_eff_nz = 0
    for idx in hidden_idx:
        shape, mask_nz, eff_nz = layer_weights[idx]
        if mask_nz is not None:
            total_mask_nz += mask_nz
        total_eff_nz += eff_nz
    
    return total_mask_nz, total_eff_nz

drugs = sorted(
    [d for d in os.listdir(RESULTS_DIR) if d.startswith('D')],
    key=lambda x: int(x[1:])
)

# Sample 3 experiments from each drug
for drug in drugs:
    drug_dir = os.path.join(RESULTS_DIR, drug)
    exps = sorted(
        [e for e in os.listdir(drug_dir) if e.startswith(drug + '_')],
        key=lambda x: int(x.split('_')[1])
    )
    samples = [exps[0], exps[len(exps)//2], exps[-1]]
    for exp in samples:
        path = os.path.join(drug_dir, exp, "best_model", "model_best.pt")
        if os.path.exists(path):
            mask_nz, eff_nz = get_nonzero_counts(path)
            print("{}/{}: mask_nonzero={}, effective_nonzero={}, density={:.6f}".format(
                drug, exp, mask_nz, eff_nz, eff_nz / TOTAL_FC_EDGES))
    sys.stdout.flush()
